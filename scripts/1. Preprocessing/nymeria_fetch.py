r"""
Fetch + synthesize the NymeriaPlus dataset (1100 seqs x 15 min = ~275 h of everyday in-the-wild
motion) into AMASS-style 60fps folders that stage-2 (`2. preprocess_all_to_imuposer_at_25fps.py`)
turns into our 25fps training files.

WHY THIS DATASET: WHIP failed as training data because of DISTRIBUTION, not quality -- its dynamic
sports motion is 2.3x more pose-diverse than DIP's everyday activity, so training on it dragged the
model off DIP's manifold (clean dose-response: more WHIP => worse dip_test). Nymeria is *everyday
motion in the wild* (cooking, cleaning, working) at ~200x DIP's scale -- same distribution, more data.

WHY IT'S CHEAP: `body_processed` is a 320 MB zip of [xdata_mhr.glb (260 MB, useless to us),
xdata_smpl_neutral.npz (60 MB, what we want)]. fbcdn serves HTTP 206, so we read the zip's central
directory remotely and pull ONLY the npz member -> 5.3x less download (66 GB not 350 GB), and we
never store the raw at all (parse in memory, keep only the synthesized tensors).

WHY THERE'S NO RETARGETING: xdata_smpl_neutral.npz is *native SMPL* -- global_orient (T,3) +
body_pose (T,69) = our exact 24-joint axis-angle layout, in AMASS's z-up frame (verified: head sits
+1.573 m above the feet on z, and the standard `amass_rot` maps it to DIP's y-up). Contrast WHIP,
where fitting SMPL to 69-joint mocap was the dominant risk.

Timestamps are microseconds at a nominal 240 fps but jitter 3-6 ms, so we resample onto a UNIFORM
60 fps grid off the real timestamps rather than striding -- `_syn_acc`'s 2nd difference scales by a
hardcoded 3600 (=60^2) and would otherwise absorb the jitter as accel noise.

  uv run python "scripts/1. Preprocessing/nymeria_fetch.py" --urls /path/urls.json --gpu 0 --shard 0/2
"""
import argparse, io, os, sys, time, zipfile, urllib.request
from pathlib import Path
import numpy as np
import torch

from imuposer.config import Config
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math as M

OUT = Path(os.environ.get("IMUPOSER_OUT_DIR", "/media/vimal/T7_2TB/CHI23/processed_imuposer_data"))
# left wrist, right wrist, left thigh, right thigh, head, pelvis  (identical to the AMASS pipeline)
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
AMASS_ROT = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
SRC_FPS, DST_FPS = 240.0, 60.0
MEMBER = "body/xdata_smpl_neutral.npz"


class RangeFile(io.RawIOBase):
    """Random-access file over HTTP range requests (fbcdn serves 206)."""

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.nbytes = url, size, 0, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=180) as f:
                    data = f.read()
                break
            except Exception:
                if attempt == 4: raise
                time.sleep(2 * (attempt + 1))
        self.pos += len(data); self.nbytes += len(data)
        return data


def fetch_smpl(url, size):
    """Pull only the SMPL npz member out of the remote body_processed zip."""
    rf = RangeFile(url, size)
    zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    with zf.open(MEMBER) as f:
        buf = io.BytesIO(f.read())
    return np.load(buf), rf.nbytes


def to_uniform_60fps(pose, tran, ts_us):
    """Resample onto a uniform 60fps grid using the real (jittery) timestamps."""
    t = (ts_us - ts_us[0]) / 1e6                       # seconds
    grid = np.arange(0, t[-1], 1.0 / DST_FPS)
    lo = np.clip(np.searchsorted(t, grid, "right") - 1, 0, len(t) - 2)
    span = np.maximum(t[lo + 1] - t[lo], 1e-9)
    w = ((grid - t[lo]) / span).astype(np.float32)
    def lerp(a):
        wv = w.reshape((-1,) + (1,) * (a.ndim - 1))
        return (a[lo] * (1 - wv) + a[lo + 1] * wv).astype(np.float32)
    return lerp(pose), lerp(tran)


def _syn_acc(v):
    """Synthesize accelerations from vertex positions (verbatim from the AMASS pipeline)."""
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    return acc


def save_chunk(seqs, bm, dev, out_dir, batch=1024):
    """FK + IMU-synthesize a chunk of sequences and write the AMASS-style folder.

    Nymeria sequences are 15 min (54k frames at 60fps) and calc_mesh=True materializes all 6890
    vertices per frame, so the whole sequence at once needs ~20 GiB. FK is per-frame independent:
    we run it in frame batches, keep only the 6 sensor vertices, then `_syn_acc` over the assembled
    sequence -- so batch boundaries do not affect the 2nd difference.
    """
    vim, jim = vi_mask.to(dev), ji_mask.to(dev)
    out = {k: [] for k in ("pose", "shape", "tran", "joint", "vrot", "vacc")}
    for pose, tran, shape in seqs:
        joints, verts, grots = [], [], []
        for b in range(0, pose.shape[0], batch):
            p = M.axis_angle_to_rotation_matrix(pose[b:b + batch].to(dev)).view(-1, 24, 3, 3)
            grot, joint, vert = bm.forward_kinematics(p, shape.to(dev), tran[b:b + batch].to(dev), calc_mesh=True)
            joints.append(joint[:, :24].contiguous().cpu()); verts.append(vert[:, vim].cpu()); grots.append(grot[:, jim].cpu())
            del grot, joint, vert, p
        out["pose"].append(pose.clone()); out["tran"].append(tran.clone()); out["shape"].append(shape.clone())
        out["joint"].append(torch.cat(joints))
        out["vacc"].append(_syn_acc(torch.cat(verts)))
        out["vrot"].append(torch.cat(grots))
        torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        torch.save(v, out_dir / f"{k}.pt")
    n = sum(p.shape[0] for p in out["pose"])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="max sequences (0=all)")
    ap.add_argument("--chunk", type=int, default=10, help="sequences per output folder")
    a = ap.parse_args()

    import json
    seqs_json = json.load(open(a.urls))["sequences"]
    names = sorted(k for k, v in seqs_json.items() if "body_processed" in v)
    si, sn = (int(x) for x in a.shard.split("/"))
    names = [n for i, n in enumerate(names) if i % sn == si]
    if a.limit: names = names[:a.limit]

    dev = torch.device(f"cuda:{a.gpu}")
    cfg = Config(project_root_dir=os.getcwd(), device=a.gpu, mkdir=False)
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    amass_dir = OUT / "processed_imuposer" / "AMASS"
    print(f"[gpu{a.gpu} shard {a.shard}] {len(names)} sequences -> {amass_dir}", flush=True)

    buf, tag, dl_mb, t0 = [], None, 0, time.time()
    for i, name in enumerate(names):
        cid = i // a.chunk
        out_dir = amass_dir / f"Nymeria_{si:02d}_{cid:03d}"
        if out_dir.exists() and not buf:
            continue                                   # resumable: chunk already written
        try:
            d, nb = fetch_smpl(seqs_json[name]["body_processed"]["download_url"],
                               seqs_json[name]["body_processed"]["file_size_bytes"])
            dl_mb += nb / 1e6
            pose = np.concatenate([d["global_orient"], d["body_pose"]], 1).astype(np.float32)
            tran = d["transl"].astype(np.float32)
            beta = np.median(d["betas"], 0)[:10].astype(np.float32)
            ts = d["timestamps"].astype(np.float64)
            if not np.isfinite(pose).all() or not np.isfinite(tran).all():
                print(f"  SKIP {name}: non-finite", flush=True); continue
            pose, tran = to_uniform_60fps(pose, tran, ts)
            pose = torch.tensor(pose).view(-1, 24, 3).clone(); pose[:, 22:24] = 0     # hands off, as AMASS
            tran = torch.tensor(tran)
            # align Nymeria's z-up frame with DIP's y-up (same rotation the AMASS path uses)
            tran = AMASS_ROT.matmul(tran.unsqueeze(-1)).view_as(tran)
            pose[:, 0] = M.rotation_matrix_to_axis_angle(
                AMASS_ROT.matmul(M.axis_angle_to_rotation_matrix(pose[:, 0])))
            buf.append((pose, tran, torch.tensor(beta)))
        except Exception as e:
            print(f"  ERR {name}: {type(e).__name__} {e}", flush=True); continue

        if len(buf) == a.chunk or i == len(names) - 1:
            n = save_chunk(buf, bm, dev, out_dir)
            el = time.time() - t0
            print(f"  [{i+1}/{len(names)}] wrote {out_dir.name}: {len(buf)} seqs {n} frames "
                  f"({n/60/3600:.2f} h) | {dl_mb/1e3:.1f} GB dl, {dl_mb/el:.1f} MB/s", flush=True)
            buf = []
    print(f"[gpu{a.gpu} shard {a.shard}] DONE {dl_mb/1e3:.1f} GB in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
