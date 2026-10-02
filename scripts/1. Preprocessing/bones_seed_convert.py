r"""
BONES-SEED (SOMA-uniform BVH @120 fps, 71k originals + 71k mirrors) -> SMPL -> synthetic IMU -> 25 fps files.

Pipeline (all closed-form, no mesh fitting -- see imuposer/datasets/soma_retarget.py):
  stream soma_uniform.tar.gz one BVH at a time (skip the "*_M.bvh" mirrored copies)
  -> parse hierarchy + MOTION block (parallel workers), decimate 120 -> 60 fps (exact stride 2)
  -> BVH forward kinematics (Euler ZYX, cm -> m, y-up as in DIP) -> world rotation of every SOMA joint
  -> SOMA -> SMPL by copying global joint rotations through the joint map with rest-pose bone alignment
  -> SMPL FK + exact 6-vertex skinning for the IMU (ori + accel), neutral body (betas = 0)
  -> 25 fps conversion (same resample + 5-tap accel average as the AMASS pipeline)
  -> BONES_<chunk>.pt (~3 h of motion each)

  uv run python "scripts/1. Preprocessing/bones_seed_convert.py" --gpu 1 [--limit 200] [--workers 3]
"""
import argparse, io, os, re, sys, tarfile, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch

from imuposer.config import Config
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math as M
from imuposer.datasets.synth_imu import amass_dir_to_25fps
from imuposer.datasets.soma_retarget import SMPLSkeleton, SOMAtoSMPL, syn_acc, soma_x_reference, mesh_calibrated_offsets

TAR = Path(os.environ.get("BONES_SEED_TAR", "/home/vimal/Downloads/kimodo/BONES-SEED/soma_uniform.tar.gz"))
OUT = Path(os.environ.get("IMUPOSER_OUT_DIR", "/home/vimal/imuposer_data"))
SRC_FPS = 120.0


# ---- BVH ---------------------------------------------------------------------------------------
def parse_hierarchy(text):
    """-> names, parents, offsets (J,3) [BVH units], channels (list of lists), motion_start (char index)."""
    names, parents, offsets, channels = [], [], [], []
    stack = []
    pos = 0
    tok = re.compile(r"\S+")
    toks = text[:text.index("MOTION")].split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            names.append(toks[i + 1]); parents.append(stack[-1] if stack else -1)
            offsets.append(None); channels.append([])
            i += 2
        elif t == "End":            # End Site { OFFSET x y z }
            i += 2  # 'Site' '{'
            assert toks[i] == "{"; i += 1
            assert toks[i] == "OFFSET"; i += 4
            assert toks[i] == "}"; i += 1
        elif t == "{":
            stack.append(len(names) - 1); i += 1
        elif t == "}":
            stack.pop(); i += 1
        elif t == "OFFSET":
            offsets[stack[-1]] = [float(toks[i + 1]), float(toks[i + 2]), float(toks[i + 3])]; i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1]); channels[stack[-1]] = toks[i + 2:i + 2 + n]; i += 2 + n
        elif t == "HIERARCHY":
            i += 1
        else:
            raise ValueError(f"unexpected token {t!r}")
    return names, parents, np.asarray(offsets, np.float32), channels


def parse_motion(text):
    m = text.index("MOTION")
    fr = re.search(r"Frames:\s*(\d+)", text[m:]); ft = re.search(r"Frame Time:\s*([0-9.eE+-]+)", text[m:])
    nfr, dt = int(fr.group(1)), float(ft.group(1))
    body = text[m + ft.end():]
    data = np.fromstring(body, dtype=np.float32, sep=" ")
    return nfr, dt, data


def worker_parse(args):
    name, raw = args
    text = raw.decode("utf-8", "replace")
    names, parents, offsets, channels = parse_hierarchy(text)
    nch = sum(len(c) for c in channels)
    nfr, dt, data = parse_motion(text)
    if data.size != nfr * nch:
        nfr = data.size // nch
        data = data[:nfr * nch]
    return name, (names, parents, offsets, channels), dt, data.reshape(nfr, nch)


def euler_to_matrix(angles_deg, order):
    """angles (N,3) in the order given by `order` (e.g. ['Zrotation','Yrotation','Xrotation']) -> R = R1 R2 R3."""
    a = torch.deg2rad(angles_deg)
    R = None
    for k, ch in enumerate(order):
        c, s = torch.cos(a[:, k]), torch.sin(a[:, k])
        o, z = torch.ones_like(c), torch.zeros_like(c)
        if ch[0] == "X":
            m = torch.stack([o, z, z, z, c, -s, z, s, c], 1)
        elif ch[0] == "Y":
            m = torch.stack([c, z, s, z, o, z, -s, z, c], 1)
        else:
            m = torch.stack([c, -s, z, s, c, z, z, z, o], 1)
        m = m.view(-1, 3, 3)
        R = m if R is None else R @ m
    return R


def bvh_fk(data, hier, device, scale=0.01):
    """data (T,C) -> world rotations (T,J,3,3), world positions (T,J,3) in meters."""
    names, parents, offsets, channels = hier
    T = data.shape[0]
    d = torch.from_numpy(data).to(device)
    off = torch.from_numpy(offsets).to(device) * scale
    G, P = [None] * len(names), [None] * len(names)
    c0 = 0
    for j, ch in enumerate(channels):
        n = len(ch)
        vals = d[:, c0:c0 + n]; c0 += n
        pos_ch = [k for k, c in enumerate(ch) if c.endswith("position")]
        rot_ch = [k for k, c in enumerate(ch) if c.endswith("rotation")]
        R = euler_to_matrix(vals[:, rot_ch], [ch[k] for k in rot_ch]) if rot_ch else torch.eye(3, device=device).expand(T, 3, 3)
        tr = vals[:, pos_ch] * scale if pos_ch else torch.zeros(T, 3, device=device)
        p = parents[j]
        if p < 0:
            G[j] = R; P[j] = off[j] + tr
        else:
            G[j] = G[p] @ R
            P[j] = P[p] + (G[p] @ (off[j] + tr).unsqueeze(-1)).squeeze(-1)
    return torch.stack(G, 1), torch.stack(P, 1)


def rest_positions(hier, scale=0.01):
    names, parents, offsets, channels = hier
    P = np.zeros_like(offsets)
    for j in range(len(names)):
        P[j] = offsets[j] * scale + (P[parents[j]] if parents[j] >= 0 else 0)
    return P


def iter_tar(tar_path, limit=0, skip_mirror=True):
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".bvh"):
                continue
            if skip_mirror and m.name.endswith("_M.bvh"):
                continue
            yield m.name, tf.extractfile(m).read()
            n += 1
            if limit and n >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk_hours", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="BONES")
    ap.add_argument("--min_frames", type=int, default=60, help="drop clips shorter than this at 120fps")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    cfg = Config(project_root_dir=str(Path(__file__).resolve().parents[2]), device=a.gpu, mkdir=False)
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    skel = SMPLSkeleton(bm, dev)
    out25 = OUT / "processed_imuposer_25fps"; out25.mkdir(parents=True, exist_ok=True)
    done = sorted(out25.glob(f"{a.tag}_*.pt"))
    if done and not a.limit:
        print(f"{len(done)} chunks already exist for {a.tag}; delete them to redo", flush=True); return

    conv = None; hier0 = None
    buf = {k: [] for k in ("pose", "shape", "tran", "joint", "vrot", "vacc")}
    cid, hours, nclips, nskip, t0 = 0, 0.0, 0, 0, time.time()
    chunk_frames = a.chunk_hours * 3600 * 60

    def flush():
        nonlocal buf, cid, hours
        if not buf["pose"]: return
        fdata = amass_dir_to_25fps(buf)
        p = out25 / f"{a.tag}_{cid:03d}.pt"
        torch.save(fdata, p.with_suffix(".pt.tmp")); os.replace(p.with_suffix(".pt.tmp"), p)
        h = sum(x.shape[0] for x in fdata["pose"]) / 25 / 3600; hours += h
        print(f"  wrote {p.name}: {len(buf['pose'])} clips, {h:.2f} h | total {hours:.1f} h, {nclips} clips, {(time.time()-t0)/60:.1f} min", flush=True)
        buf = {k: [] for k in buf}; cid += 1

    with Pool(a.workers) as pool, torch.no_grad():
        for name, hier, dt, data in pool.imap(worker_parse, iter_tar(TAR, a.limit), chunksize=4):
            if data.shape[0] < a.min_frames or not np.isfinite(data).all():
                nskip += 1; continue
            if abs(1.0 / dt - SRC_FPS) > 1:
                print(f"  SKIP {name}: frame time {dt}", flush=True); nskip += 1; continue
            if conv is None:
                hier0 = hier
                # reference T-pose frames from SOMA-X (same rig, same joint frames -- verified on idle clips);
                # reorder to the BVH joint order. Per-joint offsets calibrated from the T-pose MESHES.
                ref_names, ref_rot, ref_pos, soma = soma_x_reference(dev)
                ri = {n: i for i, n in enumerate(ref_names)}
                missing = [n for n in hier[0] if n not in ri]
                assert not missing, f"BVH joints missing from the SOMA-X rig: {missing[:5]}"
                order = torch.tensor([ri[n] for n in hier[0]], device=dev)
                conv = SOMAtoSMPL(skel, hier[0], ref_pos[order], ref_rot[order])
                conv.off = mesh_calibrated_offsets(skel, bm, soma, dev, cfg.og_smpl_model_path)
                print(f"rig: {len(hier[0])} joints, {data.shape[1]} channels; e.g. {hier[0][:4]} ... {hier[0][-3:]}", flush=True)
            elif hier[0] != hier0[0] or not np.allclose(hier[2], hier0[2], atol=1e-4):
                print(f"  SKIP {name}: different rig", flush=True); nskip += 1; continue
            data = data[::2]                                           # 120 -> 60 fps
            G, P = bvh_fk(data, hier, dev)
            aa, trn, G2, P2 = conv(G, P)
            if not (torch.isfinite(aa).all() and torch.isfinite(trn).all()):
                nskip += 1; continue
            V = skel.sensor_vertices(G2, P2)
            buf["pose"].append(aa.cpu()); buf["tran"].append(trn.cpu()); buf["shape"].append(torch.zeros(10))
            buf["joint"].append(P2.cpu()); buf["vrot"].append(G2[:, skel.ji].cpu()); buf["vacc"].append(syn_acc(V, 60.0).cpu())
            nclips += 1
            if sum(x.shape[0] for x in buf["pose"]) >= chunk_frames:
                flush()
        flush()
    print(f"DONE {a.tag}: {nclips} clips ({nskip} skipped), {hours:.1f} h in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
