r"""
MotionMillion (272-dim MotionStreamer representation, 30 fps) -> SMPL -> synthetic IMU -> 25 fps training files.

WHAT THE 272 DIMS ARE (MotionStreamer / "272-dim-Motion-Representation", verified against their code):
  [0:2]      root xz velocity in the heading-free frame
  [2:8]      per-frame heading CHANGE as a 6D rotation (accumulate to get the yaw)
  [8:74]     22 joint positions, heading-free, root at xz origin  (positions[0,1] = root height)
  [74:140]   22 joint velocities
  [140:272]  22 joint LOCAL rotations as 6D (first two ROWS of the 3x3; pytorch3d convention). Joint 0 is
             the root rotation with the heading removed.
Recovery is closed-form ("recover from rotation, no IK" in their README): undo the heading on joint 0,
integrate the xz velocities for the translation, take the root height from the positions block.
Their world is already y-up (they apply the same z-up -> y-up rotation we use for AMASS), so no extra
frame change; we only subtract SMPL's rest pelvis offset so `tran` means SMPL transl (as in AMASS).

Subsets: everything under motion_272rpr except MotionGV (video-estimated, skipped for now) and the
Motion-X subsets that already exist in the pipeline as MotionX_* (animation, haa500, humman, idea400,
kungfu, music, perform): those are re-used from the SMPL-X originals instead of this lossy copy.
Tarballs are streamed one member at a time (never fully extracted). 30 -> 60 fps by linear resample
(same as the Motion-X path), IMU synthesis at 60 fps, then the standard 25 fps conversion.

  uv run python "scripts/1. Preprocessing/motionmillion_convert.py" --gpu 1 [--subsets finedance,fit3d] [--chunk_hours 3]
"""
import argparse, io, os, sys, tarfile, time
from pathlib import Path
import numpy as np
import torch

from imuposer.config import Config
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math as M
from imuposer.datasets.synth_imu import synthesize_sequences, amass_dir_to_25fps, resample_linear

MM = Path(os.environ.get("MOTIONMILLION_DIR", "/home/vimal/Downloads/MotionMillion/motion_272rpr"))
OUT = Path(os.environ.get("IMUPOSER_OUT_DIR", "/home/vimal/imuposer_data"))
SKIP = {"animation", "haa500", "humman", "idea400", "kungfu", "music", "perform"}   # already in pipeline as MotionX_*
NJ = 22
SRC_FPS = 30.0


# ---- pytorch3d-convention helpers (rows) -----------------------------------------------------
def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def recover_smpl(x, dev=None):
    """x: (T,272) float -> pose_aa (T,24,3) torch, root_pos (T,3) torch (y-up, root JOINT position).
    Rotation math runs on `dev` (GPU) when given: the matrix->axis-angle conversion is vectorized."""
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if dev is not None:
        x = x.to(dev)
    T = x.shape[0]
    rot = rotation_6d_to_matrix(x[:, 8 + 6 * NJ:8 + 12 * NJ].reshape(T, NJ, 6))          # (T,22,3,3) local
    hd = rotation_6d_to_matrix(x[:, 2:8])                                                 # (T,3,3) heading diffs
    # accumulate R_total[t] = hd[t] @ R_total[t-1]: the diffs are pure yaw rotations about y
    # ([[c,0,s],[0,1,0],[-s,0,c]]), so the product is the yaw of the summed angles -> cumsum (vectorised).
    dyaw = torch.atan2(hd[:, 0, 2], hd[:, 0, 0])
    yaw = torch.cumsum(dyaw, 0)
    c, s = torch.cos(yaw), torch.sin(yaw)
    heading = torch.zeros(T, 3, 3, device=x.device)
    heading[:, 0, 0] = c; heading[:, 0, 2] = s; heading[:, 1, 1] = 1; heading[:, 2, 0] = -s; heading[:, 2, 2] = c
    inv = heading.transpose(1, 2)
    rot[:, 0] = inv @ rot[:, 0]
    vel = torch.zeros(T, 3, device=x.device)
    vel[:, 0] = x[:, 0]; vel[:, 2] = x[:, 1]
    vel[1:] = (inv[:-1] @ vel[1:].unsqueeze(-1)).squeeze(-1)
    root = torch.cumsum(vel, 0)
    root[:, 1] = x[:, 8 + 1]                                                              # root height
    aa = M.rotation_matrix_to_axis_angle(rot.reshape(-1, 3, 3)).view(T, NJ, 3)
    aa = torch.cat([aa, torch.zeros(T, 2, 3, device=x.device)], 1)                        # hands off (24 joints)
    return aa.cpu(), root.cpu()


def iter_members(tar_path):
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".npy"):
                f = tf.extractfile(m)
                try:
                    arr = np.load(io.BytesIO(f.read()))
                except Exception as e:
                    print(f"  bad member {m.name}: {e}", flush=True); continue
                yield m.name, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--subsets", default="", help="comma list of tar stems (default: all non-GV, minus Motion-X dupes)")
    ap.add_argument("--chunk_hours", type=float, default=3.0, help="hours of 25fps motion per output .pt")
    ap.add_argument("--min_frames", type=int, default=30, help="drop clips shorter than this at 30fps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gv", action="store_true", help="convert the MotionGV folders (video-estimated mocap) as MGV_<folder>")
    ap.add_argument("--pack_and_delete", action="store_true",
                    help="after each tarball: pack its chunks into memmap shards and delete the .pt (disk-saving; the loader reads shard-only datasets)")
    a = ap.parse_args()
    torch.set_num_threads(2)          # several converters + 2 trainings share 8 cores: avoid intra-op thread thrash
    dev = torch.device(f"cuda:{a.gpu}")
    cfg = Config(project_root_dir=str(Path(__file__).resolve().parents[2]), device=a.gpu, mkdir=False)
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    with torch.no_grad():   # SMPL rest pelvis (betas=0): tran = root_joint_pos - J0
        _, j0, _ = bm.forward_kinematics(torch.eye(3, device=dev).repeat(1, 24, 1, 1), torch.zeros(10, device=dev),
                                         torch.zeros(1, 3, device=dev), calc_mesh=True)
        J0 = j0[0, 0].cpu()
    out25 = OUT / "processed_imuposer_25fps"; out25.mkdir(parents=True, exist_ok=True)

    if a.gv:
        tars = sorted(p for p in MM.rglob("*.tar.gz") if "MotionGV" in p.parts and "Mirror" not in str(p))
    else:
        tars = sorted(p for p in MM.rglob("*.tar.gz") if "MotionGV" not in p.parts and p.stem.replace(".tar", "") not in SKIP)
    if a.subsets:
        keep = set(a.subsets.split(","))
        tars = [p for p in tars if p.name.replace(".tar.gz", "") in keep]
    print(f"{len(tars)} tarballs: {[p.name for p in tars]}", flush=True)
    if a.pack_and_delete:
        from imuposer.datasets.shards import ensure_packed, shard_root_for

    for tp in tars:
        sub = tp.name.replace(".tar.gz", "").replace("_seperate", "").replace("Datav1.1", "").replace("_smpl", "")
        tag = f"{'MGV' if a.gv else 'MM'}_{sub}"
        if a.pack_and_delete and sorted(shard_root_for(out25).glob(f"{tag}_*/meta.json")) and not (out25 / f"{tag}_000.pt").exists():
            print(f"skip {tag} (shards exist)", flush=True); continue
        if (out25 / f"{tag}_000.pt").exists():
            print(f"skip {tag} (exists)", flush=True); continue
        t0 = time.time(); buf, hours, cid, nclips, nbad = [], 0.0, 0, 0, 0
        chunk_frames = a.chunk_hours * 3600 * 60

        def flush():
            nonlocal buf, cid, hours
            if not buf: return
            out = synthesize_sequences(buf, bm, dev)
            fdata = amass_dir_to_25fps(out)
            p = out25 / f"{tag}_{cid:03d}.pt"
            torch.save(fdata, p.with_suffix(".pt.tmp")); os.replace(p.with_suffix(".pt.tmp"), p)
            h = sum(x.shape[0] for x in fdata["pose"]) / 25 / 3600; hours += h
            print(f"  wrote {p.name}: {len(buf)} clips, {h:.2f} h | total {hours:.1f} h, {time.time()-t0:.0f}s", flush=True)
            buf = []; cid += 1

        for name, arr in iter_members(tp):
            if arr.ndim != 2 or arr.shape[1] != 272 or arr.shape[0] < a.min_frames or not np.isfinite(arr).all():
                nbad += 1; continue
            aa, root = recover_smpl(arr, dev)
            if not (torch.isfinite(aa).all() and torch.isfinite(root).all()):
                nbad += 1; continue
            tran = root - J0
            aa60 = resample_linear(aa, SRC_FPS, 60.0); tran60 = resample_linear(tran, SRC_FPS, 60.0)
            buf.append((aa60, tran60, torch.zeros(10)))
            nclips += 1
            if sum(x[0].shape[0] for x in buf) >= chunk_frames:
                flush()
            if a.limit and nclips >= a.limit: break
        flush()
        if a.pack_and_delete:
            root = shard_root_for(out25)
            for p in sorted(out25.glob(f"{tag}_*.pt")):
                ensure_packed(p, root, verbose=False)
                p.unlink()
            print(f"  packed {tag} chunks into {root} and removed the .pt files", flush=True)
        print(f"DONE {tag}: {nclips} clips ({nbad} skipped), {hours:.1f} h in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
