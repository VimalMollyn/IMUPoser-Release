r"""
Re-convert the Motion-X subsets (SMPL-X 322-dim npy at 30 fps) with matrix-space rotation interpolation.

The original pipeline (preprocess_all.process_motionx) upsampled 30 -> 60 fps by lerping axis-angle poses,
which corrupts frames across the +-pi wrap (0.4 % of frames with |acc| > 50 m/s^2). Same data, same frame
handling (AMASS z-up -> DIP y-up), real betas, full-mesh synthesis; only the interpolation differs.
Writes MotionX_<subset>.pt into the NVMe data dir (replacing the symlinks to the old T7 files).

  uv run python "scripts/1. Preprocessing/motionx_reconvert.py" --gpu 1
"""
import argparse, io, os, time, zipfile
from pathlib import Path
import numpy as np
import torch

from imuposer.config import Config
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.datasets.synth_imu import synthesize_sequences, amass_dir_to_25fps, resample_linear, resample_pose_aa, zup_to_dip

RAW = Path("/media/vimal/T7_2TB/CHI23/data/motion/motion_generation/smplx322")
OUT = Path(os.environ.get("IMUPOSER_OUT_DIR", "/home/vimal/imuposer_data")) / "processed_imuposer_25fps"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="1"); ap.add_argument("--subsets", default="")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(f"cuda:{a.gpu}")
    cfg = Config(project_root_dir=str(Path(__file__).resolve().parents[2]), device=a.gpu, mkdir=False)
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    zips = sorted(RAW.glob("*.zip"))
    if a.subsets: zips = [z for z in zips if z.stem in a.subsets.split(",")]
    print(f"{len(zips)} subsets: {[z.stem for z in zips]}", flush=True)
    for zp in zips:
        name = f"MotionX_{zp.stem}"; out = OUT / f"{name}.pt"; t0 = time.time()
        seqs = []
        with zipfile.ZipFile(zp) as zf:
            for m in [n for n in zf.namelist() if n.endswith(".npy")]:
                try: arr = np.load(io.BytesIO(zf.read(m)))
                except Exception: continue
                if arr.ndim != 2 or arr.shape[1] != 322 or arr.shape[0] < 30: continue
                pose = torch.from_numpy(arr[:, :66].astype(np.float32)).view(-1, 22, 3)
                pose = torch.cat([pose, torch.zeros(pose.shape[0], 2, 3)], 1)
                tran = torch.from_numpy(arr[:, 309:312].astype(np.float32))
                beta = torch.from_numpy(arr[0, 312:322].astype(np.float32))
                pose, tran = zup_to_dip(pose, tran)
                seqs.append((resample_pose_aa(pose, 30.0, 60.0), resample_linear(tran, 30.0, 60.0), beta))
        res = synthesize_sequences(seqs, bm, dev, fast=False)             # real betas -> full-mesh path
        fdata = amass_dir_to_25fps(res)
        if out.is_symlink() or out.exists(): out.unlink()
        torch.save(fdata, out.with_suffix(".pt.tmp")); os.replace(out.with_suffix(".pt.tmp"), out)
        h = sum(x.shape[0] for x in fdata["pose"]) / 25 / 3600
        print(f"DONE {name}: {len(seqs)} clips, {h:.1f} h in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
