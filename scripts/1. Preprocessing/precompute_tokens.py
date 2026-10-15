"""Precompute TokenHMR pose tokens for packed shard datasets: for every frame, the 21 body joints (local rotations from pose_aa) are
encoded once by the frozen tokenizer into 160 code indices, stored as <shard dir>/tokens.i16 (int16 memmap, (n_frames, 160)).
The training set then reads them as an auxiliary target (AUX_TARGET=tokens) instead of re-encoding every step.

  uv run python precompute_tokens.py --tokenizer <tokenizer_ns.pth> [--datasets CMU,Nymeria_,...] [--chunk 2048]
Env: IMUPOSER_DATA_DIR / IMUPOSER_SHARD_DIR, CUDA_VISIBLE_DEVICES.
"""
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))
from pathlib import Path
from imuposer.math.angular import axis_angle_to_rotation_matrix
from imuposer.thirdparty.tokenhmr.vanilla_pose_vqvae import EncodeTokens
from imuposer.thirdparty.tokenhmr.rotation_utils import matrix_to_rotation_6d

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True); ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--force", action="store_true")
ap.add_argument("--datasets", default="CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits,Nymeria_,dip_train")
a = ap.parse_args()
SH = Path(os.environ.get("IMUPOSER_SHARD_DIR", os.path.join(os.environ.get("IMUPOSER_DATA_DIR", "/home/vimal/imuposer_data"), "shards_processed_imuposer_25fps")))
dev = "cuda"
enc = EncodeTokens(a.tokenizer).to(dev).eval()
with torch.no_grad():
    n_tok = int(enc(matrix_to_rotation_6d(torch.eye(3, device=dev).expand(1, 21, 3, 3).clone())).numel())
prefixes = [p for p in a.datasets.split(",") if p]
dirs = sorted(d for d in SH.iterdir() if (d / "meta.json").exists() and any(d.name == p or (p.endswith("_") and d.name.startswith(p)) for p in prefixes))
print(f"[tokens] {len(dirs)} datasets, {n_tok} tokens per frame", flush=True)
t0 = time.time(); tot = 0
for d in dirs:
    meta = json.loads((d / "meta.json").read_text()); n = int(meta["n_frames"])
    out = d / "tokens.i16"
    if out.exists() and not a.force and out.stat().st_size == n * n_tok * 2:
        tot += n; continue
    aa = np.memmap(d / "pose_aa.f32", dtype=np.float32, mode="r", shape=(n, 24, 3))
    tok = np.memmap(str(out) + ".tmp", dtype=np.int16, mode="w+", shape=(n, n_tok))
    with torch.no_grad():
        for s in range(0, n, a.chunk):
            x = torch.from_numpy(np.asarray(aa[s:s + a.chunk])).to(dev)
            R = axis_angle_to_rotation_matrix(x[:, 1:22].reshape(-1, 3)).view(-1, 21, 3, 3)
            ids = enc(matrix_to_rotation_6d(R)).view(-1, n_tok)
            tok[s:s + x.shape[0]] = ids.to(torch.int16).cpu().numpy()
    tok.flush(); del tok
    os.replace(str(out) + ".tmp", out)
    tot += n
    print(f"[tokens] {d.name}: {n:,} frames  ({tot/1e6:.1f} M done, {(time.time()-t0)/60:.1f} min)", flush=True)
print(f"[tokens] done: {tot:,} frames in {(time.time()-t0)/60:.1f} min", flush=True)
