"""Train a TokenHMR-style pose tokenizer (VQ-VAE, EMA-reset codebook; Dwivedi et al., CVPR 2024; vendored code in
imuposer/thirdparty/tokenhmr) on OUR poses: single frames, the 21 SMPL body joints (no root, no hands) as rotation matrices,
sampled from the packed shards of the control datasets. Loss = 6-D pose L2 + joint-position L2 (our SMPL FK, root and hands
identity) + commitment. Saves a checkpoint in the TokenHMR format ({'net', 'hparams'}) so their EncodeTokens / DecodeTokens
load it, plus a reconstruction report on the real IMUPoser poses (imuposer_all.pt), which is the manifold we care about.

  uv run python train_pose_tokenizer.py --out <ckpt.pth> [--iters 150000] [--tokens 40] [--codes 2048] [--frames 3000000]
Env: IMUPOSER_DATA_DIR (shards), IMUPOSER_25FPS_DIR (imuposer_all.pt), CUDA_VISIBLE_DEVICES.
"""
import os, sys, json, time, argparse, types
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))
from pathlib import Path
from imuposer.config import Config, amass_combos
from imuposer.math.angular import axis_angle_to_rotation_matrix, rotation_matrix_to_r6d
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.datasets.shards import ShardStore
from imuposer.thirdparty.tokenhmr.vanilla_pose_vqvae import VanillaTokenizer
from imuposer.thirdparty.tokenhmr.rotation_utils import matrix_to_rotation_6d, rotation_6d_to_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--iters", type=int, default=150000); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--tokens", type=int, default=40); ap.add_argument("--codes", type=int, default=2048); ap.add_argument("--code-dim", type=int, default=256)
ap.add_argument("--frames", type=int, default=3_000_000); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--datasets", default="CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits,Nymeria_")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"
SH = Path(os.environ.get("IMUPOSER_SHARD_DIR", os.path.join(os.environ.get("IMUPOSER_DATA_DIR", "/home/vimal/imuposer_data"), "shards_processed_imuposer_25fps")))
DD = Path(os.environ.get("IMUPOSER_25FPS_DIR", "/home/vimal/imuposer_data/processed_imuposer_25fps"))
BODY = list(range(1, 22))                                          # the 21 body joints TokenHMR tokenizes

# ---- poses: uniform subsample of frames from the control shards (axis-angle 24x3 per frame) ---------------------------
prefixes = [p for p in a.datasets.split(",") if p]
dirs = sorted(d for d in SH.iterdir() if (d / "meta.json").exists() and any(d.name == p or (p.endswith("_") and d.name.startswith(p)) for p in prefixes))
stores = [ShardStore(d) for d in dirs]
tot = sum(s.n_frames for s in stores)
per = [max(1, int(a.frames * s.n_frames / tot)) for s in stores]
print(f"[tok] {len(stores)} datasets, {tot/25/3600:.1f} h; sampling {sum(per):,} frames", flush=True)
chunks = []
for s, n in zip(stores, per):
    idx = np.sort(np.random.choice(s.n_frames, size=min(n, s.n_frames), replace=False))
    aa = s._open()["pose_aa"]
    chunks.append(torch.from_numpy(np.asarray(aa[idx], dtype=np.float32)))
POSE = torch.cat(chunks)                                            # (N, 24, 3) axis-angle
print(f"[tok] {POSE.shape[0]:,} training poses", flush=True)

# ---- real poses for the reconstruction report ------------------------------------------------------------------------
real = torch.load(DD / "imuposer_all.pt", weights_only=False)
REAL = torch.cat([p.view(-1, 24, 3, 3) for p in real["pose"]]).float()   # local rotation matrices (n, 24, 3, 3)

cfg = Config(model="GlobalModelIMUPoser", project_root_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."), joints_set=amass_combos["global"],
             normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device="0", mkdir=False)
bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
arch = types.SimpleNamespace(CODE_DIM=a.code_dim, NB_CODE=a.codes, ROT_TYPE="rot6d", QUANTIZER="ema_reset", SMPL_TYPE="smplh", DOWN_T=1, WIDTH=512, DEPTH=2,
                             DILATION_RATE=3, CB_SCALE_DOWN=2, TOKEN_SIZE_MUL=4, TOKEN_SIZE_DIV=4, NUM_TOKENS=a.tokens, NB_JOINTS=21)
model = VanillaTokenizer(arch_params=arch, input_joint_dim=6, output_joint_dim=6, mesh_inference=False, add_noise=False).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-5)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(a.iters * 0.7), int(a.iters * 0.9)], gamma=0.1)
I3 = torch.eye(3, device=dev)

def full_pose(body_R):                                                # (B,21,3,3) -> (B,24,3,3) with root + hands identity
    B = body_R.shape[0]
    R = I3.expand(B, 24, 3, 3).clone(); R[:, 1:22] = body_R
    return R

def recon_loss(body_R, out6d):
    """6-D L2 (their representation) + joint-position L2 through our FK; returns (loss, mpjpe_cm, mean angle deg)."""
    gt6 = matrix_to_rotation_6d(body_R)                               # (B,21,6)
    pred_R = rotation_6d_to_matrix(out6d.reshape(-1, 6)).view(-1, 21, 3, 3)
    l_pose = F.mse_loss(out6d, gt6)
    jg = bm.forward_kinematics(pose=full_pose(body_R).reshape(-1, 216))[1]
    jp = bm.forward_kinematics(pose=full_pose(pred_R).reshape(-1, 216))[1]
    l_jnt = F.mse_loss(jp, jg)
    with torch.no_grad():
        c = ((pred_R.transpose(-1, -2) @ body_R).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        ang = torch.rad2deg(torch.acos(c.clamp(-1, 1))).mean()
        mpjpe = (jp - jg).norm(dim=-1).mean() * 100
    return 20.0 * l_pose + 100.0 * l_jnt, mpjpe.item(), ang.item()

@torch.no_grad()
def evaluate(R24):
    model.eval(); tot_m, tot_a, n = 0.0, 0.0, 0
    for i in range(0, R24.shape[0], 2048):
        body = R24[i:i + 2048, 1:22].to(dev)
        out6d, _, _ = model(body)
        out6d = out6d.permute(0, 2, 1)                                # (B,21,6)
        _, m, ang = recon_loss(body, out6d)
        tot_m += m * body.shape[0]; tot_a += ang * body.shape[0]; n += body.shape[0]
    model.train()
    return tot_m / n, tot_a / n

t0 = time.time(); N = POSE.shape[0]
for it in range(1, a.iters + 1):
    idx = torch.randint(0, N, (a.bs,))
    body = axis_angle_to_rotation_matrix(POSE[idx][:, BODY].reshape(-1, 3).to(dev)).view(a.bs, 21, 3, 3)
    out6d, commit, perp = model(body)                                 # out6d: (B,6,21)
    loss_rec, mpjpe, ang = recon_loss(body, out6d.permute(0, 2, 1))
    loss = loss_rec + 1.0 * commit
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 2000 == 0 or it == 1:
        print(f"[tok] it {it} loss {loss.item():.4f} commit {commit.item():.4f} perplexity {float(perp):.0f} train mpjpe {mpjpe:.2f} cm angle {ang:.2f} deg  ({(time.time()-t0)/60:.1f} min)", flush=True)
    if it % 20000 == 0 or it == a.iters:
        m, ang = evaluate(REAL)
        print(f"[tok] it {it} REAL IMUPoser poses: reconstruction mpjpe {m:.2f} cm, mean joint angle error {ang:.2f} deg", flush=True)
        torch.save({"net": model.state_dict(), "hparams": types.SimpleNamespace(ARCH=arch), "iters": it, "real_mpjpe_cm": m, "real_angle_deg": ang}, a.out)
print("[tok] saved", a.out)
