r"""
Physics-IMU realism (data regen): replace the perfect-FK orientation of the TRAINING sequences with a
realistic ESKF-style orientation = true orientation + a FULL-SEQUENCE accumulated YAW drift (gravity
aids tilt, so pitch/roll stay bounded; yaw drifts as a random walk). This is the structurally-correct,
sequence-correlated version of the per-window drift I tested as an augmentation (which was iid per window).
Accel is left unchanged (already gravity-free global-frame, matching real DIP). Val/test stay clean
(symlinked). Writes a parallel data dir; train with IMUPOSER_DATA_DIR pointing at it.

  REGEN_YAW_RATE (per-sensor steady yaw-bias std, rad/s, default 0.003 ~ 0.17 deg/s -> ~7 deg over 40s)
  REGEN_YAW_RW   (additional slow random-walk on the rate, rad/s/sqrt s, default 0.0015)
"""
import os
import torch
from pathlib import Path

SRC = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
DST = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps_realistic")
DST.mkdir(parents=True, exist_ok=True)
CURATED = "CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits".split(",")
RATE = float(os.environ.get("REGEN_YAW_RATE", "0.003"))   # steady gyro yaw-bias drift rate (rad/s)
RW = float(os.environ.get("REGEN_YAW_RW", "0.0015"))      # small slow random-walk on top of the rate
DT = 1.0 / 25.0
torch.manual_seed(0)


def yaw_R(theta):                              # (T,) -> (T,3,3) rotation about world-up Y
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(-1, 3, 3)


for ds in CURATED:
    f = SRC / f"{ds}.pt"
    if not f.exists():
        continue
    d = torch.load(f, weights_only=False)
    new_ori = []
    for ori in d["ori"]:
        R = ori.view(-1, 6, 3, 3).float()
        T = R.shape[0]
        out = R.clone()
        for c in range(6):
            rate = torch.randn(1) * RATE                                # steady per-sensor yaw-bias rate
            rw = torch.cumsum(torch.randn(T) * RW * (DT ** 0.5), 0)     # small slow drift on the rate
            theta = torch.cumsum((rate + rw) * DT, 0)                   # accumulated yaw drift (T,)
            out[:, c] = torch.matmul(yaw_R(theta), R[:, c])
        new_ori.append(out.view(ori.shape))
    d["ori"] = new_ori
    torch.save(d, DST / f"{ds}.pt")
    print(f"{ds}: regenerated {len(new_ori)} sequences", flush=True)

# symlink everything else (dip_train val, dip_test/TotalCapture test, other datasets) unchanged
for x in SRC.glob("*.pt"):
    if x.stem not in CURATED and not (DST / x.name).exists():
        (DST / x.name).symlink_to(x)
print("done -> ", DST, flush=True)
