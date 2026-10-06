r"""Score a file of pose sequences ({"pred": [(T,24,3,3)], "gt": [(T,24,3,3)]}) with the protocol's metrics (same code path
as offline_fit.py: root-relative SIP / MPJRE / MPJPE / MPVPE / MPJVE / jitter, end effectors ignored). Used to score poses
post-processed outside this environment (PIP / MobilePoser physics optimizer).

  uv run python eval_poses.py --poses refined.pt [--device 0]
"""
import argparse, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from imuposer.config import Config, amass_combos
from imuposer.smpl.parametricModel import ParametricModel
import offline_fit as OF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True); ap.add_argument("--device", default="0"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device=a.device, mkdir=False)
    dev = cfg.device
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    d = torch.load(a.poses, weights_only=False)
    tot = dict(sip=0, mpjre=0, mpjpe=0, mpvpe=0, n=0, mpjve=0, nv=0, jp=0, jg=0, nj=0, perjoint=torch.zeros(24))
    for pr, gt in zip(d["pred"], d["gt"]):
        m = OF.seq_metrics(pr.to(dev).float().view(-1, 24, 3, 3), gt.to(dev).float().view(-1, 24, 3, 3), bm, dev)
        for k in tot:
            tot[k] = tot[k] + m[k]
    n, nv, nj = max(tot["n"], 1), max(tot["nv"], 1), max(tot["nj"], 1)
    print(f"{a.label}  SIP {tot['sip'] / n:7.2f} deg   MPJRE {tot['mpjre'] / n:6.2f} deg   MPJPE {tot['mpjpe'] / n:5.2f} cm   MPVPE {tot['mpvpe'] / n:5.2f} cm   "
          f"MPJVE {tot['mpjve'] / nv:5.2f} cm/s   Jit {tot['jp'] / nj:5.0f} (GT {tot['jg'] / nj:3.0f})")


if __name__ == "__main__":
    main()
