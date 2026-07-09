r"""
Package the per-clip retargeted WHIP tensors (whip_25fps/*.pt) into our DIP-style .pt files.
  whip.pt       - training clips (everything EXCEPT the held-out WHIP test)
  whip_test.pt  - held-out real WHIP benchmark: actor00 (unseen actor) + 'test_*' actions (unseen actions)

Uses WHIP's REAL calibrated orientation (the valuable sim-to-real signal), but SYNTHESISES the
acceleration from the retargeted SMPL pose (2nd-diff of the 5 sensor vertices, gravity-free global,
smoothed) -- matches the AMASS synthesis convention and, crucially, gives the HEAD an accel (WHIP's VR
head has no accelerometer). Sensor order [lw,rw,lp,rp,h,pelvis], vertices = the AMASS vi_mask.

  uv run python "scripts/1. Preprocessing/whip_package.py" --device 0
"""
import argparse, glob
from pathlib import Path
import numpy as np
import torch
from imuposer.config import Config, amass_combos
from imuposer.smpl.parametricModel import ParametricModel

SRC = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/whip_25fps")
DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
VI5 = torch.tensor([1961, 5424, 876, 4362, 411])          # lw,rw,lp,rp,h  (AMASS vi_mask[:5])
FPS = 25.0


def syn_acc(v):                                           # v (T,5,3) -> (T,5,3) m/s^2, smoothed s=5
    a = (v[2:] - 2 * v[1:-1] + v[:-2]) * (FPS ** 2)
    a = torch.cat([torch.zeros(1, *a.shape[1:]), a, torch.zeros(1, *a.shape[1:])], 0)
    pad = 2; ap = torch.cat([a[:1].repeat(pad, 1, 1), a, a[-1:].repeat(pad, 1, 1)], 0)
    return sum(ap[i:i + a.shape[0]] for i in range(5)) / 5.0


def is_test(seq, act):
    return seq.startswith("actor00_") or act.startswith("test_")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="0"); a = ap.parse_args()
    dev = torch.device(f"cuda:{a.device}")
    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=".", joints_set=amass_combos["global"], r6d=True, device=a.device, mkdir=False)
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    files = sorted(glob.glob(str(SRC / "*.pt")))
    train, test = {k: [] for k in ("acc", "ori", "pose")}, {k: [] for k in ("acc", "ori", "pose")}
    qc = []
    for f in files:
        d = torch.load(f, weights_only=False)
        pose = d["pose"].float().to(dev)                  # (T,24,3,3)
        with torch.no_grad():
            _, _, vert = bm.forward_kinematics(pose, calc_mesh=True)
            v = vert[:, VI5.to(dev)].cpu()                # (T,5,3)
        acc6 = torch.zeros(pose.shape[0], 6, 3); acc6[:, :5] = syn_acc(v)
        bucket = test if is_test(d["seq"], d["action"]) else train
        bucket["acc"].append(acc6); bucket["ori"].append(d["ori"].float()); bucket["pose"].append(d["pose"].float())
        qc.append((d["pos_err_cm"], d["ori_resid_deg"]))
    for name, data in (("whip.pt", train), ("whip_test.pt", test)):
        frames = sum(p.shape[0] for p in data["pose"])
        torch.save(data, DD / name)
        print(f"{name}: {len(data['pose'])} clips, {frames} frames ({frames/25/3600:.2f} h)")
    pes = np.array([q[0] for q in qc]); rs = np.array([q[1] for q in qc])
    print(f"QC {len(qc)} clips: pos {pes.mean():.2f}cm (p90 {np.percentile(pes,90):.2f}) | "
          f"ori resid (lw,rw,lp,rp,h) mean {rs.mean(0).round(1)} p90 {np.percentile(rs,90,axis=0).round(1)}")


if __name__ == "__main__":
    main()
