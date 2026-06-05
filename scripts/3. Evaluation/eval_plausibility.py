r"""
Plausibility evaluation for sparse-IMU pose (lw_rp_h). The standard metric rewards matching the SPECIFIC
ground-truth motion, for which the conditional MEAN is optimal — but the mean of the un-sensed limbs
(right arm, left leg for lw_rp_h) is an implausible, collapsed "average" pose. This script measures the
OTHER axis the GT-metric is blind to: how natural / on-manifold the predicted poses are.

Reports, per checkpoint, on dip_test:
  - SIP / Angle (the usual GT-match metric, for the trade-off);
  - VarRatio(un-sensed): geodesic std of predicted local rotations / std of GT, averaged over the
    un-sensed scored joints. 1.0 = natural variability; << 1 = collapsed to the mean (implausible);
  - ManifoldDist(un-sensed): mean geodesic distance (deg) from each predicted un-sensed-limb pose to
    its nearest real AMASS pose. Low = on the human-motion manifold (plausible). GT is reported as the
    reference (it IS real motion).

Usage: uv run python "scripts/3. Evaluation/eval_plausibility.py" --checkpoint <ckpt>
"""
import argparse
import os
from pathlib import Path
import torch
from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree

REPO = Path(__file__).resolve().parents[2]
# un-sensed SCORED joints for lw_rp_h: L_hip, L_knee, R_collar, R_shoulder, R_elbow
UNSENSED = [1, 4, 14, 17, 19]
SENSED = [2, 5, 16, 18, 13, 15, 12, 3, 6, 9]   # well-informed joints (feature for retrieval)
IGN = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
SIP = [1, 2, 16, 17]


def geo(A, B):
    return radian_to_degree(angle_between(A.reshape(-1, 3, 3), B.reshape(-1, 3, 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--data-dir", default="/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
    a = ap.parse_args()
    dd = Path(a.data_dir)
    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device=a.device, mkdir=False)
    cfg.processed_imu_poser_25fps = dd
    dev = cfg.device
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    I3 = torch.eye(3, device=dev)

    m = get_model(cfg)
    m.load_state_dict(torch.load(a.checkpoint, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m = m.eval().to(dev)

    # AMASS manifold DB (un-sensed local rotations), subsampled
    DB = []
    for ds in ["CMU", "KIT", "BMLmovi", "HumanEva", "SFU"]:
        f = dd / f"{ds}.pt"
        if f.exists():
            for p in torch.load(f, weights_only=False)["pose"]:
                DB.append(p.view(-1, 24, 3, 3).float()[::8, UNSENSED])
    DB = torch.cat(DB).reshape(-1, len(UNSENSED) * 9).to(dev)        # M, U*9

    data = torch.load(dd / "dip_test.pt", weights_only=False)
    idx = amass_combos["lw_rp_h"]
    P, G = [], []
    sip = ang = nf = 0.0
    with torch.no_grad():
        for acc, ori, gt in zip([d.view(-1, 6, 3)[:, :5].float() for d in data["acc"]],
                                [d.view(-1, 6, 3, 3)[:, :5].float() for d in data["ori"]],
                                [d.view(-1, 24, 3, 3).float() for d in data["pose"]]):
            n = acc.shape[0]
            ca, co = torch.zeros_like(acc), torch.zeros_like(ori)
            ca[:, idx] = acc[:, idx] / cfg.acc_scale
            co[:, idx] = ori[:, idx]
            inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
            pr = r6d_to_rotation_matrix(m(inp.unsqueeze(0), [n])[0, :, :144]).view(n, 24, 3, 3)
            # GT-metric (root-relative global angle)
            p2, t2 = pr.clone(), gt.to(dev).clone()
            p2[:, IGN] = I3
            t2[:, IGN] = I3
            gp, _, _ = bm.forward_kinematics(p2, calc_mesh=True)
            gg, _, _ = bm.forward_kinematics(t2, calc_mesh=True)
            g = geo(gp, gg).view(n, 24)
            ang += g.mean(1).sum().item()
            sip += g[:, SIP].mean(1).sum().item()
            nf += n
            P.append(pr.cpu())
            G.append(gt.cpu())
    P = torch.cat(P).to(dev)
    G = torch.cat(G).to(dev)

    # collapse: geodesic std of LOCAL rotations around the per-joint mean, un-sensed joints
    def gstd(R):                                  # R: N,J,3,3 -> per-joint std (deg)
        Rm = R.mean(0, keepdim=True)
        return geo(R, Rm.expand_as(R)).view(R.shape[0], R.shape[1]).std(0)
    sp = gstd(P[:, UNSENSED])
    sg = gstd(G[:, UNSENSED])
    var_ratio = (sp / sg).mean().item()

    # manifold distance: nearest AMASS un-sensed pose (Frobenius proxy), per frame, deg
    def mdist(R):
        q = R[:, UNSENSED].reshape(R.shape[0], -1)
        out = torch.empty(q.shape[0], device=dev)
        for s in range(0, q.shape[0], 1000):
            d = torch.cdist(q[s:s + 1000], DB)
            j = d.argmin(1)
            # geodesic distance to the matched neighbour, averaged over the un-sensed joints
            out[s:s + 1000] = geo(R[s:s + 1000, UNSENSED],
                                  DB[j].view(-1, len(UNSENSED), 3, 3)).view(-1, len(UNSENSED)).mean(1)
        return out.mean().item()

    print(f"checkpoint: {Path(a.checkpoint).parent.name}")
    print(f"  SIP {sip/nf:6.2f}   Angle {ang/nf:6.2f}   (GT-match; lower=better)")
    print(f"  VarRatio(un-sensed)   {var_ratio:5.2f}   (1.0=natural variance, <1=collapsed/implausible)")
    print(f"  ManifoldDist pred     {mdist(P):5.2f} deg   GT {mdist(G):5.2f} deg   (lower=more on-manifold)")


if __name__ == "__main__":
    main()
