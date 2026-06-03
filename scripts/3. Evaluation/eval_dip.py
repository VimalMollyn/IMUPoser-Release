r"""
Evaluate a trained GlobalModelIMUPoser checkpoint on the DIP-IMU test set using the
official IMUPoser metrics (reimplemented to match IMUPoser's IMUPoserEvaluator):

  - errors are measured ROOT-RELATIVE: the ignored joints (root, ankles, feet,
    wrists, hands) are set to identity in both pred and target before scoring;
  - SIP error  = global-rotation error on the 4 limb-root joints [1,2,16,17] (deg);
  - Angle error = global-rotation error over all (non-ignored) joints (deg);
  - Joint / Vertex error = root-aligned position error (cm);
  - LocalAngle = local-rotation error (deg).

Per IMU combo (default: all 24 sparse combos + MEAN), best read with --combos for a
subset, e.g. --combos lw_rp_h  or  --combos global.

Usage:
  uv run python "scripts/3. Evaluation/eval_dip.py" --checkpoint <ckpt.pt> [--combos lw_rp_h]
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree

REPO_ROOT = Path(__file__).resolve().parents[2]

# joints excluded from the metrics (set to identity in pred & target); incl. root -> root-relative
IGNORED = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
SIP = [1, 2, 16, 17]  # L/R hip, L/R shoulder -> global SIP error
ALL_COMBOS = ['lw', 'rw', 'lp', 'rp', 'h', 'lw_rw', 'lw_lp', 'lw_rp', 'lw_h', 'rw_lp', 'rw_rp', 'rw_h',
              'lp_rp', 'lp_h', 'rp_h', 'lw_rw_h', 'lw_rw_lp', 'lw_rw_rp', 'lw_lp_h', 'lw_lp_rp',
              'lw_rp_h', 'rw_lp_h', 'rw_lp_rp', 'rw_rp_h']


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on DIP-IMU (official IMUPoser metrics).")
    p.add_argument("--checkpoint", required=True, help="path to a GlobalModelIMUPoser .ckpt")
    p.add_argument("--combos", default="all",
                   help="'all' (24 sparse combos + MEAN), or comma-separated combo names (e.g. lw_rp_h,global)")
    p.add_argument("--data-dir", default="/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps",
                   help="folder containing dip_test.pt")
    p.add_argument("--test-file", default="dip_test.pt")
    p.add_argument("--device", default="0")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    config = Config(model="GlobalModelIMUPoser", project_root_dir=str(REPO_ROOT),
                    joints_set=amass_combos["global"], normalize="no_translation", r6d=True,
                    loss_type="mse", use_joint_loss=True, device=args.device, mkdir=False)
    config.processed_imu_poser_25fps = data_dir
    dev = config.device
    bm = ParametricModel(config.og_smpl_model_path, device=dev)
    I3 = torch.eye(3, device=dev)

    # Infer the architecture from the checkpoint's parameter names so the right model is
    # built before loading. This only affects which nn.Module is instantiated; the metric
    # computation below (and the protected forward(...)[:, :, :144] pose contract) is
    # identical for every architecture.
    sd = torch.load(args.checkpoint, map_location=dev, weights_only=False)["state_dict"]
    if any(k.startswith("recon_rnn.") for k in sd):
        config.model = "ReconIMUPoser"
    elif any(k.startswith("joint_rnn.") for k in sd):
        config.model = "StagedIMUPoser"
    elif any(k.startswith("net.step_mlp.") for k in sd):
        config.model = "DiffusionIMUPoser"
    elif any(k.startswith("net.enc.") for k in sd):
        config.model = "TransformerIMUPoser"
    elif any(k.startswith("net.blocks.") for k in sd):
        config.model = "CNN1DIMUPoser"
    elif "codebook" in sd:
        config.model = "CodebookIMUPoser"
    elif any(k.startswith("act_head.") for k in sd):
        config.model = "ActivityIMUPoser"
    model = get_model(config)
    # strict=False tolerates ONLY the (non-learned) sinusoidal positional-encoding buffer, which is
    # computed on the fly now; assert nothing else is missing/unexpected so real weight mismatches fail.
    inc = model.load_state_dict(sd, strict=False)
    bad = [k for k in list(inc.missing_keys) + list(inc.unexpected_keys) if ".pe" not in k and "_div" not in k]
    assert not bad, f"state_dict mismatch beyond positional encoding: {bad}"
    model.eval().to(dev)
    # optional PIP/PNP-style rigid-body physics refinement of the predicted pose (test-time only;
    # the metric below is unchanged — it scores the refined pose). Physics runs on a CPU body model.
    if os.environ.get("PHYS_REFINE"):
        from imuposer.physics import PhysicsRefineWrapper
        model = PhysicsRefineWrapper(model, ParametricModel(config.og_smpl_model_path, device="cpu"))

    def metrics(pred_rot, gt_rot):
        p, t = pred_rot.clone(), gt_rot.clone()
        p[:, IGNORED] = I3
        t[:, IGNORED] = I3
        je = ve = lae = gae = sip = 0.0
        n = 0
        for s in range(0, p.shape[0], 2000):
            pp, tt = p[s:s + 2000], t[s:s + 2000]
            gp, jp, vp = bm.forward_kinematics(pp, calc_mesh=True)
            gg, jg, vg = bm.forward_kinematics(tt, calc_mesh=True)
            off = jg[:, :1] - jp[:, :1]
            m = pp.shape[0]
            je += ((jp + off - jg).norm(dim=2).mean(1) * 100).sum().item()
            ve += ((vp + off - vg).norm(dim=2).mean(1) * 100).sum().item()
            lae += radian_to_degree(angle_between(pp.reshape(-1, 3, 3), tt.reshape(-1, 3, 3)).view(m, 24)).mean(1).sum().item()
            g = radian_to_degree(angle_between(gp.reshape(-1, 3, 3), gg.reshape(-1, 3, 3)).view(m, 24))
            gae += g.mean(1).sum().item()
            sip += g[:, SIP].mean(1).sum().item()
            n += m
        return je / n, ve / n, lae / n, gae / n, sip / n

    data = torch.load(data_dir / args.test_file, weights_only=False)
    accs = [d.view(-1, 6, 3)[:, :5].float() for d in data["acc"]]
    oris = [d.view(-1, 6, 3, 3)[:, :5].float() for d in data["ori"]]
    gts = [d.view(-1, 24, 3, 3).float() for d in data["pose"]]

    combos = ALL_COMBOS if args.combos == "all" else args.combos.split(",")
    print(f"checkpoint: {args.checkpoint}")
    print(f"{'combo':10s} {'SIP':>6s} {'Angle':>6s} {'Joint(cm)':>9s} {'Vert(cm)':>8s} {'LocalAng':>8s}")
    agg = []
    with torch.no_grad():
        for cid in combos:
            idx = amass_combos[cid]
            ja = va = la = ga = sa = 0.0
            nf = 0
            for acc, ori, gt in zip(accs, oris, gts):
                n = acc.shape[0]
                ca = torch.zeros_like(acc)
                co = torch.zeros_like(ori)
                ca[:, idx] = acc[:, idx] / config.acc_scale
                co[:, idx] = ori[:, idx]
                inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
                pred = model(inp.unsqueeze(0), [n])[0, :, :144]
                je, ve, lae, gae, sip = metrics(r6d_to_rotation_matrix(pred).view(n, 24, 3, 3), gt.to(dev))
                ja += je * n; va += ve * n; la += lae * n; ga += gae * n; sa += sip * n; nf += n
            ja, va, la, ga, sa = [x / nf for x in (ja, va, la, ga, sa)]
            agg.append((sa, ga, ja, va, la))
            print(f"{cid:10s} {sa:6.2f} {ga:6.2f} {ja:9.2f} {va:8.2f} {la:8.2f}")
    if len(agg) > 1:
        m = np.array(agg).mean(0)
        print(f"{'MEAN':10s} {m[0]:6.2f} {m[1]:6.2f} {m[2]:9.2f} {m[3]:8.2f} {m[4]:8.2f}")
    print("cols: SIP(deg) Angle/globalAngle(deg) Joint(cm) Vertex(cm) LocalAngle(deg)")


if __name__ == "__main__":
    main()
