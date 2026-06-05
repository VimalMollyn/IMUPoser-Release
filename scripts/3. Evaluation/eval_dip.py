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
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree, axis_angle_to_rotation_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]

# Test-time calibration augmentation (off by default). TTA_N copies of each input with a random
# per-sensor mounting-rotation offset are averaged -> marginalizes the unknown DIP calibration.
TTA_N = int(os.environ.get("TTA_N", "0"))
TTA_RAD = float(os.environ.get("TTA_RAD", "0.12217"))

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
    def _detect_model(sd):
        if any(k.startswith("recon_rnn.") for k in sd): return "ReconIMUPoser"
        if any(k.startswith("joint_rnn.") for k in sd): return "StagedIMUPoser"
        if any(k.startswith("net.step_mlp.") for k in sd): return "DiffusionIMUPoser"
        if any(k.startswith("net.enc.") for k in sd): return "TransformerIMUPoser"
        if any(k.startswith("net.blocks.") for k in sd): return "CNN1DIMUPoser"
        if "codebook" in sd: return "CodebookIMUPoser"
        if any(k.startswith("act_head.") for k in sd): return "ActivityIMUPoser"
        return "GlobalModelIMUPoser"

    def _build(sd):
        # auto-detect architecture per checkpoint -> heterogeneous ensembles work. strict=False tolerates
        # only the non-learned PE buffer / frozen distill teacher; real weight mismatches still fail.
        config.model = _detect_model(sd)
        m = get_model(config)
        inc = m.load_state_dict(sd, strict=False)
        bad = [k for k in list(inc.missing_keys) + list(inc.unexpected_keys)
               if all(t not in k for t in (".pe", "_div", "_teacher"))]
        assert not bad, f"state_dict mismatch beyond positional encoding: {bad}"
        return m.eval().to(dev)

    model = _build(torch.load(args.checkpoint, map_location=dev, weights_only=False)["state_dict"])
    # optional PIP/PNP-style rigid-body physics refinement of the predicted pose (test-time only;
    # the metric below is unchanged — it scores the refined pose). Physics runs on a CPU body model.
    if os.environ.get("PHYS_REFINE"):
        from imuposer.physics import PhysicsRefineWrapper
        model = PhysicsRefineWrapper(model, ParametricModel(config.og_smpl_model_path, device="cpu"))
    # ENSEMBLE: average the r6d pose of this checkpoint + ENSEMBLE_CKPTS. Members are auto-detected
    # independently, so HETEROGENEOUS ensembles (LSTM + transformer+IK + ...) work. Eval-construction only.
    if os.environ.get("ENSEMBLE_CKPTS"):
        members = [model]
        for cp in os.environ["ENSEMBLE_CKPTS"].split(","):
            members.append(_build(torch.load(cp, map_location=dev, weights_only=False)["state_dict"]))

        class _Ensemble(torch.nn.Module):
            def __init__(self, ms): super().__init__(); self.ms = ms
            def forward(self, x, lens): return sum(m(x, lens)[:, :, :144] for m in self.ms) / len(self.ms)
        model = _Ensemble(members)

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
                # Optional test-time calibration augmentation (TTA_N>0): the real sensor mounting
                # calibration is unknown, so marginalize over it — run the model on TTA_N copies of the
                # input, each with the present sensors' orientation rotated by a random calib offset
                # (same axis-angle(randn*rad) form as the training aug), and average the r6d. Eval-only;
                # the metric below is unchanged. TTA_N=0 -> exact original single forward.
                if TTA_N > 0:
                    preds = []
                    for _ in range(TTA_N):
                        co_t = co.clone()
                        for c in idx:
                            Rc = axis_angle_to_rotation_matrix((torch.randn(3) * TTA_RAD).unsqueeze(0))[0]
                            co_t[:, c] = torch.matmul(Rc, co_t[:, c])
                        inp = torch.cat([ca.reshape(n, -1), co_t.reshape(n, -1)], 1).to(dev)
                        preds.append(model(inp.unsqueeze(0), [n])[0, :, :144])
                    pred = sum(preds) / len(preds)
                else:
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
