r"""
Full standard metric suite (PIP/TIP/MobilePoser convention) on DIP-test, lw_rp_h, root-relative:
  SIP   (deg)  - global-rot error on hips+shoulders [1,2,16,17]
  MPJRE (deg)  - mean per-joint global ROTATION error (= "Angle"/angular error)
  MPJPE (cm)   - mean per-joint POSITION error (root-aligned)
  MPVPE (cm)   - mean per-vertex (mesh) position error (root-aligned)
  MPJVE (cm/s) - mean per-joint VELOCITY error (Δpos·fps), root-relative
  Jitter(m/s^3)- mean predicted joint JERK magnitude (3rd time-derivative); GT jitter shown as reference
Velocity/jitter are computed PER SEQUENCE (no cross-sequence boundaries). 25 fps.

  uv run python "scripts/3. Evaluation/eval_full_metrics.py" --checkpoints a.ckpt,b.ckpt   # ensemble
  uv run python "scripts/3. Evaluation/eval_full_metrics.py" --imuposer-pkl <results.pkl>   # IMUPoser saved
"""
import argparse, os
from pathlib import Path
import numpy as np
import torch
from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree

REPO = Path(__file__).resolve().parents[2]
DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
IGN = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
SIP = [1, 2, 16, 17]
FPS = 25.0


def _detect(sd):                                    # mirrors eval_dip; avatar nets load fine as TransformerIMUPoser
    if any(k.startswith("net.step_mlp.") for k in sd): return "DiffusionIMUPoser"
    if any(k.startswith("net.enc.") for k in sd): return "TransformerIMUPoser"
    if any(k.startswith("net.blocks.") for k in sd): return "CNN1DIMUPoser"
    return "GlobalModelIMUPoser"


def seq_metrics(pr, gt, bm, dev):
    """pr,gt: (N,24,3,3) local rotations. Returns per-frame sums + counts for SIP/MPJRE/MPJPE/MPVPE and
    per-sequence MPJVE / pred-jitter / gt-jitter accumulators."""
    I3 = torch.eye(3, device=dev)
    n = pr.shape[0]
    p, t = pr.clone(), gt.clone()
    p[:, IGN] = I3; t[:, IGN] = I3
    gp, jp, vp = bm.forward_kinematics(p, calc_mesh=True)
    gg, jg, vg = bm.forward_kinematics(t, calc_mesh=True)
    off = (jg[:, :1] - jp[:, :1])
    g = radian_to_degree(angle_between(gp.reshape(-1, 3, 3), gg.reshape(-1, 3, 3)).view(n, 24))
    sip = g[:, SIP].mean(1).sum().item()
    mpjre = g.mean(1).sum().item()
    jpr = (jp + off); jgr = jg                       # root-aligned joint positions (m)
    mpjpe = ((jpr - jgr).norm(dim=2).mean(1) * 100).sum().item()
    mpvpe = ((vp + off - vg).norm(dim=2).mean(1) * 100).sum().item()
    # temporal (root-RELATIVE positions: subtract own root each frame)
    jpc = (jp - jp[:, :1]).cpu().numpy(); jgc = (jg - jg[:, :1]).cpu().numpy()
    velp = np.diff(jpc, axis=0) * FPS; velg = np.diff(jgc, axis=0) * FPS
    mpjve = (np.linalg.norm(velp - velg, axis=2).mean(1) * 100).sum()        # cm/s, summed over frames
    def jerk(x):
        if x.shape[0] < 4: return 0.0, 0
        jk = (x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]) * (FPS ** 3)       # m/s^3
        return np.linalg.norm(jk, axis=2).mean(1).sum(), jk.shape[0]
    jp_sum, jn = jerk(jpc); jg_sum, _ = jerk(jgc)
    return dict(sip=sip, mpjre=mpjre, mpjpe=mpjpe, mpvpe=mpvpe, n=n,
                mpjve=mpjve, nv=max(n - 1, 0), jp=jp_sum, jg=jg_sum, nj=jn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", help="comma list (ensemble); avatar members auto-handled")
    ap.add_argument("--imuposer-pkl")
    ap.add_argument("--label", default="model")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device=a.device, mkdir=False)
    cfg.processed_imu_poser_25fps = DD
    dev = cfg.device
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    acc = dict(sip=0, mpjre=0, mpjpe=0, mpvpe=0, n=0, mpjve=0, nv=0, jp=0, jg=0, nj=0)

    def add(m):
        for k in acc: acc[k] += m[k]

    if a.imuposer_pkl:
        import pickle
        d = pickle.load(open(a.imuposer_pkl, "rb"))
        pm, tm = d["p_m"], d["t_m"]
        if isinstance(pm, dict):
            pm, tm = pm["lw_rp_h"], tm["lw_rp_h"]
        pr = torch.tensor(np.asarray(pm), dtype=torch.float32).view(-1, 24, 3, 3).to(dev)
        gt = torch.tensor(np.asarray(tm), dtype=torch.float32).view(-1, 24, 3, 3).to(dev)
        for s in range(0, pr.shape[0], 4000):       # chunk (whole-seq concat; boundary jerk negligible)
            add(seq_metrics(pr[s:s + 4000], gt[s:s + 4000], bm, dev))
    else:
        def build(cp):
            sd = torch.load(cp, map_location=dev, weights_only=False)["state_dict"]
            cfg.model = _detect(sd)
            m = get_model(cfg); m.load_state_dict(sd, strict=False); return m.eval().to(dev)
        members = [build(c) for c in a.checkpoints.split(",")]
        data = torch.load(DD / "dip_test.pt", weights_only=False)
        idx = amass_combos["lw_rp_h"]
        with torch.no_grad():
            for ai, oi, gp in zip([d.view(-1, 6, 3)[:, :5].float() for d in data["acc"]],
                                  [d.view(-1, 6, 3, 3)[:, :5].float() for d in data["ori"]],
                                  [d.view(-1, 24, 3, 3).float() for d in data["pose"]]):
                n = ai.shape[0]
                ca, co = torch.zeros_like(ai), torch.zeros_like(oi)
                ca[:, idx] = ai[:, idx] / cfg.acc_scale; co[:, idx] = oi[:, idx]
                inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
                r6 = sum(m(inp.unsqueeze(0), [n])[0, :, :144] for m in members) / len(members)
                pr = r6d_to_rotation_matrix(r6).view(n, 24, 3, 3)
                add(seq_metrics(pr, gp.to(dev), bm, dev))

    print(f"== {a.label} (lw_rp_h, dip_test, N={acc['n']}) ==")
    print(f"  SIP    {acc['sip']/acc['n']:6.2f} deg")
    print(f"  MPJRE  {acc['mpjre']/acc['n']:6.2f} deg   (angular)")
    print(f"  MPJPE  {acc['mpjpe']/acc['n']:6.2f} cm    (positional)")
    print(f"  MPVPE  {acc['mpvpe']/acc['n']:6.2f} cm    (mesh)")
    print(f"  MPJVE  {acc['mpjve']/acc['nv']:6.2f} cm/s  (velocity)")
    print(f"  Jitter {acc['jp']/acc['nj']:6.1f} m/s^3 (pred)   GT {acc['jg']/acc['nj']:5.1f} m/s^3   [/100 = 10^2 m/s^3]")


if __name__ == "__main__":
    main()
