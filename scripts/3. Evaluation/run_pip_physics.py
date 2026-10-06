r"""PIP's physics optimizer (the one MobilePoser uses) as a post-process on dumped network predictions.

Runs in the separate "physics" environment (rbdl + pybullet + qpsolvers) with PIP's own code (dynamics.py, utils.py,
articulate) on sys.path. For every sequence: reset the simulator, then per frame call PhysicsOptimizer.optimize_frame(
pose, joint_velocity, contact_logits, acc) exactly as MobilePoser does (acc is unused by the optimizer; MobilePoser
passes zeros). Our pose model has no velocity / contact heads, so, like MobilePoser's offline path derives translation
from contacts, we derive them from the prediction itself: joint velocities = finite differences of the predicted FK joint
positions (root velocity 0: metrics are root-relative), foot contact logits from foot speed (slow foot = in contact).

  python run_pip_physics.py --pip_dir PIP-main --urdf urdfmodels/physics.urdf --preds preds.pt --out refined.pt
      [--fps 25] [--contact_speed 0.3] [--max_seqs N]
"""
import argparse, json, os, sys, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pip_dir", required=True); ap.add_argument("--urdf", required=True)
    ap.add_argument("--params", default=None, help="physics_parameters.json (default: PIP's)")
    ap.add_argument("--preds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--contact_speed", type=float, default=0.3, help="foot speed (m/s) below which the foot counts as in contact")
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--upsample", type=int, default=1, help="run the simulator at upsample x fps (PIP's PD gains are tuned for 60 Hz; "
                                                            "at 25 Hz omega*dt ~ 2 sits at the explicit-integration stability limit)")
    a = ap.parse_args()

    pip = os.path.abspath(a.pip_dir); sys.path.insert(0, pip)
    # PIP's config.paths -> our files (dynamics.py reads paths.physics_model_file / physics_parameter_file at init)
    import config as pipcfg
    pipcfg.paths.physics_model_file = os.path.abspath(a.urdf)
    pipcfg.paths.physics_parameter_file = os.path.abspath(a.params or os.path.join(pip, "physics_parameters.json"))
    from dynamics import PhysicsOptimizer
    from utils import smpl_to_rbdl
    opt = PhysicsOptimizer(debug=False)
    # 25 fps data: PIP's parameters were tuned at 60 fps (delta_t 1/60); MobilePoser runs at 30. Use our frame time.
    sim_fps = a.fps * a.upsample
    opt.params["delta_t"] = 1.0 / sim_fps
    floor_y = opt.params["floor_y"]
    print(f"[pip-physics] model {a.urdf}, qdot_size {opt.model.qdot_size}, delta_t {opt.params['delta_t']:.4f}, floor_y {floor_y}, "
          f"kp_angular {opt.params['kp_angular']}, kd_angular {opt.params['kd_angular']}", flush=True)

    d = torch.load(a.preds, weights_only=False)
    out = {"pred": [], "gt": d["gt"], "names": d.get("names", []), "tran": []}
    seqs = list(zip(d["pred"], d["joints"]))
    if a.max_seqs: seqs = seqs[:a.max_seqs]
    t0 = time.time(); nfr = 0; nfail = 0
    def upsample(P, J, k):
        """k x temporal upsampling: joints by linear interpolation, rotations by lerp + re-orthonormalisation (small steps)."""
        if k == 1: return P, J
        T = P.shape[0]
        idx = torch.arange(0, T - 1 + 1e-6, 1.0 / k)
        i0 = idx.floor().long().clamp(max=T - 1); i1 = (i0 + 1).clamp(max=T - 1); w = (idx - i0.float()).view(-1, 1, 1)
        Ju = J[i0] * (1 - w) + J[i1] * w
        M = P[i0] * (1 - w.unsqueeze(-1)) + P[i1] * w.unsqueeze(-1)
        U, _, Vt = torch.linalg.svd(M)
        D = torch.diag_embed(torch.stack([torch.ones_like(U[..., 0, 0]), torch.ones_like(U[..., 0, 0]), torch.det(U @ Vt)], -1))
        return U @ D @ Vt, Ju

    for si, (P0, J0) in enumerate(seqs):
        P, J = upsample(P0, J0, a.upsample)
        T = P.shape[0]
        vel = torch.zeros_like(J)
        vel[1:] = (J[1:] - J[:-1]) * sim_fps          # m/s, root-relative joint velocities
        foot_speed = vel[:, [10, 11]].norm(dim=-1)   # LFOOT, RFOOT (SMPL 10/11)
        # soft contact probabilities (PIP's foot constraint becomes rigid above p = 0.85: cap at 0.8 so finite-difference
        # foot speeds cannot make the QP infeasible), passed as logits like a contact head would produce
        prob = (0.8 * torch.sigmoid((a.contact_speed - foot_speed) / 0.1)).clamp(0.05, 0.8)
        contact = torch.log(prob / (1 - prob))
        # root height: stand the predicted (root-relative) motion on PIP's floor so the contact logic sees the real ground
        y_off = floor_y - float(J[:, [10, 11], 1].min()) + 0.01
        opt.reset_states()
        q0 = smpl_to_rbdl(P[0].numpy(), np.array([0.0, y_off, 0.0]))[0]
        opt.q = q0; opt.qdot = np.zeros(opt.model.qdot_size)
        refined, trans = [], []
        for t in range(T):
            try:
                p, tr = opt.optimize_frame(P[t].float(), vel[t].float(), contact[t].float(), torch.zeros(6, 3))
                p = torch.as_tensor(p).reshape(24, 3, 3).float(); tr = torch.as_tensor(tr).reshape(3).float()
            except Exception:           # QP infeasible for this frame: keep the network pose and re-sync the simulator
                nfail += 1
                p, tr = P[t].float(), torch.tensor([0.0, y_off, 0.0])
                opt.q = smpl_to_rbdl(P[t].numpy(), np.array([0.0, y_off, 0.0]))[0]; opt.qdot = np.zeros(opt.model.qdot_size); opt.last_x = []
            refined.append(p); trans.append(tr)
        R = torch.stack(refined)[::a.upsample][:P0.shape[0]]; TR = torch.stack(trans)[::a.upsample][:P0.shape[0]]
        out["pred"].append(R); out["tran"].append(TR)
        nfr += T
        if si % 10 == 0:
            print(f"  seq {si + 1}/{len(seqs)}  {nfr} frames  {nfr / (time.time() - t0):.0f} fps  QP failures {nfail}", flush=True)
    out["qp_failures"] = nfail
    torch.save(out, a.out)
    print(f"[pip-physics] done: {len(out['pred'])} sequences, {nfr} frames in {time.time() - t0:.0f}s, QP failures {nfail} ({100 * nfail / max(nfr, 1):.2f} %) -> {a.out}")


if __name__ == "__main__":
    main()
