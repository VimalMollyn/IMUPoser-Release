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
    a = ap.parse_args()

    pip = os.path.abspath(a.pip_dir); sys.path.insert(0, pip)
    # PIP's config.paths -> our files (dynamics.py reads paths.physics_model_file / physics_parameter_file at init)
    import config as pipcfg
    pipcfg.paths.physics_model_file = os.path.abspath(a.urdf)
    pipcfg.paths.physics_parameter_file = os.path.abspath(a.params or os.path.join(pip, "physics_parameters.json"))
    from dynamics import PhysicsOptimizer
    opt = PhysicsOptimizer(debug=False)
    # 25 fps data: PIP's parameters were tuned at 60 fps (delta_t 1/60); MobilePoser runs at 30. Use our frame time.
    opt.params["delta_t"] = 1.0 / a.fps
    print(f"[pip-physics] model {a.urdf}, qdot_size {opt.model.qdot_size}, delta_t {opt.params['delta_t']:.4f}, "
          f"kp_angular {opt.params['kp_angular']}, kd_angular {opt.params['kd_angular']}", flush=True)

    d = torch.load(a.preds, weights_only=False)
    out = {"pred": [], "gt": d["gt"], "names": d.get("names", []), "tran": []}
    seqs = list(zip(d["pred"], d["joints"]))
    if a.max_seqs: seqs = seqs[:a.max_seqs]
    t0 = time.time(); nfr = 0
    for si, (P, J) in enumerate(seqs):
        T = P.shape[0]
        vel = torch.zeros_like(J)
        vel[1:] = (J[1:] - J[:-1]) * a.fps            # m/s, root-relative joint velocities
        foot_speed = vel[:, [10, 11]].norm(dim=-1)   # LFOOT, RFOOT (SMPL 10/11)
        contact = torch.where(foot_speed < a.contact_speed, torch.tensor(3.0), torch.tensor(-3.0))   # logits -> sigmoid
        opt.reset_states()
        refined, trans = [], []
        for t in range(T):
            p, tr = opt.optimize_frame(P[t].float(), vel[t].float(), contact[t].float(), torch.zeros(6, 3))
            refined.append(torch.as_tensor(p).reshape(24, 3, 3).float()); trans.append(torch.as_tensor(tr).reshape(3).float())
        out["pred"].append(torch.stack(refined)); out["tran"].append(torch.stack(trans))
        nfr += T
        if si % 10 == 0:
            print(f"  seq {si + 1}/{len(seqs)}  {nfr} frames  {nfr / (time.time() - t0):.0f} fps", flush=True)
    torch.save(out, a.out)
    print(f"[pip-physics] done: {len(out['pred'])} sequences, {nfr} frames in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
