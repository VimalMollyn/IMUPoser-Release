r"""
OFFLINE test-time optimization (analysis-by-synthesis) for lw_rp_h.

The realtime feed-forward ensemble is only an approximate inverse of the IMU->pose map: even
for the DIRECTLY sensed bones it leaves residual error (e.g. left-elbow joint-18 ~12 deg, while
the measured orientation matches GT to ~1 deg). Offline we can fix that: initialise from the
FT ensemble and optimise the whole-sequence SMPL pose so the FK global orientations of the
sensed bones match the measured IMU orientations, with a light temporal-smoothness term and a
prior that anchors everything (esp. the un-sensed limbs, which have no measurement and whose
metric-optimal estimate is the network's conditional mean) to the network prediction.

Sensor -> SMPL joint (orientation):  lw->18 (l-elbow), rp->2 (r-hip), h->15 (head)
Sensor -> SMPL vertex (accel, UNUSED: 25fps finite-diff too coarse, metrics are root-relative)

Only IMU (acc/ori) is used as the optimisation target -- never GT pose. Tune on fval, report dip_test.

  uv run python "scripts/3. Evaluation/offline_fit.py" --data fval.pt --iters 300 \
     --w_ori 1.0 --w_smooth 0.0 --w_anchor 0.02 [--calib] [--save-pkl out.pkl]
"""
import argparse, os, pickle, time
from pathlib import Path
import numpy as np
import torch
from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import (r6d_to_rotation_matrix, rotation_matrix_to_r6d,
                                   angle_between, radian_to_degree, axis_angle_to_rotation_matrix)

REPO = Path(__file__).resolve().parents[2]
DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
IGN = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
SIP = [1, 2, 16, 17]
FPS = 25.0
JI5 = [18, 19, 1, 2, 15]                 # 5-sensor slice [lw,rw,lp,rp,h] -> ori joints
JN = ["pelv","lhip","rhip","spin1","lknee","rknee","spin2","lank","rank","spin3","lfoot","rfoot",
      "neck","lcol","rcol","head","lsho","rsho","lelb","relb","lwri","rwri","lhnd","rhnd"]

FT = "checkpoints/autoresearch/"
DEFAULT_MEMBERS = ["exp130_ftA_exp21_avatar_s1", "exp123_finetune_avatar", "exp131_ftA_exp23_avatar_s2",
                   "exp132_ftA_exp42_avatar_s3", "exp120_finetune_lr1e4", "exp122_finetune_calib"]


def _detect(sd):
    if any(k.startswith("net.step_mlp.") for k in sd): return "DiffusionIMUPoser"
    if any(k.startswith("net.enc.") for k in sd): return "TransformerIMUPoser"
    if any(k.startswith("net.blocks.") for k in sd): return "CNN1DIMUPoser"
    return "GlobalModelIMUPoser"


def rot_fro(A, B):
    r"""Squared Frobenius (chordal) distance between rotation matrices, summed over last 2 dims.
    ||A-B||_F^2 = 6 - 2 tr(A^T B); smooth, no acos singularity."""
    return ((A - B) ** 2).sum(dim=(-1, -2))


# ---- metric accumulation (mirrors eval_full_metrics.seq_metrics) -------------------------------
def seq_metrics(pr, gt, bm, dev):
    I3 = torch.eye(3, device=dev); n = pr.shape[0]
    p, t = pr.clone(), gt.clone(); p[:, IGN] = I3; t[:, IGN] = I3
    gp, jp, vp = bm.forward_kinematics(p, calc_mesh=True)
    gg, jg, vg = bm.forward_kinematics(t, calc_mesh=True)
    off = (jg[:, :1] - jp[:, :1])
    g = radian_to_degree(angle_between(gp.reshape(-1, 3, 3), gg.reshape(-1, 3, 3)).view(n, 24))
    sip = g[:, SIP].mean(1).sum().item(); mpjre = g.mean(1).sum().item()
    jpr = (jp + off); jgr = jg
    mpjpe = ((jpr - jgr).norm(dim=2).mean(1) * 100).sum().item()
    mpvpe = ((vp + off - vg).norm(dim=2).mean(1) * 100).sum().item()
    jpc = (jp - jp[:, :1]).cpu().numpy(); jgc = (jg - jg[:, :1]).cpu().numpy()
    velp = np.diff(jpc, axis=0) * FPS; velg = np.diff(jgc, axis=0) * FPS
    mpjve = (np.linalg.norm(velp - velg, axis=2).mean(1) * 100).sum()
    def jerk(x):
        if x.shape[0] < 4: return 0.0, 0
        jk = (x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]) * (FPS ** 3)
        return np.linalg.norm(jk, axis=2).mean(1).sum(), jk.shape[0]
    jp_sum, jn = jerk(jpc); jg_sum, _ = jerk(jgc)
    perjoint = g.sum(0).cpu()                            # (24,) deg-sum for per-joint diagnostics
    return dict(sip=sip, mpjre=mpjre, mpjpe=mpjpe, mpvpe=mpvpe, n=n, mpjve=mpjve, nv=max(n - 1, 0),
                jp=jp_sum, jg=jg_sum, nj=jn, perjoint=perjoint)


def fmt(acc, label):
    print(f"== {label} (N={acc['n']}) ==")
    print(f"  SIP    {acc['sip']/acc['n']:6.2f} deg   MPJRE {acc['mpjre']/acc['n']:6.2f} deg   "
          f"MPJPE {acc['mpjpe']/acc['n']:5.2f} cm   MPVPE {acc['mpvpe']/acc['n']:5.2f} cm   "
          f"MPJVE {acc['mpjve']/acc['nv']:5.2f} cm/s   Jit {acc['jp']/acc['nj']:5.0f} (GT {acc['jg']/acc['nj']:.0f})")


def reroot_fit(r6_init, meas_ori, sens_joints, bm, wahba_w, inject_joints, smooth_root=0):
    r"""Re-estimate the pelvis orientation per frame from the measured world orientations of the
    sensed bones (weighted orthogonal-Procrustes / Wahba), using the network's relative pose to back
    out the root, then place the network relative pose under the estimated root and inject the
    measured bones. Only the sensed joints' metric orientation changes (others stay at network);
    they improve iff the estimated pelvis beats the network's effective pelvis.

      meas_ori: (N,S,3,3) world ori of sens_joints; wahba_w: (S,) weights for the root estimate."""
    N = r6_init.shape[0]
    R_net = r6d_to_rotation_matrix(r6_init).view(N, 24, 3, 3)
    Gnet = bm.forward_kinematics(R_net)[0]                         # (N,24,3,3) world
    Rnetroot = Gnet[:, :1]                                          # (N,1,3,3)
    Gloc = torch.matmul(Rnetroot.transpose(-1, -2), Gnet)          # (N,24,3,3) root-relative (metric frame)
    S = len(sens_joints)
    relpose = Gloc[:, sens_joints]                                 # (N,S,3,3) network root-relative of sensed
    w = wahba_w.to(Gnet.device).view(1, S, 1, 1)
    M = (w * torch.matmul(meas_ori, relpose.transpose(-1, -2))).sum(1)   # (N,3,3) = sum_b w meas relpose^T
    U, _, Vh = torch.linalg.svd(M)
    R_est = torch.matmul(U, Vh)
    det = torch.linalg.det(R_est)
    U2 = U.clone(); U2[:, :, -1] *= det.sign().unsqueeze(-1)       # ensure proper rotation
    R_est = torch.matmul(U2, Vh)                                   # (N,3,3) estimated pelvis
    if smooth_root and N >= 3:                                     # temporal smoothing of pelvis (rotations->mean->project)
        Re = R_est.clone()
        for _ in range(smooth_root):
            Re[1:-1] = (R_est[2:] + R_est[1:-1] + R_est[:-2]) / 3
            U3, _, Vh3 = torch.linalg.svd(Re); Re = torch.matmul(U3, Vh3)
        R_est = Re
    G = torch.matmul(R_est.unsqueeze(1), Gloc)                     # network relative pose under estimated root
    for k, j in enumerate(sens_joints):
        if j in inject_joints:
            G[:, j] = meas_ori[:, k]
    return bm.inverse_kinematics_R(G).view(N, 24, 3, 3)


def inject_measurements(r6_init, meas_ori, sens_joints, inject_joints, bm):
    r"""Overwrite the GLOBAL orientation of each injected sensed joint with its measurement, then
    IK back to a consistent local pose. Every other joint keeps the network's global orientation
    exactly (FK(IK(G))==G), so the metric for non-injected joints is unchanged and injected joints
    take the (more accurate) measured orientation. Parameter-free, strictly improves when the
    measurement beats the network on those joints."""
    N = r6_init.shape[0]
    R_net = r6d_to_rotation_matrix(r6_init).view(N, 24, 3, 3)
    G = bm.forward_kinematics(R_net)[0].clone()              # (N,24,3,3) global
    for k, j in enumerate(sens_joints):
        if j in inject_joints:
            G[:, j] = meas_ori[:, k]
    return bm.inverse_kinematics_R(G).view(N, 24, 3, 3)


def optimize_seq(r6_init, meas_ori, sens_joints, sens_w, bm, dev, args):
    r"""r6_init: (N,24,6) network init. meas_ori: (N,S,3,3) measured global rot of sens_joints.
    Returns refined R_local (N,24,3,3)."""
    N = r6_init.shape[0]
    x = r6_init.clone().detach().requires_grad_(True)
    params = [x]
    calib = None
    if args.calib:
        calib = torch.zeros(len(sens_joints), 3, device=dev, requires_grad=True)   # per-sensor axis-angle
        params.append(calib)
    opt = torch.optim.Adam(params, lr=args.lr)
    sens_w = sens_w.to(dev).view(1, -1)
    for it in range(args.iters):
        opt.zero_grad()
        R = r6d_to_rotation_matrix(x).view(N, 24, 3, 3)
        grot = bm.forward_kinematics(R)[0]                       # (N,24,3,3) global
        tgt = meas_ori
        if calib is not None:
            Rc = axis_angle_to_rotation_matrix(calib)            # (S,3,3)
            tgt = torch.einsum('sij,nsjk->nsik', Rc, meas_ori)   # rotate measurement by per-sensor calib
        fk_sens = grot[:, sens_joints]                           # (N,S,3,3)
        l_ori = (rot_fro(fk_sens, tgt) * sens_w).mean()
        loss = args.w_ori * l_ori
        if args.w_anchor > 0:
            R_init = r6d_to_rotation_matrix(r6_init).view(N, 24, 3, 3)
            loss = loss + args.w_anchor * rot_fro(R, R_init).mean()
        if args.w_smooth > 0 and N >= 3:
            acc_g = grot[2:] - 2 * grot[1:-1] + grot[:-2]        # angular accel of global rot
            loss = loss + args.w_smooth * (acc_g ** 2).sum(dim=(-1, -2)).mean()
        if calib is not None and args.calib_reg > 0:
            loss = loss + args.calib_reg * (calib ** 2).sum()
        loss.backward()
        opt.step()
    return r6d_to_rotation_matrix(x.detach()).view(N, 24, 3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fval.pt")
    ap.add_argument("--members", default=",".join(DEFAULT_MEMBERS))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--w_ori", type=float, default=1.0)
    ap.add_argument("--w_anchor", type=float, default=0.02)
    ap.add_argument("--w_smooth", type=float, default=0.0)
    ap.add_argument("--thigh_w", type=float, default=1.0, help="weight on rp(thigh) ori (5deg calib offset)")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib_reg", type=float, default=10.0)
    ap.add_argument("--inject", default="", help="'' none | 'all' sensed | 'safe' (18,15) | comma joint-idx; "
                    "overwrite sensed-joint global ori with measurement before/instead of iterating")
    ap.add_argument("--reroot", action="store_true", help="re-estimate pelvis via Wahba from sensed measurements")
    ap.add_argument("--wahba_w", default="1,0.3,1", help="per-sensor weights (lw,rp,h) for the root estimate")
    ap.add_argument("--smooth_root", type=int, default=0, help="iterations of temporal pelvis smoothing")
    ap.add_argument("--device", default="0")
    ap.add_argument("--save-pkl", default=None)
    ap.add_argument("--diag", action="store_true", help="print per-joint err delta")
    a = ap.parse_args()

    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device=a.device, mkdir=False)
    cfg.processed_imu_poser_25fps = DD
    dev = cfg.device
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)

    def build(name):
        path = name if name.endswith(".ckpt") else FT + name + "/last.ckpt"
        sd = torch.load(path, map_location=dev, weights_only=False)["state_dict"]
        cfg.model = _detect(sd); m = get_model(cfg); m.load_state_dict(sd, strict=False); return m.eval().to(dev)
    members = [build(n) for n in a.members.split(",")]

    combo = amass_combos["lw_rp_h"]                       # [0,3,4]
    sens_joints = [JI5[s] for s in combo]                 # [18,2,15]
    sens_w = torch.tensor([a.thigh_w if JI5[s] == 2 else 1.0 for s in combo])
    if a.inject == "all":   inject_joints = set(sens_joints)
    elif a.inject == "safe": inject_joints = {18, 15}
    elif a.inject:          inject_joints = set(int(x) for x in a.inject.split(","))
    else:                   inject_joints = set()
    data = torch.load(DD / a.data, weights_only=False)

    base = dict(sip=0, mpjre=0, mpjpe=0, mpvpe=0, n=0, mpjve=0, nv=0, jp=0, jg=0, nj=0, perjoint=torch.zeros(24))
    ref = {k: (0 if k != "perjoint" else torch.zeros(24)) for k in base}
    save_pr, save_gt = [], []
    t0 = time.time()
    for ai, oi, gp in zip([d.view(-1, 6, 3)[:, :5].float() for d in data["acc"]],
                          [d.view(-1, 6, 3, 3)[:, :5].float() for d in data["ori"]],
                          [d.view(-1, 24, 3, 3).float() for d in data["pose"]]):
        n = ai.shape[0]
        ca, co = torch.zeros_like(ai), torch.zeros_like(oi)
        ca[:, combo] = ai[:, combo] / cfg.acc_scale; co[:, combo] = oi[:, combo]
        inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
        with torch.no_grad():
            r6 = sum(m(inp.unsqueeze(0), [n])[0, :, :144] for m in members) / len(members)   # (n,144)
        r6_init = r6.view(n, 24, 6)
        meas_ori = oi[:, combo].to(dev)                  # (n,S,3,3) measured global rot of sens_joints
        gt = gp.to(dev)
        pr_init = r6d_to_rotation_matrix(r6_init).view(n, 24, 3, 3)
        r6_opt = r6_init
        if a.reroot:
            wahba_w = torch.tensor([float(x) for x in a.wahba_w.split(",")])
            inj = reroot_fit(r6_init, meas_ori, sens_joints, bm, wahba_w, inject_joints, a.smooth_root)
            r6_opt = rotation_matrix_to_r6d(inj.reshape(-1, 3, 3)).view(n, 24, 6)
        elif inject_joints:
            inj = inject_measurements(r6_init, meas_ori, sens_joints, inject_joints, bm)
            r6_opt = rotation_matrix_to_r6d(inj.reshape(-1, 3, 3)).view(n, 24, 6)
        if a.iters > 0 and a.w_ori > 0:
            pr_ref = optimize_seq(r6_opt, meas_ori, sens_joints, sens_w, bm, dev, a)
        else:
            pr_ref = r6d_to_rotation_matrix(r6_opt).view(n, 24, 3, 3)
        for acc, pr in ((base, pr_init), (ref, pr_ref)):
            m = seq_metrics(pr, gt, bm, dev)
            for k in acc:
                acc[k] += m[k]
        if a.save_pkl:
            save_pr.append(pr_ref.cpu().reshape(n, -1)); save_gt.append(gt.cpu().reshape(n, -1))
    print(f"[{a.data}] iters={a.iters} w_ori={a.w_ori} w_anchor={a.w_anchor} w_smooth={a.w_smooth} "
          f"calib={a.calib} thigh_w={a.thigh_w}  ({time.time()-t0:.0f}s)")
    fmt(base, "NETWORK (init)")
    fmt(ref, "OFFLINE FIT")
    d_sip = (ref['sip'] - base['sip']) / base['n']; d_re = (ref['mpjre'] - base['mpjre']) / base['n']
    print(f"  delta: SIP {d_sip:+.2f}   MPJRE {d_re:+.2f}")
    if a.diag:
        pjb = (base['perjoint'] / base['n']).numpy(); pjr = (ref['perjoint'] / ref['n']).numpy()
        print("  per-joint (deg)  base -> fit   (delta):")
        for j in range(24):
            tag = "SIP" if j in SIP else ("SENS" if j in sens_joints else "")
            if abs(pjb[j] - pjr[j]) > 0.05 or tag:
                print(f"    {j:2d} {JN[j]:5s} {pjb[j]:6.2f} -> {pjr[j]:6.2f}  ({pjr[j]-pjb[j]:+.2f}) {tag}")
    if a.save_pkl:
        pickle.dump({"p_m": {"lw_rp_h": torch.cat(save_pr).numpy()},
                     "t_m": {"lw_rp_h": torch.cat(save_gt).numpy()}}, open(a.save_pkl, "wb"))
        print("  saved", a.save_pkl)


if __name__ == "__main__":
    main()
