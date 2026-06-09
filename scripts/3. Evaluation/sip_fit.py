r"""
SIP-style offline optimization (Sparse Inertial Poser, von Marcard 2017) for lw_rp_h.

Whole-sequence energy minimisation initialised from the FT ensemble (modern hybrid). Variables:
the SMPL pose (r6d, N x 24 x 6) AND the root translation (N x 3). Energy:
  E_ori   : measured IMU orientation == FK global orientation of the sensed bone (lw->18, rp->2, h->15)
  E_acc   : measured IMU acceleration == 2nd time-diff of the sensor VERTEX world position
            (tran + reduced-vertex FK), gravity-removed global, smoothed; robust (Huber). This is the
            term offline_fit.py dropped and the term that makes SIP "see" body dynamics / the pelvis.
  E_smooth: temporal pose smoothness (SIP's motion prior, lightweight)
  E_prior : anchor to the network init (stands in for SIP's anthropomorphic pose prior; VPoser absent),
            strong on un-sensed limbs (no measurement -> keep the metric-optimal conditional mean).
Only IMU is used as the target -- never GT pose. Tune on fval, report dip_test.

  uv run python "scripts/3. Evaluation/sip_fit.py" --data fval.pt --iters 400 --w_acc 0.0   # ablate acc
  uv run python "scripts/3. Evaluation/sip_fit.py" --data fval.pt --iters 400 --w_acc 0.02  # SIP acc on
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch
from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math as M
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree

REPO = Path(__file__).resolve().parents[2]
DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
IGN = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23]); SIP = [1, 2, 16, 17]; FPS = 25.0
JI5 = [18, 19, 1, 2, 15]; VI5 = [1961, 5424, 876, 4362, 411]
FT = "checkpoints/autoresearch/"
DEFAULT = ["checkpoints/fulltrain/exp130_ftA_exp21_avatar_s1_ft2/last.ckpt",
           "checkpoints/fulltrain/exp123_finetune_avatar_ft2/last.ckpt",
           "checkpoints/fulltrain/exp131_ftA_exp23_avatar_s2_ft2/last.ckpt",
           "checkpoints/fulltrain/exp132_ftA_exp42_avatar_s3_ft2/last.ckpt",
           "checkpoints/fulltrain/exp120_finetune_lr1e4_ft2/last.ckpt",
           "checkpoints/fulltrain/exp122_finetune_calib_ft2/last.ckpt"]


def _detect(sd):
    if any(k.startswith("net.step_mlp.") for k in sd): return "DiffusionIMUPoser"
    if any(k.startswith("net.enc.") for k in sd): return "TransformerIMUPoser"
    if any(k.startswith("net.blocks.") for k in sd): return "CNN1DIMUPoser"
    return "GlobalModelIMUPoser"


def rot_fro(A, B): return ((A - B) ** 2).sum(dim=(-1, -2))


class FK:
    r"""Differentiable FK that returns global rotations, joint positions, and ONLY the sensor vertices
    (reduced skinning -> cheap enough to call every optimiser iteration)."""
    def __init__(self, bm, vsel, dev):
        self.bm = bm; self.parent = bm.parent; self.dev = dev
        j, v = bm.get_zero_pose_joint_and_vertex()            # (24,3),(6890,3) root-aligned
        self.j = j.to(dev)
        self.skin_sel = bm._skinning_weights[vsel].to(dev)    # (S,24)
        self.v_sel = v[vsel].to(dev)                          # (S,3)

    def __call__(self, pose, tran=None):
        B = pose.shape[0]
        jb = self.j.unsqueeze(0).expand(B, -1, -1)
        T_local = M.transformation_matrix(pose, self.bm.joint_position_to_bone_vector(jb))
        T_global = self.bm.forward_kinematics_T(T_local)      # (B,24,4,4)
        gR = T_global[..., :3, :3]; gp = T_global[..., :3, 3]
        # reduced mesh: re-centre then skin the selected vertices
        Tg = T_global.clone()
        Rj = torch.matmul(T_global[..., :3, :3], self.j.unsqueeze(0).unsqueeze(-1)).squeeze(-1)  # (B,24,3)
        Tg[..., :3, 3] = Tg[..., :3, 3] - Rj
        Tv = torch.einsum('njab,sj->nsab', Tg, self.skin_sel)            # (B,S,4,4)
        vh = torch.cat([self.v_sel, torch.ones(self.v_sel.shape[0], 1, device=self.dev)], -1)  # (S,4)
        verts = torch.einsum('nsab,sb->nsa', Tv, vh)[..., :3]           # (B,S,3)
        if tran is not None:
            gp = gp + tran.unsqueeze(1); verts = verts + tran.unsqueeze(1)
        return gR, gp, verts


def smooth5(a):                                               # mirror preprocessing smooth_avg(s=5)
    pad = 2; ap = torch.cat([a[:1].repeat(pad, 1, 1), a, a[-1:].repeat(pad, 1, 1)], 0)
    return sum(ap[i:i + a.shape[0]] for i in range(5)) / 5.0


def synth_acc(pos):                                          # pos (N,S,3) -> (N,S,3) m/s^2, smoothed
    a = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (FPS ** 2)
    a = torch.cat([torch.zeros(1, *a.shape[1:], device=a.device), a, torch.zeros(1, *a.shape[1:], device=a.device)], 0)
    return smooth5(a)


def seq_metrics(pr, gt, bm, dev):
    I3 = torch.eye(3, device=dev); n = pr.shape[0]
    p, t = pr.clone(), gt.clone(); p[:, IGN] = I3; t[:, IGN] = I3
    gp, jp, vp = bm.forward_kinematics(p, calc_mesh=True); gg, jg, vg = bm.forward_kinematics(t, calc_mesh=True)
    off = (jg[:, :1] - jp[:, :1])
    g = radian_to_degree(angle_between(gp.reshape(-1, 3, 3), gg.reshape(-1, 3, 3)).view(n, 24))
    jpr = (jp + off)
    mpjpe = ((jpr - jg).norm(dim=2).mean(1) * 100).sum().item()
    mpvpe = ((vp + off - vg).norm(dim=2).mean(1) * 100).sum().item()
    jpc = (jp - jp[:, :1]).cpu().numpy(); jgc = (jg - jg[:, :1]).cpu().numpy()
    velp = np.diff(jpc, axis=0) * FPS; velg = np.diff(jgc, axis=0) * FPS
    mpjve = (np.linalg.norm(velp - velg, axis=2).mean(1) * 100).sum()
    jk = lambda x: (np.linalg.norm((x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]) * FPS ** 3, axis=2).mean(1).sum(), x.shape[0] - 3)
    jps, nj = jk(jpc); jgs, _ = jk(jgc)
    return dict(sip=g[:, SIP].mean(1).sum().item(), mpjre=g.mean(1).sum().item(), mpjpe=mpjpe, mpvpe=mpvpe,
                n=n, mpjve=mpjve, nv=n - 1, jp=jps, jg=jgs, nj=nj, perjoint=g.sum(0).cpu())


def fmt(a, label):
    print(f"== {label} (N={a['n']}) ==  SIP {a['sip']/a['n']:6.2f}  MPJRE {a['mpjre']/a['n']:6.2f}  "
          f"MPJPE {a['mpjpe']/a['n']:5.2f}  MPVPE {a['mpvpe']/a['n']:5.2f}  MPJVE {a['mpjve']/a['nv']:5.2f}  "
          f"Jit {a['jp']/a['nj']:4.0f}(GT {a['jg']/a['nj']:.0f})")


def sip_optimize(r6_init, meas_ori, meas_acc, sens_joints, fk, dev, ar):
    N = r6_init.shape[0]
    x = r6_init.clone().detach().requires_grad_(True)
    tran = torch.zeros(N, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([x, tran], lr=ar.lr)
    R_init = r6d_to_rotation_matrix(r6_init).view(N, 24, 3, 3)
    anc = torch.full((24,), ar.anchor_unsensed, device=dev); anc[sens_joints] = ar.anchor_sensed
    anc[[0, 3, 6, 9, 13, 16, 12, 14, 15, 1, 2, 5, 8, 11]] = ar.anchor_sensed   # sensed kinematic chains
    for it in range(ar.iters):
        opt.zero_grad()
        R = r6d_to_rotation_matrix(x).view(N, 24, 3, 3)
        gR, gp, verts = fk(R, tran)
        l_ori = rot_fro(gR[:, sens_joints], meas_ori).mean()
        loss = ar.w_ori * l_ori + (anc * rot_fro(R, R_init)).mean()
        if ar.w_acc > 0:
            a_syn = synth_acc(verts)
            l_acc = torch.nn.functional.huber_loss(a_syn[2:-2], meas_acc[2:-2], delta=ar.huber)
            loss = loss + ar.w_acc * l_acc
        if ar.w_smooth > 0 and N >= 3:
            loss = loss + ar.w_smooth * ((gR[2:] - 2 * gR[1:-1] + gR[:-2]) ** 2).sum(dim=(-1, -2)).mean()
        loss.backward(); opt.step()
    return r6d_to_rotation_matrix(x.detach()).view(N, 24, 3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fval.pt"); ap.add_argument("--members", default=",".join(DEFAULT))
    ap.add_argument("--iters", type=int, default=400); ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--w_ori", type=float, default=1.0); ap.add_argument("--w_acc", type=float, default=0.02)
    ap.add_argument("--w_smooth", type=float, default=0.0); ap.add_argument("--huber", type=float, default=2.0)
    ap.add_argument("--anchor_sensed", type=float, default=0.01); ap.add_argument("--anchor_unsensed", type=float, default=0.1)
    ap.add_argument("--device", default="0"); ap.add_argument("--diag", action="store_true")
    a = ap.parse_args()
    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device=a.device, mkdir=False)
    cfg.processed_imu_poser_25fps = DD; dev = cfg.device
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    combo = amass_combos["lw_rp_h"]; sens_joints = [JI5[s] for s in combo]; vsel = [VI5[s] for s in combo]
    fk = FK(bm, vsel, dev)

    def build(name):
        path = name if name.endswith(".ckpt") else FT + name + "/last.ckpt"
        sd = torch.load(path, map_location=dev, weights_only=False)["state_dict"]
        cfg.model = _detect(sd); m = get_model(cfg); m.load_state_dict(sd, strict=False); return m.eval().to(dev)
    members = [build(n) for n in a.members.split(",")]
    data = torch.load(DD / a.data, weights_only=False)
    base = {k: (0 if k != "perjoint" else torch.zeros(24)) for k in
            ["sip", "mpjre", "mpjpe", "mpvpe", "n", "mpjve", "nv", "jp", "jg", "nj", "perjoint"]}
    ref = {k: v.clone() if torch.is_tensor(v) else v for k, v in base.items()}
    t0 = time.time()
    for ai, oi, gp in zip([d.view(-1, 6, 3)[:, :5].float() for d in data["acc"]],
                          [d.view(-1, 6, 3, 3)[:, :5].float() for d in data["ori"]],
                          [d.view(-1, 24, 3, 3).float() for d in data["pose"]]):
        n = ai.shape[0]; ca, co = torch.zeros_like(ai), torch.zeros_like(oi)
        ca[:, combo] = ai[:, combo] / cfg.acc_scale; co[:, combo] = oi[:, combo]
        inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
        with torch.no_grad():
            r6 = sum(m(inp.unsqueeze(0), [n])[0, :, :144] for m in members) / len(members)
        r6_init = r6.view(n, 24, 6)
        meas_ori = oi[:, combo].to(dev); meas_acc = ai[:, combo].to(dev)          # raw m/s^2
        gt = gp.to(dev)
        pr_init = r6d_to_rotation_matrix(r6_init).view(n, 24, 3, 3)
        pr_ref = sip_optimize(r6_init, meas_ori, meas_acc, sens_joints, fk, dev, a)
        for acc, pr in ((base, pr_init), (ref, pr_ref)):
            m = seq_metrics(pr, gt, bm, dev)
            for k in acc: acc[k] += m[k]
    print(f"[{a.data}] iters={a.iters} w_ori={a.w_ori} w_acc={a.w_acc} w_smooth={a.w_smooth} "
          f"anc_s={a.anchor_sensed} anc_u={a.anchor_unsensed}  ({time.time()-t0:.0f}s)")
    fmt(base, "NETWORK"); fmt(ref, "SIP FIT")
    print(f"  delta: SIP {(ref['sip']-base['sip'])/base['n']:+.2f}  MPJRE {(ref['mpjre']-base['mpjre'])/base['n']:+.2f}  "
          f"MPJPE {(ref['mpjpe']-base['mpjpe'])/base['n']:+.2f}  MPJVE {(ref['mpjve']-base['mpjve'])/base['nv']:+.2f}")
    if a.diag:
        JN = ["pelv","lhip","rhip","spin1","lknee","rknee","spin2","lank","rank","spin3","lfoot","rfoot","neck","lcol","rcol","head","lsho","rsho","lelb","relb","lwri","rwri","lhnd","rhnd"]
        pjb = base['perjoint'] / base['n']; pjr = ref['perjoint'] / ref['n']
        for j in range(24):
            if abs(pjb[j] - pjr[j]) > 0.05 or j in SIP or j in sens_joints:
                print(f"    {j:2d} {JN[j]:5s} {pjb[j]:6.2f} -> {pjr[j]:6.2f} ({pjr[j]-pjb[j]:+.2f}) {'SIP' if j in SIP else ('SENS' if j in sens_joints else '')}")


if __name__ == "__main__":
    main()
