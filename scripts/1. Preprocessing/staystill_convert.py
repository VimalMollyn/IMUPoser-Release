r"""
StayStill (SCA 2026; LaFAN1-skeleton BVH @30 fps, 1634 clips / 50 subjects / ~6 h of idle motion: standing idle,
idle with a phone, 18 idle actions such as looking around, checking a watch or phone, scratching, stretching, yawning,
shifting weight) -> SMPL -> synthetic IMU -> 25 fps training files STAYSTILL_<chunk>.pt (+ memmap shards).

Source: https://zenodo.org/records/18741736 (MIT), unpacked under $STAYSTILL_DIR/lafan/{idle,phone,actions}/*.bvh.
The Freemocap folder (raw markerless skeleton) is ignored; the LaFAN folder is the authors' retarget to the LaFAN1 rig.

Retarget (closed-form, same idea as bones_seed_convert.py / soma_retarget.SOMAtoSMPL):
  G_smpl[k] = G_rig[m(k)] @ A[k]      with A[k] = Rot(bone_axis_rig, theta_k) @ align(bone_dir_smpl[k] -> bone_axis_rig)
The LaFAN rig has NO T-pose reference (its zero pose is a straight line along +X: every offset lies on the joint's
local X axis), so the per-joint constant A[k] cannot be read off a reference frame as for SOMA. Instead: the
bone-direction alignment fixes A[k] up to a twist about the bone, and the twist theta_k is calibrated from the data:
  * pelvis and chest (spine3): the twist is pinned by the rig's off-axis children (hip joints resp. collars), which
    must land where SMPL's rest offsets put them (closed-form 1-D search);
  * every other joint: theta_k makes the dataset-mean twist of that joint's SMPL LOCAL rotation (about its own bone)
    zero -- exact for hinge joints (knees, elbows), a zero-mean assumption for ball joints (hips, shoulders, head).
30 -> 60 fps by matrix-space rotation interpolation (resample_pose_aa), IMU synthesis at 60 fps on the SMPL mesh
vertices (synthesize_sequences, betas = 0), then the standard 25 fps conversion (amass_dir_to_25fps).

  uv run python "scripts/1. Preprocessing/staystill_convert.py" --gpu 1 [--limit 20] [--plot]
"""
import argparse, importlib.util, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

from imuposer.config import Config
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math as M
from imuposer.datasets.synth_imu import synthesize_sequences, amass_dir_to_25fps, resample_linear, resample_pose_aa
from imuposer.datasets.soma_retarget import SMPLSkeleton, SMPL_PARENTS, SMPL_CHILD, align_rotation

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bones_seed_convert", HERE / "bones_seed_convert.py")
_b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_b)
parse_hierarchy, parse_motion, bvh_fk = _b.parse_hierarchy, _b.parse_motion, _b.bvh_fk

SRC = Path(os.environ.get("STAYSTILL_DIR", "/media/vimal/NVME990Pro/imuposer_data/raw/staystill")) / "lafan"
OUT = Path(os.environ.get("IMUPOSER_OUT_DIR", "/home/vimal/imuposer_data"))
SRC_FPS = 30.0

# SMPL joint -> LaFAN joint (22-joint rig: Hips, {Left,Right}{UpLeg,Leg,Foot,Toe}, Spine, Spine1, Spine2, Neck, Head,
# {Left,Right}{Shoulder,Arm,ForeArm,Hand})
RIG_MAP = {0: "Hips", 1: "LeftUpLeg", 2: "RightUpLeg", 3: "Spine", 4: "LeftLeg", 5: "RightLeg", 6: "Spine1",
           7: "LeftFoot", 8: "RightFoot", 9: "Spine2", 10: "LeftToe", 11: "RightToe", 12: "Neck",
           13: "LeftShoulder", 14: "RightShoulder", 15: "Head", 16: "LeftArm", 17: "RightArm",
           18: "LeftForeArm", 19: "RightForeArm", 20: "LeftHand", 21: "RightHand", 22: None, 23: None}
# rig joint -> child whose offset defines the joint's bone axis (leaves: +X, the rig's bone axis convention)
RIG_CHILD = {"Hips": "Spine", "LeftUpLeg": "LeftLeg", "RightUpLeg": "RightLeg", "Spine": "Spine1", "LeftLeg": "LeftFoot",
             "RightLeg": "RightFoot", "Spine1": "Spine2", "LeftFoot": "LeftToe", "RightFoot": "RightToe", "Spine2": "Neck",
             "Neck": "Head", "LeftShoulder": "LeftArm", "RightShoulder": "RightArm", "LeftArm": "LeftForeArm",
             "RightArm": "RightForeArm", "LeftForeArm": "LeftHand", "RightForeArm": "RightHand"}
# joints whose twist is pinned by a hinge child: SMPL joint -> (hinge child joint, SMPL hinge axis in the joint's frame)
HINGE = {16: (18, (0., -1., 0.)), 17: (19, (0., 1., 0.))}   # knees: the idle data has too little flexion (sway dominates), thighs use the zero-mean rule


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def rot_about(axis, theta):
    """Rotation matrix about unit `axis` (3,) by `theta` (scalar tensor or float)."""
    th = torch.as_tensor(theta, dtype=torch.float32, device=axis.device)
    return M.axis_angle_to_rotation_matrix((axis * th).view(1, 3))[0]


def twist_angle(L, d):
    """Twist of rotations L (N,3,3) about unit axis d (3,) in the swing-twist decomposition L = swing @ twist."""
    aa = M.rotation_matrix_to_axis_angle(L.reshape(-1, 3, 3))
    ang = aa.norm(dim=-1)
    axis = aa / ang.clamp_min(1e-9)[:, None]
    w = torch.cos(ang / 2); p = (axis * torch.sin(ang / 2)[:, None]) @ d
    return 2 * torch.atan2(p, w)


class LafanToSMPL:
    def __init__(self, skel, hier, G, P, verbose=True):
        """hier: parsed BVH hierarchy; G (N,J,3,3), P (N,J,3): world rotations / positions of calibration frames."""
        self.skel = skel; dev = skel.device
        names, parents, offsets, _ = hier
        self.names = names; self.idx = {n: i for i, n in enumerate(names)}
        off = torch.as_tensor(offsets, device=dev)
        J = skel.J
        d_smpl = {k: J[c] - J[k] for k, c in SMPL_CHILD.items()}
        d_smpl[20] = J[22] - J[20]; d_smpl[21] = J[23] - J[21]            # wrist -> hand
        d_smpl[15] = torch.tensor([0., 1., 0.], device=dev)                  # head: up
        d_smpl[10] = J[10] - J[7]; d_smpl[11] = J[11] - J[8]                 # toes: along the foot
        self.d_smpl = {k: unit(v) for k, v in d_smpl.items()}
        self.d_rig = {}
        for n in names:
            c = RIG_CHILD.get(n)
            self.d_rig[n] = unit(off[self.idx[c]]) if c else torch.tensor([1., 0., 0.], device=dev)
        self.map_idx = [self.idx[RIG_MAP[k]] if RIG_MAP[k] is not None else -1 for k in range(24)]
        self.A = torch.eye(3, device=dev).repeat(24, 1, 1)
        self.theta = torch.zeros(24)
        N = G.shape[0]
        Gs = torch.eye(3, device=dev).repeat(N, 24, 1, 1)
        print(f"  SMPL rest: toe-ankle (facing) {(J[10]-J[7]).tolist()}, l_hip-pelvis {(J[1]-J[0]).tolist()}, spine1-pelvis {(J[3]-J[0]).tolist()}", flush=True)
        report = []
        for k in range(22):
            n = RIG_MAP[k]; m = self.idx[n]
            base = align_rotation(self.d_smpl[k], self.d_rig[n])              # d_smpl -> d_rig
            axis = self.d_rig[n]
            if k in (0, 9):
                # twist pinned by the off-axis children: the left-to-right axis through the hip joints (pelvis) resp.
                # the collars (chest) must agree, projected perpendicular to the bone axis
                lk, rk = (1, 2) if k == 0 else (13, 14)
                d = self.d_smpl[k]
                src = J[lk] - J[rk]; src = unit(src - (src @ d) * d)                           # SMPL rest left-right axis
                v = off[self.idx[RIG_MAP[lk]]] - off[self.idx[RIG_MAP[rk]]]                     # rig local left-right axis
                v = base.T @ v; v = unit(v - (v @ d) * d)                                      # in the SMPL-k frame (theta=0)
                th = -torch.atan2(torch.cross(v, src, dim=-1) @ d, v @ src)                   # Rot(d,-th) v = src
                how = f"pinned by the {RIG_MAP[lk]}-{RIG_MAP[rk]} axis"
            elif k in HINGE:
                # twist pinned by the hinge plane of the child joint (knee / elbow): the child bone swings in a plane
                # whose normal must be SMPL's hinge axis (knees flex about +x, left elbow about -y, right elbow about +y)
                c, tgt = HINGE[k]
                cm, cc = self.idx[RIG_MAP[c]], self.idx[RIG_CHILD[RIG_MAP[c]]]
                w = unit(P[:, cc] - P[:, cm])                                                 # world child-bone dir
                s = torch.einsum("nji,nj->ni", G[:, m] @ base, w)                             # in the SMPL-k frame (theta=0)
                d = self.d_smpl[k]
                cr = torch.cross(d.expand_as(s), s, dim=-1)                                   # |cr| = sin(bend)
                hn = unit(cr.sum(0))                                                          # hinge axis (theta=0 frame)
                t = torch.tensor(tgt, device=dev, dtype=torch.float32)
                tp = unit(t - (t @ d) * d)                                                    # target axis, perp to bone
                th = -torch.atan2(torch.cross(hn, tp, dim=-1) @ d, hn @ tp)                   # Rot(d,-th) hn = tp
                bend = torch.rad2deg(torch.asin(cr.norm(dim=-1).clamp(max=1)))
                how = f"pinned by the hinge plane of {c} (mean bend {bend.mean():.1f} deg, max {bend.max():.0f})"
            else:
                p = SMPL_PARENTS[k]
                L0 = Gs[:, p].transpose(1, 2) @ G[:, m] @ base
                tau = twist_angle(L0, self.d_smpl[k])
                th = -torch.atan2(torch.sin(tau).mean(), torch.cos(tau).mean())
                dev_ = torch.atan2(torch.sin(tau + th), torch.cos(tau + th))
                how = f"zero-mean twist (std {torch.rad2deg(dev_).std().item():.1f} deg over {N} frames)"
            self.A[k] = rot_about(axis, th) @ base
            self.theta[k] = th
            Gs[:, k] = G[:, m] @ self.A[k]
            report.append(f"  A[{k:2d}] {n:14s} twist {torch.rad2deg(th).item():7.1f} deg  {how}")
        if verbose:
            print("\n".join(report), flush=True)

    @torch.no_grad()
    def __call__(self, G, P):
        """G (T,J,3,3), P (T,J,3) rig world rotations / positions (m, y-up) -> pose_aa (T,24,3), transl (T,3)."""
        T = G.shape[0]; dev = self.skel.device
        Gs = torch.eye(3, device=dev).repeat(T, 24, 1, 1)
        for k in range(22):
            Gs[:, k] = G[:, self.map_idx[k]] @ self.A[k]
        L = torch.empty_like(Gs)
        L[:, 0] = Gs[:, 0]
        for k in range(1, 24):
            L[:, k] = Gs[:, SMPL_PARENTS[k]].transpose(1, 2) @ Gs[:, k]
        L[:, 22:24] = torch.eye(3, device=dev)
        transl = P[:, self.idx["Hips"]] - self.skel.J[0]
        aa = M.rotation_matrix_to_axis_angle(L.reshape(-1, 3, 3)).view(T, 24, 3)
        return aa, transl


def load_bvh(path):
    text = Path(path).read_text(errors="replace")
    hier = parse_hierarchy(text)
    nch = sum(len(c) for c in hier[3])
    nfr, dt, data = parse_motion(text)
    if data.size != nfr * nch:
        nfr = data.size // nch; data = data[:nfr * nch]
    return hier, dt, data.reshape(nfr, nch)


def check(conv, skel, hier, G, P, tag=""):
    """Numeric sanity checks on calibration frames: joint positions of the retargeted SMPL pose vs the rig (pelvis-
    relative, cm), and the dominant local rotation axes of knees / elbows (SMPL: knees flex about +x, elbows about y)."""
    aa, tr = conv(G, P)
    L = M.axis_angle_to_rotation_matrix(aa.reshape(-1, 3)).view(-1, 24, 3, 3)
    Gs, Ps = skel.fk(L, tr)
    err = []
    for k in range(22):
        m = conv.map_idx[k]
        e = ((Ps[:, k] - Ps[:, 0]) - (P[:, m] - P[:, conv.idx["Hips"]])).norm(dim=-1).mean() * 100
        err.append(f"{k}:{e:.1f}")
    print(f"{tag}joint position error vs rig (pelvis-relative, cm): " + " ".join(err), flush=True)
    for k, (c, tgt) in HINGE.items():
        # hinge normal of the child bone in the FINAL parent frame (should be the SMPL hinge axis `tgt`)
        cm, cc = conv.idx[RIG_MAP[c]], conv.idx[RIG_CHILD[RIG_MAP[c]]]
        w = unit(P[:, cc] - P[:, cm])
        s = torch.einsum("nji,nj->ni", Gs[:, k], w)
        d = conv.d_smpl[k]
        hn = unit(torch.cross(d.expand_as(s), s, dim=-1).sum(0))
        ang = aa[:, c].norm(dim=-1); ax = (unit(aa[:, c]) * ang[:, None]).sum(0); ax = unit(ax)
        print(f"{tag}hinge {k}->{c}: plane normal in the final frame ({hn[0]:+.2f},{hn[1]:+.2f},{hn[2]:+.2f}) target {tgt}; "
              f"angle-weighted local rotation axis of {c}: ({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f}); rest dirs d_k ({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}) "
              f"d_c ({conv.d_smpl[c][0]:+.2f},{conv.d_smpl[c][1]:+.2f},{conv.d_smpl[c][2]:+.2f})", flush=True)
    # facing: horizontal direction of each SMPL frame's +z axis vs the rig's mean foot direction (toe - ankle)
    ft = unit(((P[:, conv.idx["LeftToe"]] - P[:, conv.idx["LeftFoot"]]) + (P[:, conv.idx["RightToe"]] - P[:, conv.idx["RightFoot"]])) * torch.tensor([1., 0., 1.], device=P.device))
    for k, nm in ((0, "pelvis"), (1, "l_thigh"), (2, "r_thigh"), (9, "chest"), (15, "head")):
        z = Gs[:, k, :, 2] * torch.tensor([1., 0., 1.], device=P.device); z = unit(z)
        yaw = torch.rad2deg(torch.atan2(torch.cross(ft, z, dim=-1)[:, 1], (ft * z).sum(-1)))
        print(f"{tag}facing of SMPL {nm:8s} +z vs rig feet: mean yaw {yaw.mean():+.1f} deg (std {yaw.std():.1f})", flush=True)
    # knee hinge normal expressed in the PELVIS frame (should be +x if the pelvis yaw is right)
    for c, (cm, cc) in ((4, ("LeftLeg", "LeftFoot")), (5, ("RightLeg", "RightFoot"))):
        tw = unit(P[:, conv.idx[cm]] - P[:, conv.idx[RIG_MAP[SMPL_PARENTS[c]]]]); sw = unit(P[:, conv.idx[cc]] - P[:, conv.idx[cm]])
        hn_w = torch.cross(tw, sw, dim=-1); hn_p = unit(torch.einsum("nji,nj->ni", Gs[:, 0], hn_w).sum(0))
        print(f"{tag}knee {c} hinge normal in the pelvis frame: ({hn_p[0]:+.2f},{hn_p[1]:+.2f},{hn_p[2]:+.2f})", flush=True)
    for k, nm in ((4, "l_knee"), (5, "r_knee"), (18, "l_elbow"), (19, "r_elbow"), (1, "l_hip"), (2, "r_hip"), (16, "l_shoulder"), (15, "head")):
        v = aa[:, k]; big = v.norm(dim=-1) > math.radians(15)
        mean_axis = unit(v[big]).mean(0) if big.any() else torch.zeros(3)
        print(f"{tag}{nm:10s}: {int(big.sum())}/{len(v)} frames > 15 deg, mean rotation axis (x,y,z) = "
              f"({mean_axis[0]:+.2f}, {mean_axis[1]:+.2f}, {mean_axis[2]:+.2f}), mean |angle| {torch.rad2deg(v.norm(dim=-1).mean()):.1f} deg", flush=True)
    return aa, tr, Ps


def plot(conv, skel, hier, G, P, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    aa, tr, Ps = check(conv, skel, hier, G, P, tag="[plot] ")
    names, parents, _, _ = hier
    fig, axes = plt.subplots(2, G.shape[0], figsize=(3.2 * G.shape[0], 6.5))
    for i in range(G.shape[0]):
        for row, (X, Y) in enumerate(((0, 1), (2, 1))):
            ax = axes[row, i]
            p = P[i].cpu().numpy(); q = Ps[i].cpu().numpy()
            for j, pa in enumerate(parents):
                if pa >= 0: ax.plot([p[j, X], p[pa, X]], [p[j, Y], p[pa, Y]], "b-", lw=2, alpha=.6)
            for j, pa in enumerate(SMPL_PARENTS):
                if pa >= 0: ax.plot([q[j, X], q[pa, X]], [q[j, Y], q[pa, Y]], "r--", lw=1.5)
            ax.set_aspect("equal"); ax.set_title(f"frame {i} {'XY (front)' if row == 0 else 'ZY (side)'}", fontsize=8)
    fig.suptitle("blue: LaFAN rig   red dashed: retargeted SMPL (FK of the pose targets)")
    fig.tight_layout(); fig.savefig(out_png, dpi=80); print("wrote", out_png, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--tag", default="STAYSTILL")
    ap.add_argument("--chunk_hours", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N clips (per folder order)")
    ap.add_argument("--calib_clips", type=int, default=120, help="clips sampled (evenly over the dataset) for the twist calibration")
    ap.add_argument("--min_frames", type=int, default=30)
    ap.add_argument("--plot", default="", help="write a stick-figure check PNG (rig vs retargeted SMPL) to this path")
    ap.add_argument("--pack", action="store_true", help="also pack the chunks into memmap shards")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(f"cuda:{a.gpu}")
    cfg = Config(project_root_dir=str(Path(__file__).resolve().parents[2]), device=a.gpu, mkdir=False)
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    skel = SMPLSkeleton(bm, dev)
    files = sorted(SRC.glob("*/*.bvh"))
    assert files, f"no BVH under {SRC}"
    print(f"{len(files)} clips under {SRC}", flush=True)
    out25 = OUT / "processed_imuposer_25fps"; out25.mkdir(parents=True, exist_ok=True)
    if sorted(out25.glob(f"{a.tag}_*.pt")) and not a.limit and not a.plot:
        print(f"chunks already exist for {a.tag}; delete them to redo", flush=True); return

    # ---- calibration frames: every k-th clip, every 10th frame
    step = max(1, len(files) // a.calib_clips)
    hier0 = None; Gc, Pc = [], []
    with torch.no_grad():
        for f in files[::step]:
            hier, dt, data = load_bvh(f)
            if hier0 is None:
                hier0 = hier
            elif hier[0] != hier0[0] or not np.allclose(hier[2], hier0[2], atol=1e-3):
                print(f"  calib: SKIP {f.name} (different rig)", flush=True); continue
            G, P = bvh_fk(data[::10], hier, dev)
            Gc.append(G); Pc.append(P)
        Gc = torch.cat(Gc); Pc = torch.cat(Pc)
        print(f"calibrating the 22 joint frames on {Gc.shape[0]} frames from {len(files[::step])} clips "
              f"(rig: {len(hier0[0])} joints, frame time {dt:.4f})", flush=True)
        conv = LafanToSMPL(skel, hier0, Gc, Pc)
        check(conv, skel, hier0, Gc[::7], Pc[::7], tag="[calib] ")
        if a.plot:
            sel = torch.linspace(0, Gc.shape[0] - 1, 5).long()
            plot(conv, skel, hier0, Gc[sel], Pc[sel], a.plot)
            if not a.limit:
                return

    buf, hours, cid, nclips, nskip, t0 = [], 0.0, 0, 0, 0, time.time()
    chunk_frames = a.chunk_hours * 3600 * 60

    def flush():
        nonlocal buf, cid, hours
        if not buf: return
        out = synthesize_sequences(buf, bm, dev)
        fdata = amass_dir_to_25fps(out, device=dev)
        p = out25 / f"{a.tag}_{cid:03d}.pt"
        torch.save(fdata, p.with_suffix(".pt.tmp")); os.replace(p.with_suffix(".pt.tmp"), p)
        h = sum(x.shape[0] for x in fdata["pose"]) / 25 / 3600; hours += h
        acc = torch.cat([x[:, :2].norm(dim=-1).flatten() for x in fdata["acc"]])
        print(f"  wrote {p.name}: {len(buf)} clips, {h:.2f} h | total {hours:.1f} h, {nclips} clips, {(time.time()-t0)/60:.1f} min; "
              f"wrist |acc| p50/p95 {acc.quantile(.5):.1f}/{acc.quantile(.95):.1f} m/s^2", flush=True)
        buf = []; cid += 1

    with torch.no_grad():
        for f in files:
            hier, dt, data = load_bvh(f)
            if data.shape[0] < a.min_frames or not np.isfinite(data).all() or abs(1 / dt - SRC_FPS) > 1:
                print(f"  SKIP {f.name}: {data.shape[0]} frames, dt {dt}", flush=True); nskip += 1; continue
            if hier[0] != hier0[0] or not np.allclose(hier[2], hier0[2], atol=1e-3):
                print(f"  SKIP {f.name}: different rig", flush=True); nskip += 1; continue
            G, P = bvh_fk(data, hier, dev)
            aa, tr = conv(G, P)
            if not (torch.isfinite(aa).all() and torch.isfinite(tr).all()):
                nskip += 1; continue
            aa60 = resample_pose_aa(aa, SRC_FPS, 60.0).cpu(); tr60 = resample_linear(tr.cpu(), SRC_FPS, 60.0)
            buf.append((aa60, tr60, torch.zeros(10)))
            nclips += 1
            if sum(x[0].shape[0] for x in buf) >= chunk_frames:
                flush()
            if a.limit and nclips >= a.limit:
                break
        flush()
    print(f"DONE {a.tag}: {nclips} clips ({nskip} skipped), {hours:.2f} h in {(time.time()-t0)/60:.1f} min", flush=True)
    if a.pack:
        from imuposer.datasets.shards import ensure_packed, shard_root_for
        root = shard_root_for(out25)
        for p in sorted(out25.glob(f"{a.tag}_*.pt")):
            ensure_packed(p, root, verbose=False)
        print(f"packed {a.tag} chunks into {root}", flush=True)


if __name__ == "__main__":
    main()
