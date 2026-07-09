r"""
Retarget the WHIP dataset (ECCV 2026, real wearable IMU + 69-joint mocap) into our SMPL / IMUPoser
format so it can serve as REAL training data for the lw_rp_h (and every) sensor combo.

WHIP gives, per clip: 69-joint mocap positions, 4 real IMUs (watch_left/right, phone_left/right; each
orientation + gravity-removed accel in g + gyro) and a VR head 6-DoF pose. We fit SMPL (shape + per-frame
pose + per-sensor two-sided calibration) to the mocap joints AND the IMU orientations jointly -- the IMUs
supply the bone TWIST that joint positions leave unconstrained. Output matches the 25fps DIP format:
  pose (T,24,3,3) SMPL local rot ; ori (T,6,3,3) calibrated REAL sensor ori in SMPL-global (bone) frame ;
  acc (T,6,3) REAL accel rotated to SMPL-global, m/s^2, gravity-removed.  Sensor order [lw,rw,lp,rp,h,(pelvis=0)].

Sensor -> SMPL joint (same ji_mask as our synth): lw->18 rw->19 lp->1 rp->2 h->15.
30fps -> 25fps by linear resample (axis-angle pose; matrices re-orthonormalised).

  uv run python "scripts/1. Preprocessing/whip_retarget.py" --gpu 0 --shard 0/2 --iters 1000
"""
import argparse, glob, io, os, tarfile, time
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, "/tmp/whip_repo")
from whip.skeleton import DATASET_KEYPOINTS
from imuposer.config import Config, amass_combos
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import (r6d_to_rotation_matrix, rotation_matrix_to_r6d,
                                   rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix,
                                   angle_between, radian_to_degree)

WHIP = Path("/media/vimal/T7_2TB/CHI23/whip/data")
OUT = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/whip_25fps")
G = 9.81
SRC_FPS, DST_FPS = 30.0, 25.0
# WHIP mocap name -> SMPL joint idx (18 correspondences for the position term)
JMAP = {'Hips': 0, 'LeftUpLeg': 1, 'RightUpLeg': 2, 'LeftLeg': 4, 'RightLeg': 5, 'LeftFoot': 7, 'RightFoot': 8,
        'LeftToeBase': 10, 'RightToeBase': 11, 'Neck': 12, 'Head': 15, 'LeftArm': 16, 'RightArm': 17,
        'LeftForeArm': 18, 'RightForeArm': 19, 'LeftHand': 20, 'RightHand': 21, 'Spine2': 6}
# 5 sensors, in our [lw,rw,lp,rp,h] order: (whip source, SMPL bone joint)
SENSORS = [("watch_left", 18), ("watch_right", 19), ("phone_left", 1), ("phone_right", 2), ("vr", 15)]
EYE6 = torch.tensor([1., 0, 0, 0, 1, 0])


def load_member(tar, m, pickle=False):
    return np.load(io.BytesIO(tar.extractfile(m).read()), allow_pickle=pickle)


def load_clip(path):
    with tarfile.open(path) as t:
        J = load_member(t, "body_tracking/joints_3D.npz")["translations"].astype(np.float32)   # (T,69,3)
        imu = {}
        for dev in ("watch_left", "watch_right", "phone_left", "phone_right"):
            d = load_member(t, f"imu/{dev}.npz")
            imu[dev] = (d["orientation"].astype(np.float32), d["acceleration"].astype(np.float32))
        vr = load_member(t, "vr/poses.npz")
        imu["vr"] = (vr["rotations"].astype(np.float32), None)                                   # head: no accel
    return J, imu


def resample_lin(x, src=SRC_FPS, dst=DST_FPS):
    n = x.shape[0]; idx = np.arange(0, n - 1e-6, src / dst)
    lo = np.floor(idx).astype(np.int64); hi = np.minimum(lo + 1, n - 1)
    w = (idx - lo).reshape((-1,) + (1,) * (x.ndim - 1)).astype(np.float32)
    return x[lo] * (1 - w) + x[hi] * w


def retarget(J, imu, bm, dev, wi, si, iters, w_ori=5.0):
    T = J.shape[0]
    tgt = torch.tensor(J[:, wi], device=dev); tgt = tgt - tgt[:, :1]
    meas = [torch.tensor(imu[s][0], device=dev) for s, _ in SENSORS]        # (T,3,3) each
    beta = torch.zeros(1, 10, device=dev, requires_grad=True)
    r6 = rotation_matrix_to_r6d(torch.eye(3, device=dev).repeat(T, 24, 1, 1)).view(T, 24, 6).clone().requires_grad_(True)
    cal = EYE6.to(dev).repeat(len(SENSORS), 2, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([r6, beta, cal], lr=0.03)
    si_t = torch.tensor(si, device=dev); bj = [j for _, j in SENSORS]
    for it in range(iters):
        opt.zero_grad()
        R = r6d_to_rotation_matrix(r6.reshape(-1, 6)).view(T, 24, 3, 3)
        grot, jp = bm.forward_kinematics(R, shape=beta.expand(T, 10)); jp = jp - jp[:, :1]
        Lpos = ((jp[:, si_t] - tgt) * 100).pow(2).sum(-1).mean()
        Rc = r6d_to_rotation_matrix(cal.reshape(-1, 6)).view(len(SENSORS), 2, 3, 3)
        Lori = sum(((torch.einsum('ij,njk,kl->nil', Rc[k, 0], meas[k], Rc[k, 1]) - grot[:, bj[k]]) ** 2).sum((-1, -2)).mean()
                   for k in range(len(SENSORS)))
        loss = Lpos + w_ori * Lori + 0.02 * ((r6[2:] - 2 * r6[1:-1] + r6[:-2]) ** 2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_([r6, beta, cal], 5.0); opt.step()
    with torch.no_grad():
        R = r6d_to_rotation_matrix(r6.reshape(-1, 6)).view(T, 24, 3, 3)
        grot, jp = bm.forward_kinematics(R, shape=beta.expand(T, 10)); jp = jp - jp[:, :1]
        pe = (jp[:, si_t] - tgt).norm(dim=-1).mean().item() * 100
        Rc = r6d_to_rotation_matrix(cal.reshape(-1, 6)).view(len(SENSORS), 2, 3, 3)
        ori6 = torch.zeros(T, 6, 3, 3, device=dev); ori6[:, :, [0, 1, 2]] = torch.eye(3, device=dev)[None, None]
        acc6 = torch.zeros(T, 6, 3, device=dev); resid = []
        for k, (s, j) in enumerate(SENSORS):
            cori = torch.einsum('ij,njk,kl->nil', Rc[k, 0], meas[k], Rc[k, 1])   # calibrated real ori (SMPL-global bone)
            ori6[:, k] = cori
            resid.append(radian_to_degree(angle_between(cori, grot[:, j])).mean().item())
            if imu[s][1] is not None:                                            # accel -> SMPL-global m/s^2
                a = torch.tensor(imu[s][1], device=dev)
                acc6[:, k] = torch.einsum('ij,njk,nk->ni', Rc[k, 0], meas[k], a) * G
        return R.cpu(), ori6.cpu(), acc6.cpu(), pe, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--iters", type=int, default=1000); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=os.getcwd(), joints_set=amass_combos["global"],
                 r6d=True, device=a.gpu, mkdir=False)
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    wi = [DATASET_KEYPOINTS.index(n) for n in JMAP]; si = [JMAP[n] for n in JMAP]
    OUT.mkdir(parents=True, exist_ok=True)
    clips = sorted(glob.glob(str(WHIP / "*/actions/*.tar")))
    si_idx, sn = (int(x) for x in a.shard.split("/"))
    clips = [c for i, c in enumerate(clips) if i % sn == si_idx]
    if a.limit: clips = clips[:a.limit]
    print(f"[gpu{a.gpu} shard {a.shard}] {len(clips)} clips", flush=True)
    qc = []
    for ci, path in enumerate(clips):
        seq = Path(path).parents[1].name; act = Path(path).stem
        out = OUT / f"{seq}__{act}.pt"
        if out.exists(): continue
        t0 = time.time()
        try:
            J, imu = load_clip(path)
            pose, ori, acc, pe, resid = retarget(J, imu, bm, dev, wi, si, a.iters)
            pose = torch.as_tensor(resample_lin(pose.numpy()))     # (T',24,3,3)
            ori = torch.as_tensor(resample_lin(ori.numpy())); acc = torch.as_tensor(resample_lin(acc.numpy()))
            torch.save({"pose": pose, "ori": ori, "acc": acc, "seq": seq, "action": act,
                        "pos_err_cm": pe, "ori_resid_deg": resid}, out)
            qc.append((seq, act, pe, resid))
            if ci % 20 == 0:
                print(f"  [{ci}/{len(clips)}] {seq}/{act} pos={pe:.2f}cm resid={[round(r,1) for r in resid]} ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  ERR {seq}/{act}: {e}", flush=True)
    if qc:
        pes = np.array([q[2] for q in qc]); rs = np.array([q[3] for q in qc])
        print(f"[gpu{a.gpu}] done {len(qc)} clips | pos {pes.mean():.2f}cm | resid mean {rs.mean(0).round(1)} (lw,rw,lp,rp,h)", flush=True)


if __name__ == "__main__":
    main()
