r"""
Shared IMU-synthesis helpers for new motion datasets (BONES-SEED, form-hoi, MotionMillion, ...).

One code path, identical to the AMASS / Nymeria pipeline:
  SMPL pose (axis-angle, DIP y-up world frame) + translation + betas
    -> SMPL forward kinematics (6890-vertex mesh)
    -> 6 virtual IMUs [lw, rw, lp, rp, h, pelvis]: global bone orientation + 2nd-difference vertex accel
    -> AMASS-style 60 fps folder {pose, shape, tran, joint, vrot, vacc}.pt
    -> 25 fps training file (lerp resample + 5-tap accel average), same as stage 2 of the pipeline.

`synthesize_sequences` runs FK in frame batches so long sequences never materialise the full mesh.
"""
from pathlib import Path

import numpy as np
import torch

from imuposer import math as M

# left wrist, right wrist, left thigh, right thigh, head, pelvis  (identical to the AMASS pipeline)
VI_MASK = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
JI_MASK = torch.tensor([18, 19, 1, 2, 15, 0])
# AMASS / SMPL z-up  ->  DIP y-up (the standard `amass_rot`)
AMASS_ROT = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]])


def syn_acc(v, fps=60.0):
    """2nd-difference accel from vertex positions (verbatim from the AMASS pipeline at 60 fps)."""
    f2 = fps * fps
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * f2 for i in range(0, v.shape[0] - 2)])
    return torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))


def resample_linear(arr, src_fps, dst_fps):
    """Linear resample of a (N, ...) array/tensor from src_fps to dst_fps (float-safe)."""
    is_t = torch.is_tensor(arr)
    a = arr.numpy() if is_t else np.asarray(arr)
    n = a.shape[0]
    # same grid + end clamp as _resample60 and resample_pose_aa, so all fields of a sequence keep one length
    idx = torch.arange(0, n, src_fps / dst_fps).numpy()
    lo = np.minimum(np.floor(idx).astype(np.int64), n - 1)
    hi = np.minimum(np.ceil(idx).astype(np.int64), n - 1)
    w = (idx - lo).reshape((-1,) + (1,) * (a.ndim - 1)).astype(np.float32)
    out = (a[lo] * (1 - w) + a[hi] * w).astype(np.float32)
    return torch.from_numpy(out) if is_t else out


def _project_so3(Rm):
    """Project (...,3,3) near-rotations onto SO(3) by Gram-Schmidt on the first two columns."""
    c1, c2 = Rm[..., :, 0], Rm[..., :, 1]
    b1 = c1 / c1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c2 = c2 - (b1 * c2).sum(-1, keepdim=True) * b1
    b2 = c2 / c2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def resample_pose_aa(aa, src_fps, dst_fps):
    """Resample an axis-angle pose sequence (T,J,3) by interpolating ROTATION MATRICES (then projecting back
    onto SO(3)) instead of axis-angle vectors.

    WHY: linear interpolation of axis-angle is wrong whenever consecutive frames straddle the +-pi wrap
    (the representation flips by ~2 pi): the interpolated vector collapses toward the identity, the limb
    snaps to rest for one frame, and the 2nd-difference accel explodes (1.7-10.8 % of frames above 50 m/s^2
    in form-hoi / MotionMillion, max in the thousands, vs 0.00 % for the stride-decimated BONES-SEED).
    Matrix interpolation + projection has no such discontinuity (for the 2x upsample it is the midpoint
    rotation to within 1e-4)."""
    is_t = torch.is_tensor(aa)
    a = aa if is_t else torch.from_numpy(np.asarray(aa))
    T, J = a.shape[0], a.shape[1]
    R = M.axis_angle_to_rotation_matrix(a.reshape(-1, 3).float()).view(T, J, 3, 3)
    # same index grid as _resample60 / resample_linear (torch.arange(0, T, step) with the end index clamped),
    # so every field of a sequence keeps the same number of frames
    idx = torch.arange(0, T, src_fps / dst_fps).numpy()
    lo = torch.from_numpy(np.minimum(np.floor(idx).astype(np.int64), T - 1))
    hi = torch.from_numpy(np.minimum(np.ceil(idx).astype(np.int64), T - 1))
    w = torch.from_numpy((idx - np.floor(idx)).astype(np.float32)).view(-1, 1, 1, 1)
    Rm = R[lo] * (1 - w) + R[hi] * w
    Rp = _project_so3(Rm)
    out = M.rotation_matrix_to_axis_angle(Rp.reshape(-1, 3, 3)).view(-1, J, 3)
    return out if is_t else out.numpy()


def zup_to_dip(pose_aa, tran):
    """Rotate a z-up SMPL sequence (AMASS convention) into DIP's y-up frame. In place on copies."""
    pose_aa = pose_aa.clone(); tran = tran.clone()
    tran = AMASS_ROT.matmul(tran.unsqueeze(-1)).squeeze(-1)
    pose_aa[:, 0] = M.rotation_matrix_to_axis_angle(
        AMASS_ROT.unsqueeze(0).matmul(M.axis_angle_to_rotation_matrix(pose_aa[:, 0])))
    return pose_aa, tran


_SKEL_CACHE = {}


def _skeleton(body_model, device):
    from imuposer.datasets.soma_retarget import SMPLSkeleton
    key = (id(body_model), str(device))
    if key not in _SKEL_CACHE:
        _SKEL_CACHE[key] = SMPLSkeleton(body_model, device)
    return _SKEL_CACHE[key]


@torch.no_grad()
def synthesize_sequences(seqs, body_model, device, batch=2048, fps=60.0, fast=True):
    """seqs: iterable of (pose_aa (T,24,3), tran (T,3), shape (10,)) at `fps`, DIP y-up frame.
    Returns dict of lists {pose, shape, tran, joint, vrot, vacc} (CPU tensors), AMASS-style.

    fast=True (default): mesh-free path -- FK + exact linear-blend skinning of ONLY the 6 sensor vertices
    (soma_retarget.SMPLSkeleton; verified identical to the full 6890-vertex mesh to ~1e-4 relative, ~70x
    faster). Requires betas = 0 (all new datasets use the neutral body, like Nymeria). fast=False runs the
    original full-mesh ParametricModel path."""
    vim, jim = VI_MASK.to(device), JI_MASK.to(device)
    out = {k: [] for k in ("pose", "shape", "tran", "joint", "vrot", "vacc")}
    skel = _skeleton(body_model, device) if fast else None
    for pose, tran, shape in seqs:
        T = pose.shape[0]
        if T <= 12:
            continue
        if fast:
            from imuposer.datasets.soma_retarget import imu_from_pose
            assert float(shape.abs().max()) == 0.0, "fast path assumes betas=0"
            P, vrot, vacc = imu_from_pose(skel, pose.to(device), tran.to(device), fps)
            out["pose"].append(pose.clone().float()); out["tran"].append(tran.clone().float()); out["shape"].append(shape.clone().float())
            out["joint"].append(P.cpu()); out["vrot"].append(vrot.cpu()); out["vacc"].append(vacc.cpu())
            continue
        joints, verts, grots = [], [], []
        for b in range(0, T, batch):
            p = M.axis_angle_to_rotation_matrix(pose[b:b + batch].reshape(-1, 3).to(device)).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape.to(device), tran[b:b + batch].to(device), calc_mesh=True)
            joints.append(joint[:, :24].contiguous().cpu())
            verts.append(vert[:, vim].cpu())
            grots.append(grot[:, jim].cpu())
            del grot, joint, vert, p
        out["pose"].append(pose.clone().float())
        out["tran"].append(tran.clone().float())
        out["shape"].append(shape.clone().float())
        out["joint"].append(torch.cat(joints))
        out["vacc"].append(syn_acc(torch.cat(verts), fps))
        out["vrot"].append(torch.cat(grots))
    return out


def save_amass_style(out, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        torch.save(v, out_dir / f"{k}.pt")
    return sum(p.shape[0] for p in out["pose"])


# ---- stage 2: 60 fps folder -> 25 fps training file (mirrors "2. preprocess_all_to_imuposer_at_25fps.py")
def _smooth_avg(acc, s=3):
    nan_tensor = torch.zeros((s // 2, acc.shape[1], acc.shape[2])) * torch.nan
    acc = torch.cat((nan_tensor, acc, nan_tensor))
    tensors = []
    for i in range(s):
        L = acc.shape[0]
        tensors.append(acc[i:L - (s - i - 1)])
    return torch.stack(tensors).nanmean(dim=0)


def _resample60(tensor, target_fps):
    indices = torch.arange(0, tensor.shape[0], 60 / target_fps)
    start_indices = torch.floor(indices).long()
    end_indices = torch.ceil(indices).long()
    end_indices[end_indices >= tensor.shape[0]] = tensor.shape[0] - 1
    start = tensor[start_indices]
    end = tensor[end_indices]
    floats = indices - start_indices
    for _ in range(len(tensor.shape) - 1):
        floats = floats.unsqueeze(1)
    weights = torch.ones_like(start) * floats
    return torch.lerp(start, end, weights)


def amass_dir_to_25fps(out, target_fps=25):
    """`out` is the dict returned by synthesize_sequences (or loaded from an AMASS-style folder)."""
    fdata = {
        "joint": [_resample60(x, target_fps) for x in out["joint"]],
        # pose targets: interpolate rotations in matrix space (see resample_pose_aa), not axis-angle
        "pose": [M.axis_angle_to_rotation_matrix(resample_pose_aa(x, 60.0, float(target_fps)).reshape(-1, 3).contiguous()).view(-1, 24, 3, 3)
                 for x in out["pose"]],
        "shape": out["shape"],
        "tran": [_resample60(x, target_fps) for x in out["tran"]],
        "acc": [_smooth_avg(_resample60(x, target_fps), s=5) for x in out["vacc"]],
        "ori": [_resample60(x, target_fps) for x in out["vrot"]],
    }
    return fdata


def load_amass_dir(d):
    d = Path(d)
    return {k: torch.load(d / f"{k}.pt", weights_only=False) for k in ("pose", "shape", "tran", "joint", "vrot", "vacc")}
