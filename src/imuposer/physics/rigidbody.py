r"""
Reduced-coordinate articulated rigid-body simulator for physics refinement (PIP/PNP-style).

PIP (Physics-based Inertial Poser, Yi et al. CVPR'22) refines the network's kinematic pose with a
rigid-body physics tracker: PD-servo torques drive a dynamics model toward the kinematic target while
gravity / contact keep it physical. We have no physics engine, so this implements the core in reduced
coordinates: each SMPL segment is a rigid body (mass ∝ bone length); every joint is a critically-damped
2nd-order rotational system actuated by a PD term toward the network's target local rotation, plus a
gravity torque from the joint's subtree weight. Integrated over the sequence with semi-implicit Euler.

We drop the contact/translation QP from PIP on purpose: our metric is ROOT-RELATIVE rotation (SIP), so
ground contact / global-translation correction (what contact mainly fixes) is not scored. The root
rotation is taken from the network (it sets the gravity direction) and is not simulated.

Test-time only: physics_refine(R_local_seq, model) -> refined R_local_seq. Knobs via env:
PHYS_OMEGA0 (tracking stiffness rad/s, 40), PHYS_ZETA (damping ratio, 1.0), PHYS_GRAVITY (gravity
scale, 1.0), PHYS_SUBSTEPS (per-frame substeps, 4), PHYS_FPS (25).
"""
import os
import torch
import torch.nn as nn

_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
_LEAF_TO_ROOT = sorted(range(24), key=lambda i: -i)       # children before parents


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0],
                        -v[..., 1], v[..., 0], z], -1).reshape(v.shape[:-1] + (3, 3))


def _expmap(w):                                # axis-angle -> R
    ang = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    K = _skew(w / ang)
    I = torch.eye(3, device=w.device).expand(K.shape)
    s = torch.sin(ang)[..., None]; c = torch.cos(ang)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def _logmap(R):                                # R -> axis-angle
    cos = (((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.acos(cos)[..., None]
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1)
    return v / (2 * torch.sin(ang).clamp(min=1e-7)) * ang


def _inertia(model, device):
    j, _ = model.get_zero_pose_joint_and_vertex()
    J = j[:24].to(device)
    blen = torch.tensor([float((J[i] - J[_PARENT[i]]).norm()) if _PARENT[i] >= 0 else 0.0
                         for i in range(24)], device=device)
    m = blen + 0.02                                        # per-segment mass proxy ∝ bone length
    children = [[k for k in range(24) if _PARENT[k] == i] for i in range(24)]
    M = m.clone()
    for i in _LEAF_TO_ROOT:
        for k in children[i]:
            M[i] = M[i] + M[k]
    I = (M * blen.clamp(min=0.05) ** 2).clamp(min=1e-3)    # effective rotational inertia about joint
    return m, M, I, children


@torch.no_grad()
def physics_refine(R_local, model, device=None):
    r"""R_local: (T,24,3,3) network local rotations -> physics-refined (T,24,3,3)."""
    device = device or R_local.device
    R_local = R_local.to(device)
    T = R_local.shape[0]
    m, M, I, children = _inertia(model, device)
    omega0 = float(os.environ.get("PHYS_OMEGA0", "40"))
    zeta = float(os.environ.get("PHYS_ZETA", "1.0"))
    gscale = float(os.environ.get("PHYS_GRAVITY", "1.0"))
    S = int(os.environ.get("PHYS_SUBSTEPS", "4"))
    fps = float(os.environ.get("PHYS_FPS", "25"))
    dt = 1.0 / (fps * S)
    g = torch.tensor([0., -9.81, 0.], device=device)
    sim = torch.zeros(24, dtype=torch.bool, device=device)
    for i in range(24):
        if _PARENT[i] >= 0:
            sim[i] = True

    R = R_local[0].clone()
    w = torch.zeros(24, 3, device=device)
    out = torch.empty_like(R_local)
    out[0] = R_local[0]
    for t in range(1, T):
        tgt = R_local[t]
        for _ in range(S):
            grot, pos = model.forward_kinematics(pose=R.reshape(1, 216))[:2]
            grot, pos = grot[0], pos[0]                    # (24,3,3),(24,3)
            # subtree COM (mass-weighted) via leaf->root accumulation
            num = m[:, None] * pos                          # numerator Σ m_k pos_k
            num = num.clone()
            for i in _LEAF_TO_ROOT:
                for k in children[i]:
                    num[i] = num[i] + num[k]
            Csub = num / M[:, None]                          # subtree COM about world
            tau_g = torch.cross(Csub - pos, M[:, None] * g, dim=-1) * gscale   # world gravity torque
            tau_g_local = torch.einsum('jab,jb->ja', grot.transpose(-1, -2), tau_g)
            err = _logmap(torch.einsum('jab,jbc->jac', R.transpose(-1, -2), tgt))   # body-frame error
            alpha = omega0 ** 2 * err - 2 * zeta * omega0 * w + tau_g_local / I[:, None]
            w_new = w + alpha * dt
            # body-frame angular velocity -> RIGHT-multiply (R @ exp), matching the body-frame error
            R_new = torch.einsum('jab,jbc->jac', R, _expmap(w_new * dt))
            w = torch.where(sim[:, None], w_new, torch.zeros_like(w))
            R = torch.where(sim[:, None, None], R_new, tgt)
        out[t] = R
    return out


class PhysicsRefineWrapper(nn.Module):
    r"""Wrap a base pose model so forward() returns the PHYSICS-REFINED r6d pose. Used at eval only
    (PHYS_REFINE=1); the protected metric is unchanged — it just scores a different (refined) pose."""
    def __init__(self, base, body_model):
        super().__init__()
        self.base = base
        self.pm = body_model

    def forward(self, imu_inputs, imu_lens):
        from imuposer.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_r6d
        dev = imu_inputs.device
        pred = self.base(imu_inputs, imu_lens)[:, :, :144]     # (B,T,144) r6d
        B, T, _ = pred.shape
        R = r6d_to_rotation_matrix(pred).view(B, T, 24, 3, 3).cpu()  # physics on CPU (no launch overhead)
        ref = torch.stack([physics_refine(R[b], self.pm, device="cpu") for b in range(B)])
        return rotation_matrix_to_r6d(ref.reshape(B * T, 216)).view(B, T, 144).to(dev)
