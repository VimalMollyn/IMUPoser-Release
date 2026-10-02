r"""
Closed-form SOMA -> SMPL retarget (no mesh fitting) + mesh-free IMU synthesis.

WHY NOT SOMA-X's PoseInversion: it poses the SOMA mesh, bridges it to SMPL topology and fits SMPL
rotations to the vertices (~270 fps on a TITAN X: 16 h for BONES-SEED alone) and still left 7-13 deg
of bone-direction error on form-hoi. We only need two things from SOMA: the GLOBAL rotation of each
body joint and the joint positions, both of which forward kinematics gives directly.

RETARGET: for every SMPL joint k with a SOMA counterpart m(k),
    G_smpl(k) = G_soma(m(k)) @ Off(k),    Off(k) = R_align(d_smpl_rest(k) -> d_soma_rest(k))
where d_*_rest are the world-frame bone directions (joint -> child) of the two rigs in their zero pose
(both are T-poses; the offset absorbs the few-degree differences in rest limb directions). Local SMPL
rotations are then L(k) = G_smpl(parent k)^T G_smpl(k). By construction the SMPL bone directions equal
the SOMA ones, and the twist about each bone is inherited from SOMA. Hands (22, 23) are zero as in the
rest of the pipeline. transl = SOMA Hips world position - SMPL rest pelvis.

IMU WITHOUT LBS: the 6 sensor vertices are attached rigidly to their bone (rest offset in the bone
frame, from the SMPL rest mesh): p_v(t) = p_j(t) + G_j(t) r_v. `validate_rigid_accel` checks this
against the true skinned-mesh accel on an AMASS file (see the conversion scripts' logs).
"""
import numpy as np
import torch

from imuposer import math as M

# SMPL kinematic tree (24 joints)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_NAMES = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3",
              "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow",
              "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"]
# SMPL joint -> SOMA joint name (None = no counterpart -> identity local rotation)
SOMA_MAP = {0: "Hips", 1: "LeftLeg", 2: "RightLeg", 3: "Spine1", 4: "LeftShin", 5: "RightShin", 6: "Spine2",
            7: "LeftFoot", 8: "RightFoot", 9: "Chest", 10: "LeftToeBase", 11: "RightToeBase", 12: "Neck1",
            13: "LeftShoulder", 14: "RightShoulder", 15: "Head", 16: "LeftArm", 17: "RightArm", 18: "LeftForeArm",
            19: "RightForeArm", 20: "LeftHand", 21: "RightHand", 22: None, 23: None}
# bone used to define each joint's rest DIRECTION (joint -> child); SOMA child names likewise
SMPL_CHILD = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19,
              18: 20, 19: 21}
SOMA_CHILD = {"Hips": "Spine1", "LeftLeg": "LeftShin", "RightLeg": "RightShin", "Spine1": "Spine2", "LeftShin": "LeftFoot",
              "RightShin": "RightFoot", "Spine2": "Chest", "LeftFoot": "LeftToeBase", "RightFoot": "RightToeBase",
              "Chest": "Neck1", "Neck1": "Head", "LeftShoulder": "LeftArm", "RightShoulder": "RightArm",
              "LeftArm": "LeftForeArm", "RightArm": "RightForeArm", "LeftForeArm": "LeftHand", "RightForeArm": "RightHand"}
# sensors: vertex id, bone joint (same as the AMASS pipeline: lw, rw, lp, rp, h, pelvis)
VI_MASK = [1961, 5424, 876, 4362, 411, 3021]
JI_MASK = [18, 19, 1, 2, 15, 0]


def align_rotation(a, b):
    """Minimal rotation taking unit vector a to unit vector b (both (...,3) tensors)."""
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1, keepdim=True)
    s = v.norm(dim=-1, keepdim=True)
    ang = torch.atan2(s, c)
    axis = v / s.clamp_min(1e-9)
    aa = axis * ang
    aa = torch.where(s < 1e-7, torch.zeros_like(aa), aa)      # parallel: identity (anti-parallel not expected)
    return M.axis_angle_to_rotation_matrix(aa.reshape(-1, 3)).view(*a.shape[:-1], 3, 3)


class SMPLSkeleton:
    """Rest joints / FK for SMPL (betas=0) without the mesh. Also the rigid sensor-vertex offsets."""

    def __init__(self, body_model, device):
        self.device = device
        with torch.no_grad():
            I = torch.eye(3, device=device).repeat(1, 24, 1, 1)
            _, j, v = body_model.forward_kinematics(I, torch.zeros(10, device=device), torch.zeros(1, 3, device=device), calc_mesh=True)
        self.J = j[0, :24].clone()                                   # (24,3) rest joints (betas=0), world == pelvis frame
        V = v[0]                                                     # (6890,3) rest mesh
        self.parents = SMPL_PARENTS
        self.ji = torch.tensor(JI_MASK, device=device)
        # EXACT linear blend skinning for just the 6 sensor vertices (the ParametricModel used for IMU
        # synthesis has pose blendshapes off, so this reproduces its mesh vertices to float precision):
        #   v_s = sum_j w_sj ( G_j (v_rest_s - J_j) + P_j )
        self.W = body_model._skinning_weights[VI_MASK].to(device)                       # (6,24)
        self.vrest = V[VI_MASK].clone()                                                  # (6,3)
        self.rv_all = self.vrest[:, None, :] - self.J[None, :, :]                        # (6,24,3) offsets per joint
        # rest bone directions
        self.rest_dir = {k: (self.J[c] - self.J[k]) for k, c in SMPL_CHILD.items()}

    def fk(self, local_R, transl):
        """local_R (T,24,3,3), transl (T,3) -> global_R (T,24,3,3), joints (T,24,3) (SMPL convention)."""
        T = local_R.shape[0]
        G = [local_R[:, 0]]
        P = [self.J[0].expand(T, 3) + transl]
        for k in range(1, 24):
            p = self.parents[k]
            G.append(G[p] @ local_R[:, k])
            P.append(P[p] + (G[p] @ (self.J[k] - self.J[p])))
        return torch.stack(G, 1), torch.stack(P, 1)

    def sensor_vertices(self, G, P):
        """(T,6,3) sensor vertex positions by exact LBS over the 24 joints (no full mesh needed)."""
        rot = torch.einsum("tjik,sjk->tsji", G, self.rv_all)                             # (T,6,24,3): G_j (v - J_j)
        return torch.einsum("sj,tsji->tsi", self.W, rot + P[:, None, :, :])


def syn_acc(v, fps=60.0):
    f2 = fps * fps
    acc = (v[:-2] + v[2:] - 2 * v[1:-1]) * f2
    return torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))


class SOMAtoSMPL:
    """Build once per SOMA rig (needs the rig's zero-pose world joint positions to get rest bone directions)."""

    def __init__(self, skel: SMPLSkeleton, soma_names, soma_rest_joints, soma_rest_rot=None):
        """soma_names: list of SOMA joint names; soma_rest_joints: (J,3) world joint positions in the rig's
        reference T-pose; soma_rest_rot: (J,3,3) world joint rotations in that same T-pose (None = identity,
        i.e. a rig whose zero pose IS the reference pose). The retarget uses the rotation RELATIVE to this
        reference: G_rel = G_soma @ rest_rot^T, so G_rel = I in the T-pose for every joint."""
        self.skel = skel
        self.names = [str(n) for n in soma_names]
        self.idx = {n: i for i, n in enumerate(self.names)}
        dev = skel.device
        soma_rest_joints = torch.as_tensor(soma_rest_joints, dtype=torch.float32, device=dev)
        J = len(self.names)
        self.rest_rot_T = (torch.eye(3, device=dev).repeat(J, 1, 1) if soma_rest_rot is None
                           else torch.as_tensor(soma_rest_rot, dtype=torch.float32, device=dev)).transpose(1, 2).contiguous()
        self.map_idx = [self.idx[SOMA_MAP[k]] if SOMA_MAP[k] is not None else -1 for k in range(24)]
        off = torch.eye(3, device=dev).repeat(24, 1, 1)
        for k in range(24):
            n = SOMA_MAP[k]
            if n is None or k not in SMPL_CHILD or n not in SOMA_CHILD:
                continue
            d_smpl = skel.rest_dir[k]
            d_soma = soma_rest_joints[self.idx[SOMA_CHILD[n]]] - soma_rest_joints[self.idx[n]]
            off[k] = align_rotation(d_smpl, d_soma)
        self.off = off                                               # (24,3,3)
        self.hips = self.idx["Hips"]

    @torch.no_grad()
    def __call__(self, G_soma, P_soma, world_rot=None):
        """G_soma (T,J,3,3) world rotations, P_soma (T,J,3) world positions (meters, y-up).
        world_rot (3,3) optional extra rotation applied to the whole world (e.g. tilt correction).
        -> pose_aa (T,24,3), transl (T,3), G_smpl (T,24,3,3), joints (T,24,3)."""
        T = G_soma.shape[0]
        dev = self.skel.device
        if world_rot is not None:
            G_soma = world_rot @ G_soma
            P_soma = (world_rot @ P_soma.unsqueeze(-1)).squeeze(-1)
        G = torch.eye(3, device=dev).repeat(T, 24, 1, 1)
        for k in range(24):
            m = self.map_idx[k]
            if m >= 0:
                G[:, k] = G_soma[:, m] @ self.rest_rot_T[m] @ self.off[k]
        # locals
        L = torch.empty_like(G)
        L[:, 0] = G[:, 0]
        for k in range(1, 24):
            p = SMPL_PARENTS[k]
            L[:, k] = G[:, p].transpose(1, 2) @ G[:, k]
        L[:, 22:24] = torch.eye(3, device=dev)                       # hands off
        transl = P_soma[:, self.hips] - self.skel.J[0]
        G2, P2 = self.skel.fk(L, transl)                             # recompose (hands zeroed) for joints
        aa = M.rotation_matrix_to_axis_angle(L.reshape(-1, 3, 3)).view(T, 24, 3)
        return aa, transl, G2, P2


BASE_FIT = "/home/vimal/Downloads/kimodo/BONES-SEED/soma_shapes/soma_base_fit_mhr_params.npz"


@torch.no_grad()
def soma_x_reference(device, identity_model_type="mhr", base_fit=BASE_FIT):
    """The SOMA rig's reference T-pose from SOMA-X (zero pose, BONES-SEED's uniform MHR identity):
    names (78, 'Root' first), world joint rotations (78,3,3) and positions (78,3). Shared by the form-hoi
    (SOMA-X FK) and BONES-SEED (BVH FK) converters so both express rotations relative to the same reference
    frames. (The BVH rig uses the same joint frames: its stored rotations are the 'absolute' local rotations
    including the joint orients, so BVH forward kinematics lands in SOMA-X's world joint frames.)"""
    from soma import SOMALayer
    from soma.assets import get_assets_dir
    from soma.smpl.transfer import _pose_layer
    from soma.units import Unit
    soma = SOMALayer(get_assets_dir(), identity_model_type=identity_model_type, low_lod=False, device=device,
                     mode="warp", output_unit=Unit.METERS)
    names = [str(n) for n in soma.public_joint_names]
    nj = len(soma.public_joint_names) - 1          # poses exclude Root
    z = np.load(base_fit)
    ident = torch.from_numpy(np.asarray(z["identity_params"], np.float32)).to(device)
    scale = torch.from_numpy(np.asarray(z["scale_params"], np.float32)).to(device)
    soma.prepare_identity(ident, scale_params=scale)
    rest = _pose_layer(soma, torch.zeros(1, nj, 3, device=device), torch.zeros(1, 3, device=device), pose2rot=True,
                       absolute_pose=False, extra_kwargs={"apply_correctives": False, "fk_only": True})
    tr = rest["transforms"][0]
    return names, tr[:, :3, :3].clone(), tr[:, :3, 3].clone(), soma


def _kabsch(A, B):
    """A,B (N,3) -> R (3,3) minimising |R A_c - B_c| (Kabsch / Procrustes rotation)."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    H = Ac.T @ Bc
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1., 1., float(d)], device=A.device))
    return Vt.T @ D @ U.T


@torch.no_grad()
def mesh_calibrated_offsets(skel: SMPLSkeleton, body_model, soma, device, smpl_pkl, base_fit=BASE_FIT):
    """Per-SMPL-joint rest offsets Off(k) (24,3,3) from the MESHES: SOMA-X bridges the SOMA T-pose mesh
    (BONES-SEED uniform identity) into SMPL topology, so for every SMPL body part (vertices whose dominant
    skinning joint is k) the Kabsch rotation from SMPL's rest part onto SOMA's rest part gives the full
    rigid alignment -- bone DIRECTION and TWIST. Direction-only alignment left 20-30 deg of twist error
    on forearms/pelvis; this brings posed per-part errors to ~1-3 deg (thighs, head, pelvis) and ~6-10 deg
    (forearms, where SOMA's twist joints that we do not model deform the mesh)."""
    from soma.smpl import create_smpl_family_layer
    from soma.smpl.transfer import SMPLFamilyTopologyBridge, _pose_layer
    from soma.assets import get_assets_dir
    from soma.units import Unit
    smpl = create_smpl_family_layer("smpl", get_assets_dir(), device=device, mode="warp", output_unit=Unit.METERS,
                                    gender="male", model_path=smpl_pkl)
    bridge = SMPLFamilyTopologyBridge(soma, smpl)
    z = np.load(base_fit)
    soma.prepare_identity(torch.from_numpy(np.asarray(z["identity_params"], np.float32)).to(device),
                          scale_params=torch.from_numpy(np.asarray(z["scale_params"], np.float32)).to(device))
    nj = len(soma.public_joint_names) - 1
    so_rest = _pose_layer(soma, torch.zeros(1, nj, 3, device=device), torch.zeros(1, 3, device=device), pose2rot=True,
                          absolute_pose=False, extra_kwargs={"apply_correctives": False})
    V_soma = bridge(so_rest["vertices"])[0]
    I = torch.eye(3, device=device).repeat(1, 24, 1, 1)
    _, _, V_smpl = body_model.forward_kinematics(I, torch.zeros(10, device=device), torch.zeros(1, 3, device=device), calc_mesh=True)
    V_smpl = V_smpl[0]
    part = body_model._skinning_weights.argmax(1)
    off = torch.eye(3, device=device).repeat(24, 1, 1)
    for k in range(24):
        if SOMA_MAP[k] is None:
            continue
        m = part == k
        if int(m.sum()) < 20:
            continue
        off[k] = _kabsch(V_smpl[m], V_soma[m])
    return off


@torch.no_grad()
def imu_from_pose(skel: SMPLSkeleton, pose_aa, transl, fps=60.0):
    """Mesh-free synthesis: pose_aa (T,24,3), transl (T,3) -> joint (T,24,3), vrot (T,6,3,3), vacc (T,6,3)."""
    T = pose_aa.shape[0]
    L = M.axis_angle_to_rotation_matrix(pose_aa.reshape(-1, 3)).view(T, 24, 3, 3)
    G, P = skel.fk(L, transl)
    V = skel.sensor_vertices(G, P)
    return P, G[:, skel.ji], syn_acc(V, fps)


@torch.no_grad()
def validate_rigid_accel(body_model, skel, pose_aa, transl, device, fps=60.0):
    """Compare mesh-free sensor accel/positions with the true LBS mesh on one sequence. Returns dict of stats."""
    T = pose_aa.shape[0]
    L = M.axis_angle_to_rotation_matrix(pose_aa.reshape(-1, 3).to(device)).view(T, 24, 3, 3)
    grot, joint, vert = body_model.forward_kinematics(L, torch.zeros(10, device=device), transl.to(device), calc_mesh=True)
    v_true = vert[:, VI_MASK]
    G, P = skel.fk(L, transl.to(device))
    v_rig = skel.sensor_vertices(G, P)
    a_true, a_rig = syn_acc(v_true, fps), syn_acc(v_rig, fps)
    dj = (joint[:, :24] - P).norm(dim=-1).mean().item()
    dg = M.radian_to_degree(M.angle_between(grot[:, :24].reshape(-1, 3, 3), G.reshape(-1, 3, 3))).mean().item()
    pos_err = (v_true - v_rig).norm(dim=-1).mean(0)                                  # (6,) m
    acc_rel = ((a_true - a_rig).norm(dim=-1).mean(0) / a_true.norm(dim=-1).mean(0).clamp_min(1e-6))
    return {"joint_pos_err_mm": dj * 1000, "global_rot_err_deg": dg,
            "sensor_pos_err_mm": (pos_err * 1000).cpu().numpy().round(2).tolist(),
            "sensor_acc_rel_err": acc_rel.cpu().numpy().round(4).tolist(),
            "acc_mean_true": a_true.norm(dim=-1).mean(0).cpu().numpy().round(3).tolist()}
