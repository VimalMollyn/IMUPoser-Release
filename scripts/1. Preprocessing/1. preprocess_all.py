r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

"""

# %load_ext autoreload
# %autoreload 2

import torch
import os
import io
import pickle
import zipfile
import numpy as np
from tqdm import tqdm
import glob

from imuposer.config import Config, amass_datasets
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math

config = Config(project_root_dir="../../")

# left wrist, right wrist, left thigh, right thigh, head, pelvis
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])


def _syn_acc(v):
    r"""
    Synthesize accelerations from vertex positions.
    """
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    return acc


def _resample_to_60fps(arr, src_fps):
    r"""
    Linearly resample a (N, ...) sequence to 60fps. Used for datasets whose
    framerate is not an integer multiple of 60 (e.g. LARa @ 200fps, Motion-X @
    30fps). For integer multiples this is equivalent to plain striding, so the
    existing 120/60fps datasets are processed identically.
    """
    n = arr.shape[0]
    idx = np.arange(0, n, src_fps / 60.0)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(np.ceil(idx).astype(np.int64), n - 1)
    w = (idx - lo).reshape((-1,) + (1,) * (arr.ndim - 1)).astype(np.float32)
    return (arr[lo] * (1 - w) + arr[hi] * w).astype(np.float32)


def _synthesize_and_save(pose, shape, tran, length, body_model, out_dir, device=torch.device('cpu')):
    r"""
    Run forward kinematics over each sequence, synthesize the 6 IMU
    accelerations/orientations and save the AMASS-style tensors to out_dir.

    pose:   (total_frames, 24, 3) axis-angle, global frame already aligned to DIP
    shape:  (num_seqs, 10) SMPL betas (one per sequence)
    tran:   (total_frames, 3) root translation
    length: per-sequence frame counts (sums to total_frames)

    FK runs on `device`; only the current sequence is moved there at a time so
    memory stays bounded, and every saved tensor is moved back to the CPU.
    """
    print('Synthesizing IMU accelerations and orientations')
    vim, jim = vi_mask.to(device), ji_mask.to(device)
    cpu_model = [None]  # lazily-built CPU fallback for sequences that OOM the GPU

    def _fk(pose_seq, shape_i, tran_seq):
        try:
            p = math.axis_angle_to_rotation_matrix(pose_seq.to(device)).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape_i.to(device), tran_seq.to(device), calc_mesh=True)
            return joint[:, :24].contiguous().cpu(), _syn_acc(vert[:, vim]).cpu(), grot[:, jim].cpu()
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower() or device.type != 'cuda':
                raise
            torch.cuda.empty_cache()
            print('\tCUDA OOM on a long sequence, falling back to CPU for it')
            if cpu_model[0] is None:
                cpu_model[0] = ParametricModel(config.og_smpl_model_path, device=torch.device('cpu'))
            p = math.axis_angle_to_rotation_matrix(pose_seq).view(-1, 24, 3, 3)
            grot, joint, vert = cpu_model[0].forward_kinematics(p, shape_i, tran_seq, calc_mesh=True)
            return joint[:, :24].contiguous(), _syn_acc(vert[:, vi_mask]), grot[:, ji_mask]

    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        l = int(l)
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        joint24, vacc, vrot = _fk(pose[b:b + l], shape[i], tran[b:b + l])
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3 (CPU original)
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint24)  # N, 24, 3
        out_vacc.append(vacc)  # N, 6, 3
        out_vrot.append(vrot)  # N, 6, 3, 3
        b += l

    print('Saving')
    out_dir.mkdir(exist_ok=True, parents=True)
    torch.save(out_pose, out_dir / 'pose.pt')
    torch.save(out_shape, out_dir / 'shape.pt')
    torch.save(out_tran, out_dir / 'tran.pt')
    torch.save(out_joint, out_dir / 'joint.pt')
    torch.save(out_vrot, out_dir / 'vrot.pt')
    torch.save(out_vacc, out_dir / 'vacc.pt')
    print('Saved to', str(out_dir))

def process_amass():
    body_model = ParametricModel(config.og_smpl_model_path, device=config.device)

    try:
        processed = [fpath.name for fpath in (config.processed_imu_poser / "AMASS").iterdir()]
    except:
        processed = []

    for ds_name in amass_datasets:
        if ds_name in processed:
            continue
        data_pose, data_trans, data_beta, length = [], [], [], []
        print('\rReading', ds_name)
        # Older AMASS releases use "*_poses.npz"; newer "stageii" releases (GRAB,
        # SOMA, WEIZMANN, MOYO, LARa) use "*_stageii.npz" and can be nested at
        # arbitrary depth, so glob recursively for both conventions.
        npz_fnames = sorted(
            glob.glob(os.path.join(config.raw_amass_path, ds_name, '**', '*_poses.npz'), recursive=True) +
            glob.glob(os.path.join(config.raw_amass_path, ds_name, '**', '*_stageii.npz'), recursive=True))
        for npz_fname in tqdm(npz_fnames):
            try: cdata = np.load(npz_fname)
            except: continue

            # framerate key was renamed "mocap_framerate" -> "mocap_frame_rate"
            fr_key = 'mocap_framerate' if 'mocap_framerate' in cdata else 'mocap_frame_rate'
            framerate = round(float(cdata[fr_key]))

            pose_i = cdata['poses'].astype(np.float32)
            tran_i = cdata['trans'].astype(np.float32)
            if framerate in (60, 59):
                pass
            elif framerate % 60 == 0:
                step = framerate // 60
                pose_i, tran_i = pose_i[::step], tran_i[::step]
            else:
                # non-integer multiple of 60 (e.g. LARa @ 200fps): interpolate
                pose_i = _resample_to_60fps(pose_i, framerate)
                tran_i = _resample_to_60fps(tran_i, framerate)

            data_pose.extend(pose_i)
            data_trans.extend(tran_i)
            data_beta.append(cdata['betas'][:10])
            length.append(pose_i.shape[0])

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32))

        # SMPL+H (52 joints, 156-dim) and SMPL-X (55 joints, 165-dim) share the
        # same body joints 0-21. Keep the SMPL 24-joint body layout and zero the
        # two hand joints so every dataset is treated identically (no hands).
        n_joints = pose.shape[1] // 3
        pose = pose.view(-1, n_joints, 3)[:, :24].clone()
        pose[:, 22:24] = 0

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        _synthesize_and_save(pose, shape, tran, length, body_model,
                             config.processed_imu_poser / "AMASS" / ds_name, config.device)

def process_motionx():
    r"""
    Synthesize IMUs from the Motion-X dataset (https://github.com/IDEA-Research/Motion-X).

    Uses the "motion_generation/smplx322" representation: each clip is a
    (N, 322) SMPL-X motion array at 30fps, where [0:3]=root_orient,
    [3:66]=pose_body (joints 1-21), [309:312]=trans, [312:322]=betas. Hands /
    face are dropped to match the hand-free SMPL 24-joint poses used elsewhere.
    Each Motion-X subset (haa500, idea400, ...) is written as its own
    AMASS-style folder so the 25fps step and the dataloader pick it up.
    """
    body_model = ParametricModel(config.og_smpl_model_path, device=config.device)
    smplx_dir = config.raw_amass_path / "motion" / "motion_generation" / "smplx322"
    amass_dir = config.processed_imu_poser / "AMASS"

    try:
        processed = [fpath.name for fpath in amass_dir.iterdir()]
    except:
        processed = []

    for zip_path in sorted(smplx_dir.glob("*.zip")):
        ds_name = f"MotionX_{zip_path.stem}"
        if ds_name in processed:
            continue
        data_pose, data_trans, data_beta, length = [], [], [], []
        print('\rReading', ds_name)
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if n.endswith('.npy')]
            for name in tqdm(members):
                try:
                    arr = np.load(io.BytesIO(zf.read(name)))
                except:
                    continue
                if arr.ndim != 2 or arr.shape[1] != 322:
                    continue
                pose_i = arr[:, :66].astype(np.float32)        # root_orient + pose_body (22 joints)
                tran_i = arr[:, 309:312].astype(np.float32)
                beta_i = arr[0, 312:322].astype(np.float32)    # betas are static across frames
                # Motion-X is 30fps -> upsample to the pipeline's 60fps
                pose_i = _resample_to_60fps(pose_i, 30)
                tran_i = _resample_to_60fps(tran_i, 30)
                data_pose.extend(pose_i)
                data_trans.extend(tran_i)
                data_beta.append(beta_i)
                length.append(pose_i.shape[0])

        if len(data_pose) == 0:
            print(f"Motion-X subset, {ds_name} has no usable clips")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))

        # 22 body joints (root + body) -> SMPL 24-joint layout with hands zeroed
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 22, 3)
        pose = torch.cat([pose, torch.zeros(pose.shape[0], 2, 3)], dim=1).clone()

        # align global frame with DIP (Motion-X SMPL-X uses the same frame as AMASS)
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        _synthesize_and_save(pose, shape, tran, length, body_model, amass_dir / ds_name, config.device)

def process_dipimu(split="test"):
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        return acc
    
    imu_mask = [7, 8, 9, 10, 0, 2]
    if split == "test":
        test_split = ['s_09', 's_10']
    else:
        test_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    accs, oris, poses, trans, shapes, joints, vrots, vaccs = [], [], [], [], [], [], [], []
    
    body_model = ParametricModel(config.og_smpl_model_path)
    
    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(config.raw_dip_path, subject_name)):
            path = os.path.join(config.raw_dip_path, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            # zero the two SMPL hand joints to match the hand-free AMASS poses
            pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)
            pose[:, 22:24] = 0
            pose = pose.reshape(-1, 72)

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            shape = torch.ones((10))
            tran = torch.zeros(pose.shape[0], 3) # dip-imu does not contain translations
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(tran.clone())  
                
                shapes.append(shape.clone()) # default shape
                
                # forward kinematics to get the joint position
                p = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                vacc = _syn_acc(vert[:, vi_mask])
                vrot = grot[:, ji_mask]
                
                joints.append(joint)
                vaccs.append(vacc)
                vrots.append(vrot)
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))
                
    path_to_save = config.processed_imu_poser / f"DIP_IMU/{split}"
    path_to_save.mkdir(exist_ok=True, parents=True)
    
    torch.save(poses, path_to_save / 'pose.pt')
    torch.save(shapes, path_to_save / 'shape.pt')
    torch.save(trans, path_to_save / 'tran.pt')
    torch.save(joints, path_to_save / 'joint.pt')
    torch.save(vrots, path_to_save / 'vrot.pt')
    torch.save(vaccs, path_to_save / 'vacc.pt')
    torch.save(oris, path_to_save / 'oris.pt')
    torch.save(accs, path_to_save / 'accs.pt')
    
    print('Preprocessed DIP-IMU dataset is saved at', path_to_save)

def _to_25fps(config):
    r"""Run stage 2 (resample 60fps -> 25fps) from the sibling script."""
    import importlib.util
    p2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "2. preprocess_all_to_imuposer_at_25fps.py")
    spec = importlib.util.spec_from_file_location("preprocess_25fps", p2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.to_25fps(config)


if __name__ == '__main__':
    from pathlib import Path

    # Write the synthesized dataset to an external folder (outside the repo) so
    # it can be symlinked in later. Override the location with IMUPOSER_OUT_DIR.
    out_dir = Path(os.environ.get("IMUPOSER_OUT_DIR",
                                  "/media/vimal/T7_2TB/CHI23/processed_imuposer_data"))
    config.processed_imu_poser = out_dir / "processed_imuposer"
    config.processed_imu_poser_25fps = out_dir / "processed_imuposer_25fps"
    config.processed_imu_poser.mkdir(parents=True, exist_ok=True)
    config.processed_imu_poser_25fps.mkdir(parents=True, exist_ok=True)
    print(f"=== output: {out_dir} | device: {config.device} ===", flush=True)

    # stage 1: synthesize the 60fps datasets (DIP only if its raw data is present)
    if config.raw_dip_path.exists():
        print("=== DIP-IMU (raw present) ===", flush=True)
        process_dipimu(split="test")
        process_dipimu(split="train")
    else:
        print(f"=== DIP-IMU raw not found at {config.raw_dip_path}; skipping ===", flush=True)

    print("=== AMASS ===", flush=True)
    process_amass()

    print("=== Motion-X ===", flush=True)
    process_motionx()

    # stage 2: resample everything to 25fps
    print("=== resample -> 25fps ===", flush=True)
    _to_25fps(config)

    print("=== DONE ===", flush=True)
