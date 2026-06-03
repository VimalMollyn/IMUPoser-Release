import os
import torch
from torch.utils.data import Dataset
from imuposer import math
from imuposer.config import Config, amass_combos

class GlobalModelDataset(Dataset):
    r"""
    Each training sample is a (window, IMU-combo) pair: every windowed sequence is
    paired with all entries in ``amass_combos`` (zeroing out the IMUs not in the combo).

    To avoid holding ~25x the dataset in RAM, the base windows are stored once and the
    combo masking is applied lazily in ``__getitem__`` instead of being materialized up
    front. ``idx`` maps to ``(window_idx, combo_idx)`` via integer div/mod.
    """
    def __init__(self, split="train", config:Config=None, data_files=None):
        super().__init__()

        # load the data
        self.train = split
        self.config = config
        # train on all IMU combos (generalist) unless a single combo is pinned
        # (config.train_combo, e.g. "lw_rp_h") -> specialist for that sensor set
        _tc = getattr(config, "train_combo", None)
        self.combos = [amass_combos[_tc]] if _tc else list(amass_combos.values())
        # explicit list of .pt filenames to load (used to keep validation drawn
        # only from the original datasets); None -> auto-discover all non-dip files
        self.data_files = data_files
        # IMU input augmentation (domain randomization) — applied ONLY to the present
        # sensors, ONLY when self.augment is set (get_dataset turns it on for the train
        # set, never val/test). Magnitudes from env so experiments can sweep them.
        self.augment = False
        self.aug_acc_std = float(os.environ.get("AUG_ACC_STD", "0"))
        self.aug_ori_std = float(os.environ.get("AUG_ORI_STD", "0"))
        # GlobalPose-style per-sensor calibration / mounting rotation error (radians,
        # per-axis std; constant over a window). Real IMUs are mis-mounted/mis-calibrated
        # vs the body segment, which our perfect FK orientation lacks. GlobalPose uses
        # ~0.1*sqrt(pi/8) ≈ 0.063 rad (~3.6 deg/axis).
        self.aug_calib = float(os.environ.get("AUG_CALIB_RAD", "0"))
        # GlobalPose-style orientation DRIFT: real IMU orientation comes from gyro
        # integration and drifts over time. Model it as a per-sensor constant angular-
        # velocity bias (rad/s) integrated over the window -> a time-growing rotation.
        self.aug_drift = float(os.environ.get("AUG_DRIFT_RAD_S", "0"))
        self.data = self.load_data()

    def load_data(self):
        # an explicit file list (the canonical split) always wins; otherwise fall
        # back to auto-discovery (all non-dip for train, dip_test for test)
        if self.data_files is not None:
            data_files = list(self.data_files)
        elif self.train == "train":
            data_files = sorted(x.name for x in self.config.processed_imu_poser_25fps.iterdir() if "dip" not in x.name)
        else:
            data_files = ["dip_test.pt"]

        # silently skip any requested file that isn't present (e.g. dip_test.pt
        # when DIP-IMU hasn't been regenerated yet)
        data_files = [f for f in data_files if (self.config.processed_imu_poser_25fps / f).exists()]

        # base windows, stored once (combo masking happens lazily in __getitem__)
        acc_windows = []
        ori_windows = []
        pose_windows = []
        joint_windows = []
        tran_windows = []
        _aux = getattr(self.config, "aux_target", None)
        need_joint = _aux == "joint"
        need_tran = _aux == "tran"

        window_length = self.config.max_sample_len * 25 // 60

        for fname in data_files:
            fdata = torch.load(self.config.processed_imu_poser_25fps / fname, weights_only=False)

            for i in range(len(fdata["acc"])):
                # inputs
                facc = fdata["acc"][i]
                fori = fdata["ori"][i]

                # load all the data
                glb_acc = facc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]] / self.config.acc_scale
                glb_ori = fori.view(-1, 6, 3, 3)[:, [0, 1, 2, 3, 4]]

                acc = glb_acc           # N, 5, 3
                ori = glb_ori           # N, 5, 3, 3

                # outputs
                fpose = fdata["pose"][i]
                fpose = fpose.reshape(fpose.shape[0], -1)

                # clip the data into windows (25 is the data sampling rate)
                acc_windows.extend(torch.split(acc, window_length))
                ori_windows.extend(torch.split(ori, window_length))
                pose_windows.extend(torch.split(fpose, window_length))
                if need_joint:
                    joint_windows.extend(torch.split(fdata["joint"][i].view(-1, 24, 3), window_length))
                if need_tran:
                    tran_windows.extend(torch.split(fdata["tran"][i].view(-1, 3), window_length))

        self.acc_windows = acc_windows
        self.ori_windows = ori_windows
        self.pose_windows = pose_windows
        self.joint_windows = joint_windows
        self.tran_windows = tran_windows
        self.num_windows = len(pose_windows)
        self.num_combos = len(self.combos)

    def __getitem__(self, idx):
        window_idx = idx // self.num_combos
        combo = self.combos[idx % self.num_combos]

        acc = self.acc_windows[window_idx]      # W, 5, 3
        ori = self.ori_windows[window_idx]      # W, 5, 3, 3

        # zero out the IMUs not present in this combo
        _combo_acc = torch.zeros_like(acc)
        _combo_ori = torch.zeros_like(ori)
        _combo_acc[:, combo] = acc[:, combo]
        _combo_ori[:, combo] = ori[:, combo]

        # domain randomization: perturb the PRESENT sensors only (absent stay zero)
        if self.augment:
            # GlobalPose-style calibration/mounting error: rotate each present sensor's
            # orientation by a random rotation, constant over the window (per-sensor).
            if self.aug_calib > 0:
                for c in combo:
                    aa = torch.randn(3) * self.aug_calib                       # axis-angle (rad)
                    Rc = math.axis_angle_to_rotation_matrix(aa.unsqueeze(0))[0]  # 3x3
                    _combo_ori[:, c] = torch.matmul(Rc, _combo_ori[:, c])      # (W,3,3)
            # orientation drift: per-sensor constant angular-velocity bias integrated over time
            if self.aug_drift > 0:
                W = _combo_ori.shape[0]
                t = (torch.arange(W, dtype=torch.float32) / 25.0).unsqueeze(1)  # seconds, (W,1)
                for c in combo:
                    bias = torch.randn(3) * self.aug_drift                     # rad/s
                    Rd = math.axis_angle_to_rotation_matrix(bias.unsqueeze(0) * t)  # (W,3,3)
                    _combo_ori[:, c] = torch.matmul(Rd, _combo_ori[:, c])
            if self.aug_acc_std > 0:
                _combo_acc[:, combo] += torch.randn_like(_combo_acc[:, combo]) * self.aug_acc_std
            if self.aug_ori_std > 0:
                _combo_ori[:, combo] += torch.randn_like(_combo_ori[:, combo]) * self.aug_ori_std

        _input = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1).float()

        _pose = self.pose_windows[window_idx].float()
        if self.config.r6d == True:
            _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        else:
            _output = _pose

        # auxiliary target, appended AFTER the pose target (sliced back out by the
        # staged/recon models). Pose stays first so eval_dip.py [...,:144] is unaffected.
        aux = getattr(self.config, "aux_target", None)
        if aux == "imu":
            # full CLEAN 5-IMU set (reconstruct the absent sensors + denoise present)
            full_imu = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1).float()
            _output = torch.cat([_output, full_imu], dim=1)
        elif aux == "joint":
            jp = self.joint_windows[window_idx].float()        # W, 24, 3
            jp = (jp - jp[:, :1]).reshape(jp.shape[0], -1)      # root-relative, W, 72
            _output = torch.cat([_output, jp], dim=1)
        elif aux == "tran":
            tr = self.tran_windows[window_idx].float()         # W, 3 (root translation)
            _output = torch.cat([_output, tr], dim=1)

        return _input, _output

    def __len__(self):
        return self.num_windows * self.num_combos
