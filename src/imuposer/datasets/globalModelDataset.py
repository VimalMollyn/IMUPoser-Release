import os
import torch
from torch.utils.data import Dataset
from imuposer import math
from imuposer.config import Config, amass_combos


def _kmeans(x, k, iters=25, seed=0):
    r"""Tiny Lloyd k-means (torch). x: (N,D) -> labels (N,). Deterministic init from a fixed stride."""
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(n, generator=g)[:k]].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, c)                     # N,k
        labels = d.argmin(1)
        for j in range(k):
            m = labels == j
            if m.any():
                c[j] = x[m].mean(0)
    return labels

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
        # accelerometer realism (scaled units, i.e. m/s^2 / acc_scale): per-sensor constant BIAS and
        # per-axis SCALE-factor error (constant over the window), the imperfections our perfect kinematic
        # accel lacks. AUG_ACC_BIAS ~0.02 (~0.6 m/s^2), AUG_ACC_SCALE ~0.03 (3% gain error).
        self.aug_acc_bias = float(os.environ.get("AUG_ACC_BIAS", "0"))
        self.aug_acc_scale = float(os.environ.get("AUG_ACC_SCALE", "0"))
        # GlobalPose-style per-sensor calibration / mounting rotation error (radians,
        # per-axis std; constant over a window). Real IMUs are mis-mounted/mis-calibrated
        # vs the body segment, which our perfect FK orientation lacks. GlobalPose uses
        # ~0.1*sqrt(pi/8) ≈ 0.063 rad (~3.6 deg/axis).
        self.aug_calib = float(os.environ.get("AUG_CALIB_RAD", "0"))
        # GlobalPose-style orientation DRIFT: real IMU orientation comes from gyro
        # integration and drifts over time. Model it as a per-sensor constant angular-
        # velocity bias (rad/s) integrated over the window -> a time-growing rotation.
        self.aug_drift = float(os.environ.get("AUG_DRIFT_RAD_S", "0"))
        # GlobalPose-style realistic gyro-integration orientation: real IMU orientation = integral of a
        # NOISY gyro (random-walk bias + white noise), giving a sqrt-time random-walk DRIFT (esp. yaw) that
        # our perfect-FK orientation and the crude constant-bias drift both lack. AUG_GYRO_RW = bias
        # random-walk rate (rad/s/sqrt(s)); AUG_GYRO_N = white gyro noise (rad/s). Vectorized via cumsum.
        self.aug_gyro_rw = float(os.environ.get("AUG_GYRO_RW", "0"))
        self.aug_gyro_n = float(os.environ.get("AUG_GYRO_N", "0"))
        self.aug_gyro_yawonly = bool(os.environ.get("AUG_GYRO_YAWONLY"))  # ESKF tilt-correction: drift yaw only
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
        need_act = _aux == "activity"

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

        # pseudo-ACTIVITY labels: cluster windows by mean pose (no real activity labels exist).
        # K from ACT_K (default 16). One label per window; used to supervise the activity head.
        if need_act:
            K = int(os.environ.get("ACT_K", "16"))
            feats = torch.stack([pw.float().mean(0).flatten() for pw in pose_windows])  # (Nwin, 216)
            self.activity_labels = _kmeans(feats, K)
            self.n_activities = K

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
            # GlobalPose-style realistic gyro-integration drift: random-walk bias + white gyro noise,
            # integrated -> sqrt-time random-walk orientation error (the real-IMU drift; richer than the
            # constant-bias model above). Vectorized: accumulated rotation-vector via cumsum (small-error
            # approx ignores non-commutativity, fine at these magnitudes).
            if self.aug_gyro_rw > 0 or self.aug_gyro_n > 0:
                W = _combo_ori.shape[0]
                dt = 1.0 / 25.0
                for c in combo:
                    bias = torch.cumsum(torch.randn(W, 3) * self.aug_gyro_rw * (dt ** 0.5), dim=0)  # rad/s
                    noise = torch.randn(W, 3) * self.aug_gyro_n                                      # rad/s
                    d = torch.cumsum((bias + noise) * dt, dim=0)               # accumulated drift rot-vec (W,3)
                    if self.aug_gyro_yawonly:
                        # ESKF/accelerometer tilt-correction: real IMUs correct pitch/roll via gravity,
                        # leaving only YAW (world-up = Y axis) to drift. Keep only the up-axis component.
                        d = d * torch.tensor([0., 1., 0.])
                    D = math.axis_angle_to_rotation_matrix(d)                  # (W,3,3)
                    _combo_ori[:, c] = torch.matmul(D, _combo_ori[:, c])
            # accelerometer bias (constant per sensor) + per-axis scale-factor error (constant per sensor)
            if self.aug_acc_bias > 0 or self.aug_acc_scale > 0:
                for c in combo:
                    if self.aug_acc_scale > 0:
                        _combo_acc[:, c] = _combo_acc[:, c] * (1 + torch.randn(3) * self.aug_acc_scale)
                    if self.aug_acc_bias > 0:
                        _combo_acc[:, c] = _combo_acc[:, c] + torch.randn(3) * self.aug_acc_bias
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
            # per-frame root VELOCITY (m/frame) — learnable from IMU (single-integrate accel), unlike
            # absolute position which has no global reference. TransPose-style translation target.
            vel = torch.zeros_like(tr)
            vel[1:] = tr[1:] - tr[:-1]
            _output = torch.cat([_output, vel], dim=1)
        elif aux == "activity":
            # per-window pseudo-activity label, broadcast over frames as one extra column
            lab = float(self.activity_labels[window_idx])
            _output = torch.cat([_output, _output.new_full((_output.shape[0], 1), lab)], dim=1)

        return _input, _output

    def __len__(self):
        return self.num_windows * self.num_combos
