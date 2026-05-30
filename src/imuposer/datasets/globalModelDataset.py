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
        self.combos = list(amass_combos.values())
        # explicit list of .pt filenames to load (used to keep validation drawn
        # only from the original datasets); None -> auto-discover all non-dip files
        self.data_files = data_files
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

        self.acc_windows = acc_windows
        self.ori_windows = ori_windows
        self.pose_windows = pose_windows
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

        _input = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1).float()

        _pose = self.pose_windows[window_idx].float()
        if self.config.r6d == True:
            _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        else:
            _output = _pose

        return _input, _output

    def __len__(self):
        return self.num_windows * self.num_combos
