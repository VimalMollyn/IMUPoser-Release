r"""
Dataset util functions
"""

import os
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from imuposer.datasets import *
from imuposer.config import val_datasets, test_datasets, original_amass_datasets

def train_val_split(dataset, train_pct):
    # get the train and val split
    total_size = len(dataset)
    train_size = int(train_pct * total_size)
    val_size = total_size - train_size
    return train_size, val_size

def get_split_files(config):
    r"""
    Resolve the canonical dataset-level split into lists of 25fps .pt filenames.

    Whole datasets are assigned to exactly one split; a dataset used for val/test
    is never in train. Test = held-out AMASS (config.test_datasets) + DIP-IMU.
    """
    all_files = sorted(x.name for x in config.processed_imu_poser_25fps.iterdir()
                       if x.name.endswith(".pt") and "dip" not in x.name)
    val_set, test_set = set(val_datasets), set(test_datasets)

    val_files = [f for f in all_files if f[:-len(".pt")] in val_set]
    test_files = [f for f in all_files if f[:-len(".pt")] in test_set] + ["dip_test.pt"]
    train_files = [f for f in all_files if f[:-len(".pt")] not in val_set and f[:-len(".pt")] not in test_set]

    # VAL_FILES env override: validate on explicit file(s), e.g. real DIP for the autoresearch
    # loop (VAL_FILES=dip_train.pt). Selection-only — these are never added to train_files.
    _val_override = os.environ.get("VAL_FILES")
    if _val_override:
        val_files = [v if v.endswith(".pt") else v + ".pt" for v in _val_override.split(",")]

    # original-data-only ablation: drop the 5 newer AMASS datasets + Motion-X from
    # training (val/test are unchanged so the comparison stays controlled)
    if getattr(config, "original_train_only", False):
        orig = set(original_amass_datasets)
        train_files = [f for f in train_files if f[:-len(".pt")] in orig]

    # TRAIN_DATASETS env: restrict training to an explicit curated set of dataset
    # names (e.g. only DIP-relevant everyday/locomotion motion). Comma-separated.
    _train_only = os.environ.get("TRAIN_DATASETS")
    if _train_only:
        keep = set(_train_only.split(","))
        train_files = [f for f in train_files if f[:-len(".pt")] in keep]
    return train_files, val_files, test_files

def get_dataset(config=None, test_only=False):
    model = config.model
    # load the dataset (Recon/Staged share the same GlobalModelDataset; they only
    # differ in the auxiliary target appended via config.aux_target)
    if model in ("GlobalModelIMUPoser", "ReconIMUPoser", "StagedIMUPoser"):
        train_files, val_files, test_files = get_split_files(config)

        test_dataset = GlobalModelDataset("test", config, data_files=test_files)
        if test_only:
            return test_dataset

        train_dataset = GlobalModelDataset("train", config, data_files=train_files)
        train_dataset.augment = True   # domain randomization on TRAIN only (val/test stay clean)
        val_dataset = GlobalModelDataset("train", config, data_files=val_files)
        return train_dataset, test_dataset, val_dataset

    elif model == "GlobalModelIMUPoserFineTuneDIP":
        if not test_only:
            train_dataset = GlobalModelDatasetFineTuneDIP("train", config)
        test_dataset = GlobalModelDatasetFineTuneDIP("test", config)
        if test_only:
            return test_dataset
        train_size, val_size = train_val_split(train_dataset, train_pct=config.train_pct)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        return train_dataset, test_dataset, val_dataset

    else:
        print("Enter a valid model")
        return

def get_datamodule(config):
    model = config.model
    # load the dataset
    if model in ["GlobalModelIMUPoser", "GlobalModelIMUPoserFineTuneDIP", "ReconIMUPoser", "StagedIMUPoser"]:
        return IMUPoserDataModule(config)
    else:
        print("Enter a valid model")

def pad_seq(batch):
    inputs = [item[0] for item in batch]
    outputs = [item[1] for item in batch]
    
    input_lens = [item.shape[0] for item in inputs]
    output_lens = [item.shape[0] for item in outputs]
    
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    outputs = nn.utils.rnn.pad_sequence(outputs, batch_first=True)
    return inputs, outputs, input_lens, output_lens

class IMUPoserDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset, self.test_dataset, self.val_dataset = get_dataset(self.config)
        print("Done with setup")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)
