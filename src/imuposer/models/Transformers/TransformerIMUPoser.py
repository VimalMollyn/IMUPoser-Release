r"""
Transformer-encoder pose model (à la Transformer Inertial Poser, TIP — SIGGRAPH Asia 2022).

TIP's encoder is a Transformer-Encoder (optionally + biLSTM, "TE-biLSTM") feeding an MLP decoder
to SMPL params; its stationary-point loss only concerns *root* motion, which our root-relative
metrics ignore, so we drop it. This is a clean attention-vs-recurrence swap: same data, same
MSE(+FK joint) loss, same optimizer budget as the LSTM baseline — only the network changes.

MODEL=TransformerIMUPoser. Optional hybrid TE-biLSTM via TF_BILSTM=1 (transformer encoder ->
biLSTM -> head), which is TIP's exact recipe. Knobs: TF_DMODEL, TF_LAYERS, TF_HEADS, TF_FF, TF_DROPOUT.

forward(imu, lens) returns per-frame r6d pose (B,T,144), padded positions zeroed to match the
LSTM's pack/pad behaviour exactly, so eval_dip.py forward(...)[:, :, :144] is unchanged.
"""
import os
import math
import torch
import torch.nn as nn
import lightning.pytorch as pl

from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # 1, max_len, d_model

    def forward(self, x):  # x: B,T,d
        return x + self.pe[:, :x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(self, n_input, n_output, d_model=256, nhead=8, n_layers=4,
                 dim_ff=1024, dropout=0.1, use_bilstm=False):
        super().__init__()
        self.inp = nn.Linear(n_input, d_model)
        self.pe = _SinusoidalPE(d_model)
        # pre-LN (norm_first) trains stably from scratch without LR warmup
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.use_bilstm = use_bilstm
        head_in = d_model
        if use_bilstm:
            self.bilstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, bidirectional=True)
            head_in = d_model * 2
        self.out = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, n_output))

    def forward(self, x, lens):
        T = x.size(1)
        dev = x.device
        lens_t = torch.as_tensor(lens, device=dev)
        pad_mask = torch.arange(T, device=dev)[None, :] >= lens_t[:, None]  # B,T True=pad
        h = self.pe(self.inp(x))
        h = self.enc(h, src_key_padding_mask=pad_mask)
        if self.use_bilstm:
            h, _ = self.bilstm(h)
        y = self.out(h)
        # zero padded positions so downstream loss/FK is identical to the LSTM's pack/pad output
        return y.masked_fill(pad_mask.unsqueeze(-1), 0.0)


class TransformerIMUPoser(pl.LightningModule):
    r"""Attention-based pose model; same loss/training loop as IMUPoserModel (fair A/B vs the LSTM)."""
    def __init__(self, config: Config):
        super().__init__()
        n_input = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        self.config = config
        self.net = _TransformerNet(
            n_input, self.n_pose_output,
            d_model=int(os.environ.get("TF_DMODEL", "256")),
            nhead=int(os.environ.get("TF_HEADS", "8")),
            n_layers=int(os.environ.get("TF_LAYERS", "4")),
            dim_ff=int(os.environ.get("TF_FF", "1024")),
            dropout=float(os.environ.get("TF_DROPOUT", "0.1")),
            use_bilstm=bool(os.environ.get("TF_BILSTM")))
        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.loss = nn.MSELoss() if config.loss_type == "mse" else nn.L1Loss()
        self.lr = float(os.environ.get("TF_LR", "3e-4"))
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=["config"])

    def on_fit_start(self):
        if self.config.use_joint_loss:
            self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def forward(self, imu_inputs, imu_lens):
        return self.net(imu_inputs, imu_lens)

    def _step(self, batch):
        imu, target, lens, _ = batch
        pred_pose = self(imu, lens)[:, :, :self.n_pose_output]
        target_pose = target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            tj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1]
            loss = loss + self.loss(pj, tj)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        self.training_step_outputs.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.batch_size)
        self.validation_step_outputs.append(loss.item())
        return {"loss": loss}

    def on_train_epoch_end(self):
        self._epoch_end(self.training_step_outputs, "train"); self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self._epoch_end(self.validation_step_outputs, "val"); self.validation_step_outputs.clear()

    def _epoch_end(self, outputs, loop):
        if outputs:
            self.log(f"{loop}_loss", sum(outputs) / len(outputs), prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
