r"""
1D temporal-convolutional (TCN-style) pose model.

A dilated residual 1D-CNN over time: IMU features (B,T,60) -> per-frame r6d pose (B,T,144). Non-causal
'same' padding (the task is offline, like the bidirectional LSTM) and exponentially-growing dilations
give a receptive field that covers a ~125-frame window. Unlike the transformer, a CNN's receptive field
is FIXED and local, so it is length-agnostic — full DIP takes evaluate fine with no sliding-window hack.

Same MSE(+FK joint) loss / data / optimizer family as the other models (fair A/B vs the LSTM), plus the
optional AvatarPoser-style IK orientation-consistency loss (IK_LOSS=1) so a CNN+IK variant can be tried.

MODEL=CNN1DIMUPoser. Knobs: CNN_CHANNELS (256), CNN_BLOCKS (6), CNN_KERNEL (3), CNN_DROPOUT (0.1),
TF_LR (3e-4), IK_LOSS / AP_IK_W (shared).
"""
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl

from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config


class _TCNBlock(nn.Module):
    def __init__(self, C, k, dilation, dropout):
        super().__init__()
        pad = dilation * (k - 1) // 2                          # 'same' (odd k) -> non-causal, symmetric
        self.conv1 = nn.Conv1d(C, C, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(C, C, k, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.act(self.conv1(x)))
        h = self.act(self.conv2(h))
        return x + h


class _TCN(nn.Module):
    def __init__(self, n_input, n_output, channels=256, blocks=6, k=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_input, channels, 1)
        self.blocks = nn.ModuleList([_TCNBlock(channels, k, 2 ** i, dropout) for i in range(blocks)])
        self.out = nn.Conv1d(channels, n_output, 1)

    def forward(self, x, lens):
        T = x.size(1)
        h = self.inp(x.transpose(1, 2))                        # B,C,T
        for b in self.blocks:
            h = b(h)
        y = self.out(h).transpose(1, 2)                        # B,T,n_output
        # zero padded positions so the loss matches the LSTM's pack/pad output
        dev = x.device
        pad = torch.arange(T, device=dev)[None, :] >= torch.as_tensor(lens, device=dev)[:, None]
        return y.masked_fill(pad.unsqueeze(-1), 0.0)


class CNN1DIMUPoser(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        n_input = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        self.net = _TCN(n_input, self.n_pose_output,
                        channels=int(os.environ.get("CNN_CHANNELS", "256")),
                        blocks=int(os.environ.get("CNN_BLOCKS", "6")),
                        k=int(os.environ.get("CNN_KERNEL", "3")),
                        dropout=float(os.environ.get("CNN_DROPOUT", "0.1")))
        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.loss = nn.MSELoss() if config.loss_type == "mse" else nn.L1Loss()
        self.lr = float(os.environ.get("TF_LR", "3e-4"))
        self.ik_loss = bool(os.environ.get("IK_LOSS"))
        self.ik_w = float(os.environ.get("AP_IK_W", "1.0"))
        self.register_buffer("imu_joints", torch.tensor([18, 19, 1, 2, 15]), persistent=False)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=["config"])

    def on_fit_start(self):
        if self.config.use_joint_loss:
            self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def forward(self, imu_inputs, imu_lens):
        return self.net(imu_inputs, imu_lens)

    def _ik_consistency(self, imu_inputs, pred_pose):
        B, T = imu_inputs.shape[:2]
        grot = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[0]
        pred_so = grot[:, self.imu_joints].view(B, T, 5, 3, 3)
        obs_ori = imu_inputs[:, :, 15:60].reshape(B, T, 5, 3, 3)
        m = (obs_ori.abs().flatten(3).sum(-1) > 0).unsqueeze(-1).unsqueeze(-1)
        if not m.any():
            return pred_pose.new_zeros(())
        return self.ik_w * self.loss(pred_so * m, obs_ori * m)

    def _step(self, batch):
        imu, target, lens, _ = batch
        pred_pose = self(imu, lens)[:, :, :self.n_pose_output]
        target_pose = target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            tj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1]
            loss = loss + self.loss(pj, tj)
        if self.ik_loss:
            loss = loss + self._ik_consistency(imu, pred_pose)
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
