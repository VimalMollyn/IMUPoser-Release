r"""
Activity-conditioned pose model: predict the activity from IMU, then condition pose on it.

Hypothesis (user): knowing the activity constrains the pose manifold -> better pose. Our data has no
activity labels, so the dataset assigns PSEUDO-activities by k-means on mean-pose (AUX_TARGET=activity,
ACT_K clusters). Stage 1: an activity head predicts the activity from window-pooled IMU features
(cross-entropy vs the pseudo-label). Stage 2: the pose head (a second RNN) is conditioned on the
predicted-activity embedding (concatenated to the per-frame IMU). End-to-end.

MODEL=ActivityIMUPoser. Knobs: ACT_K (clusters, 16), ACT_EMB (embed dim, 32), ACT_W (CE weight, 0.1).
forward(imu,lens) predicts activity internally + returns pose, so eval is unchanged (auto-detected via
the act_head key). ACT_K/ACT_EMB must match at eval (defaults do).
"""
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl

from ..LSTMs.RNN import RNN
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config


class ActivityIMUPoser(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        n_input = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        self.K = int(os.environ.get("ACT_K", "16"))
        self.emb = int(os.environ.get("ACT_EMB", "32"))
        self.act_w = float(os.environ.get("ACT_W", "0.1"))
        self.enc = RNN(n_input=n_input, n_output=256, n_hidden=512, bidirectional=True)   # IMU features
        self.act_head = nn.Linear(256, self.K)
        self.act_embed = nn.Parameter(torch.randn(self.K, self.emb) * 0.1)
        self.pose_rnn = RNN(n_input=n_input + self.emb, n_output=self.n_pose_output, n_hidden=512, bidirectional=True)
        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.loss = nn.MSELoss() if config.loss_type == "mse" else nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.lr = 3e-4
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=["config"])

    def on_fit_start(self):
        if self.config.use_joint_loss:
            self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def _encode_activity(self, imu, lens):
        feats, _, _ = self.enc(imu, lens)                       # B,T,256
        T = feats.shape[1]
        m = (torch.arange(T, device=feats.device)[None, :] < torch.as_tensor(lens, device=feats.device)[:, None]).float()
        pooled = (feats * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)  # B,256 (masked mean)
        logits = self.act_head(pooled)                          # B,K
        a = torch.softmax(logits, dim=-1) @ self.act_embed      # B,emb (soft activity embedding)
        return logits, a

    def forward(self, imu_inputs, imu_lens):
        _, a = self._encode_activity(imu_inputs, imu_lens)
        T = imu_inputs.shape[1]
        cond = torch.cat([imu_inputs, a[:, None, :].expand(-1, T, -1)], dim=2)
        pose, _, _ = self.pose_rnn(cond, imu_lens)
        return pose

    def _step(self, batch):
        imu, target, lens, _ = batch
        logits, a = self._encode_activity(imu, lens)
        T = imu.shape[1]
        cond = torch.cat([imu, a[:, None, :].expand(-1, T, -1)], dim=2)
        pred_pose, _, _ = self.pose_rnn(cond, lens)
        pred_pose = pred_pose[:, :, :self.n_pose_output]
        target_pose = target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            tj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1]
            loss = loss + self.loss(pj, tj)
        if target.shape[2] > self.n_pose_output:                # activity label appended -> supervise
            act_lab = target[:, 0, self.n_pose_output].long()
            loss = loss + self.act_w * self.ce(logits, act_lab)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
