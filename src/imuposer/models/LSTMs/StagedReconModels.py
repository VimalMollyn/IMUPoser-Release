r"""
Two architectural experiments past the input-augmentation plateau (see ADDITIONAL_EXPERIMENTS.md):

- ReconIMUPoserModel: reconstruct the FULL 5-IMU set from the present (masked) sensors as an
  auxiliary stage, then regress pose from the reconstructed full set. The pose head always sees a
  "completed" input regardless of which combo is present. Requires config.aux_target == "imu"
  (dataset appends the full 5-IMU target after the pose target).

- StagedIMUPoserModel: TransPose-style cascade IMU -> joint positions -> pose, with intermediate
  supervision on (root-relative) joint positions. Requires config.aux_target == "joint".

Both: forward(imu, lens) returns the pose (r6d) so scripts/3. Evaluation/eval_dip.py works unchanged.
Selection metric = pose-only val loss (the aux/recon loss is not used for checkpoint selection).
"""
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl

from .RNN import RNN
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config


class _StagedBase(pl.LightningModule):
    """Shared Lightning boilerplate (loss, FK joint loss, epoch hooks, optimizer)."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_input = 12 * len(config.joints_set)                       # 60 (5 IMUs)
        self.n_pose = len(config.pred_joints_set) * (6 if config.r6d else 9)  # 144
        self.loss = nn.MSELoss() if config.loss_type == "mse" else nn.L1Loss()
        self.lr = 3e-4
        self.batch_size = config.batch_size
        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_fit_start(self):
        # rebuild SMPL body model on this rank's device (DDP / single-GPU correctness)
        if self.config.use_joint_loss:
            self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def _pose_loss(self, pred_pose, target_pose):
        l = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            tj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1]
            l = l + self.loss(pj, tj)
        return l

    def training_step(self, batch, batch_idx):
        loss, pose_loss = self._losses(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        self.training_step_outputs.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, pose_loss = self._losses(batch)
        # select on the POSE loss only (aux loss is a means, not the objective)
        self.log("validation_step_loss", pose_loss.item(), batch_size=self.batch_size)
        self.validation_step_outputs.append(pose_loss.item())
        return {"loss": pose_loss}

    def on_train_epoch_end(self):
        self._epoch_end(self.training_step_outputs, "train"); self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self._epoch_end(self.validation_step_outputs, "val"); self.validation_step_outputs.clear()

    def _epoch_end(self, outputs, loop):
        if outputs:
            self.log(f"{loop}_loss", sum(outputs) / len(outputs), prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ReconIMUPoserModel(_StagedBase):
    """present IMUs -> reconstruct full 5-IMU -> pose."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.recon_rnn = RNN(n_input=self.n_input, n_output=self.n_input, n_hidden=512, bidirectional=True)
        self.pose_rnn = RNN(n_input=self.n_input, n_output=self.n_pose, n_hidden=512, bidirectional=True)
        self.recon_w = float(os.environ.get("RECON_W", "1.0"))
        self.save_hyperparameters(ignore=["config"])

    def forward(self, imu, lens):
        recon, _, _ = self.recon_rnn(imu, lens)
        pose, _, _ = self.pose_rnn(recon, lens)
        return pose

    def _losses(self, batch):
        imu, target, lens, _ = batch
        recon, _, _ = self.recon_rnn(imu, lens)
        pred_pose, _, _ = self.pose_rnn(recon, lens)
        target_pose = target[:, :, :self.n_pose]
        target_imu = target[:, :, self.n_pose:self.n_pose + self.n_input]
        pose_loss = self._pose_loss(pred_pose, target_pose)
        recon_loss = self.loss(recon, target_imu)
        return pose_loss + self.recon_w * recon_loss, pose_loss


class StagedIMUPoserModel(_StagedBase):
    """TransPose-style: IMU -> joint positions -> pose."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.n_joint = len(config.pred_joints_set) * 3                   # 72
        self.joint_rnn = RNN(n_input=self.n_input, n_output=self.n_joint, n_hidden=512, bidirectional=True)
        self.pose_rnn = RNN(n_input=self.n_input + self.n_joint, n_output=self.n_pose, n_hidden=512, bidirectional=True)
        self.stage1_w = float(os.environ.get("STAGE1_W", "1.0"))
        self.save_hyperparameters(ignore=["config"])

    def forward(self, imu, lens):
        jp, _, _ = self.joint_rnn(imu, lens)
        pose, _, _ = self.pose_rnn(torch.cat([imu, jp], dim=2), lens)
        return pose

    def _losses(self, batch):
        imu, target, lens, _ = batch
        jp, _, _ = self.joint_rnn(imu, lens)
        pred_pose, _, _ = self.pose_rnn(torch.cat([imu, jp], dim=2), lens)
        target_pose = target[:, :, :self.n_pose]
        target_jp = target[:, :, self.n_pose:self.n_pose + self.n_joint]
        pose_loss = self._pose_loss(pred_pose, target_pose)
        joint_loss = self.loss(jp, target_jp)
        return pose_loss + self.stage1_w * joint_loss, pose_loss
