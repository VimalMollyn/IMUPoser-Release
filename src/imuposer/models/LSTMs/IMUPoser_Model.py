r"""
IMUPoser Model
"""

import torch.nn as nn
import torch
import lightning.pytorch as pl
from .RNN import RNN
from imuposer.models.loss_functions import *
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config

class IMUPoserModel(pl.LightningModule):
    r"""
    Inputs - N IMUs, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config:Config):
        super().__init__()
        n_input = 12 * len(config.joints_set)

        n_output_joints = len(config.pred_joints_set)
        self.n_output_joints = n_output_joints
        self.n_pose_output = n_output_joints * (6 if config.r6d == True else 9)

        n_output = self.n_pose_output

        self.batch_size = config.batch_size
        
        self.dip_model = RNN(n_input=n_input, n_output=n_output, n_hidden=512, bidirectional=True)

        self.config = config

        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)

        if config.loss_type == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        self.lr = 3e-4

        # PyTorch Lightning >= 2.0 removed the `outputs` argument from the epoch-end
        # hooks, so we accumulate per-step losses ourselves.
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters(ignore=["config"])

    def forward(self, imu_inputs, imu_lens):
        pred_pose, _, _ = self.dip_model(imu_inputs, imu_lens)
        return pred_pose

    def training_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss

        self.log(f"training_step_loss", loss.item(), batch_size=self.batch_size)

        # store the scalar value (float), NOT the tensor: accumulating per-step CUDA
        # tensors here pins ~one graph's worth of GPU memory per step and OOMs.
        self.training_step_outputs.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss

        self.log(f"validation_step_loss", loss.item(), batch_size=self.batch_size)

        self.validation_step_outputs.append(loss.item())
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss

        return {"loss": loss.item(), "pred": pred_pose, "true": target_pose}

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_outputs, loop_type="train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_outputs, loop_type="val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.epoch_end_callback(self.test_step_outputs, loop_type="test")
        self.test_step_outputs.clear()

    def epoch_end_callback(self, outputs, loop_type="train"):
        if len(outputs) == 0:
            return

        # agg the losses (outputs are plain floats)
        avg_loss = sum(outputs) / len(outputs)
        self.log(f"{loop_type}_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
