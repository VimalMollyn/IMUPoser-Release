r"""
IMUPoser Model
"""

import os
import torch.nn as nn
import torch
import lightning.pytorch as pl
from .RNN import RNN
from imuposer.models.loss_functions import *
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config


class _GradReverse(torch.autograd.Function):
    """Gradient-reversal: identity forward, negated (scaled) gradient backward. Lets a discriminator
    train normally while the upstream pose model is pushed in the OPPOSITE (adversarial) direction,
    so adversarial training fits in standard single-optimizer automatic optimization."""
    @staticmethod
    def forward(ctx, x, w):
        ctx.w = w
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.w * g, None


class IMUPoserModel(pl.LightningModule):
    r"""
    Inputs - N IMUs, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config:Config):
        super().__init__()
        n_input = 12 * len(config.joints_set)
        self.n_input = n_input

        # privileged-information DISTILLATION: a frozen 5-IMU teacher (DISTILL_TEACHER=<ckpt>) supervises
        # this (3-IMU) student to match its pose. Teacher runs on the full clean 5-IMU (AUX_TARGET=imu
        # appends it to the target). DISTILL_W weights the teacher-matching MSE. Loaded in on_fit_start.
        self.distill_teacher = os.environ.get("DISTILL_TEACHER")
        self.distill_w = float(os.environ.get("DISTILL_W", "1.0"))
        self._teacher = None
        # SELECTIVE_DISTILL=1: only distill the joints lw_rp_h CAN infer (have a nearby sensor: head/spine,
        # left arm via lw, right leg via rp, root); mask the un-inferable right-arm/left-leg (no rw/lp),
        # which naive distillation forced the student toward and wasted capacity on.
        self.distill_selective = bool(os.environ.get("SELECTIVE_DISTILL"))
        _inferable = [0, 2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 18, 20]
        _jw = torch.zeros(24); _jw[_inferable] = 1.0
        self.register_buffer("distill_jw", _jw.repeat_interleave(6), persistent=False)  # 144-d (r6d)

        n_output_joints = len(config.pred_joints_set)
        self.n_output_joints = n_output_joints
        self.n_pose_output = n_output_joints * (6 if config.r6d == True else 9)

        # multi-task TRANSLATION head (TRANS_LOSS=1): widen the RNN output by 3 to also predict the
        # root translation (paired with AUX_TARGET=tran in the dataset). Tests whether translation as an
        # auxiliary task helps pose via shared features (the metric is root-relative, so any gain is indirect).
        self.trans_loss = bool(os.environ.get("TRANS_LOSS"))
        self.trans_w = float(os.environ.get("TRANS_W", "1.0"))
        n_output = self.n_pose_output + (3 if self.trans_loss else 0)

        self.batch_size = config.batch_size

        # LSTM_HIDDEN / LSTM_LAYERS knobs for the capacity-check experiment (default 512 / 2 = baseline)
        _h = int(os.environ.get("LSTM_HIDDEN", "512"))
        _l = int(os.environ.get("LSTM_LAYERS", "2"))
        self.dip_model = RNN(n_input=n_input, n_output=n_output, n_hidden=_h, n_rnn_layer=_l, bidirectional=True)

        self.config = config

        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)

        # Optional AvatarPoser-style IK orientation-consistency loss (IK_LOSS=1): the predicted FK
        # GLOBAL orientation at the IMU joints ([18,19,1,2,15]) must match the OBSERVED sensor
        # orientation. Default OFF so the baseline LSTM is byte-for-byte unchanged.
        self.ik_loss = bool(os.environ.get("IK_LOSS"))
        self.ik_w = float(os.environ.get("AP_IK_W", "1.0"))
        # physics-informed ACCELERATION-consistency loss (ACC_LOSS=1): the predicted motion's synthetic
        # acceleration at the IMU joints must match the OBSERVED accelerometer signal. Uses the accel half
        # of the IMU (the IK term uses only orientation). Joint-proxy for the mounting vertex; 25fps^2 to
        # match _syn_acc's true-accel units (which were /acc_scale in the input).
        self.acc_loss = bool(os.environ.get("ACC_LOSS"))
        self.acc_w = float(os.environ.get("ACC_W", "1.0"))
        self.acc_scale = config.acc_scale
        self.register_buffer("imu_joints", torch.tensor([18, 19, 1, 2, 15]), persistent=False)

        # Optional SIP-weighted pose loss (SIP_LOSS_W>0): add an extra MSE term on the r6d of the
        # 4 SIP joints ([1,2,16,17] = L/R hip, L/R shoulder) that dominate the reported SIP metric,
        # so the optimizer spends more capacity where the headline error lives. Default OFF.
        self.sip_w = float(os.environ.get("SIP_LOSS_W", "0"))
        self.register_buffer("_sip_dims",
                             torch.tensor([d for j in (1, 2, 16, 17) for d in range(j * 6, j * 6 + 6)]),
                             persistent=False)

        # Adversarial POSE PRIOR (ADV_W>0): a per-frame discriminator learns to tell real AMASS poses
        # from predicted ones; via the gradient-reversal layer the pose model is pushed to make its
        # output indistinguishable from real poses. The signal bites hardest where the L2 gradient is
        # flat — the un-sensed limbs that otherwise collapse to the implausible conditional mean — so it
        # restores plausibility there at low cost to the well-sensed joints. Default OFF.
        self.adv_w = float(os.environ.get("ADV_W", "0"))
        if self.adv_w:
            self.discriminator = nn.Sequential(
                nn.Linear(self.n_pose_output, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1))

        if config.loss_type == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        self.lr = float(os.environ.get("LR", "3e-4"))   # LR / OPTIMIZER / WEIGHT_DECAY / LR_SCHED knobs

        # PyTorch Lightning >= 2.0 removed the `outputs` argument from the epoch-end
        # hooks, so we accumulate per-step losses ourselves.
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters(ignore=["config"])

    def forward(self, imu_inputs, imu_lens):
        pred_pose, _, _ = self.dip_model(imu_inputs, imu_lens)
        return pred_pose

    def _ik_consistency(self, imu_inputs, pred_pose):
        # AvatarPoser-style: predicted FK global orientation at the IMU joints must match the
        # OBSERVED sensor orientation (present sensors only; absent ones are zero-masked in the input).
        B, T = imu_inputs.shape[:2]
        grot = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[0]
        pred_so = grot[:, self.imu_joints].view(B, T, 5, 3, 3)
        obs_ori = imu_inputs[:, :, 15:60].reshape(B, T, 5, 3, 3)
        m = (obs_ori.abs().flatten(3).sum(-1) > 0).unsqueeze(-1).unsqueeze(-1)
        if not m.any():
            return pred_pose.new_zeros(())
        return self.ik_w * self.loss(pred_so * m, obs_ori * m)

    def _acc_consistency(self, imu_inputs, pred_pose):
        # predicted joint-acceleration (2nd time-diff x 25fps^2 / acc_scale) vs observed IMU accel
        B, T = imu_inputs.shape[:2]
        jpos = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
        sp = jpos[:, self.imu_joints].view(B, T, 5, 3)
        a = torch.zeros_like(sp)
        a[:, 1:-1] = (sp[:, :-2] + sp[:, 2:] - 2 * sp[:, 1:-1]) * (25.0 ** 2) / self.acc_scale
        obs_acc = imu_inputs[:, :, :15].reshape(B, T, 5, 3)
        obs_ori = imu_inputs[:, :, 15:60].reshape(B, T, 5, 3, 3)
        m = (obs_ori.abs().flatten(3).sum(-1) > 0).unsqueeze(-1)
        if not m.any():
            return pred_pose.new_zeros(())
        return self.acc_w * self.loss(a * m, obs_acc * m)

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
        if self.sip_w:
            loss = loss + self.sip_w * self.loss(pred_pose[..., self._sip_dims], target_pose[..., self._sip_dims])
        if self.adv_w and self.training:
            # discriminator trains real(target)->1, fake(pred)->0; the gradient-reversal pushes the pose
            # model to fool it -> predicted poses (esp. the unconstrained un-sensed limbs) move onto the
            # real-pose manifold instead of collapsing to the mean. Guarded by self.training so the
            # validation_step_loss (checkpoint-selection signal) stays the clean pose error.
            real = self.discriminator(target_pose.reshape(-1, self.n_pose_output))
            fake = self.discriminator(_GradReverse.apply(pred_pose.reshape(-1, self.n_pose_output), self.adv_w))
            loss = loss + (nn.functional.binary_cross_entropy_with_logits(real, torch.ones_like(real)) +
                           nn.functional.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))
        if self.ik_loss:
            loss = loss + self._ik_consistency(imu_inputs, pred_pose)
        if self.acc_loss:
            loss = loss + self._acc_consistency(imu_inputs, pred_pose)
        if self.trans_loss:
            np_ = self.n_pose_output
            loss = loss + self.trans_w * self.loss(_pred[:, :, np_:np_+3], _target[:, :, np_:np_+3])
        if self._teacher is not None:
            np_ = self.n_pose_output
            loss = loss + self._distill_loss(_target[:, :, np_:np_+self.n_input], input_lengths, pred_pose)

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
        if self.sip_w:
            loss = loss + self.sip_w * self.loss(pred_pose[..., self._sip_dims], target_pose[..., self._sip_dims])
        if self.adv_w and self.training:
            # discriminator trains real(target)->1, fake(pred)->0; the gradient-reversal pushes the pose
            # model to fool it -> predicted poses (esp. the unconstrained un-sensed limbs) move onto the
            # real-pose manifold instead of collapsing to the mean. Guarded by self.training so the
            # validation_step_loss (checkpoint-selection signal) stays the clean pose error.
            real = self.discriminator(target_pose.reshape(-1, self.n_pose_output))
            fake = self.discriminator(_GradReverse.apply(pred_pose.reshape(-1, self.n_pose_output), self.adv_w))
            loss = loss + (nn.functional.binary_cross_entropy_with_logits(real, torch.ones_like(real)) +
                           nn.functional.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))
        if self.ik_loss:
            loss = loss + self._ik_consistency(imu_inputs, pred_pose)
        if self.acc_loss:
            loss = loss + self._acc_consistency(imu_inputs, pred_pose)
        if self.trans_loss:
            np_ = self.n_pose_output
            loss = loss + self.trans_w * self.loss(_pred[:, :, np_:np_+3], _target[:, :, np_:np_+3])
        if self._teacher is not None:
            np_ = self.n_pose_output
            loss = loss + self._distill_loss(_target[:, :, np_:np_+self.n_input], input_lengths, pred_pose)

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

    def on_fit_start(self):
        # The SMPL body model holds plain (non-registered) tensors, so Lightning's
        # device move doesn't touch it. Rebuild it on this rank's device so the joint
        # loss works under multi-GPU (DDP), where each rank uses a different GPU.
        if self.config.use_joint_loss:
            self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)
        if self.distill_teacher and self._teacher is None:
            # frozen 5-IMU teacher (a GlobalModelIMUPoser RNN, default 512/2)
            t = RNN(n_input=self.n_input, n_output=self.n_pose_output, n_hidden=512, n_rnn_layer=2, bidirectional=True)
            sd = torch.load(self.distill_teacher, map_location="cpu")["state_dict"]
            t.load_state_dict({k[len("dip_model."):]: v for k, v in sd.items() if k.startswith("dip_model.")})
            t.eval().to(self.device)
            for p in t.parameters():
                p.requires_grad_(False)
            self._teacher = t

    def _distill_loss(self, full_imu, lens, student_pose):
        # teacher pose from the full clean 5-IMU (AUX_TARGET=imu), match it (privileged distillation)
        with torch.no_grad():
            tp = self._teacher(full_imu, lens)[0][:, :, :self.n_pose_output]
        if self.distill_selective:
            w = self.distill_jw.to(student_pose.device)
            return self.distill_w * self.loss(student_pose * w, tp * w)
        return self.distill_w * self.loss(student_pose, tp)

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
        opt_name = os.environ.get("OPTIMIZER", "adam").lower()
        wd = float(os.environ.get("WEIGHT_DECAY", "0"))
        if opt_name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=(wd or 1e-4))
        elif opt_name == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=wd, nesterov=True)
        elif opt_name == "rmsprop":
            opt = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=wd)
        elif opt_name == "radam":
            opt = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=wd)
        else:
            opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=wd)
        if os.environ.get("LR_SCHED", "") == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(os.environ.get("EPOCHS", "30")))
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
