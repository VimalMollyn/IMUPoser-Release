r"""
AvatarPoser-style pose model (Jiang et al., ECCV 2022, arXiv:2207.13784).

AvatarPoser = Transformer encoder (linear embed -> 256-d) + FC decoder -> SMPL rotations, PLUS an
inverse-kinematics step that refines the arms so the predicted *end-effectors match the observed
sparse trackers*. We reuse the transformer encoder (same as TransformerIMUPoser) and add the
AvatarPoser-distinctive piece as a training loss: the predicted pose's forward-kinematics GLOBAL
orientations at the IMU joints ([18,19,1,2,15] = L/R elbow, L/R hip, head) must match the OBSERVED
sensor orientations (the IMU analog of "predicted end-effectors == measured trackers"). AvatarPoser
runs IK as test-time optimization to match tracker *positions*; we don't have positions (only acc+ori),
so we enforce the orientation consistency in training — same principle, no per-sequence optimization.

MODEL=AvatarPoserModel. Knobs: TF_DMODEL/TF_LAYERS/TF_HEADS/TF_FF/TF_DROPOUT (shared with the
transformer), AP_IK_W (consistency weight, default 1.0). Windowed inference like the transformer.
"""
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl

from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config
from .TransformerIMUPoser import _TransformerNet

# SMPL joints the 5 IMUs sit on (= ji_mask[:5] from preprocessing): L/R elbow, L/R hip, head.
_IMU_JOINTS = [18, 19, 1, 2, 15]


class AvatarPoserModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        n_input = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        self.net = _TransformerNet(
            n_input, self.n_pose_output,
            d_model=int(os.environ.get("TF_DMODEL", "256")),
            nhead=int(os.environ.get("TF_HEADS", "8")),
            n_layers=int(os.environ.get("TF_LAYERS", "4")),
            dim_ff=int(os.environ.get("TF_FF", "1024")),
            dropout=float(os.environ.get("TF_DROPOUT", "0.1")),
            use_bilstm=bool(os.environ.get("TF_BILSTM")))
        # AvatarPoser always uses the FK/IK geometric supervision, so force the body model on.
        self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.loss = nn.MSELoss() if config.loss_type == "mse" else nn.L1Loss()
        self.ik_w = float(os.environ.get("AP_IK_W", "1.0"))
        self.lr = float(os.environ.get("TF_LR", "3e-4"))
        self.register_buffer("imu_joints", torch.tensor(_IMU_JOINTS), persistent=False)
        # SIP-weighted pose loss (SIP_LOSS_W>0): upweight the r6d of the 4 SIP joints [1,2,16,17].
        # Same knob as the LSTM; the −1.1° SIP win there transfers if the transformer responds too.
        self.sip_w = float(os.environ.get("SIP_LOSS_W", "0"))
        self.register_buffer("_sip_dims",
                             torch.tensor([d for j in (1, 2, 16, 17) for d in range(j * 6, j * 6 + 6)]),
                             persistent=False)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=["config"])

    def on_fit_start(self):
        self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def forward(self, imu_inputs, imu_lens):
        T = imu_inputs.size(1)
        W = int(os.environ.get("TF_EVAL_WINDOW", "125"))
        if self.training or T <= W:
            return self.net(imu_inputs, imu_lens)
        outs = []
        for s in range(0, T, W):
            chunk = imu_inputs[:, s:s + W]
            clen = [int(min(max(l - s, 0), chunk.size(1))) for l in imu_lens]
            outs.append(self.net(chunk, clen))
        return torch.cat(outs, dim=1)

    def _step(self, batch):
        imu, target, lens, _ = batch
        B, T, _ = imu.shape
        pred_pose = self(imu, lens)[:, :, :self.n_pose_output]
        target_pose = target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.sip_w:
            loss = loss + self.sip_w * self.loss(pred_pose[..., self._sip_dims], target_pose[..., self._sip_dims])
        # FK once: global rotations (grot) AND joint positions
        grot, jpos = self.bodymodel.forward_kinematics(
            pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[:2]
        tgrot, tjpos = self.bodymodel.forward_kinematics(
            pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[:2]
        loss = loss + self.loss(jpos, tjpos)                              # FK joint-position loss
        # AvatarPoser IK consistency: predicted grot at the IMU joints == OBSERVED sensor orientation
        obs_ori = imu[:, :, 15:60].reshape(B, T, 5, 3, 3)                 # observed sensor rotations
        present = obs_ori.abs().flatten(3).sum(-1) > 0                    # B,T,5 (absent = zero-masked)
        pred_so = grot[:, self.imu_joints].view(B, T, 5, 3, 3)           # predicted grot at sensor joints
        m = present.unsqueeze(-1).unsqueeze(-1)
        if m.any():
            loss = loss + self.ik_w * self.loss(pred_so * m, obs_ori * m)
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
