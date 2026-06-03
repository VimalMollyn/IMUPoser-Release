r"""
Codebook pose model — inspired by AI4Animation SIGGRAPH 2024 "Categorical Codebook Matching for
Embodied Character Controllers" (Starke et al.).

Their controller matches sparse 3-point input to a learned categorical codebook of motions (sampled
for diverse, responsive character control). For our *accuracy* metric we use the soft/expectation
version: a bidirectional LSTM produces per-frame logits over K learned r6d-pose codewords; the output
pose is the softmax-weighted blend of codewords. This biases predictions toward a learned pose prior /
dictionary (a regularizer), the accuracy-oriented analog of categorical codebook matching.

MODEL=CodebookIMUPoser. Knobs: CB_SIZE (codewords K, default 256), CB_TEMP (softmax temperature, 1.0).
Same MSE(+FK joint) loss as the LSTM. eval_dip auto-detects via the 'codebook' key.
"""
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl

from ..LSTMs.RNN import RNN
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config


class CodebookIMUPoser(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        n_input = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        self.K = int(os.environ.get("CB_SIZE", "256"))
        self.temp = float(os.environ.get("CB_TEMP", "1.0"))
        self.rnn = RNN(n_input=n_input, n_output=self.K, n_hidden=512, bidirectional=True)
        self.codebook = nn.Parameter(torch.randn(self.K, self.n_pose_output) * 0.1)  # K learned r6d poses
        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.loss = nn.MSELoss() if config.loss_type == "mse" else nn.L1Loss()
        self.lr = 3e-4
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=["config"])

    def on_fit_start(self):
        if self.config.use_joint_loss:
            self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def forward(self, imu_inputs, imu_lens):
        logits, _, _ = self.rnn(imu_inputs, imu_lens)          # B,T,K
        w = torch.softmax(logits / self.temp, dim=-1)
        return torch.matmul(w, self.codebook)                  # B,T,n_pose (soft codeword blend)

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
