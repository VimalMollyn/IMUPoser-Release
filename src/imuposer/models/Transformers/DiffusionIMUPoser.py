r"""
Conditional diffusion pose model (à la EgoEgo, Li et al. CVPR 2023, arXiv:2212.04636).

EgoEgo's stage-2 is a conditional diffusion model: a transformer denoiser over the full-body motion
sequence, conditioned by CONCATENATING the condition in the feature dim (cosine schedule, T=1000,
DDPM). Here the condition is the per-frame IMU feature (60-d) and the "motion" being denoised is the
r6d pose (144-d) — i.e. we model p(pose | IMU) and sample it.

Parameterization: predict x0 (the clean pose). This lets us keep the SAME MSE(+FK joint) loss as the
LSTM/Transformer baselines (applied to the predicted clean pose), so the only thing that changes vs the
transformer is the *iterative denoising* — a clean test of "does generative refinement help sparse-IMU
pose?". Training samples a random diffusion step t and denoises; eval runs DDIM sampling.

MODEL=DiffusionIMUPoser. forward(imu, lens) runs DDIM sampling and returns the r6d pose (B,T,144), so
eval_dip.py's forward(...)[:, :, :144] contract is unchanged. Knobs: DIFF_T (train steps), DIFF_SAMPLE_STEPS
(DDIM steps at eval), TF_DMODEL/TF_LAYERS/TF_HEADS/TF_FF/TF_DROPOUT (shared with the transformer).
"""
import os
import math
import torch
import torch.nn as nn
import lightning.pytorch as pl

from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.config import Config
from .TransformerIMUPoser import _SinusoidalPE


def _cosine_acp(T, s=0.008):
    x = torch.linspace(0, T, T + 1)
    acp = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    return acp / acp[0].clone()


class _Denoiser(nn.Module):
    r"""Transformer denoiser: [x_t(pose) ⊕ imu_cond] + step-emb -> predicted clean pose."""
    def __init__(self, n_cond, n_pose, d_model=256, nhead=8, n_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_pose + n_cond, d_model)
        self.pe = _SinusoidalPE(d_model)
        self.step_mlp = nn.Sequential(_SinusoidalStepEmb(d_model), nn.Linear(d_model, d_model),
                                      nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_pose))

    def forward(self, x_t, cond, t, pad_mask):
        h = self.inp(torch.cat([x_t, cond], dim=-1))          # B,T,d
        h = self.pe(h) + self.step_mlp(t)[:, None, :]          # add per-sequence step embedding
        h = self.enc(h, src_key_padding_mask=pad_mask)
        return self.out(h)


class _SinusoidalStepEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,) long
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1))
        a = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class DiffusionIMUPoser(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        n_cond = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        self.T = int(os.environ.get("DIFF_T", "1000"))
        self.sample_steps = int(os.environ.get("DIFF_SAMPLE_STEPS", "50"))
        self.net = _Denoiser(n_cond, self.n_pose_output,
                             d_model=int(os.environ.get("TF_DMODEL", "256")),
                             nhead=int(os.environ.get("TF_HEADS", "8")),
                             n_layers=int(os.environ.get("TF_LAYERS", "4")),
                             dim_ff=int(os.environ.get("TF_FF", "1024")),
                             dropout=float(os.environ.get("TF_DROPOUT", "0.1")))
        acp = _cosine_acp(self.T)                              # (T+1,)
        self.register_buffer("acp", acp)
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

    def _pad_mask(self, lens, T, dev):
        return torch.arange(T, device=dev)[None, :] >= torch.as_tensor(lens, device=dev)[:, None]

    # ---- training: denoise at a random diffusion step, predict clean pose ----
    def _step(self, batch):
        imu, target, lens, _ = batch
        x0 = target[:, :, :self.n_pose_output]
        B, T, _ = x0.shape
        dev = x0.device
        pad = self._pad_mask(lens, T, dev)
        t = torch.randint(1, self.T + 1, (B,), device=dev)
        a = self.acp[t][:, None, None]                         # B,1,1
        noise = torch.randn_like(x0)
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
        pred_x0 = self.net(x_t, imu, t, pad).masked_fill(pad.unsqueeze(-1), 0.0)
        loss = self.loss(pred_x0, x0)
        if self.config.use_joint_loss:
            pj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_x0).view(-1, 216))[1]
            tj = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(x0).view(-1, 216))[1]
            loss = loss + self.loss(pj, tj)
        return loss

    # ---- inference: DDIM sampling conditioned on the IMU ----
    @torch.no_grad()
    def forward(self, imu_inputs, imu_lens):
        # DIFF_SAMPLE_AVG>1 averages K independent DDIM samples -> approaches the conditional MEAN
        # pose, the right point estimate for a per-frame accuracy metric (a single generative sample
        # is high-variance). Averaging r6d then orthonormalizing (in the evaluator) is the usual trick.
        K = int(os.environ.get("DIFF_SAMPLE_AVG", "1"))
        if K == 1:
            return self._windowed(imu_inputs, imu_lens)
        acc = self._windowed(imu_inputs, imu_lens)
        for _ in range(K - 1):
            acc = acc + self._windowed(imu_inputs, imu_lens)
        return acc / K

    def _windowed(self, imu_inputs, imu_lens):
        T = imu_inputs.size(1)
        W = int(os.environ.get("TF_EVAL_WINDOW", "125"))   # = training window
        # Same length-generalization fix as the transformer: the denoiser was trained on <=125-frame
        # windows, so sample in training-length chunks rather than over the whole take at once.
        if T <= W:
            return self._sample(imu_inputs, imu_lens)
        outs = []
        for s in range(0, T, W):
            chunk = imu_inputs[:, s:s + W]
            clen = [int(min(max(l - s, 0), chunk.size(1))) for l in imu_lens]
            outs.append(self._sample(chunk, clen))
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def _sample(self, imu_inputs, imu_lens):
        B, T, _ = imu_inputs.shape
        dev = imu_inputs.device
        pad = self._pad_mask(imu_lens, T, dev)
        x_t = torch.randn(B, T, self.n_pose_output, device=dev)
        steps = torch.linspace(self.T, 1, self.sample_steps, device=dev).round().long()
        for i in range(len(steps)):
            t = steps[i].repeat(B)
            a_t = self.acp[t][:, None, None]
            pred_x0 = self.net(x_t, imu_inputs, t, pad).masked_fill(pad.unsqueeze(-1), 0.0)
            if i == len(steps) - 1:
                x_t = pred_x0
                break
            a_prev = self.acp[steps[i + 1].repeat(B)][:, None, None]
            eps = (x_t - a_t.sqrt() * pred_x0) / (1 - a_t).clamp(min=1e-8).sqrt()
            x_t = a_prev.sqrt() * pred_x0 + (1 - a_prev).sqrt() * eps   # DDIM (deterministic)
        return x_t

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        self.training_step_outputs.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # denoising (training-objective) loss as the cheap per-epoch val signal; final model is
        # what gets scored by SIP via eval_dip.py (sampling), consistent with the fixed-epoch protocol.
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
