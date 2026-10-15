r"""
Generative pose models for sparse-IMU tracking on one shared DiT-style backbone (re-implementation, 2026-10-12).

  RegressIMUPoser       deterministic regression with this backbone (the architecture control)
  FlowIMUPoser          conditional flow matching / rectified flow (Lipman et al. 2023, Liu et al. 2023):
                        y_t = (1-t) z + t y1, the network predicts the velocity v = y1 - z, Euler/midpoint sampling
  DiffusionV2IMUPoser   variance-preserving diffusion with the cosine alpha/sigma schedule, v-prediction
                        (Salimans & Ho 2022) and deterministic DDIM sampling

Time convention for both generative models: t in [0, 1], t = 0 is pure noise, t = 1 is clean data.

What is different from the June DiffusionIMUPoser (and why that one was a weak test):
  * pose targets are standardised per r6d dimension (mean/std measured on the training set at fit start and
    stored as buffers), so unit noise is commensurate with the data instead of swamping dimensions whose std
    is 0.05-0.3 (which made almost every diffusion step a "pure noise" step and the sampler useless);
  * the time step enters every block through adaLN-zero modulation (DiT) instead of one additive embedding;
  * the training target is the velocity / v, not x0 (an x0 head copies its input at low noise, so sampling
    errors were never corrected);
  * t is drawn logit-normal (SD3) by default so the mid-path, where the network has to do the work, is
    sampled most;
  * checkpoint selection uses the POSE error of a deterministic few-step estimate (the same MSE + FK + IK loss
    as AvatarPoserModel), not the noisy training objective;
  * the FK/IK geometric loss is applied to the x1 estimate implied by the prediction, so the generative models
    get exactly the geometric supervision the regression baseline gets;
  * eval exposes the solver, the number of steps, K-sample averaging (the conditional mean, the right point
    estimate for a mean-error metric) and classifier-free guidance.

Env knobs (train): TF_DMODEL / TF_HEADS / TF_LAYERS / TF_FF / TF_DROPOUT / TF_BILSTM (backbone, as AvatarPoser),
  GEN_T_DIST=logitnormal|uniform, GEN_T_M / GEN_T_S (logit-normal location / scale, 0 / 1),
  GEN_GEOM_W (weight of the FK/IK loss on the x1 estimate, 1), GEN_COND_DROP (p of zeroing the IMU condition
  for classifier-free guidance, 0), GEN_STD_MIN (floor of the per-dim std, 0.05), GEN_STATS_BATCHES (40),
  GEN_VAL_STEPS (steps of the validation estimate, 1), AP_IK_W (IK term weight, 1), TF_LR, TF_WD, LR_SCHED, ...
Env knobs (eval): GEN_STEPS (8), GEN_SOLVER=euler|midpoint (flow; diffusion always DDIM), GEN_SAMPLE_AVG (K, 1),
  GEN_CFG (guidance scale, 1 = off), GEN_SEED (0), TF_EVAL_WINDOW / TF_EVAL_STRIDE (as the transformer).

forward(imu, lens) returns the r6d pose (B, T, 144) like every other model, so offline_fit.py / eval_dip.py
work unchanged; offline_fit detects the class from the `gen_kind` buffer.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_r6d
from imuposer.config import Config
from .TransformerIMUPoser import _SinusoidalPE, windowed_inference
from .AvatarPoserModel import AvatarPoserModel, _IMU_JOINTS


def _env_f(k, d): return float(os.environ.get(k, d))
def _env_i(k, d): return int(os.environ.get(k, d))


class _TimeEmb(nn.Module):
    """Sinusoidal embedding of t in [0, 1] (scaled by 1000) followed by a 2-layer MLP."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):                      # t: (B,) or (B,T) float in [0, 1] -> (B,d) or (B,T,d)
        half = self.d // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        a = (t.float() * 1000.0)[..., None] * freqs
        e = torch.cat([a.sin(), a.cos()], dim=-1)
        if e.shape[-1] < self.d:
            e = F.pad(e, (0, self.d - e.shape[-1]))
        return self.mlp(e)


class _MoEFFN(nn.Module):
    """Mixture-of-experts feed-forward (the LLM recipe): E expert MLPs, a linear router per frame, top-k softmax gating, and a
    Switch-style load-balancing loss (E * sum_e f_e * P_e, f = fraction of frames routed to e, P = mean router probability of e).
    Experts are computed densely and masked (fine at our sizes); the aux loss is stored in self.aux for the module to add."""
    def __init__(self, d, ff, n_exp, k, dropout):
        super().__init__()
        self.n_exp, self.k = n_exp, k
        self.router = nn.Linear(d, n_exp)
        self.linear1 = nn.ModuleList([nn.Linear(d, ff) for _ in range(n_exp)])
        self.linear2 = nn.ModuleList([nn.Linear(ff, d) for _ in range(n_exp)])
        self.drop = nn.Dropout(dropout)
        self.aux = torch.zeros(())

    def forward(self, x, valid):               # x: (B,T,d)  valid: (B,T) bool (unpadded frames)
        logits = self.router(x)                                        # (B,T,E)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)                        # (B,T,k)
        gate = torch.zeros_like(probs).scatter(-1, topi, topv)
        gate = gate / gate.sum(-1, keepdim=True).clamp(min=1e-9)       # renormalise the kept experts
        out = torch.zeros_like(x)
        for e in range(self.n_exp):
            g = gate[..., e:e + 1]
            if g.any():
                out = out + g * self.linear2[e](self.drop(F.gelu(self.linear1[e](x))))
        if self.training:
            v = valid.unsqueeze(-1).float()
            n = v.sum().clamp(min=1)
            frac = ((gate > 0).float() * v).sum((0, 1)) / n              # fraction of frames routed to each expert
            pmean = (probs * v).sum((0, 1)) / n                          # mean router probability per expert
            self.aux = self.n_exp * (frac * pmean).sum()
        return out


class _Block(nn.Module):
    """Pre-LN transformer block with adaLN-zero modulation (DiT): scale/shift/gate per sub-layer from the time embedding."""
    def __init__(self, d, nhead, ff, dropout, n_exp=0, k=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.moe = _MoEFFN(d, ff, n_exp, k, dropout) if n_exp > 0 else None
        if self.moe is None:
            self.linear1 = nn.Linear(d, ff)    # names match nn.TransformerEncoderLayer so offline_fit reads FF off the weights
            self.linear2 = nn.Linear(ff, d)
        self.drop = nn.Dropout(dropout)
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, h, c, pad):              # h: (B,T,d)  c: (B,d) one t per window, or (B,T,d) one t per frame  pad: (B,T) True = padded
        mod = self.ada(F.silu(c))
        if mod.dim() == 2:
            mod = mod.unsqueeze(1)
        s1, b1, g1, s2, b2, g2 = mod.chunk(6, dim=-1)
        x = self.norm1(h) * (1 + s1) + b1
        a = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]
        h = h + g1 * self.drop(a)
        x = self.norm2(h) * (1 + s2) + b2
        if self.moe is not None:
            h = h + g2 * self.drop(self.moe(x, ~pad))
        else:
            h = h + g2 * self.drop(self.linear2(self.drop(F.gelu(self.linear1(x)))))
        return h


class _GenNet(nn.Module):
    """[x_t ⊕ imu] per frame -> sinusoidal PE -> N adaLN blocks -> adaLN final norm -> (biLSTM) -> MLP head."""
    def __init__(self, n_in, n_out, d_model=256, nhead=8, n_layers=4, dim_ff=1024, dropout=0.1, use_bilstm=False, n_exp=0, k=2):
        super().__init__()
        self.inp = nn.Linear(n_in, d_model)    # `net.inp.weight` -> offline_fit reads d_model off it
        self.pe = _SinusoidalPE(d_model)
        self.t_emb = _TimeEmb(d_model)
        self.enc = nn.Module()
        self.enc.layers = nn.ModuleList([_Block(d_model, nhead, dim_ff, dropout, n_exp, k) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ada_out = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.ada_out.weight); nn.init.zeros_(self.ada_out.bias)
        self.use_bilstm = use_bilstm
        head_in = d_model
        if use_bilstm:
            self.bilstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, bidirectional=True)
            head_in = d_model * 2
        self.out = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, n_out))

    def moe_aux(self):
        return sum(blk.moe.aux for blk in self.enc.layers if blk.moe is not None)

    def forward(self, x, t, pad):
        full = pad.all(dim=1)                  # a fully padded row would make attention NaN; its output is zeroed anyway
        if full.any():
            pad = pad & ~full[:, None]
        c = self.t_emb(t)
        h = self.pe(self.inp(x))
        for blk in self.enc.layers:
            h = blk(h, c, pad)
        mod = self.ada_out(F.silu(c))
        if mod.dim() == 2:
            mod = mod.unsqueeze(1)
        s, b = mod.chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + s) + b
        if self.use_bilstm:
            h, _ = self.bilstm(h)
        y = self.out(h)
        return y.masked_fill(pad.unsqueeze(-1), 0.0)


class _GenPoser(pl.LightningModule):
    KIND = 0                                   # 0 regression, 1 flow matching, 2 diffusion (stored in the `gen_kind` buffer)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_cond = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        self.generative = self.KIND != 0
        n_in = self.n_cond + (self.n_pose_output if self.generative else 0)
        # GEN_MOE_E > 0: mixture-of-experts feed-forward in every block (GEN_MOE_E experts, top-GEN_MOE_K routing, load-balancing
        # loss weight GEN_MOE_AUX), the LLM recipe, trained jointly from scratch
        self.moe_e, self.moe_k, self.moe_aux_w = _env_i("GEN_MOE_E", 0), _env_i("GEN_MOE_K", 2), _env_f("GEN_MOE_AUX", 0.01)
        self.net = _GenNet(n_in, self.n_pose_output,
                           d_model=_env_i("TF_DMODEL", 256), nhead=_env_i("TF_HEADS", 8), n_layers=_env_i("TF_LAYERS", 4),
                           dim_ff=_env_i("TF_FF", 1024), dropout=_env_f("TF_DROPOUT", 0.1), use_bilstm=bool(os.environ.get("TF_BILSTM")),
                           n_exp=self.moe_e, k=self.moe_k)
        self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.loss = nn.MSELoss() if config.loss_type == "mse" else nn.L1Loss()
        self.ik_w = _env_f("AP_IK_W", 1.0)
        self.lr = _env_f("TF_LR", 3e-4)
        self.register_buffer("imu_joints", torch.tensor(_IMU_JOINTS), persistent=False)
        self.register_buffer("gen_kind", torch.tensor(self.KIND))
        self.register_buffer("x_mean", torch.zeros(self.n_pose_output))
        self.register_buffer("x_std", torch.ones(self.n_pose_output))
        self.register_buffer("stats_ready", torch.tensor(0))
        self.t_dist = os.environ.get("GEN_T_DIST", "logitnormal").lower()
        self.t_m, self.t_s = _env_f("GEN_T_M", 0.0), _env_f("GEN_T_S", 1.0)
        self.geom_w = _env_f("GEN_GEOM_W", 1.0)
        self.cond_drop = _env_f("GEN_COND_DROP", 0.0)
        self.std_min = _env_f("GEN_STD_MIN", 0.05)
        self.val_steps = _env_i("GEN_VAL_STEPS", 1)
        # HISTORY-CONDITIONED (autoregressive) mode, the DiffusionPoser / EgoEgo way: GEN_HIST=1 trains with per-frame time
        # levels where a random prefix of the window is the clean (or slightly noised) pose history and only the rest is
        # generated; GEN_AR=1 at eval then samples a sequence chunk by chunk (GEN_CHUNK new frames per step, the last
        # TF_EVAL_WINDOW - GEN_CHUNK generated frames as fixed context, the IMU of the whole window as condition).
        self.hist = _env_i("GEN_HIST", 0)
        self.hist_p = _env_f("GEN_HIST_P", 0.7)           # share of training windows that get a history prefix
        self.ctx_noise_p = _env_f("GEN_CTX_NOISE_P", 0.5) # share of those whose context is slightly noised (t in [0.7, 1))
        self.chunk = _env_i("GEN_CHUNK", 25)              # new frames per autoregressive step (1 s at 25 fps)
        self.min_new = 8
        # SPATIAL INPAINTING (GEN_INPAINT=1): the target is the GLOBAL rotation of every joint (the metric's space) and the
        # sensed joints' columns are "known pixels": their measured orientation replaces the noisy state at every sampling step
        # (RePaint-style) and in training (except a GEN_INPAINT_NOISE_P share of windows, where they are noised like the rest,
        # for robustness). GEN_INPAINT_OUT=1 (eval) also writes the measured orientation into the final output for those joints.
        self.inpaint = _env_i("GEN_INPAINT", 0)
        self.inpaint_out = _env_i("GEN_INPAINT_OUT", 1)
        self.inpaint_noise_p = _env_f("GEN_INPAINT_NOISE_P", 0.2)
        self.training_step_outputs, self.validation_step_outputs, self.test_step_outputs = [], [], []
        self.save_hyperparameters(ignore=["config"])
        n_par = sum(p.numel() for p in self.parameters())
        print(f"[gen] {type(self).__name__}: {n_par/1e6:.2f} M params, t_dist={self.t_dist} geom_w={self.geom_w} "
              f"cond_drop={self.cond_drop} val_steps={self.val_steps} hist={self.hist} (p={self.hist_p}, ctx_noise_p={self.ctx_noise_p}, chunk={self.chunk}) "
              f"inpaint={self.inpaint} (out={self.inpaint_out}, noise_p={self.inpaint_noise_p}) moe={self.moe_e}x top-{self.moe_k} (aux {self.moe_aux_w})", flush=True)

    # ---- helpers -------------------------------------------------------------------------------------------
    def _pad_mask(self, lens, T, dev):
        return torch.arange(T, device=dev)[None, :] >= torch.as_tensor(lens, device=dev)[:, None]

    def _norm(self, x): return (x - self.x_mean) / self.x_std
    def _denorm(self, y): return y * self.x_std + self.x_mean

    # ---- global <-> local rotation representations (spatial inpainting works in global-rotation space) -------
    def _to_global(self, x_local):              # (B,T,144) local r6d -> (B,T,144) global r6d
        B, T, _ = x_local.shape
        R = r6d_to_rotation_matrix(x_local.reshape(-1, 24, 6)).view(-1, 24, 3, 3)
        G = self.bodymodel.forward_kinematics_R(R)
        return rotation_matrix_to_r6d(G).view(B, T, 24 * 6)

    def _to_local(self, x_global):              # (B,T,144) global r6d -> (B,T,144) local r6d
        B, T, _ = x_global.shape
        G = r6d_to_rotation_matrix(x_global.reshape(-1, 24, 6)).view(-1, 24, 3, 3)
        par = torch.as_tensor(self.bodymodel.parent[1:], device=G.device)
        L = G.clone()
        L[:, 1:] = G[:, par].transpose(-1, -2) @ G[:, 1:]
        return rotation_matrix_to_r6d(L).view(B, T, 24 * 6)

    def _known(self, imu):
        """Sensed joints as known pixels: (B,T,144) bool mask over r6d columns and the measured global rotations as r6d."""
        B, T, _ = imu.shape
        obs = imu[:, :, 15:60].reshape(B, T, 5, 3, 3)
        present = obs.abs().flatten(3).sum(-1) > 0                                   # (B,T,5)
        r6 = rotation_matrix_to_r6d(obs).view(B, T, 5, 6)
        mask = torch.zeros(B, T, 24, dtype=torch.bool, device=imu.device)
        vals = torch.zeros(B, T, 24, 6, device=imu.device)
        for k, j in enumerate(_IMU_JOINTS):
            mask[:, :, j] = present[:, :, k]
            vals[:, :, j] = torch.where(present[:, :, k, None], r6[:, :, k], vals[:, :, j])
        return mask.repeat_interleave(6, dim=-1), vals.view(B, T, 24 * 6)

    def on_fit_start(self):
        self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def on_train_start(self):
        # the train dataloader exists from here on (fit_loop.setup_data ran); a fresh run measures the target stats once,
        # a resumed / warm-started run keeps the buffers that came with its checkpoint
        if self.generative and int(self.stats_ready.item()) == 0:
            self._measure_stats()

    @torch.no_grad()
    def _measure_stats(self):
        """Per-dimension mean / std of the r6d targets over the first GEN_STATS_BATCHES training batches (valid frames only)."""
        dl = self.trainer.train_dataloader
        if dl is None:
            dl = self.trainer.datamodule.train_dataloader()
        n_b = _env_i("GEN_STATS_BATCHES", 40)
        s = torch.zeros(self.n_pose_output, dtype=torch.float64); ss = torch.zeros_like(s); cnt = 0
        for i, batch in enumerate(dl):
            _, target, lens, _ = batch
            x = target[:, :, :self.n_pose_output]
            if self.inpaint:
                x = self._to_global(x.to(self.device)).cpu()
            m = torch.arange(x.shape[1])[None, :] < torch.as_tensor(lens)[:, None]
            x = x[m].double()
            s += x.sum(0); ss += (x * x).sum(0); cnt += x.shape[0]
            if i + 1 >= n_b:
                break
        del dl
        mean = s / max(cnt, 1)
        std = (ss / max(cnt, 1) - mean * mean).clamp(min=0).sqrt().clamp(min=self.std_min)
        self.x_mean.copy_(mean.float().to(self.device)); self.x_std.copy_(std.float().to(self.device)); self.stats_ready.fill_(1)
        print(f"[gen] target stats from {cnt} frames: std min {std.min():.3f} median {std.median():.3f} max {std.max():.3f} "
              f"(floor {self.std_min}); |mean| max {mean.abs().max():.3f}", flush=True)

    def _pose_loss(self, imu, pred_pose, target_pose, B, T):
        """AvatarPoser's default supervision: r6d MSE + FK joint-position MSE + IK sensor-orientation consistency."""
        loss = self.loss(pred_pose, target_pose)
        grot, jpos = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[:2]
        tgrot, tjpos = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[:2]
        loss = loss + self.loss(jpos, tjpos)
        obs_ori = imu[:, :, 15:60].reshape(B, T, 5, 3, 3)
        present = obs_ori.abs().flatten(3).sum(-1) > 0
        pred_so = grot[:, self.imu_joints].view(B, T, 5, 3, 3)
        m = present.unsqueeze(-1).unsqueeze(-1)
        if m.any():
            loss = loss + self.ik_w * self.loss(pred_so * m, obs_ori * m)
        return loss

    def _sample_t(self, B, dev):
        if self.t_dist == "uniform":
            return torch.rand(B, device=dev)
        return torch.sigmoid(torch.randn(B, device=dev) * self.t_s + self.t_m)       # logit-normal

    # ---- the three objectives (subclasses override _train_loss / _sample_step) ------------------------------
    def _train_loss(self, batch):
        imu, target, lens, _ = batch
        x1 = target[:, :, :self.n_pose_output]
        B, T, _ = x1.shape
        pad = self._pad_mask(lens, T, x1.device)
        if not self.generative:
            pred = self.net(imu, torch.zeros(B, device=x1.device), pad)
            return self._pose_loss(imu, pred, x1, B, T)
        if self.cond_drop > 0:
            keep = (torch.rand(B, device=x1.device) >= self.cond_drop).float()[:, None, None]
            imu = imu * keep
        if self.inpaint:
            # global-rotation target; the sensed joints' measured orientations are clean known columns (except for a
            # GEN_INPAINT_NOISE_P share of windows); the velocity loss is taken on the unknown columns, the geometric loss on
            # the implied clean pose with the known columns written back, converted to local rotations
            dev = x1.device
            y1 = self._norm(self._to_global(x1))
            kmask, kvals = self._known(imu)
            keep_clean = (torch.rand(B, device=dev) >= self.inpaint_noise_p)[:, None, None] & kmask
            z = torch.randn_like(y1)
            t = self._sample_t(B, dev)
            y_t, target_v = self._path(y1, z, t)
            y_t = torch.where(keep_clean, self._norm(kvals), y_t)
            pred = self.net(torch.cat([y_t, imu], dim=-1), t, pad)
            valid = (~pad).unsqueeze(-1) & ~keep_clean
            loss = (((pred - target_v) ** 2) * valid).sum() / valid.sum().clamp(min=1)
            if self.geom_w > 0:
                x1g = self._x1_from(y_t, pred, t)
                x1g = torch.where(keep_clean, self._norm(kvals), x1g)
                x1_hat = self._to_local(self._denorm(x1g)).masked_fill(pad.unsqueeze(-1), 0.0)
                loss = loss + self.geom_w * self._pose_loss(imu, x1_hat, x1, B, T)
            return loss
        y1 = self._norm(x1)
        z = torch.randn_like(y1)
        t = self._sample_t(B, x1.device)
        if self.hist:
            # per-frame time levels: a random prefix of k frames is the pose history (clean, t = 1, or slightly noised),
            # the remaining frames share the sampled t; the loss is taken on the non-clean frames only
            dev = x1.device
            use_hist = torch.rand(B, device=dev) < self.hist_p
            k = torch.randint(0, max(T - self.min_new, 0) + 1, (B,), device=dev)
            ctx_t = torch.where(torch.rand(B, device=dev) < self.ctx_noise_p, 0.7 + 0.3 * torch.rand(B, device=dev), torch.ones(B, device=dev))
            is_ctx = (torch.arange(T, device=dev)[None, :] < k[:, None]) & use_hist[:, None]
            t_frames = torch.where(is_ctx, ctx_t[:, None], t[:, None])                 # (B,T)
            y_t, target_v = self._path(y1, z, t_frames)
            pred = self.net(torch.cat([y_t, imu], dim=-1), t_frames, pad)
            valid = (~pad) & (t_frames < 1.0 - 1e-6)
            loss = (((pred - target_v) ** 2) * valid.unsqueeze(-1)).sum() / (valid.sum() * pred.shape[-1]).clamp(min=1)
            t_for_x1 = t_frames
        else:
            y_t, target_v = self._path(y1, z, t)
            pred = self.net(torch.cat([y_t, imu], dim=-1), t, pad)
            target_v = target_v.masked_fill(pad.unsqueeze(-1), 0.0)
            loss = self.loss(pred, target_v)
            t_for_x1 = t
        if self.geom_w > 0:
            x1_hat = self._denorm(self._x1_from(y_t, pred, t_for_x1)).masked_fill(pad.unsqueeze(-1), 0.0)
            loss = loss + self.geom_w * self._pose_loss(imu, x1_hat, x1, B, T)
        return loss

    @staticmethod
    def _tt(t):
        """Broadcast a time tensor to (B,1,1) (one t per window) or (B,T,1) (one t per frame)."""
        return t[:, None, None] if t.dim() == 1 else t[:, :, None]

    def _path(self, y1, z, t):          # -> (y_t, training target)
        raise NotImplementedError

    def _x1_from(self, y_t, pred, t):   # x1 estimate implied by the prediction at time t
        raise NotImplementedError

    def _sample_step(self, y, imu, t0, t1, pad, w):
        raise NotImplementedError

    def _predict(self, y, imu, t, pad, w):
        """Network prediction with optional classifier-free guidance (w > 1 mixes in an unconditional prediction).
        t: a float (one time for the whole window) or a (B,T) tensor of per-frame times."""
        B = y.shape[0]
        tt = t if torch.is_tensor(t) else torch.full((B,), float(t), device=y.device)
        p = self.net(torch.cat([y, imu], dim=-1), tt, pad)
        if w != 1.0:
            pu = self.net(torch.cat([y, torch.zeros_like(imu)], dim=-1), tt, pad)
            p = pu + w * (p - pu)
        return p

    @torch.no_grad()
    def _sample(self, imu, lens, steps, gen, w):
        B, T, _ = imu.shape
        dev = imu.device
        pad = self._pad_mask(lens, T, dev)
        y = torch.randn(B, T, self.n_pose_output, generator=gen, device=dev)
        ts = torch.linspace(0.0, 1.0, steps + 1).tolist()
        if self.inpaint:
            kmask, kvals = self._known(imu)
            yk = self._norm(kvals)
            for i in range(steps):
                y = torch.where(kmask, yk, y)                      # known pixels re-imposed before every step
                y = self._sample_step(y, imu, ts[i], ts[i + 1], pad, w)
            if self.inpaint_out:
                y = torch.where(kmask, yk, y)
            return self._to_local(self._denorm(y)).masked_fill(pad.unsqueeze(-1), 0.0)
        for i in range(steps):
            y = self._sample_step(y, imu, ts[i], ts[i + 1], pad, w)
        return self._denorm(y).masked_fill(pad.unsqueeze(-1), 0.0)

    @torch.no_grad()
    def _sample_ar(self, imu, lens, steps, gen, w):
        """Autoregressive sampling of a whole sequence (the DiffusionPoser way): the first window is sampled from noise as
        usual; after that, each step keeps the last W - C generated frames as clean context (t = 1) and denoises only the
        next C frames from noise, with the IMU of the whole window as condition. Padded frames are not supported here
        (evaluation feeds one unpadded sequence at a time)."""
        B, T, _ = imu.shape
        dev = imu.device
        W = _env_i("TF_EVAL_WINDOW", 125)
        C = max(1, min(self.chunk, W - 1))
        n = self.n_pose_output
        ts = torch.linspace(0.0, 1.0, steps + 1).tolist()
        out = torch.zeros(B, T, n, device=dev)
        first = min(W, T)
        pad0 = torch.zeros(B, first, dtype=torch.bool, device=dev)
        y = torch.randn(B, first, n, generator=gen, device=dev)
        for i in range(steps):
            y = self._sample_step(y, imu[:, :first], ts[i], ts[i + 1], pad0, w)
        out[:, :first] = y
        pos = first
        # GEN_CTX_T < 1: instead of fixing the generated history as clean context, re-noise it to time t_c and refine it
        # together with the new frames (SDEdit-style); the context then steps from t_c to 1 while the new frames go 0 -> 1
        t_c = _env_f("GEN_CTX_T", 1.0)
        while pos < T:
            s = max(0, pos - (W - C)); e = min(pos + C, T)
            L_ctx, L_new = pos - s, e - pos
            ctx = out[:, s:pos]
            if t_c < 1.0:
                ctx = (1 - t_c) * torch.randn(B, L_ctx, n, generator=gen, device=dev) + t_c * ctx
            y_new = torch.randn(B, L_new, n, generator=gen, device=dev)
            pad_w = torch.zeros(B, e - s, dtype=torch.bool, device=dev)
            t_frames = torch.ones(B, e - s, device=dev)
            for i in range(steps):
                t_frames[:, L_ctx:] = ts[i]
                tc0, tc1 = t_c + (1 - t_c) * ts[i], t_c + (1 - t_c) * ts[i + 1]
                t_frames[:, :L_ctx] = tc0
                pred = self._predict(torch.cat([ctx, y_new], dim=1), imu[:, s:e], t_frames, pad_w, w)
                y_new = self._step_from_pred(y_new, pred[:, L_ctx:], ts[i], ts[i + 1])
                if t_c < 1.0:
                    ctx = self._step_from_pred(ctx, pred[:, :L_ctx], tc0, tc1)
            if t_c < 1.0:
                out[:, s:pos] = ctx                       # the refined context replaces the earlier estimate of those frames
            out[:, pos:e] = y_new
            pos = e
        return self._denorm(out)

    def _sample_avg(self, imu, lens, steps=None, K=None, seed=None, w=None, ar=False):
        steps = _env_i("GEN_STEPS", 8) if steps is None else steps
        K = _env_i("GEN_SAMPLE_AVG", 1) if K is None else K
        seed = _env_i("GEN_SEED", 0) if seed is None else seed
        w = _env_f("GEN_CFG", 1.0) if w is None else w
        acc = None
        for k in range(K):
            gen = torch.Generator(device=imu.device); gen.manual_seed(seed + k)
            s = self._sample_ar(imu, lens, steps, gen, w) if ar else self._sample(imu, lens, steps, gen, w)
            acc = s if acc is None else acc + s
        return acc / K

    # ---- Lightning API ----------------------------------------------------------------------------------------
    def forward(self, imu_inputs, imu_lens):
        if not self.generative:
            f = lambda c, l: self.net(c, torch.zeros(c.shape[0], device=c.device), self._pad_mask(l, c.shape[1], c.device))
        else:
            if not self.training and _env_i("GEN_AR", 0):
                return self._sample_avg(imu_inputs, imu_lens, ar=True)     # whole sequence, chunk by chunk
            f = lambda c, l: self._sample_avg(c, l)
        T = imu_inputs.size(1)
        W = _env_i("TF_EVAL_WINDOW", 125)
        if self.training or T <= W:
            return f(imu_inputs, imu_lens)
        return windowed_inference(f, imu_inputs, imu_lens, W)

    def training_step(self, batch, batch_idx):
        loss = self._train_loss(batch)
        if self.moe_e > 0:
            loss = loss + self.moe_aux_w * self.net.moe_aux()
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        self.training_step_outputs.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # POSE error of a deterministic estimate (fixed noise, GEN_VAL_STEPS steps) with the regression loss, so the
        # best-val checkpoint is the best POSE model, comparable with AvatarPoser's validation_step_loss
        imu, target, lens, _ = batch
        x1 = target[:, :, :self.n_pose_output]
        B, T, _ = x1.shape
        if self.generative:
            pred = self._sample_avg(imu, lens, steps=self.val_steps, K=1, seed=10_000 + batch_idx, w=1.0)
        else:
            pred = self.net(imu, torch.zeros(B, device=x1.device), self._pad_mask(lens, T, x1.device))
        loss = self._pose_loss(imu, pred, x1, B, T)
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

    configure_optimizers = AvatarPoserModel.configure_optimizers


class RegressIMUPoser(_GenPoser):
    """Deterministic regression on the DiT-style backbone: the architecture control for the two generative models."""
    KIND = 0


class FlowIMUPoser(_GenPoser):
    """Rectified flow: y_t = (1-t) z + t y1, v = y1 - z. Sampling integrates dy/dt = v from t=0 (noise) to t=1 (data)."""
    KIND = 1

    def _path(self, y1, z, t):
        tt = self._tt(t)
        return (1 - tt) * z + tt * y1, y1 - z

    def _x1_from(self, y_t, v, t):
        return y_t + (1 - self._tt(t)) * v

    def _step_from_pred(self, y, v, t0, t1):
        return y + (t1 - t0) * v                 # Euler step of dy/dt = v

    def _sample_step(self, y, imu, t0, t1, pad, w):
        h = t1 - t0
        v = self._predict(y, imu, t0, pad, w)
        if os.environ.get("GEN_SOLVER", "euler").lower() == "midpoint" and t1 < 1.0 + 1e-9 and h > 0:
            y_mid = y + 0.5 * h * v
            v = self._predict(y_mid, imu, t0 + 0.5 * h, pad, w)
        return self._step_from_pred(y, v, t0, t1)


class DiffusionV2IMUPoser(_GenPoser):
    """VP diffusion, cosine schedule alpha_t = sin(pi t / 2), sigma_t = cos(pi t / 2) (t = 1 clean), v-prediction, DDIM."""
    KIND = 2

    @staticmethod
    def _ab(t):
        return math.sin(0.5 * math.pi * t), math.cos(0.5 * math.pi * t)

    def _path(self, y1, z, t):
        a = self._tt(torch.sin(0.5 * math.pi * t)); s = self._tt(torch.cos(0.5 * math.pi * t))
        return a * y1 + s * z, a * z - s * y1

    def _x1_from(self, y_t, v, t):
        a = self._tt(torch.sin(0.5 * math.pi * t)); s = self._tt(torch.cos(0.5 * math.pi * t))
        return a * y_t - s * v

    def _step_from_pred(self, y, v, t0, t1):
        a0, s0 = self._ab(t0); a1, s1 = self._ab(t1)
        y1_hat = a0 * y - s0 * v
        z_hat = s0 * y + a0 * v
        return a1 * y1_hat + s1 * z_hat          # deterministic DDIM update (eta = 0)

    def _sample_step(self, y, imu, t0, t1, pad, w):
        a0, s0 = self._ab(t0); a1, s1 = self._ab(t1)
        v = self._predict(y, imu, t0, pad, w)
        y1_hat = a0 * y - s0 * v
        z_hat = s0 * y + a0 * v
        return a1 * y1_hat + s1 * z_hat          # deterministic DDIM update (eta = 0)
