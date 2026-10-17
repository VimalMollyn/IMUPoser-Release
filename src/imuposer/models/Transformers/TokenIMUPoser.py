r"""
TokenIMUPoser: IMU -> pose TOKENS, the TokenHMR idea (Dwivedi et al., CVPR 2024) adapted to sparse-IMU tracking.

A frozen TokenHMR-style pose tokenizer (VQ-VAE over the 21 SMPL body joints, see imuposer/thirdparty/tokenhmr; either the
official checkpoint or one trained on our poses with train_pose_tokenizer.py) turns each frame's body pose into N discrete
tokens from a codebook of K codes. The IMU model is the shared DiT-style backbone (GenIMUPoser._GenNet, regression mode) with a
factorised token head: per frame, N x K logits. Training: cross-entropy against the tokenizer's encoding of the ground-truth pose
+ the geometric loss on the SOFT decode (softmax over the codebook per token -> weighted code -> decoder -> body pose), as in
TokenHMR's logit-based decoding. Inference: soft decode (a continuous output, so it ensembles like every other model).

The decoded pose covers the 21 body joints; the root and the two hands are set to identity (the evaluator sets those joints to
identity before scoring, so nothing is lost on the metric; an application would need a separate root head). Output is the usual
(B, T, 144) local r6d in this codebase's convention, so offline_fit / eval_dip work unchanged.

Env: TOKENIZER_CKPT (path, required), TF_* (backbone), TOK_HEAD_DIM (factorised head width, 64), TOK_CE_W (1.0), GEN_GEOM_W (1.0),
     TOK_TAU (softmax temperature for the soft decode at eval, 1.0).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_r6d
from imuposer.config import Config
from .GenIMUPoser import _GenNet, _TimeEmb, _env_i, _env_f
from .TransformerIMUPoser import windowed_inference
from .AvatarPoserModel import AvatarPoserModel
from imuposer.thirdparty.tokenhmr.vanilla_pose_vqvae import EncodeTokens, DecodeTokens
from imuposer.thirdparty.tokenhmr.rotation_utils import rotation_6d_to_matrix, matrix_to_rotation_6d

BODY = list(range(1, 22))


class TokenIMUPoser(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_cond = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        ck = os.environ.get("TOKENIZER_CKPT")
        assert ck and os.path.exists(ck), f"TOKENIZER_CKPT must point to a TokenHMR-style tokenizer checkpoint (got {ck!r})"
        self.tok_enc = EncodeTokens(ck); self.tok_dec = DecodeTokens(ck, mesh_inference=False)
        for p in list(self.tok_enc.parameters()) + list(self.tok_dec.parameters()):
            p.requires_grad_(False)
        self.tok_enc.eval(); self.tok_dec.eval()
        self.n_code, self.code_dim = self.tok_dec.quantizer.nb_code, self.tok_dec.quantizer.code_dim
        hd = _env_i("TOK_HEAD_DIM", 64)
        self.net = _GenNet(self.n_cond, 1,   # the backbone's own head is unused; the token head reads its hidden state
                           d_model=_env_i("TF_DMODEL", 256), nhead=_env_i("TF_HEADS", 8), n_layers=_env_i("TF_LAYERS", 4),
                           dim_ff=_env_i("TF_FF", 1024), dropout=_env_f("TF_DROPOUT", 0.1), use_bilstm=bool(os.environ.get("TF_BILSTM")),
                           n_exp=_env_i("GEN_MOE_E", 0), k=_env_i("GEN_MOE_K", 2))
        self.net.out = nn.Identity()
        d = _env_i("TF_DMODEL", 256)
        # the token count is fixed by the tokenizer: probe it once with a dummy pose
        with torch.no_grad():
            self.n_tok = int(self.tok_enc(matrix_to_rotation_6d(torch.eye(3).expand(1, 21, 3, 3).clone())).numel())   # flat (frames*tokens,) output
        self.head_in = nn.Linear(d, self.n_tok * hd)
        self.head_out = nn.Linear(hd, self.n_code)
        self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.loss = nn.MSELoss()
        self.ce_w, self.geom_w, self.tau = _env_f("TOK_CE_W", 1.0), _env_f("GEN_GEOM_W", 1.0), _env_f("TOK_TAU", 1.0)
        self.tok_frames = _env_i("TOK_FRAMES", 32)                     # frames per window used for the token losses
        self.geom_frames = _env_i("TOK_GEOM_FRAMES", 8)                # of those, frames that also get the soft-decode geometric loss
        self.lr = _env_f("TF_LR", 3e-4)
        self.register_buffer("gen_kind", torch.tensor(3))
        self.training_step_outputs, self.validation_step_outputs, self.test_step_outputs = [], [], []
        self.save_hyperparameters(ignore=["config"])
        n_par = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[tok] TokenIMUPoser: {n_par/1e6:.2f} M trainable params, tokenizer {ck} ({self.n_tok} tokens x {self.n_code} codes, dim {self.code_dim}), "
              f"head dim {hd}, ce_w {self.ce_w} geom_w {self.geom_w}", flush=True)

    # ---- helpers ---------------------------------------------------------------------------------------------
    def on_fit_start(self):
        self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def _pad_mask(self, lens, T, dev):
        return torch.arange(T, device=dev)[None, :] >= torch.as_tensor(lens, device=dev)[:, None]

    def _logits(self, imu, pad):                                       # (B,T,n_tok,n_code)
        B, T, _ = imu.shape
        h = self.net(imu, torch.zeros(B, device=imu.device), pad)     # (B,T,d) (the Identity head returns the hidden state)
        z = self.head_in(h).view(B, T, self.n_tok, -1)
        return self.head_out(F.gelu(z))

    def _decode(self, logits, tau=1.0):
        """Soft decode: softmax over the codebook per token -> body pose as OUR 24-joint local r6d (root, hands identity)."""
        B, T, N, K = logits.shape
        probs = (logits / tau).softmax(-1).reshape(B * T, N, K)
        body6 = self.tok_dec(probs)                                    # (B*T, 21, 6) TokenHMR 6-D (rows)
        R = rotation_6d_to_matrix(body6.reshape(-1, 6)).view(B * T, 21, 3, 3)
        full = torch.eye(3, device=R.device).expand(B * T, 24, 3, 3).clone()
        full[:, 1:22] = R
        return rotation_matrix_to_r6d(full).view(B, T, 24 * 6)

    @torch.no_grad()
    def _gt_tokens(self, x1):                                          # x1: (B,T,144) our local r6d -> (B*T, n_tok) code ids
        B, T, _ = x1.shape
        R = r6d_to_rotation_matrix(x1.reshape(-1, 24, 6)).view(B * T, 24, 3, 3)[:, 1:22]
        return self.tok_enc(matrix_to_rotation_6d(R)).view(B * T, -1)     # the encoder takes TokenHMR's 6-D (rows) form

    def _geom_loss(self, pred, target):
        """r6d MSE + FK joint positions, both with root / hands identity (the metric's view); no IK term (root unknown)."""
        B, T, _ = target.shape
        Rt = r6d_to_rotation_matrix(target.reshape(-1, 24, 6)).view(B * T, 24, 3, 3).clone()
        Rt[:, 0] = torch.eye(3, device=Rt.device); Rt[:, 22:] = torch.eye(3, device=Rt.device)
        tg = rotation_matrix_to_r6d(Rt).view(B, T, 144)
        loss = self.loss(pred, tg)
        jp = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred).view(-1, 216))[1]
        jg = self.bodymodel.forward_kinematics(pose=Rt.view(-1, 216))[1]
        return loss + self.loss(jp, jg)

    def _subset_logits(self, imu, x1, pad, return_idx=False):
        """160 x 2048 logits per frame are too big to keep for a whole batch of windows, so the token head and the losses run on
        TOK_FRAMES random valid frames per window; the backbone still attends over the full window."""
        B, T, _ = x1.shape
        h = self.net(imu, torch.zeros(B, device=imu.device), pad)     # (B,T,d)
        nf = min(self.tok_frames, T)
        idx = torch.rand(B, T, device=imu.device).masked_fill(pad, -1.0).topk(nf, dim=1).indices      # (B,nf)
        sel_valid = (~pad).gather(1, idx)
        hs = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        x1s = x1.gather(1, idx.unsqueeze(-1).expand(-1, -1, x1.shape[-1]))
        logits = self.head_out(F.gelu(self.head_in(hs).view(B, nf, self.n_tok, -1)))                 # (B,nf,n_tok,K)
        return (logits, x1s, sel_valid, idx) if return_idx else (logits, x1s, sel_valid)

    def _train_loss(self, batch):
        imu, target, lens, _ = batch
        x1 = target[:, :, :self.n_pose_output]
        B, T, _ = x1.shape
        pad = self._pad_mask(lens, T, x1.device)
        logits, x1s, sel_valid, idx = self._subset_logits(imu, x1, pad, return_idx=True)
        nf = logits.shape[1]
        if target.shape[-1] >= self.n_pose_output + self.n_tok:        # precomputed tokens (AUX_TARGET=tokens): no encoder pass
            gt = target[:, :, self.n_pose_output:self.n_pose_output + self.n_tok].gather(1, idx.unsqueeze(-1).expand(-1, -1, self.n_tok)).reshape(B * nf, self.n_tok).long()
        else:
            gt = self._gt_tokens(x1s)                                  # (B*nf, n_tok)
        m = sel_valid.reshape(-1)
        ce = F.cross_entropy(logits.reshape(B * nf, self.n_tok, self.n_code)[m].reshape(-1, self.n_code), gt[m].reshape(-1))
        loss = self.ce_w * ce
        if self.geom_w > 0:
            # the soft decode is the second-biggest cost: take it on TOK_GEOM_FRAMES of the sampled frames
            ng = min(self.geom_frames, nf)
            inv = (~sel_valid[:, :ng]).unsqueeze(-1)
            pred = self._decode(logits[:, :ng]).masked_fill(inv, 0.0)
            loss = loss + self.geom_w * self._geom_loss(pred, x1s[:, :ng].masked_fill(inv, 0.0))
        return loss, ce

    # ---- Lightning API ----------------------------------------------------------------------------------------
    def forward(self, imu_inputs, imu_lens):
        f = lambda c, l: self._decode(self._logits(c, self._pad_mask(l, c.shape[1], c.device)), tau=self.tau)
        T = imu_inputs.size(1)
        W = _env_i("TF_EVAL_WINDOW", 125)
        if self.training or T <= W:
            return f(imu_inputs, imu_lens)
        return windowed_inference(f, imu_inputs, imu_lens, W)

    def training_step(self, batch, batch_idx):
        loss, ce = self._train_loss(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        self.log("token_ce", ce.item(), batch_size=self.batch_size)
        self.training_step_outputs.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imu, target, lens, _ = batch
        x1 = target[:, :, :self.n_pose_output]
        B, T, _ = x1.shape
        pad = self._pad_mask(lens, T, x1.device)
        logits, x1s, sel_valid = self._subset_logits(imu, x1, pad)
        inv = (~sel_valid).unsqueeze(-1)
        pred = self._decode(logits).masked_fill(inv, 0.0)
        loss = self._geom_loss(pred, x1s.masked_fill(inv, 0.0))       # pose error of the decoded pose, comparable across epochs
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
        # same AdamW / schedule as the other models, over the trainable parameters only (tokenizer frozen)
        params = [p for p in self.parameters() if p.requires_grad]
        beta2 = float(os.environ.get("ADAM_BETA2", "0.999")); eps = float(os.environ.get("ADAM_EPS", "1e-8"))
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=float(os.environ.get("TF_WD", "1e-4")), betas=(0.9, beta2), eps=eps)
        warm = int(os.environ.get("LR_WARMUP_STEPS", "0") or 0)
        if warm == 0:
            return opt
        total = int(self.trainer.estimated_stepping_batches)
        cos = os.environ.get("LR_SCHED", "").lower() == "cosine"; min_frac = float(os.environ.get("LR_MIN_FRAC", "0.01"))
        import math
        def lam(step):
            w = min(1.0, (step + 1) / warm)
            if not cos: return w
            p = min(1.0, max(0, step - warm) / max(1, total - warm))
            return w * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lam), "interval": "step"}}


class FlowTokenIMUPoser(pl.LightningModule):
    r"""Flow matching in the frozen TokenHMR code-embedding space (the "flow + TokenHMR" idea).

    The plain TokenIMUPoser predicts discrete codes with cross-entropy; here we instead FLOW-MATCH the CONTINUOUS
    code-embedding field. The transformer backbone (GenIMUPoser._GenNet) encodes IMU into per-frame features h ONCE; a
    lightweight per-token flow head v(y_t, t | h, token-position) generates the (n_tok, code_dim) embedding field by
    integrating a rectified-flow ODE from Gaussian noise. The field is decoded to body pose by the frozen tokenizer decoder
    (decode_feat -> decoder -> 21 joints), then written as this codebase's (B,T,144) 24-joint local r6d with root and the two
    hands set to identity (the evaluator ignores those). Continuous output, so it ensembles like the rest.

    Target = normalized codebook embeddings of the ground-truth codes (precomputed, AUX_TARGET=tokens). Losses: rectified-flow
    MSE (v = y1 - z) on TOK_FRAMES subsampled frames, plus the geometric loss (FK joints + r6d) on the decoded x1 estimate of
    TOK_GEOM_FRAMES of them. Inference: Euler (optionally midpoint) over GEN_STEPS, then decode. gen_kind = 4.
    Env: TOKENIZER_CKPT, TF_* (backbone), TOK_POS_DIM (per-token id width, 64), TOK_HEAD_DIM (flow-head width, 512),
    GEN_GEOM_W (1.0), GEN_STEPS (8), GEN_T_M/GEN_T_S/GEN_T_DIST, GEN_SEED, GEN_SOLVER.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_cond = 12 * len(config.joints_set)
        self.n_pose_output = len(config.pred_joints_set) * (6 if config.r6d else 9)
        self.batch_size = config.batch_size
        ck = os.environ.get("TOKENIZER_CKPT")
        assert ck and os.path.exists(ck), f"TOKENIZER_CKPT must point to a TokenHMR-style tokenizer checkpoint (got {ck!r})"
        self.tok_enc = EncodeTokens(ck); self.tok_dec = DecodeTokens(ck, mesh_inference=False)
        for p in list(self.tok_enc.parameters()) + list(self.tok_dec.parameters()):
            p.requires_grad_(False)
        self.tok_enc.eval(); self.tok_dec.eval()
        self.n_code, self.code_dim = self.tok_dec.quantizer.nb_code, self.tok_dec.quantizer.code_dim
        d = _env_i("TF_DMODEL", 256)
        self.net = _GenNet(self.n_cond, 1,
                           d_model=d, nhead=_env_i("TF_HEADS", 8), n_layers=_env_i("TF_LAYERS", 4),
                           dim_ff=_env_i("TF_FF", 1024), dropout=_env_f("TF_DROPOUT", 0.1), use_bilstm=bool(os.environ.get("TF_BILSTM")),
                           n_exp=_env_i("GEN_MOE_E", 0), k=_env_i("GEN_MOE_K", 2))
        self.net.out = nn.Identity()
        with torch.no_grad():
            self.n_tok = int(self.tok_enc(matrix_to_rotation_6d(torch.eye(3).expand(1, 21, 3, 3).clone())).numel())
        d_tok = _env_i("TOK_POS_DIM", 64); hid = _env_i("TOK_HEAD_DIM", 512)
        self.tok_pos = nn.Parameter(torch.randn(self.n_tok, d_tok) * 0.02)
        self.h_proj = nn.Linear(d, d)
        self.t_emb = _TimeEmb(d)
        self.head = nn.Sequential(nn.Linear(d + d_tok + self.code_dim + d, hid), nn.SiLU(),
                                  nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, self.code_dim))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)   # start at v=0 (identity flow)
        self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)
        self.loss = nn.MSELoss()
        self.geom_w = _env_f("GEN_GEOM_W", 1.0)
        self.tok_frames = _env_i("TOK_FRAMES", 32)
        self.geom_frames = _env_i("TOK_GEOM_FRAMES", 8)
        self.steps = _env_i("GEN_STEPS", 8)
        self.t_dist = os.environ.get("GEN_T_DIST", "logitnormal").lower()
        self.t_m, self.t_s = _env_f("GEN_T_M", 0.0), _env_f("GEN_T_S", 1.0)
        self.lr = _env_f("TF_LR", 3e-4)
        cb = self.tok_dec.quantizer.codebook.float()                   # (n_code, code_dim), frozen
        self.register_buffer("emb_mean", cb.mean(0))
        self.register_buffer("emb_std", cb.std(0).clamp_min(1e-4))
        self.register_buffer("gen_kind", torch.tensor(4))
        self.training_step_outputs, self.validation_step_outputs, self.test_step_outputs = [], [], []
        self.save_hyperparameters(ignore=["config"])
        n_par = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[flowtok] FlowTokenIMUPoser: {n_par/1e6:.2f} M trainable params, tokenizer {ck} "
              f"({self.n_tok} tokens x {self.n_code} codes, dim {self.code_dim}), head dim {hid}, "
              f"geom_w {self.geom_w} steps {self.steps} t_m {self.t_m}", flush=True)

    # ---- helpers ---------------------------------------------------------------------------------------------
    def on_fit_start(self):
        self.bodymodel = ParametricModel(self.config.og_smpl_model_path, device=self.device)

    def _pad_mask(self, lens, T, dev):
        return torch.arange(T, device=dev)[None, :] >= torch.as_tensor(lens, device=dev)[:, None]

    def _codebook(self):
        return self.tok_dec.quantizer.codebook.to(self.emb_mean.dtype)

    def _norm(self, e):   return (e - self.emb_mean) / self.emb_std
    def _denorm(self, y): return y * self.emb_std + self.emb_mean

    def _sample_t(self, B, dev):
        if self.t_dist == "uniform":
            return torch.rand(B, device=dev)
        return torch.sigmoid(torch.randn(B, device=dev) * self.t_s + self.t_m)       # logit-normal (SD3)

    @torch.no_grad()
    def _gt_codes(self, target):                                       # (B,T,n_tok) long
        B, T, D = target.shape
        if D >= self.n_pose_output + self.n_tok:                       # precomputed tokens (AUX_TARGET=tokens)
            return target[:, :, self.n_pose_output:self.n_pose_output + self.n_tok].long()
        x1 = target[:, :, :self.n_pose_output]
        R = r6d_to_rotation_matrix(x1.reshape(-1, 24, 6)).view(B * T, 24, 3, 3)[:, 1:22]
        return self.tok_enc(matrix_to_rotation_6d(R)).view(B, T, self.n_tok).long()

    def _gt_emb(self, codes):                                          # (B,T,n_tok) -> normalized (B,T,n_tok,code_dim)
        return self._norm(F.embedding(codes, self._codebook()))

    def _flow(self, h, y_t, t):                                        # h:(B,M,d) y_t:(B,M,N,C) t:(B,) -> v:(B,M,N,C)
        B, M, N, C = y_t.shape
        hc = self.h_proj(h).unsqueeze(2).expand(B, M, N, -1)
        tp = self.tok_pos.view(1, 1, N, -1).expand(B, M, N, -1)
        te = self.t_emb(t)[:, None, None, :].expand(B, M, N, -1)       # (B,M,N,d)
        return self.head(torch.cat([hc, tp, y_t, te], dim=-1))

    def _decode_emb(self, e):                                          # e:(B,M,N,C) denormalized -> (B,M,144) r6d
        B, M, N, C = e.shape
        feat = e.reshape(B * M, N, C).permute(0, 2, 1)                 # (B*M, C, N)
        body6 = self.tok_dec.decoder(feat)["pred_pose_body_6d"]       # (B*M, 21, 6)
        R = rotation_6d_to_matrix(body6.reshape(-1, 6)).view(B * M, 21, 3, 3)
        full = torch.eye(3, device=R.device).expand(B * M, 24, 3, 3).clone()
        full[:, 1:22] = R
        return rotation_matrix_to_r6d(full).view(B, M, 24 * 6)

    def _geom_loss(self, pred, target):                               # both (B,M,144); root/hands set to identity
        B, M, _ = target.shape
        Rt = r6d_to_rotation_matrix(target.reshape(-1, 24, 6)).view(B * M, 24, 3, 3).clone()
        Rt[:, 0] = torch.eye(3, device=Rt.device); Rt[:, 22:] = torch.eye(3, device=Rt.device)
        tg = rotation_matrix_to_r6d(Rt).view(B, M, 144)
        jp = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred).view(-1, 216))[1]
        jg = self.bodymodel.forward_kinematics(pose=Rt.view(-1, 216))[1]
        return self.loss(pred, tg) + self.loss(jp, jg)

    def _backbone(self, imu, pad):
        B = imu.shape[0]
        return self.net(imu, torch.zeros(B, device=imu.device), pad)   # (B,T,d); Identity head returns hidden state

    # ---- training --------------------------------------------------------------------------------------------
    def _train_loss(self, batch):
        imu, target, lens, _ = batch
        B, T, _ = imu.shape
        pad = self._pad_mask(lens, T, imu.device)
        y1_full = self._gt_emb(self._gt_codes(target))                # (B,T,N,C)
        h = self._backbone(imu, pad)
        nf = min(self.tok_frames, T)
        idx = torch.rand(B, T, device=imu.device).masked_fill(pad, -1.0).topk(nf, dim=1).indices    # (B,nf)
        sel = (~pad).gather(1, idx)                                    # (B,nf) valid
        hs = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        y1 = y1_full.gather(1, idx[:, :, None, None].expand(-1, -1, self.n_tok, self.code_dim))
        z = torch.randn_like(y1)
        t = self._sample_t(B, imu.device)
        tt = t[:, None, None, None]
        y_t = (1 - tt) * z + tt * y1
        v = self._flow(hs, y_t, t)
        m = sel.reshape(-1)
        fl = (((v - (y1 - z)) ** 2).mean(dim=-1).reshape(-1, self.n_tok)[m]).mean()
        loss = fl
        if self.geom_w > 0:
            ng = min(self.geom_frames, nf)
            x1 = y_t[:, :ng] + (1 - tt) * v[:, :ng]                    # rectified-flow x1 estimate
            pred = self._decode_emb(self._denorm(x1))
            gt_pose = target[:, :, :144].gather(1, idx[:, :ng].unsqueeze(-1).expand(-1, -1, 144))
            inv = (~sel[:, :ng]).unsqueeze(-1)
            loss = loss + self.geom_w * self._geom_loss(pred.masked_fill(inv, 0.0), gt_pose.masked_fill(inv, 0.0))
        return loss, fl

    @torch.no_grad()
    def _sample_frames(self, h, gen=None):                            # h:(B,M,d) -> embeddings (B,M,N,C)
        B, M, _ = h.shape
        y = torch.randn(B, M, self.n_tok, self.code_dim, device=h.device, generator=gen)
        midp = os.environ.get("GEN_SOLVER", "euler").lower() == "midpoint"
        for i in range(self.steps):
            t0, t1 = i / self.steps, (i + 1) / self.steps
            v = self._flow(h, y, torch.full((B,), t0, device=h.device))
            if midp and t1 < 1.0 + 1e-9:
                y_mid = y + 0.5 * (t1 - t0) * v
                v = self._flow(h, y_mid, torch.full((B,), t0 + 0.5 * (t1 - t0), device=h.device))
            y = y + (t1 - t0) * v
        return y

    def forward(self, imu_inputs, imu_lens):
        W = _env_i("TF_EVAL_WINDOW", 125)
        def f(c, l):
            pad = self._pad_mask(l, c.shape[1], c.device)
            g = None
            if os.environ.get("GEN_SEED"):
                g = torch.Generator(device=c.device); g.manual_seed(int(os.environ["GEN_SEED"]))
            y = self._sample_frames(self._backbone(c, pad), gen=g)
            return self._decode_emb(self._denorm(y)).masked_fill(pad.unsqueeze(-1), 0.0)
        T = imu_inputs.size(1)
        if self.training or T <= W:
            return f(imu_inputs, imu_lens)
        return windowed_inference(f, imu_inputs, imu_lens, W)

    def training_step(self, batch, batch_idx):
        loss, fl = self._train_loss(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        self.log("flow_mse", fl.item(), batch_size=self.batch_size)
        self.training_step_outputs.append(loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imu, target, lens, _ = batch
        B, T, _ = imu.shape
        pad = self._pad_mask(lens, T, imu.device)
        h = self._backbone(imu, pad)
        nf = min(self.tok_frames, T)
        idx = torch.rand(B, T, device=imu.device).masked_fill(pad, -1.0).topk(nf, dim=1).indices
        sel = (~pad).gather(1, idx)
        hs = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        y = self._sample_frames(hs)
        pred = self._decode_emb(self._denorm(y))
        gt_pose = target[:, :, :144].gather(1, idx.unsqueeze(-1).expand(-1, -1, 144))
        inv = (~sel).unsqueeze(-1)
        loss = self._geom_loss(pred.masked_fill(inv, 0.0), gt_pose.masked_fill(inv, 0.0))
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
        params = [p for p in self.parameters() if p.requires_grad]
        beta2 = float(os.environ.get("ADAM_BETA2", "0.999")); eps = float(os.environ.get("ADAM_EPS", "1e-8"))
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=float(os.environ.get("TF_WD", "1e-4")), betas=(0.9, beta2), eps=eps)
        warm = int(os.environ.get("LR_WARMUP_STEPS", "0") or 0)
        if warm == 0:
            return opt
        total = int(self.trainer.estimated_stepping_batches)
        cos = os.environ.get("LR_SCHED", "").lower() == "cosine"; min_frac = float(os.environ.get("LR_MIN_FRAC", "0.01"))
        import math
        def lam(step):
            w = min(1.0, (step + 1) / warm)
            if not cos: return w
            p = min(1.0, max(0, step - warm) / max(1, total - warm))
            return w * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lam), "interval": "step"}}
