# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
# Variable-length sequence batches fragment the CUDA caching allocator's reserved pool;
# expandable segments keep reserved memory bounded. Set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import lightning.pytorch as pl

# The streaming loader memory-maps 5 files per dataset and the large arms have 400+ datasets, so every worker holds
# 2000+ descriptors: raise the soft open-files limit to the hard limit and share tensors through the file system
# instead of descriptors (fig2's default soft limit of 1024 crashed the loader with "Too many open files").
try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _soft < _hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
        print(f"[fd] open-files limit raised {_soft} -> {_hard}", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[fd] could not raise the open-files limit: {_e}", flush=True)
torch.multiprocessing.set_sharing_strategy("file_system")
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything

from pathlib import Path

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule, get_split_files
from imuposer.utils import get_parser

# set the random seed (override with SEED env for seed-replication / variance estimates)
seed_everything(int(os.environ.get("SEED", "42")), workers=True)

parser = get_parser()
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment

# GPUs to train on, e.g. GPUS="0,1" for 2-GPU DDP. Default single GPU.
gpus = [int(g) for g in os.environ.get("GPUS", "0").split(",") if g != ""]

# %%
# MODEL selects the architecture: GlobalModelIMUPoser (default), ReconIMUPoser
# (reconstruct full 5-IMU as an aux stage), or StagedIMUPoser (IMU->joints->pose).
_model = os.environ.get("MODEL", "GlobalModelIMUPoser")
config = Config(experiment=f"{_experiment}_{combo_id}", model=_model,
                project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device=str(gpus[0]))

# AUX_TARGET appends an auxiliary supervision target to each sample's output:
#   "imu"   -> full clean 5-IMU (ReconIMUPoser)
#   "joint" -> root-relative 24-joint positions (StagedIMUPoser)
config.aux_target = os.environ.get("AUX_TARGET")

# Under DDP each rank uses `batch_size` and gradients are averaged across ranks,
# so split the per-GPU batch to keep the EFFECTIVE batch size constant (=256).
# BATCH_SIZE / ACCUM: micro-batch and gradient-accumulation steps (effective batch = BATCH_SIZE*ACCUM;
# keep it 256 to match every other run). Lets a big model fit next to another job on the same GPU.
config.batch_size = int(os.environ.get("BATCH_SIZE", config.batch_size)) // len(gpus)
_accum = int(os.environ.get("ACCUM", "1"))

# read the synthesized data from the external folder (override with IMUPOSER_DATA_DIR).
# The canonical dataset-level split (config.val_datasets / config.test_datasets) is
# applied automatically by get_datamodule -> get_dataset.
config.processed_imu_poser_25fps = Path(os.environ.get(
    "IMUPOSER_DATA_DIR", "/media/vimal/T7_2TB/CHI23/processed_imuposer_data")) \
    / os.environ.get("IMUPOSER_FPS_SUBDIR", "processed_imuposer_25fps")

# Pin the checkpoint dir to a known location (used by the auto-restart watchdog so
# resumes always read/write the same last.ckpt). Default is the timestamped dir.
_ckpt_dir = os.environ.get("CHECKPOINT_DIR")
if _ckpt_dir:
    config.checkpoint_path = Path(_ckpt_dir)
    config.checkpoint_path.mkdir(parents=True, exist_ok=True)

# ORIGINAL_ONLY=1 trains on the 20 original AMASS datasets only (minus val/test),
# excluding the 5 newer AMASS + Motion-X. Same val/test => controlled ablation.
config.original_train_only = bool(os.environ.get("ORIGINAL_ONLY"))

# MAX_SAMPLE_LEN: window length knob (default 300 -> 125-frame windows at 25fps). Sweep temporal context.
config.max_sample_len = int(os.environ.get("MAX_SAMPLE_LEN", config.max_sample_len))

# TRAIN_COMBO=<combo> trains a SPECIALIST on a single IMU combo (e.g. lw_rp_h)
# instead of the all-combo generalist; train + val both use only that combo.
config.train_combo = os.environ.get("TRAIN_COMBO")

train_files, val_files, test_files = get_split_files(config)
print(f"data: {config.processed_imu_poser_25fps}", flush=True)
print(f"TRAIN ({len(train_files)}): {sorted(f[:-3] for f in train_files)}", flush=True)
print(f"VAL   ({len(val_files)}): {sorted(f[:-3] for f in val_files)}", flush=True)
print(f"TEST  ({len(test_files)}): {test_files}", flush=True)

# %%
# instantiate model and data
model = get_model(config)
# optional warm-start: load weights from a previous checkpoint to continue training
_continue_from = os.environ.get("CONTINUE_FROM")
if _continue_from:
    model.load_state_dict(torch.load(_continue_from, map_location="cpu")["state_dict"])
    print(f"warm-started weights from {_continue_from}", flush=True)
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path

# %%
# To CONTINUE the same W&B run's charts after a crash/restart, set WANDB_RUN_ID to the
# existing run id (resume="allow" appends to it). Pair with RESUME_FROM=<last.ckpt> below
# so the epoch/step counter continues instead of resetting to 0.
_wandb_id = os.environ.get("WANDB_RUN_ID")
wandb_logger = WandbLogger(project=config.experiment, name=os.environ.get("WANDB_RUN_NAME"),
                           id=_wandb_id, resume=("allow" if _wandb_id else None),
                           save_dir=str(checkpoint_path))

# Train for a FIXED number of epochs (no early stopping). Override with EPOCHS env var.
max_epochs = int(os.environ.get("EPOCHS", "50"))

# Full checkpoints (not weights-only) + last.ckpt so runs can be cleanly resumed.
checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False,
                                      save_top_k=3, dirpath=checkpoint_path, save_weights_only=False,
                                      save_last=True,
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

# SNAPSHOT_EPOCHS="20,40": after epoch N (1-based) copy the best-so-far checkpoint to snap_epN.ckpt. With a
# constant LR and a fixed seed the first N epochs of a long run ARE the N-epoch run, so one 60-epoch run also
# yields the 20-epoch budget point (fine-tuned later via run_newdata.sh SNAPSHOT_FROM=<tag>:N).
class _Snapshot(pl.Callback):
    def __init__(self, epochs, ckpt_cb, dirpath):
        self.epochs, self.ckpt_cb, self.dirpath = set(epochs), ckpt_cb, Path(dirpath)

    def on_train_epoch_end(self, trainer, pl_module):   # fires after this epoch's validation + checkpointing
        n = trainer.current_epoch + 1
        if n in self.epochs and self.ckpt_cb.best_model_path:
            import shutil
            dst = self.dirpath / f"snap_ep{n}.ckpt"
            shutil.copy(self.ckpt_cb.best_model_path, dst)
            (self.dirpath / f"snap_ep{n}.txt").write_text(f"{self.ckpt_cb.best_model_path}\n")
            print(f"[snapshot] epoch {n}: best so far {Path(self.ckpt_cb.best_model_path).name} -> {dst.name}", flush=True)

_snap = [int(x) for x in os.environ.get("SNAPSHOT_EPOCHS", "").split(",") if x.strip()]
_callbacks = [checkpoint_callback] + ([_Snapshot(_snap, checkpoint_callback, checkpoint_path)] if _snap else [])

# NOTE: deterministic="warn" (not True): the bidirectional CuDNN LSTM backward has no
# deterministic implementation, so deterministic=True raises at the first backward pass.
# "warn" keeps the seeded run reproducible where possible and only warns on those ops.
strategy = "ddp" if len(gpus) > 1 else "auto"
# PRECISION: "32-true" (default, bit-identical to all earlier runs) or "16-mixed" (fp16 autocast + loss
# scaling on the Volta/Pascal tensor cores). LIMIT_TRAIN_BATCHES: cap steps/epoch (benchmarking only).
_precision = os.environ.get("PRECISION", "32-true")
_limit_tb = os.environ.get("LIMIT_TRAIN_BATCHES")
# GRAD_CLIP=<norm>: clip the global gradient norm (off by default = every result before 2026-10-05). The L model
# (25.6 M) at 3e-4 showed 25-30x loss spikes (L60-control step 11799, L20 on 709 h step 5749) that M never did;
# pairs with ACC_CLAMP in the loader (synthetic accel capped at a real sensor's range).
_clip = float(os.environ.get("GRAD_CLIP", "0") or 0)
if _clip or os.environ.get("ACC_CLAMP"):
    print(f"[stab] GRAD_CLIP={_clip or 'off'} ACC_CLAMP={os.environ.get('ACC_CLAMP', 'off')} m/s^2", flush=True)
trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=wandb_logger, max_epochs=max_epochs,
                     accelerator="gpu", devices=gpus, strategy=strategy, precision=_precision,
                     limit_train_batches=(int(_limit_tb) if _limit_tb else 1.0), accumulate_grad_batches=_accum,
                     gradient_clip_val=(_clip or None), callbacks=_callbacks, deterministic="warn")

# %%
# RESUME_FROM=<last.ckpt> resumes optimizer + epoch/global_step (full Lightning resume),
# so charts continue seamlessly when paired with WANDB_RUN_ID above.
trainer.fit(model, datamodule=datamodule, ckpt_path=(os.environ.get("RESUME_FROM") or None))

# %%
with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
