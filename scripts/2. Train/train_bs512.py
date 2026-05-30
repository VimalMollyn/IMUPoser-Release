r"""Launch a batch-512 global-model run (LR linearly scaled 3e-4 -> 6e-4) without
editing the shared config.py / model files, so the batch-256 baseline run and the
committed code stay untouched. Mirrors '1. Train Global Model.py' otherwise.

Run on GPU 1:  CUDA_VISIBLE_DEVICES=1 uv run python train_bs512.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule

seed_everything(42, workers=True)

BATCH = 512
LR = 6e-4  # linear scaling from 3e-4 at batch 256

# Log into the SAME wandb project as the batch-256 baseline (GPU 0) so both runs
# show up on the same charts; distinguish this one by its run name.
WANDB_PROJECT = "IMU2PoseGlobalFinal_global"
WANDB_RUN_NAME = "bs512_lr6e-4"

config = Config(experiment="IMU2PoseGlobalFinal_bs512_global", model="GlobalModelIMUPoser",
                project_root_dir="../../", joints_set=amass_combos["global"], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device="0")
config.batch_size = BATCH

model = get_model(config)
model.lr = LR
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path

wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_RUN_NAME, save_dir=str(checkpoint_path))

early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                        min_delta=0.00001, patience=5)
checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False,
                                      save_top_k=5, dirpath=checkpoint_path, save_weights_only=True,
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

trainer = pl.Trainer(logger=wandb_logger, max_epochs=1000, accelerator="gpu", devices=[0],
                     callbacks=[early_stopping_callback, checkpoint_callback], deterministic="warn")

print(f"batch_size={config.batch_size}  lr={model.lr}", flush=True)
trainer.fit(model, datamodule=datamodule)

with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
