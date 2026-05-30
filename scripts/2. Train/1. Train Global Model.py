# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
# Variable-length sequence batches fragment the CUDA caching allocator's reserved pool;
# expandable segments keep reserved memory bounded. Set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything

from pathlib import Path

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule, get_split_files
from imuposer.utils import get_parser

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment

# %%
config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoser",
                project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device="0")

# read the synthesized data from the external folder (override with IMUPOSER_DATA_DIR).
# The canonical dataset-level split (config.val_datasets / config.test_datasets) is
# applied automatically by get_datamodule -> get_dataset.
config.processed_imu_poser_25fps = Path(os.environ.get(
    "IMUPOSER_DATA_DIR", "/media/vimal/T7_2TB/CHI23/processed_imuposer_data")) / "processed_imuposer_25fps"

train_files, val_files, test_files = get_split_files(config)
print(f"data: {config.processed_imu_poser_25fps}", flush=True)
print(f"TRAIN ({len(train_files)}): {sorted(f[:-3] for f in train_files)}", flush=True)
print(f"VAL   ({len(val_files)}): {sorted(f[:-3] for f in val_files)}", flush=True)
print(f"TEST  ({len(test_files)}): {test_files}", flush=True)

# %%
# instantiate model and data
model = get_model(config)
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path 

# %%
wandb_logger = WandbLogger(project=config.experiment, name=os.environ.get("WANDB_RUN_NAME"),
                           save_dir=str(checkpoint_path))

early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                        min_delta=0.00001, patience=5)
checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
                                      save_top_k=5, dirpath=checkpoint_path, save_weights_only=True, 
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

# NOTE: deterministic="warn" (not True): the bidirectional CuDNN LSTM backward has no
# deterministic implementation, so deterministic=True raises at the first backward pass.
# "warn" keeps the seeded run reproducible where possible and only warns on those ops.
trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=wandb_logger, max_epochs=1000, accelerator="gpu", devices=[0],
                     callbacks=[early_stopping_callback, checkpoint_callback], deterministic="warn")

# %%
trainer.fit(model, datamodule=datamodule)

# %%
with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
