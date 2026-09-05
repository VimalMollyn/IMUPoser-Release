#!/usr/bin/env bash
# QUEUED: 50 Hz lw_rw_lp (loose pose model + calibrator). Runs AFTER the current 25 Hz lw_rw_lp jobs
# finish (waits on their DONE markers), because both GPUs + RAM are busy and a full-Nymeria base needs
# ~32 GB. Steps: re-fetch Nymeria @60fps (intermediates were reclaimed) -> resample everything to 50fps
# -> train loose lw_rw_lp base @50fps -> FT -> eval; and the 50fps calibrator. Idempotent/resumable.
set -u
REPO=/media/vimal/T7_2TB/CHI23/IMUPoser-Release
DATA=/media/vimal/T7_2TB/CHI23/processed_imuposer_data
D50="$DATA/processed_imuposer_50fps"
URLS=/media/vimal/T7_2TB/CHI23/nymeria_plus_urls.json
LOOSE="AUG_LOOSE_SENSORS=2 AUG_LOOSE_CALIB_RAD=0.30 AUG_LOOSE_DRIFT_RAD_S=0.04 AUG_LOOSE_RESEAT_RAD=0.30 AUG_LOOSE_ACC_STD=0.15"
log(){ echo "[50hz] $*"; }

# 0. wait for the current 25 Hz lw_rw_lp pipeline to fully finish (both GPUs free)
log "waiting for 25 Hz lw_rw_lp jobs (loose+clean) to finish ..."
while ! grep -q "\[gpu0\] LOOSE lw_rw_lp DONE" /tmp/lwrwlp_gpu0.log 2>/dev/null \
   || ! grep -q "\[gpu1\] CLEAN lw_rw_lp DONE" /tmp/lwrwlp_gpu1.log 2>/dev/null; do sleep 60; done
sleep 60
log "25 Hz jobs done. Starting 50 Hz build."

# 1. re-fetch Nymeria @60fps (regenerate the intermediates we reclaimed). 4 shards, 2 per GPU.
if [ "$(ls -d "$DATA"/processed_imuposer/AMASS/Nymeria_* 2>/dev/null | wc -l)" -lt 80 ]; then
  log "re-fetching Nymeria @60fps ..."
  cd "$REPO"
  pids=()
  for s in 0 1 2 3; do g=$((s % 2))
    uv run python "scripts/1. Preprocessing/nymeria_fetch.py" \
      --urls "$URLS" --gpu $g --shard $s/4 > "/tmp/nym50_fetch_$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  log "Nymeria re-fetch done: $(ls -d "$DATA"/processed_imuposer/AMASS/Nymeria_* | wc -l) chunks"
else
  log "Nymeria 60fps intermediates already present, skipping re-fetch"
fi

# 2. resample ALL 60fps datasets (curated AMASS + Nymeria + DIP) -> 50fps set
log "resampling to 50 fps -> $D50"
cd "$REPO"
uv run python - << PY
import importlib.util
from pathlib import Path
from imuposer.config import Config
cfg = Config(project_root_dir='.', mkdir=False)
cfg.processed_imu_poser = Path("$DATA")/"processed_imuposer"
cfg.processed_imu_poser_25fps = Path("$D50")            # reuse the resampler, writing to the 50fps dir
spec = importlib.util.spec_from_file_location('p2', 'scripts/1. Preprocessing/2. preprocess_all_to_imuposer_at_25fps.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.target_fps = 50                                        # <-- 50 fps instead of 25
m.to_25fps(cfg)
print("50fps resample complete")
PY

# 3. reclaim the 60fps Nymeria intermediates (keep the 50fps .pt)
log "reclaiming 60fps Nymeria intermediates"
for d in "$DATA"/processed_imuposer/AMASS/Nymeria_*; do
  b=$(basename "$d"); [ -f "$D50/${b}.pt" ] && rm -f "$d"/*.pt
done

# 4. reproduce the ftrain/fval split (a fixed 32/9 permutation of dip_train) at 50fps by matching each
#    25fps ftrain/fval sequence back to its dip_train index, then slicing the 50fps dip_train.
log "building 50fps ftrain/fval (mirroring the 25fps split)"
cd "$REPO"
D25="$DATA/processed_imuposer_25fps" uv run python - << PY
import os, torch
from pathlib import Path
D25 = Path(os.environ["D25"]); D50 = Path("$D50")
dt25 = torch.load(D25/"dip_train.pt", weights_only=False)
dt50 = torch.load(D50/"dip_train.pt", weights_only=False)
assert len(dt25["pose"]) == len(dt50["pose"]), "dip_train seq count differs across fps"
def idx_of(seq_pose, pool):                    # exact match a 25fps seq to a dip_train index
    for i, p in enumerate(pool):
        if p.shape[0] == seq_pose.shape[0] and torch.equal(p, seq_pose):
            return i
    raise RuntimeError("no match")
for name in ("ftrain", "fval"):
    src = torch.load(D25/f"{name}.pt", weights_only=False)
    idxs = [idx_of(p, dt25["pose"]) for p in src["pose"]]
    out = {k: [dt50[k][i] for i in idxs] if isinstance(dt50[k], list) else dt50[k] for k in dt50}
    torch.save(out, D50/f"{name}.pt")
    print(f"50fps {name}: {len(idxs)} seqs")
PY
ln -sf "dip_train.pt" "$D50/alltrain.pt"

# 5. LOOSE lw_rw_lp @50fps: base -> FT -> eval  (GPU0). 50fps knobs: IMUPOSER_FPS=50, 250-frame windows.
COMMON50="IMUPOSER_FPS_SUBDIR=processed_imuposer_50fps IMUPOSER_FPS=50 TF_EVAL_WINDOW=250 MAX_SAMPLE_LEN=300"
cd "$REPO/scripts/2. Train"
log "training loose lw_rw_lp base @50fps"
env $COMMON50 TRAIN_COMBO=lw_rw_lp MODEL=AvatarPoserModel EPOCHS=60 AUG_CALIB_RAD=0.12217 $LOOSE \
    TRAIN_DATASETS="$(ls "$D50" | grep -E '^(CMU|BioMotionLab_NTroje|BMLmovi|KIT|EKUT|Transitions_mocap|HumanEva|SFU|HUMAN4D|SSM_synced|MPI_mosh|MPI_Limits|Nymeria_).*\.pt$' | sed 's/\.pt$//' | paste -sd, -)" \
    VAL_FILES=dip_train.pt GPUS=0 CHECKPOINT_DIR="$REPO/checkpoints/nymeria/base_lwrwlp_loose_50hz_s1" \
    WANDB_RUN_NAME=base_lwrwlp_loose_50hz_s1 \
    uv run python "1. Train Global Model.py" --combo_id global --experiment nymeria50 \
    > "$REPO/checkpoints/nymeria/base_lwrwlp_loose_50hz_s1.log" 2>&1

BEST="$(head -1 "$REPO/checkpoints/nymeria/base_lwrwlp_loose_50hz_s1/best_model.txt" 2>/dev/null)"
[ -f "$BEST" ] || BEST="$REPO/checkpoints/nymeria/base_lwrwlp_loose_50hz_s1/last.ckpt"
log "loose FT @50fps from $BEST"
env $COMMON50 TRAIN_COMBO=lw_rw_lp MODEL=AvatarPoserModel TF_LR=1e-4 EPOCHS=60 AUG_CALIB_RAD=0.12217 $LOOSE \
    TRAIN_DATASETS=ftrain VAL_FILES=fval.pt CONTINUE_FROM="$BEST" GPUS=0 \
    CHECKPOINT_DIR="$REPO/checkpoints/nymeria/ft_lwrwlp_loose_50hz_s1" WANDB_RUN_NAME=ft_lwrwlp_loose_50hz_s1 \
    uv run python "1. Train Global Model.py" --combo_id global --experiment nymeria50 \
    > "$REPO/checkpoints/nymeria/ft_lwrwlp_loose_50hz_s1.log" 2>&1

log "50fps loose guardrail eval:"
cd "$REPO"
IMUPOSER_25FPS_DIR="$D50" TF_EVAL_WINDOW=250 CUDA_VISIBLE_DEVICES=0 \
  uv run python "scripts/3. Evaluation/offline_fit.py" --data dip_test.pt --combo lw_rw_lp \
  --members "$REPO/checkpoints/nymeria/ft_lwrwlp_loose_50hz_s1/last.ckpt" --iters 0 2>&1 | grep -E "^  SIP"

# 6. calibrator @50fps (GPU1, light; can overlap step 5's base but simplest to run after)
log "training 50fps calibrator lw_rw_lp (slot 2)"
IMUPOSER_25FPS_DIR="$D50" IMUPOSER_FPS=50 CAL_NYM_CHUNKS=3 CUDA_VISIBLE_DEVICES=1 \
  WANDB_PROJECT=imu_calibrator WANDB_RUN_NAME=cal_lwrwlp_50hz_s1 \
  uv run python "scripts/2. Train/train_calibrator.py" --combo lw_rw_lp --loose-slot 2 --epochs 60 \
  --out "$REPO/checkpoints/calibrator/cal_lwrwlp_50hz_s1"

log "50 Hz lw_rw_lp DONE"
