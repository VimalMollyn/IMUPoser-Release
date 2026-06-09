#!/usr/bin/env bash
# FT-augmentation sweep: warm-start an AMASS avatar base, fine-tune on ftrain (fval HELD OUT) with
# different domain-randomization configs, to test whether stronger augmentation during fine-tuning
# improves generalization to held-out subjects (measured on fval). Valid for iteration: fval not trained on.
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
BASE="$REPO/checkpoints/autoresearch/exp21_avatar_s1/epoch=epoch=37-val_loss=validation_step_loss=0.03638.ckpt"
OUT="$REPO/checkpoints/ftaug"
mkdir -p "$OUT"
COMMON="TRAIN_DATASETS=ftrain VAL_FILES=fval.pt TRAIN_COMBO=lw_rp_h EPOCHS=60 MODEL=AvatarPoserModel TF_LR=1e-4"

GPU="${1:?gpu}"; shift
for spec in "$@"; do
  IFS='|' read -r tag augenv <<< "$spec"
  dir="$OUT/$tag"; mkdir -p "$dir"
  env $COMMON $augenv GPUS="$GPU" CONTINUE_FROM="$BASE" CHECKPOINT_DIR="$dir" \
      WANDB_RUN_NAME="ftaug_$tag" \
      uv run python "1. Train Global Model.py" --combo_id global --experiment ftaug \
      > "$dir/train.log" 2>&1
  echo "DONE $tag -> $dir"
done
