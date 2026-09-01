#!/usr/bin/env bash
# Fine-tune (warm-start from an AMASS avatar base) on a configurable real-data mix that can include the
# retargeted WHIP data. Used to test whether WHIP real data beats the DIP-only 18.37 floor.
# Recipe search runs on ftrain (+whip), selecting on the held-out fval; the final run uses alltrain+whip.
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
BASE="${BASE_CKPT:-$REPO/checkpoints/autoresearch/exp21_avatar_s1/epoch=epoch=37-val_loss=validation_step_loss=0.03638.ckpt}"
OUT="${OUT_DIR:-$REPO/checkpoints/whipft}"
mkdir -p "$OUT"

GPU="${1:?gpu}"; shift
for spec in "$@"; do
  IFS='|' read -r tag traindata valfiles extra <<< "$spec"
  dir="$OUT/$tag"; mkdir -p "$dir"
  env MODEL=AvatarPoserModel TF_LR=1e-4 EPOCHS=60 TRAIN_COMBO="${TRAIN_COMBO:-lw_rp_h}" \
      TRAIN_DATASETS="$traindata" VAL_FILES="$valfiles" $extra \
      GPUS="$GPU" CONTINUE_FROM="$BASE" CHECKPOINT_DIR="$dir" WANDB_RUN_NAME="whipft_$tag" \
      uv run python "1. Train Global Model.py" --combo_id global --experiment whipft \
      > "$dir/train.log" 2>&1
  echo "DONE $tag -> $dir"
done
