#!/usr/bin/env bash
# Continue-fine-tune each existing FT ensemble member on the FULL dip_train (s01-s08, 41 seqs
# = ftrain 32 + fval 9) at low LR. Adds the 9 held-out fval seqs that were reserved only as a
# research-phase selection signal; dip_test (s09/s10) stays held out. Single test readout after.
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
CK="$REPO/checkpoints/autoresearch"
OUT="$REPO/checkpoints/fulltrain"
mkdir -p "$OUT"
COMMON="TRAIN_DATASETS=alltrain VAL_FILES=fval.pt TRAIN_COMBO=lw_rp_h EPOCHS=30"

run() {  # gpu member model lrenv extra
  local gpu="$1" member="$2" model="$3" lrenv="$4" extra="$5"
  local dir="$OUT/${member}_ft2"
  mkdir -p "$dir"
  env $COMMON $extra GPUS="$gpu" MODEL="$model" $lrenv \
      CONTINUE_FROM="$CK/$member/last.ckpt" CHECKPOINT_DIR="$dir" \
      WANDB_RUN_NAME="ft2_${member}" \
      uv run python "1. Train Global Model.py" --combo_id global --experiment fulltrain \
      > "$dir/train.log" 2>&1
  echo "DONE $member -> $dir"
}

GPU="${1:?gpu}"; shift
for spec in "$@"; do
  IFS='|' read -r member model lrenv extra <<< "$spec"
  run "$GPU" "$member" "$model" "$lrenv" "$extra"
done
