#!/usr/bin/env bash
# Interim read-out of a base run that is still training: fine-tune its best-so-far checkpoint on DIP ftrain
# (select on fval) and evaluate on dip_test, into ft_<tag>_interim_ep<N>. Runs alongside the base training.
#   interim_eval.sh <gpu> <base_tag>
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
export IMUPOSER_DATA_DIR="${IMUPOSER_DATA_DIR:-/home/vimal/imuposer_data}"
DD="$IMUPOSER_DATA_DIR/processed_imuposer_25fps"
GPU="${1:?gpu}"; TAG="${2:?base tag}"
BDIR="$REPO/checkpoints/newdata/base_$TAG"
BEST="$(ls "$BDIR"/epoch=epoch=*-val_loss=*.ckpt | sort -t= -k4 -n | head -1)"
EP="$(echo "$BEST" | grep -oE 'epoch=epoch=[0-9]+' | grep -oE '[0-9]+$')"
ftdir="$REPO/checkpoints/newdata/ft_${TAG}_interim_ep${EP}"; mkdir -p "$ftdir"
echo "[$(date -Is)] interim FT of $TAG from epoch $EP: $BEST"
env MODEL=AvatarPoserModel TF_LR=1e-4 EPOCHS=60 TRAIN_COMBO=lw_rw_rp SEED=1 NUM_WORKERS=2 \
    TRAIN_DATASETS=ftrain VAL_FILES=fval.pt \
    GPUS="$GPU" CONTINUE_FROM="$BEST" CHECKPOINT_DIR="$ftdir" WANDB_RUN_NAME="newdata_ft_${TAG}_interim_ep${EP}" \
    uv run python "1. Train Global Model.py" --combo_id global --experiment newdata > "$ftdir/train.log" 2>&1
( cd "$REPO" && CUDA_VISIBLE_DEVICES="$GPU" IMUPOSER_25FPS_DIR="$DD" \
    uv run python "scripts/3. Evaluation/offline_fit.py" --data dip_test.pt --combo lw_rw_rp \
    --members "$ftdir/last.ckpt" --iters 0 --device 0 > "$ftdir/eval_dip_test.log" 2>&1 )
grep -m1 "SIP" "$ftdir/eval_dip_test.log"
echo "[$(date -Is)] DONE interim $TAG ep$EP"
