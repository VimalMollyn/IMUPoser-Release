#!/usr/bin/env bash
# Fine-tune + evaluate ONE given pretraining checkpoint (e.g. an epoch snapshot of a run still training) into
# ft_<tag>_interim_<label>, with the unchanged FT recipe. Model-size env (TF_DMODEL/TF_LAYERS/TF_FF) must be exported.
#   interim_ckpt.sh <gpu> <tag> <ckpt> <label>        e.g. interim_ckpt.sh 0 dlall_xl20 /path/snap_ep10.ckpt ep10snap
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
export IMUPOSER_DATA_DIR="${IMUPOSER_DATA_DIR:-/home/vimal/imuposer_data}"
DD="$IMUPOSER_DATA_DIR/processed_imuposer_25fps"
GPU="${1:?gpu}"; TAG="${2:?tag}"; CKPT="${3:?ckpt}"; LABEL="${4:?label}"
ftdir="$REPO/checkpoints/newdata/ft_${TAG}_interim_${LABEL}"; mkdir -p "$ftdir"
echo "[$(date -Is)] interim FT of $TAG from $CKPT -> $ftdir"
env MODEL=AvatarPoserModel TF_LR=1e-4 EPOCHS=60 TRAIN_COMBO=lw_rw_rp SEED=1 NUM_WORKERS=2 \
    TRAIN_DATASETS=ftrain VAL_FILES=fval.pt \
    GPUS="$GPU" CONTINUE_FROM="$CKPT" CHECKPOINT_DIR="$ftdir" WANDB_RUN_NAME="newdata_ft_${TAG}_interim_${LABEL}" \
    uv run python "1. Train Global Model.py" --combo_id global --experiment newdata > "$ftdir/train.log" 2>&1
( cd "$REPO" && CUDA_VISIBLE_DEVICES="$GPU" IMUPOSER_25FPS_DIR="$DD" \
    uv run python "scripts/3. Evaluation/offline_fit.py" --data dip_test.pt --combo lw_rw_rp \
    --members "$ftdir/last.ckpt" --iters 0 --device 0 > "$ftdir/eval_dip_test.log" 2>&1 )
grep -m1 "SIP" "$ftdir/eval_dip_test.log"
echo "[$(date -Is)] DONE interim $TAG $LABEL"
