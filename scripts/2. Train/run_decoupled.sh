#!/usr/bin/env bash
# DECOUPLED / part-based model: each sensor predicts ONLY its own body part, trained as an independent
# single-sensor specialist with a region-masked loss (PART_LOSS_JOINTS). Contrast the JOINT model
# (lw_rw_rp) where all 3 sensors together predict the whole body.
#   lw -> left arm  {13,16,18,20};  rw -> right arm {14,17,19,21};  rp -> legs {0,1,2,4,5,7,8,10,11}
# Torso/neck/head {3,6,9,12,15} have NO sensor in this config -> filled with rest pose at assembly.
# First pass on curated-12 (fast); promote to Nymeria+FT if it is competitive with the joint model.
# spec = "tag|combo|part_joints|extra"  (empty part_joints => full-body joint model, the baseline).
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
DD="${IMUPOSER_25FPS_DIR:-/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps}"
OUT="$REPO/checkpoints/decoupled"; mkdir -p "$OUT"
CURATED="CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits"

GPU="${1:?gpu}"; shift
for spec in "$@"; do
  IFS='|' read -r tag combo part extra <<< "$spec"
  dir="$OUT/$tag"; mkdir -p "$dir"
  env MODEL=AvatarPoserModel EPOCHS=60 TRAIN_COMBO="$combo" AUG_CALIB_RAD=0.12217 \
      ${part:+PART_LOSS_JOINTS=$part} TRAIN_DATASETS="$CURATED" VAL_FILES=dip_train.pt $extra \
      GPUS="$GPU" CHECKPOINT_DIR="$dir" WANDB_RUN_NAME="decoupled_$tag" \
      uv run python "1. Train Global Model.py" --combo_id global --experiment decoupled \
      > "$dir/train.log" 2>&1
  echo "DONE $tag -> $dir"
done
