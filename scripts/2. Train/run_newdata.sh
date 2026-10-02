#!/usr/bin/env bash
# New-motion-data experiment (2026-10): does adding BONES-SEED (originals) + form-hoi + MotionMillion
# (non-GV subsets) to pretraining improve the lw_rw_rp 25 Hz deliverable?
#
# Single-variable, two-stage design (same as the Nymeria study):
#   stage 1  AvatarPoser base, 60 ep, AUG_CALIB_RAD=0.12217, TRAIN_COMBO=lw_rw_rp, select on dip_train
#            control   = curated-12 AMASS + Nymeria (the current best recipe, dip_test 17.32)
#            treatment = control + BONES_* + FORMHOI_* + MM_* + MotionX_*   (the new data)
#   stage 2  fine-tune on real DIP ftrain (60 ep, TF_LR=1e-4), select on fval, eval last.ckpt on dip_test
# Both arms read the SAME files through the streaming (memmap-shard) loader from IMUPOSER_DATA_DIR.
#
#   run_newdata.sh <gpu> "<tag>|<control|treatment>|<seed>[|EXTRA=env ...]" ...
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
export IMUPOSER_DATA_DIR="${IMUPOSER_DATA_DIR:-/home/vimal/imuposer_data}"
DD="$IMUPOSER_DATA_DIR/processed_imuposer_25fps"
OUT="${OUT_DIR:-$REPO/checkpoints/newdata}"
mkdir -p "$OUT"

CURATED="CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits"
NYM="$(ls "$DD" | grep -E '^Nymeria_.*\.pt$' | sed 's/\.pt$//' | paste -sd, -)"
NEW="$(ls "$DD" | grep -E '^(BONES_|FORMHOI_|MM_|MotionX_).*\.pt$' | sed 's/\.pt$//' | paste -sd, -)"
# MotionGV (video-estimated MotionMillion) lives as shard-only datasets (no .pt): list its shard dirs
SH="${IMUPOSER_SHARD_DIR:-$IMUPOSER_DATA_DIR/shards_processed_imuposer_25fps}"
GV="$(ls "$SH" 2>/dev/null | grep -E '^MGV_' | grep -v '\.lock$\|\.packing$' | paste -sd, -)"
[ -n "$NYM" ] || { echo "no Nymeria_*.pt in $DD" >&2; exit 1; }

GPU="${1:?gpu}"; shift
for spec in "$@"; do
  IFS='|' read -r tag arm seed extra <<< "$spec"
  case "$arm" in
    control)   DATA="$CURATED,$NYM" ;;
    treatment) [ -n "$NEW" ] || { echo "no new-data files in $DD" >&2; exit 1; }; DATA="$CURATED,$NYM,$NEW" ;;
    treatment_gv) [ -n "$NEW" ] && [ -n "$GV" ] || { echo "missing new-data or MGV_ shards" >&2; exit 1; }; DATA="$CURATED,$NYM,$NEW,$GV" ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  echo "[$(date -Is)] START base_$tag arm=$arm seed=$seed gpu=$GPU"
  dir="$OUT/base_$tag"; mkdir -p "$dir"
  if [ ! -f "$dir/best_model.txt" ]; then
    env MODEL=AvatarPoserModel EPOCHS=60 TRAIN_COMBO=lw_rw_rp AUG_CALIB_RAD=0.12217 SEED="$seed" \
        TRAIN_DATASETS="$DATA" VAL_FILES=dip_train.pt ${extra:-} \
        GPUS="$GPU" CHECKPOINT_DIR="$dir" WANDB_RUN_NAME="newdata_base_$tag" \
        uv run python "1. Train Global Model.py" --combo_id global --experiment newdata \
        > "$dir/train.log" 2>&1
  fi
  BEST="$(head -1 "$dir/best_model.txt" 2>/dev/null)"; [ -f "$BEST" ] || BEST="$dir/last.ckpt"
  echo "[$(date -Is)] base done: $BEST"

  # model-architecture env (TF_DMODEL/TF_LAYERS/...) must also reach the FT stage and the evaluator, which
  # rebuild the model from env; the base-stage EPOCHS override must NOT (FT is always 60 ep)
  extra_model="$(echo "${extra:-}" | tr ' ' '\n' | grep -v '^EPOCHS=' | grep -v '^$' | paste -sd' ' -)"
  ftdir="$OUT/ft_$tag"; mkdir -p "$ftdir"
  if [ ! -f "$ftdir/best_model.txt" ]; then
    env MODEL=AvatarPoserModel TF_LR=1e-4 EPOCHS=60 TRAIN_COMBO=lw_rw_rp SEED=1 \
        TRAIN_DATASETS=ftrain VAL_FILES=fval.pt $extra_model \
        GPUS="$GPU" CONTINUE_FROM="$BEST" CHECKPOINT_DIR="$ftdir" WANDB_RUN_NAME="newdata_ft_$tag" \
        uv run python "1. Train Global Model.py" --combo_id global --experiment newdata \
        > "$ftdir/train.log" 2>&1
  fi
  echo "[$(date -Is)] FT done, evaluating $tag on dip_test"
  ( cd "$REPO" && env $extra_model CUDA_VISIBLE_DEVICES="$GPU" IMUPOSER_25FPS_DIR="$DD" \
      uv run python "scripts/3. Evaluation/offline_fit.py" --data dip_test.pt --combo lw_rw_rp \
      --members "$ftdir/last.ckpt" --iters 0 --device 0 > "$ftdir/eval_dip_test.log" 2>&1 )
  grep -E "SIP" "$ftdir/eval_dip_test.log" | head -3
  echo "[$(date -Is)] DONE $tag"
done
