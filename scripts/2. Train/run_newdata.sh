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
  # arms may append dataset groups by name prefix: "control+FORMHOI_,BONES_" or "treatment+MGV_" (prefixes are
  # matched against .pt files in $DD and shard-only dataset dirs in $SH)
  base_arm="${arm%%+*}"; plus="${arm#*+}"; [ "$plus" = "$arm" ] && plus=""
  case "$base_arm" in
    curated)   DATA="$CURATED" ;;
    control)   DATA="$CURATED,$NYM" ;;
    treatment) [ -n "$NEW" ] || { echo "no new-data files in $DD" >&2; exit 1; }; DATA="$CURATED,$NYM,$NEW" ;;
    treatment_gv) [ -n "$NEW" ] && [ -n "$GV" ] || { echo "missing new-data or MGV_ shards" >&2; exit 1; }; DATA="$CURATED,$NYM,$NEW,$GV" ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  if [ -n "$plus" ]; then
    for pre in $(echo "$plus" | tr ',' ' '); do
      ADD="$( { ls "$DD" | grep -E "^${pre}.*\.pt$" | sed 's/\.pt$//'; ls "$SH" 2>/dev/null | grep -E "^${pre}" | grep -v '\.lock$\|\.packing$'; } | sort -u | paste -sd, -)"
      [ -n "$ADD" ] || { echo "no datasets with prefix $pre" >&2; exit 1; }
      DATA="$DATA,$ADD"
    done
  fi
  # FT-stage knobs (defaults = the recipe every run so far used). BASE_FROM=<tag> reuses another run's finished
  # base checkpoint (FT-only variants: FT_LR / FT_EPOCHS / FT_SEED / FT_SCHED), so the base is trained once.
  getk(){ echo "${extra:-}" | tr ' ' '\n' | grep "^$1=" | tail -1 | cut -d= -f2-; }
  BASE_FROM="$(getk BASE_FROM)"; FT_LR="$(getk FT_LR)"; FT_EPOCHS="$(getk FT_EPOCHS)"; FT_SEED="$(getk FT_SEED)"; FT_SCHED="$(getk FT_SCHED)"
  # SNAPSHOT_FROM=<tag>:<N>: fine-tune the epoch-N snapshot (snap_epN.ckpt, see SNAPSHOT_EPOCHS in the trainer) of
  # another run = the N-epoch budget point of that run, without training it again
  SNAPSHOT_FROM="$(getk SNAPSHOT_FROM)"; SNAP_EP=""
  if [ -n "$SNAPSHOT_FROM" ]; then BASE_FROM="${SNAPSHOT_FROM%%:*}"; SNAP_EP="${SNAPSHOT_FROM##*:}"; fi
  # CONTINUE_FROM_TAG=<tag>: warm-start the base stage from that run's best checkpoint (curriculum: broad -> narrow)
  CFT="$(getk CONTINUE_FROM_TAG)"
  if [ -n "$CFT" ]; then
    CFB="$(head -1 "$OUT/base_$CFT/best_model.txt" 2>/dev/null)"
    [ -f "$CFB" ] || { echo "CONTINUE_FROM_TAG=$CFT: no finished base" >&2; exit 1; }
    extra="$(echo "${extra:-}" | tr ' ' '\n' | grep -v '^CONTINUE_FROM_TAG=' | paste -sd' ' -) CONTINUE_FROM=$CFB"
    echo "[$(date -Is)] $tag warm-starts from $CFB"
  fi
  echo "[$(date -Is)] START base_$tag arm=$arm seed=$seed gpu=$GPU"
  dir="$OUT/base_${BASE_FROM:-$tag}"; mkdir -p "$dir"
  if [ -n "$SNAP_EP" ]; then
    [ -f "$dir/snap_ep$SNAP_EP.ckpt" ] || { echo "SNAPSHOT_FROM=$SNAPSHOT_FROM: no $dir/snap_ep$SNAP_EP.ckpt" >&2; exit 1; }
  elif [ -n "$BASE_FROM" ] && [ ! -f "$dir/best_model.txt" ]; then echo "BASE_FROM=$BASE_FROM has no finished base in $dir" >&2; exit 1; fi
  if [ ! -f "$dir/best_model.txt" ]; then
    # an interrupted base run (killed for a drive swap, crash, ...) resumes from its last epoch checkpoint
    RESUME=""; [ -f "$dir/last.ckpt" ] && RESUME="$dir/last.ckpt" && echo "[$(date -Is)] resuming $tag from last.ckpt"
    env MODEL=AvatarPoserModel EPOCHS=60 TRAIN_COMBO=lw_rw_rp AUG_CALIB_RAD=0.12217 SEED="$seed" \
        TRAIN_DATASETS="$DATA" VAL_FILES=dip_train.pt ${extra:-} RESUME_FROM="$RESUME" \
        GPUS="$GPU" CHECKPOINT_DIR="$dir" WANDB_RUN_NAME="newdata_base_$tag" \
        uv run python "1. Train Global Model.py" --combo_id global --experiment newdata \
        > "$dir/train.log" 2>&1
  fi
  BEST="$(head -1 "$dir/best_model.txt" 2>/dev/null)"; [ -f "$BEST" ] || BEST="$dir/last.ckpt"
  [ -n "$SNAP_EP" ] && BEST="$dir/snap_ep$SNAP_EP.ckpt"
  echo "[$(date -Is)] base done: $BEST"

  # model-architecture env (TF_DMODEL/TF_LAYERS/...) must also reach the FT stage and the evaluator, which
  # rebuild the model from env; the base-stage EPOCHS / LR schedule / data weighting overrides must NOT
  # (FT is always 60 ep, constant TF_LR=1e-4, on ftrain) so the FT recipe stays identical across arms
  extra_model="$(echo "${extra:-}" | tr ' ' '\n' | grep -v '^EPOCHS=\|^LR_SCHED=\|^LR_MIN_FRAC=\|^DATASET_REPEAT=\|^DATASET_FRACTION=\|^DATASET_KEEP=\|^CONTINUE_FROM=\|^TF_WD=\|^BASE_FROM=\|^SNAPSHOT_\|^FT_' | grep -v '^$' | paste -sd' ' -)"
  ftdir="$OUT/ft_$tag"; mkdir -p "$ftdir"
  if [ -n "$BASE_FROM$FT_LR$FT_EPOCHS$FT_SEED$FT_SCHED" ]; then
    printf '{"base_from": "%s", "ft_lr": "%s", "ft_epochs": "%s", "ft_seed": "%s", "ft_sched": "%s", "snapshot_epoch": %s, "snapshot_src": "%s"}\n' \
      "${BASE_FROM:-$tag}" "${FT_LR:-1e-4}" "${FT_EPOCHS:-60}" "${FT_SEED:-1}" "${FT_SCHED:-const}" "${SNAP_EP:-null}" \
      "$( [ -n "$SNAP_EP" ] && basename "$(head -1 "$dir/snap_ep$SNAP_EP.txt" 2>/dev/null)" )" > "$ftdir/ft_meta.json"
  fi
  if [ ! -f "$ftdir/best_model.txt" ]; then
    env MODEL=AvatarPoserModel TF_LR="${FT_LR:-1e-4}" EPOCHS="${FT_EPOCHS:-60}" TRAIN_COMBO=lw_rw_rp SEED="${FT_SEED:-1}" \
        LR_SCHED="${FT_SCHED:-}" TRAIN_DATASETS=ftrain VAL_FILES=fval.pt $extra_model \
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
