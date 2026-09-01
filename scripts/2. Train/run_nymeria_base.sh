#!/usr/bin/env bash
# Nymeria experiment: does pretraining on ~100 h of DIP-LIKE everyday motion (NymeriaPlus) beat
# pretraining on curated-12 AMASS alone?
#
# WHY THE PRETRAIN STAGE (and not the FT stage, where WHIP was tried and failed): Nymeria is
# synthetic-IMU motion data, exactly like AMASS -- its value is motion DISTRIBUTION, not sensor
# realism. The FT stage exists to adapt to real IMU noise; injecting synthetic data there dilutes
# it, which is part of why WHIP lost. Diagnostics back the placement: Nymeria's own-mean pose
# diversity is 28.5-29.7 deg vs DIP's 30.9-32.5 (WHIP: 50.5), and its distance from DIP's mean
# pose is 42 deg (WHIP: 61). Same manifold, ~77x the data.
#
# exp21's exact TRAIN_DATASETS is not recorded in wandb, so we train BOTH arms here rather than
# reuse exp21 as the control -- otherwise "Nymeria added" confounds with "data changed".
# Stage 2 (FT on real DIP) then runs via run_whip_ft.sh with BASE_CKPT=<this base>/last.ckpt.
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
DD="${IMUPOSER_25FPS_DIR:-/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps}"
OUT="$REPO/checkpoints/nymeria"
mkdir -p "$OUT"

CURATED="CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits"
NYM="$(ls "$DD" | grep -E '^Nymeria_.*\.pt$' | sed 's/\.pt$//' | paste -sd, -)"
if [ -z "$NYM" ]; then echo "no Nymeria_*.pt in $DD" >&2; exit 1; fi
echo "Nymeria chunks: $(echo "$NYM" | tr ',' '\n' | wc -l)"
# nym80 = the original 80.9h dose = the FIRST 34 chunks (shards 00_00x/01_00x/02_00x/03_00x from round 1),
# i.e. those with a two-digit shard <04. Deterministic subset so the 80.9h dose is reproducible.
NYM80="$(echo "$NYM" | tr ',' '\n' | grep -E '^Nymeria_0[0-3]_00[0-9]$' | paste -sd, -)"

GPU="${1:?gpu}"; shift
for spec in "$@"; do
  IFS='|' read -r tag arm extra <<< "$spec"
  case "$arm" in
    curated)  DATA="$CURATED" ;;
    nymeria)  DATA="$CURATED,$NYM" ;;
    nym80)    DATA="$CURATED,$NYM80" ;;
    nymonly)  DATA="$NYM" ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  dir="$OUT/$tag"; mkdir -p "$dir"
  env MODEL=AvatarPoserModel EPOCHS=60 TRAIN_COMBO="${TRAIN_COMBO:-lw_rp_h}" AUG_CALIB_RAD=0.12217 \
      TRAIN_DATASETS="$DATA" VAL_FILES=dip_train.pt $extra \
      GPUS="$GPU" CHECKPOINT_DIR="$dir" WANDB_RUN_NAME="nymbase_$tag" \
      uv run python "1. Train Global Model.py" --combo_id global --experiment nymeria \
      > "$dir/train.log" 2>&1
  echo "DONE $tag -> $dir"
done
