#!/usr/bin/env bash
# Wait until (a) the control run on this GPU has fully finished (its run log says DONE) and (b) all three
# conversions have written their DONE lines, then pack the new files into shards and launch the treatment
# arm on the same GPU. Keeps the GPU busy without babysitting.
#   chain_treatment.sh <gpu> <control_run_log> <treatment_tag> <seed>
set -u
cd "$(dirname "$0")"
GPU="${1:?gpu}"; CLOG="${2:?control run log}"; TAG="${3:?tag}"; SEED="${4:?seed}"
LOGS=/home/vimal/imuposer_data/logs
log(){ echo "[chain $TAG $(date -Is)] $*"; }

log "waiting for control run ($CLOG) and conversions ..."
while ! grep -q "^\[.*\] DONE " "$CLOG" 2>/dev/null; do sleep 60; done
log "control run done"
while ! { grep -q "^DONE MM_PhantomDance" "$LOGS/motionmillion_convert.log" 2>/dev/null \
       && grep -q "^DONE BONES" "$LOGS/bones_seed_convert.log" 2>/dev/null \
       && grep -q "^DONE FORMHOI" "$LOGS/formhoi_convert.log" 2>/dev/null; }; do sleep 60; done
log "conversions done; packing shards"
NEW="$(ls /home/vimal/imuposer_data/processed_imuposer_25fps | grep -E '^(BONES_|FORMHOI_|MM_|MotionX_).*\.pt$' | sed 's/\.pt$//' | paste -sd, -)"
NPROC=3 uv run python /home/vimal/.claude/jobs/a291a823/tmp/prepack.py "$NEW" > "$LOGS/prepack_new_$TAG.log" 2>&1
log "packed; launching treatment $TAG on GPU $GPU"
bash run_newdata.sh "$GPU" "$TAG|treatment|$SEED"
log "COMPLETE"
