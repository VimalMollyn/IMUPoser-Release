#!/usr/bin/env bash
# Wait for a run log to say DONE <tag>, then run more run_newdata.sh specs on the same GPU (keeps a GPU busy).
#   chain_after.sh <gpu> <run_log> <tag_to_wait_for> "<spec>" ["<spec>" ...]
set -u
cd "$(dirname "$0")"
GPU="${1:?gpu}"; RLOG="${2:?run log}"; WAIT="${3:?tag}"; shift 3
echo "[chain_after gpu$GPU $(date -Is)] waiting for DONE $WAIT in $RLOG"
while ! grep -q "DONE $WAIT" "$RLOG" 2>/dev/null; do sleep 120; done
echo "[chain_after gpu$GPU $(date -Is)] launching: $*"
bash run_newdata.sh "$GPU" "$@"
echo "[chain_after gpu$GPU $(date -Is)] COMPLETE"
