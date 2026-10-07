#!/usr/bin/env bash
# Pull the small per-run files of fig2's checkpoints/newdata into this machine's checkpoints/newdata every 10 min
# (train.log, best_model.txt, eval_dip_test.log, ft_meta.json, snap_*.txt; never the .ckpt files), so the report
# and the results log see fig2's runs as if they were local. Fetch a specific last.ckpt by hand when it is needed
# for an ensemble. Also mirrors fig2's queue logs (queue_fig2_gpu*.log, distinct names so nothing local is overwritten).
#   mirror_fig2.sh            (loops forever; run under setsid/nohup)
set -u
FIG2="vimal@fig2.andrew.cmu.edu"
SSH="ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=20"
SRC="$FIG2:/media/vimal/Taejun4TB/vimal_imuposer/IMUPoser-Release/checkpoints/newdata/"
DST="$(cd "$(dirname "$0")/../.." && pwd)/checkpoints/newdata/"
LOGS=/home/vimal/imuposer_data/logs
while true; do
  rsync -a --update -e "$SSH" --include='*/' --include='train.log' --include='best_model.txt' --include='eval_dip_test.log' \
        --include='ft_meta.json' --include='snap_*.txt' --exclude='*' --prune-empty-dirs \
        "$SRC" "$DST" 2>>"$LOGS/mirror_fig2.err" || echo "[$(date -Is)] rsync failed" >> "$LOGS/mirror_fig2.err"
  # no remote globs: fig2's login shell is fish, which aborts on an unmatched wildcard; filter on the receiving side
  rsync -a -e "$SSH" --include='queue_fig2_gpu*.log' --exclude='*' \
        "$FIG2:/media/vimal/Taejun4TB/vimal_imuposer/imuposer_data/logs/" "$LOGS/" 2>>"$LOGS/mirror_fig2.err"
  [ -n "${ONCE:-}" ] && exit 0
  sleep 600
done
