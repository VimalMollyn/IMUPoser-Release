#!/usr/bin/env bash
# Run ON fig2: start one file-driven queue runner per GPU with the data root on the 4 TB drive.
# Queue files: /media/vimal/Taejun4TB/vimal_imuposer/imuposer_data/queue_fig2_gpu{0,1}.txt (edit/append any time).
# Logs: .../imuposer_data/logs/queue_fig2_gpu{0,1}.log (mirrored to the main machine by mirror_fig2.sh).
set -u
cd "$(dirname "$0")"
export IMUPOSER_DATA_DIR=/media/vimal/Taejun4TB/vimal_imuposer/imuposer_data
# the 4 TB drive is NTFS (no symlinks): the uv environment lives on the root disk, the project/data/checkpoints on the drive
export UV_PROJECT_ENVIRONMENT=/home/vimal/venvs/imuposer-newdata
D="$IMUPOSER_DATA_DIR"; mkdir -p "$D/logs"
for g in 0 1; do
  touch "$D/queue_fig2_gpu$g.txt"
  if pgrep -f "queue_runner.sh $g $D/queue_fig2_gpu$g.txt" > /dev/null; then echo "runner for gpu$g already running"; continue; fi
  setsid nohup bash queue_runner.sh "$g" "$D/queue_fig2_gpu$g.txt" >> "$D/logs/queue_fig2_gpu$g.log" 2>&1 &
  echo "started runner gpu$g (pid $!)"
done
