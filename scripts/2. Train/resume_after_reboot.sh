#!/usr/bin/env bash
# Relaunch the whole experiment queue after a reboot (2026-10-02). Every base run resumes from its last epoch
# checkpoint (run_newdata.sh picks up base_<tag>/last.ckpt); finished stages are skipped (best_model.txt).
# Priority per the user: scaling grid first, then the treatment arms, then MotionGV and the extra control seeds.
#   bash scripts/2. Train/resume_after_reboot.sh          (from the newdata worktree; uses its .venv)
set -u
cd "$(dirname "$0")"
LOGS=/home/vimal/imuposer_data/logs; mkdir -p "$LOGS"
M="TF_DMODEL=384 TF_LAYERS=6 TF_FF=1536"
L="TF_DMODEL=512 TF_LAYERS=8 TF_FF=2048 BATCH_SIZE=128 ACCUM=2"
XL="TF_DMODEL=768 TF_LAYERS=8 TF_FF=3072 BATCH_SIZE=64 ACCUM=4"

# GPU0 (TITAN V, fast)
setsid nohup bash run_newdata.sh 0 \
  "scale_l20_ctrl|control|1|$L EPOCHS=20" \
  "scale_m60_ctrl|control|1|$M" \
  "scale_xl20_ctrl|control|1|$XL EPOCHS=20" \
  "scale_m20_trt|treatment|1|$M EPOCHS=20" \
  "treatment_s1|treatment|1" \
  "treatment_gv_s1|treatment_gv|1" \
  "control_s2|control|2" "control_s3|control|3" \
  > "$LOGS/run_resume_gpu0.log" 2>&1 &

# GPU1 (TITAN X Pascal, ~1.8x slower; fewer loader workers)
NUM_WORKERS=4 setsid nohup bash run_newdata.sh 1 \
  "scale_s60_cur|curated|1" \
  "scale_m60_cur|curated|1|$M" \
  "scale_l60_cur|curated|1|$L" \
  "scale_s20_trt|treatment|1|EPOCHS=20" \
  "treatment_s2|treatment|2" \
  "treatment_s3|treatment|3" \
  > "$LOGS/run_resume_gpu1.log" 2>&1 &
sleep 2
echo "launched $(pgrep -f 'run_newdata.sh' | wc -l) runners; logs: $LOGS/run_resume_gpu{0,1}.log"
echo "results page: uv run python 'scripts/3. Evaluation/newdata_report.py'  then publish autoresearch/newdata_report.html"
