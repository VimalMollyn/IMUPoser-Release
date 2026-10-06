#!/usr/bin/env bash
# Append fig2 queue jobs as each shard-transfer stage completes (a job must never start before its shards are all
# there). Stages are logged by the prioritised rsync in rsync_shards_fig2.log: control -> new_mocap -> gv_filtered -> gv_raw.
#   fig2_stage_handoff.sh    (run once under setsid/nohup; exits after the last hand-off)
set -u
LOG=/home/vimal/imuposer_data/logs/rsync_shards_fig2.log
FIG2="vimal@fig2.andrew.cmu.edu"; SSH="ssh -p 2222 -o BatchMode=yes"
FD=/media/vimal/Taejun4TB/vimal_imuposer/imuposer_data
K="$FD/logs"
append(){ # append <gpu> <lines...>
  local g="$1"; shift
  printf '%s\n' "$@" | $SSH "$FIG2" "cat >> $FD/queue_fig2_gpu$g.txt"
  echo "[$(date -Is)] appended to fig2 gpu$g: $*" | cut -c1-200
}
wait_for(){ until grep -q "$1" "$LOG" 2>/dev/null; do sleep 120; done; }

# mocap stage complete == the gv_filtered stage has started
wait_for "stage: gv_filtered"
append 0 "# --- new mocap data landed: mixing ratio of the new data (natural ratio = scale_s20_trt2 16.62), per-dataset ablations" \
  "mix025_s20_trt|treatment|1|EPOCHS=20 DATASET_REPEAT=BONES_=0.25,FORMHOI_=0.25,MM_=0.25,MotionX_=0.25 PRECISION=bf16-mixed" \
  "mix050_s20_trt|treatment|1|EPOCHS=20 DATASET_REPEAT=BONES_=0.5,FORMHOI_=0.5,MM_=0.5,MotionX_=0.5 PRECISION=bf16-mixed" \
  "mix200_s20_trt|treatment|1|EPOCHS=20 DATASET_REPEAT=BONES_=2,FORMHOI_=2,MM_=2,MotionX_=2 PRECISION=bf16-mixed" \
  "abl_formhoi2_s20|control+FORMHOI_|1|EPOCHS=20 PRECISION=bf16-mixed" \
  "abl_mm_s20|control+MM_,MotionX_|1|EPOCHS=20 PRECISION=bf16-mixed" \
  "abl_bonesformhoi_s20|control+BONES_,FORMHOI_|1|EPOCHS=20 PRECISION=bf16-mixed" \
  "treatment_rew_s20|treatment|1|EPOCHS=20 DATASET_REPEAT=FORMHOI_=3,Nymeria_=2,BONES_=0.5,MM_=0.5,MotionX_=0.5 PRECISION=bf16-mixed" \
  "abl_amassrest_s20|control+ACCAD,BMLhandball,DanceDB,Eyes_Japan,GRAB,LARa,MOYO,MPI_HDM05,TotalCapture,WEIZMANN,DFaust,SOMA,TCD_hand|1|EPOCHS=20 PRECISION=bf16-mixed"

# GV filtered complete == gv_raw stage has started
wait_for "stage: gv_raw"
append 0 "# --- MotionGV filtered landed: M20 data recipes on the best set (DIP-like mocap + DIP-like GV)" \
  "dlall_m20_stab|treatment+MGV_|1|TF_DMODEL=384 TF_LAYERS=6 TF_FF=1536 EPOCHS=20 GRAD_CLIP=1.0 ACC_CLAMP=160 PRECISION=bf16-mixed DATASET_KEEP=$K/diplike_keep_all_d20.json" \
  "trtgvdl_m20|treatment+MGV_|1|TF_DMODEL=384 TF_LAYERS=6 TF_FF=1536 EPOCHS=20 PRECISION=bf16-mixed DATASET_KEEP=$K/diplike_keep_mgv_d20_a22_v4.json" \
  "dlall_m20_cos|treatment+MGV_|1|TF_DMODEL=384 TF_LAYERS=6 TF_FF=1536 EPOCHS=20 LR_SCHED=cosine PRECISION=bf16-mixed DATASET_KEEP=$K/diplike_keep_all_d20.json"

wait_for "ALL STAGES DONE"
append 0 "# --- MotionGV raw landed" \
  "dlallraw_m20|treatment+MGVRAW_|1|TF_DMODEL=384 TF_LAYERS=6 TF_FF=1536 EPOCHS=20 PRECISION=bf16-mixed DATASET_KEEP=$K/diplike_keep_allraw_d20.json"
echo "[$(date -Is)] hand-off complete"
