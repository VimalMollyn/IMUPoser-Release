# AutoResearch backlog — ALL-DAY continuous run (2026-06-03), keep BOTH GPUs busy, never idle

## WHIP thread (2026-07-09) — QUEUED (user: "do this later")
- [~] IN PROGRESS: retarget WHIP real IMU -> SMPL, fine-tune lw_rp_h on DIP+WHIP, eval dip_test (beat 18.37?).
- [ ] **Joint-position IMUPoser (user idea).** Retrain IMUPoser to predict 3D JOINT POSITIONS directly
      (like WHIP; MPJPE loss/metric) instead of SMPL rotations. Then compare: does adding WHIP data help in
      that rotation-free formulation? Key advantage — **no mocap->SMPL retargeting needed for WHIP's GT**
      (WHIP joints_3D are native), so it removes the retargeting error and gives an apples-to-apples
      comparison on WHIP's own benchmark. Needs: joint-position head + loss on IMUPoser, a joints target in
      the dataset (we already have joint FK), and a shared joint set between DIP-SMPL-joints and WHIP-69-joints.
      Compare DIP-only vs DIP+WHIP on both dip_test-joints and whip_test MPJPE.


Best recipe env: `TRAIN_COMBO=lw_rp_h AUG_CALIB_RAD=0.12217 VAL_FILES=dip_train.pt` + curated-12
`TRAIN_DATASETS=CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits`.
Baselines: plain LSTM val SIP 26.79(s1)/26.93(s2); noise floor ~1.5° → seed-replicate wins.
Eval finished + launch next BEFORE evaluating, so GPUs never idle.

## Established (don't redo)
calibration aug −4.3° (saturates 7–10°); AvatarPoser IK loss Angle −1.2°/Joint−0.3cm (transformer-only,
generalizes to test). NOT helping: transformer/diffusion/CNN/codebook/deeper, recon/staged/IK-on-LSTM,
accel-consistency(hurts), physics-refine(neutral), translation(hurts), activity(neutral).

## DATA-REALISM thread (primary — the approved lever)
- [~] gyro-drift sweep AUG_GYRO_RW: 0.01, 0.02 RUNNING (exp45/46). then 0.005, 0.04.
- [ ] **ESKF tilt-correction gyro** (AUG_GYRO + correct pitch/roll toward true gravity, leave YAW drifting
      — the realistic behavior; all-axis drift over-perturbs tilt). NEEDS CODE.
- [ ] **accel realism**: per-sensor accel bias + scale-factor error + white noise (AUG_ACC_BIAS/SCALE/N).
      NEEDS CODE (have AUG_ACC_STD noise only).
- [ ] **per-sensor heterogeneous calibration** (different calib magnitude per sensor; real sensors differ).
- [ ] **full realistic combo**: best calib + best gyro-drift + accel realism together (the full IMU model).
- [ ] seed-replicate the best realism config (2–3 seeds), stack on AvatarPoser, eval on test.
- [ ] **per-sequence ESKF data-regen** (heaviest, most faithful — full-sequence drift) IF on-the-fly wins.

## OTHER untested levers
- [ ] **combo-dropout training** (random combo/window vs all-24; regularization + ~25× faster epochs ->
      more epochs). NEEDS dataset code (doc TODO).
- [ ] **loss-function variants**: geodesic rotation loss; SIP-weighted loss (upweight limb-root joints
      1,2,16,17). NEEDS model code.
- [ ] **bigger/deeper LSTM** (hidden 768/1024, 3 layers) — capacity check. NEEDS RNN hidden-size knob.
- [ ] **seed-ensemble at eval** (average the 3-seed LSTM / AvatarPoser pose predictions) — cheap, eval-only.
- [ ] **window length** sweep (max_sample_len 150/300/600) — temporal context.
- [ ] **output representation**: r6d vs quaternion vs axis-angle.
- [ ] **noise curriculum** (anneal augmentation strength over epochs).
- [ ] LSTM + light self-attention hybrid.

## Capstone
- [ ] best-of-everything model, 3-seed, final dip_test number + updated graph.

## NEW threads (2026-06-03 afternoon, user ideas)
- [~] **5-IMU teacher -> 3-IMU DISTILLATION** (DISTILL_TEACHER + AUX_TARGET=imu). Train teacher (TRAIN_COMBO=global)
      first, then distilled student (lw_rp_h). HIGH priority.
- [ ] **value/uncertainty-weighted distillation** (teacher confidence weights the distill loss) = the
      tractable version of the "RL teacher provides value" idea. (RL proper is high-variance for this
      regression; distillation is the supervised equivalent.)
- [ ] **accel realism** AUG_ACC_BIAS/SCALE (other IMU-signal axis) — ready.
- [ ] **LR / optimizer / weight-decay / cosine-sched sweeps** (LR, OPTIMIZER=adam/adamw/sgd/radam,
      WEIGHT_DECAY, LR_SCHED=cosine) — ready, env-only.
- [ ] **bigger LSTM** (LSTM_HIDDEN 768/1024) — exp49/50 running.
