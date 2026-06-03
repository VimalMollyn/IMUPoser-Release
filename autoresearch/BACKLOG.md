# AutoResearch backlog (continuous autonomous run, started 2026-06-03 ~01:00)

Durable queue so the 8-hour run survives context compaction. Mark items `[x]` when logged in
results.jsonl. Keep BOTH GPUs busy; commit+push + update graph/doc every experiment. Noise floor
~1.5° SIP → seed-replicate (≥2 seeds), compare to seed-matched LSTM (26.79 s1 / 26.93 s2).

Best recipe env: `TRAIN_COMBO=lw_rp_h AUG_CALIB_RAD=0.12217 AUG_DRIFT_RAD_S=0.05 VAL_FILES=dip_train.pt`
+ curated-12 `TRAIN_DATASETS=CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits`.

## Established (don't redo)
- calibration-error aug −4.3° (the big lever). curation/drift within noise.
- AvatarPoser (transformer+IK loss) −0.34° vs LSTM → win is the **IK orientation-consistency loss**.
- recon/staged ~0; transformer & diffusion & deeper variants all ≥ LSTM (worse). depth ~ mild overfit.

## User-requested models / ideas
- [x] Transformer (TIP) + deeper(8L) — ≥ LSTM (worse)
- [x] Diffusion (EgoEgo) + deeper(8L); 1-step sampling best — ≥ LSTM (worse)
- [x] AvatarPoser (transformer + IK) — beats LSTM (the IK loss, not attention)
- [x] IK-on-LSTM (exp26/27) — inconsistent (+0.88/-0.47); IK is transformer-specific, doesn't transfer
- [ ] **1D-CNN (TCN)** — exp28/29 RUNNING (seeds 1,2)
- [ ] **CNN + IK** (only if CNN competitive)
- [ ] **AI4Animation SIGGRAPH 2024** = "Categorical Codebook Matching" (VQ codebook + MLP, 3-pt VR char
      control). Transferable = discrete VQ pose codebook / categorical pose prior. MED effort. LOW-MED fit
      (it's a generative controller, not an estimator).
- [ ] **arXiv 2603.04090** = "EgoPoseFormer v2" (egocentric VIDEO pose; different domain). Transferable =
      uncertainty-weighted loss, temporal smoothing. LOW fit — deprioritize; maybe just the smoothing.
- [ ] **Physics refinement (TransPose/PIP/PNP)**: full PIP simulator is impractical here AND our metric is
      root-relative rotation (physics mainly fixes global translation/foot-skate). Tractable, metric-relevant
      pieces: (a) ACCEL-CONSISTENCY loss, (b) eval-time temporal smoothing. See idea queue.
- [ ] **Estimate translation** (user): add translation/velocity as a multi-task AUX output+loss. Metric is
      root-relative so it can only help via shared features. → TRANS_LOSS flag on LSTM (use fdata["tran"]).

## My idea queue
- [ ] **Acceleration-consistency loss** (ACC_LOSS, physics-informed): predicted pose's synthetic acc
      (finite-diff of FK joint positions ×3600 / acc_scale) must match OBSERVED IMU acc at sensor joints.
      Uses the accelerometer half of the signal (IK uses only orientation). HIGH priority — most novel lever.
- [ ] **IK weight sweep** AP_IK_W ∈ {0.5,2,4} on AvatarPoser (the IK win) — optimize it.
- [ ] **Deterministic LSTM** / seed-average N=3 — shrink the ~1.5° noise floor so finer levers show.
- [ ] **Bigger/deeper LSTM** (hidden 768/1024 or 3 layers) — one capacity check.
- [ ] **Best-model combo + seed-average** → final headline number.
- [ ] **Calibration re-sweep** (does the optimal calib magnitude change with new losses?).
NOTE: best backbone is the LSTM. Most promising untried levers are physics/sensor-consistency losses
(accel-consistency, translation aux) ON the LSTM, and the VQ codebook model.
