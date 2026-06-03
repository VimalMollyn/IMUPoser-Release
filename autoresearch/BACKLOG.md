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

## User-requested models
- [x] Transformer (TIP) + deeper(8L)
- [x] Diffusion (EgoEgo) + deeper(8L); 1-step sampling best
- [x] AvatarPoser (transformer + IK)
- [ ] **1D-CNN (TCN)** — exp28/29 RUNNING (seeds 1,2)
- [ ] **AI4Animation SIGGRAPH 2024** (Starke) — RESEARCH then implement. https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2024/ReadMe.md
- [ ] **arXiv 2603.04090** — fetch, extract ideas, implement the promising one.

## My idea queue (work when user queue is empty)
- [ ] **IK-on-LSTM** (LSTM + IK loss) — exp26/27 DONE, evaluating. If it wins → new best, confirms IK transferable.
- [ ] **Acceleration-consistency loss**: enforce predicted pose's synthetic acc (finite-diff of FK joint
      positions ×3600/scale) matches the OBSERVED IMU acc at sensor joints. Uses the acc half of the IMU
      signal (IK currently uses only orientation). Strong physical constraint. → ACC_LOSS flag on LSTM.
- [ ] **IK weight sweep** AP_IK_W ∈ {0.5, 2, 4} on the best base (LSTM+IK) — optimize the lever.
- [ ] **IK + acc consistency combined** on LSTM (best of both).
- [ ] **CNN + IK** if CNN is competitive.
- [ ] **Deterministic LSTM** (disable CuDNN nondeterminism / torch.use_deterministic_algorithms) to
      shrink the ~1.5° noise floor so finer levers become detectable; or seed-average N=3.
- [ ] **Bigger/deeper LSTM** (hidden 768/1024, 3 layers) — one capacity check.
- [ ] **Best-model combo**: LSTM + IK(+acc) at best AP_IK_W, seed-averaged → final headline number.
- [ ] **Ensemble** of LSTM+IK across seeds (average pose predictions).
- [ ] **Calibration re-sweep** with the IK loss on (does IK change the optimal calib magnitude?).
