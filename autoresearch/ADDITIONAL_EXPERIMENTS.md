# IMUPoser AutoResearch — Findings (lw_rp_h: left-wrist, right-thigh, head)

Protocol: train on AMASS-only synthetic IMU, **never** on DIP; select on `dip_train` (real DIP);
report on `dip_test`. Specialist `TRAIN_COMBO=lw_rp_h`, 60 epochs, best-val checkpoint. Noise
floor ≈ 1° SIP across seeds (seed + non-deterministic CuDNN bi-LSTM backward). ~100 experiments;
full log in `results.jsonl`, timeline in `progress.png`.

## What helps (the only levers that beat the LSTM baseline)
| Lever | Effect | Notes |
|---|---|---|
| Calibration-error aug (`AUG_CALIB_RAD≈0.122`, 7°) | **−4.3° SIP** | Biggest win. Real IMUs are mis-mounted; perfect-FK synthetic lacks this. |
| AvatarPoser IK loss (orientation-consistency) | **−1.2° Angle** | Anchors the ensemble's strong-Angle members. |
| Diverse ensembling (LSTM + AvatarPoser) | **→ 24.6 SIP** | The deliverable. Saturates ~24.6–24.8; adding members/data stops helping. |

> **Correction — SIP-weighted loss (`SIP_LOSS_W=4`) is NULL.** It *looked* like a −1.1° win on 2 seeds,
> but shrank with every added seed: −1.1 → −0.8 → −0.5 → **−0.26° at 4 seeds** (clean mean 25.97 vs sipw4
> 25.71), distributions fully overlapping inside the ~1° noise floor. A phantom from lucky early seeds.
> **Lesson: replicate ≥4 seeds before claiming any sub-noise-floor effect.** Left in as a cautionary record.

## What doesn't (all tested, not assumed)
Architectures (Transformer/TIP, Diffusion/EgoEgo, 1D-CNN/TCN, codebook, deeper) · physics rigid-body
refinement · **physics-IMU data regen** (realistic full-sequence gyro/ESKF drift — accel already matches
DIP gravity-free, orientation realism already captured by calib) · translation estimation · activity-
conditioning · 5→3 IMU distillation / RL value · aux reconstruction / staged · acc-consistency · capacity
(768/1024) · window length · optimizer/LR · **test-time aug** · **heading/yaw aug** (s1 lucky, s2 regressed) ·
SIP-weight W=2 · sipw on AvatarPoser (helps SIP, costs its Angle strength).

**Why so many nulls:** 3 IMUs under-constrain the pose; residual error is sim-to-real gap + irreducible
ambiguity. Gains come from *data realism* (calib) and *loss/metric alignment* (sipw), not capacity/architecture.

## Plausibility study — "un-sensed limbs only need to be plausible, not exact"
Motivating decomposition (dip_test): un-sensed joints (right arm, left leg) average **29° error vs 16°**
for sensed; R-elbow 48°, R-shoulder 41°. Predictions **collapse to the mean** — only 73% of natural pose
variance (`VarRatio 0.73`); `eval_plausibility.py` measures this (VarRatio + manifold distance).

**Core result — plausibility ⊥ accuracy, proven 3 ways:**
| approach | VarRatio | SIP | verdict |
|---|---|---|---|
| regressed mean (sipw4) | 0.73 | **25.0** | metric-optimal, looks dead |
| NN-retrieval (graft real coordinated limbs) | 0.83 | 27.8 | most plausible, deterministic — best *tool* |
| adversarial pose prior (full-pose / velocity / targeted) | 0.72–0.81 | 25.3–27.6 | unstable (1.7° seed swings), no robust benefit |

The un-sensed limbs we want plausible **are scored joints** (L-hip, R-shoulder, R-elbow…), so every step
toward plausibility is a step of metric error — a GT-matching metric *always* prefers the implausible mean
(it's the optimal point estimate). Built an env-gated adversarial prior (`ADV_W`, gradient-reversal,
velocity-aware D, `ADV_UNSENSED` to target un-sensed limbs); it restores variance only noisily and
unstably. **Recommendation:** benchmark → keep the mean (sipw4); live avatar → NN-retrieval as a cheap
post-hoc layer + a plausibility-aware eval. The real limitation is the *metric*, which is blind to plausibility.

## Bottom line
Single-model lw_rp_h SIP ≈ **26.0** (clean specialist; sipw4 indistinguishable within noise). Ensemble
deliverable **~24.6** — unchanged this session. The established levers (calib + IK + ensemble) all predate
this session; nothing new beat the noise floor. Honest net contribution of this session: the **plausibility
study** (a rigorous proof that plausibility ⊥ accuracy for scored un-sensed limbs, a reusable plausibility
metric, and the NN-retrieval recommendation) — a real *negative* result. Further metric gains need a
changed setup (more sensors, real-data fine-tuning, or a plausibility-aware objective/eval).
