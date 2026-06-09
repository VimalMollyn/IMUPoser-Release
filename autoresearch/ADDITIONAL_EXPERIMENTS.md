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

## DIP fine-tuning — the real unlock (protocol change, user-directed)
After exhausting AMASS-only levers (ceiling ~24.6 ensemble), fine-tuning on **real DIP** crossed the
sim-to-real gap that capped everything. Done safely: split `dip_train` (41 seqs) → `ftrain` (32) +
`fval` (9) by sequence; warm-start an AMASS model (`CONTINUE_FROM`), train on `ftrain`, select on `fval`,
**`dip_test` (s09/s10) fully held out**. Fine-tuning is tiny (~1 min/model).

| stage | SIP | Angle |
|---|---|---|
| AMASS-only single | 26.09 | 22.28 |
| AMASS-only ensemble (old best) | 24.63 | 20.85 |
| FT single (LSTM / AvatarPoser) | 19.85 / 19.09 | 18.98 / 18.0 |
| **FT ensemble (curated 4 avatar + 2 LSTM)** | **18.59** | **17.58** |

**−6.0° SIP off the old deliverable.** AvatarPoser fine-tunes best; calib aug during FT helps slightly;
ensemble saturates ~18.5. This single lever dwarfs every AMASS-only finding combined.

## Offline optimization — analysis-by-synthesis (user-directed: "no realtime constraint, best fit ever")
Built `scripts/3. Evaluation/offline_fit.py`: per-sequence offline fit that initialises from the FT
ensemble and uses ONLY the IMU (never GT) — fits the FK global orientations of the sensed bones
(lw→joint 18, rp→joint 2, h→joint 15) to the measured DIP orientations, with temporal smoothness, a
network-prior anchor, optional per-sensor calibration, plus a parameter-free **measurement injection**
(overwrite a sensed bone's global ori with its measurement via IK) and a **Wahba re-root** (re-estimate
the pelvis from the sensed measurements). Acceleration term dropped: 25 fps finite-difference accel is
aliasing-dominated and all metrics are root-relative (translation unscored).

**Result — offline post-hoc optimization does NOT beat the trained ensemble (SIP neutral).** Tuned on
held-out `fval`, confirmed on `dip_test`: every variant lands at ΔSIP ≈ 0 (injection +0.05, reroot +0.03,
iterative +0.12), with only a tiny ΔMPJRE ≈ −0.15 (the directly-sensed joints get nudged).

Why (the diagnosis that matters): the network's error on the **directly measured** joints is already
*worse than the raw measurement* (joint 18 net 12° vs meas 1.2°; joint 15 net 11° vs 0.9°; joint 2 net
10.5° vs 5.1°) — the feed-forward net degrades signals it gets as input — **but you cannot exploit this**
because (a) the metric is root-relative and zeros the pelvis, while measurements are world-frame, so
injecting a world ori onto the network's pelvis just re-expresses the network's ~11° pelvis error; (b) the
pelvis is **un-sensed** (sensors on left-forearm / right-thigh / head) and can't be recovered better than
the network — a Wahba estimate from the net's own relative pose is circular. An **oracle** with the *true*
pelvis only reaches SIP 15.36 on fval (−1.2), and even then the un-sensed shoulders get *worse* (the net's
pose is internally consistent with its own pelvis). The dominant error is the irreducible un-sensed right
arm (relb/rsho 33/26°) + left hip — no measurement, conditional-mean-optimal. **18.x SIP is the floor for
lw_rp_h; offline fitting confirms it rather than beating it.** The offline fitter is kept as a tool and a
rigorous negative result.

## Data lever — full dip_train fine-tune (NEW BEST 18.37)
The steepest lever is still *real-data realism*. The FT ensemble trained on `ftrain` (32 seqs) and selected
on `fval` (9) — but `fval` was only a research-phase selection signal; the standard DIP split trains on all
of s01–s08. Continue-fine-tuned each of the 6 members on the **full dip_train** (41 seqs) at low LR
(5e-5, 30 ep); `dip_test` (s09/s10) stays fully held out. Single test readout (last.ckpt, apples-to-apples):

| metric | old (ftrain) | **new (full dip_train)** |
|---|---|---|
| SIP   | 18.60 | **18.37** |
| MPJRE | 17.56 | 17.36 |
| MPJPE | 8.13  | 8.03 |
| MPVPE | 9.56  | 9.49 |
| MPJVE | 29.33 | 28.84 |

Consistent −0.2–0.5 across every metric from +9 real seqs → the FT regime is still real-data-limited
(suggests more real IMU, e.g. TotalCapture, as the next lever). New deliverable arc: 24.6 AMASS-ens →
18.59 FT-ens → **18.37 full-dip_train FT-ens**.

## Bottom line
Single-model lw_rp_h SIP ≈ **26.0** (clean specialist; sipw4 indistinguishable within noise). Ensemble
deliverable **~24.6** — unchanged this session. The established levers (calib + IK + ensemble) all predate
this session; nothing new beat the noise floor. Honest net contribution of this session: the **plausibility
study** (a rigorous proof that plausibility ⊥ accuracy for scored un-sensed limbs, a reusable plausibility
metric, and the NN-retrieval recommendation) — a real *negative* result. Further metric gains need a
changed setup (more sensors, real-data fine-tuning, or a plausibility-aware objective/eval).
