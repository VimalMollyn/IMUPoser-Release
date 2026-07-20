# Nymeria for sparse-IMU pose (lw_rp_h) — results

**Status: POSITIVE — NEW BEST-EVER (18.27 dip_test SIP, beats prior 18.37).**
Adding 242.6 h of NymeriaPlus everyday in-the-wild motion to AMASS pretraining improves the
`lw_rp_h` DIP/SIP deliverable by **~0.34° SIP** (Welch p=0.019, 3 seeds), the first clean positive
data lever after a long run of neutral/negative ones. It is the mirror image of the WHIP result.

Last updated 2026-07-20. Data files: `autoresearch/results.jsonl` (raw), checkpoints under
`checkpoints/nymeria/`.

---

## 1. Motivation — why Nymeria, right after WHIP failed

WHIP (real wearable IMU + mocap) was integrated first and lost on **every** arm — cleanly and
monotonically (dip_test SIP 19.20 → 19.73 → 20.68 → 21.15 as more WHIP was added; WHIP-pretrain→FT
also 19.6). The diagnosis was **distribution, not quality**: WHIP's dynamic sports motion is 2.3× more
pose-diverse than DIP's everyday activity (own-mean pose diversity 50.5° vs DIP 32.5°; distance from
DIP's mean pose 60.6°). Training on it dragged the model off DIP's motion manifold.

That diagnosis makes a prediction: a real-world dataset of **everyday** motion, matched to DIP's
distribution but at large scale, should help where WHIP couldn't. NymeriaPlus is exactly that —
1100 sequences × ~15 min = ~275 h of cooking / cleaning / working / walking in real homes, captured
with head + wrist devices and XSens body mocap, released as native SMPL.

## 2. Data acquisition — `scripts/1. Preprocessing/nymeria_fetch.py`

- **Native SMPL, zero retargeting.** `body/xdata_smpl_neutral.npz` = `global_orient (T,3)` +
  `body_pose (T,69)` = our exact 24-joint axis-angle layout, at 240 fps (uniform-resampled to 60).
  This removes the single biggest risk from the WHIP work (fitting SMPL to 69-joint mocap).
- **Frame verified empirically** (not assumed): head sits +1.573 m above the feet on **z** → Nymeria
  is AMASS's z-up frame, and the existing `amass_rot` maps it to DIP's y-up (+1.573 on y).
- **Range-fetch trick.** `body_processed` is a 320 MB zip of `[xdata_mhr.glb 260 MB (useless),
  xdata_smpl_neutral.npz 60 MB (wanted)]`. fbcdn serves HTTP 206, so we read each remote zip's
  central directory and stream only the npz member → **~19 % of the bytes** (66 GB not 350 GB), and
  the raw is never stored (parsed in memory, only synthesized tensors kept).
- **Full set fetched:** 242.6 h across 91 chunks (23 seqs skipped for missing SMPL), 19.4 GB down in
  ~14 min per 400-seq round (4 shards). 60 fps intermediates reclaimed after 25 fps conversion; the
  25 fps training set is 28 GB.

## 3. QC — the data is clean and DIP-matched

Root-relative pose diversity (the metric that must be root-relative; raw global rotations measure
heading spread and made Nymeria look 2× worse than WHIP until corrected — the same heading-invariance
trap as the WHIP 59° zero-shot scare):

| dataset | own-mean diversity | distance from DIP | GT jerk | accel mean / p95 | tran range |
|---|---|---|---|---|---|
| dip_train | 32.5° | 32.5° | 93.0 | 1.94 / 7.52 (REAL) | 0.0 |
| dip_test | 30.9° | 32.7° | 108.6 | 2.88 / 11.65 (REAL) | 0.0 |
| WHIP | 50.5° | **60.6°** | — | 1.28 / 4.14 | — |
| CMU | — | — | — | 3.81 / 10.76 | 6.8 m |
| BioMotionLab | — | — | — | 3.53 / 9.28 | 5.1 m |
| **Nymeria** | **28.5–29.7°** | **42.2–42.6°** | 79.5 | 3.67 / 5.45 | 26.8 m |

Nymeria's diversity ≈ DIP's (WHIP was far higher), it is 18° closer to DIP than WHIP, its GT jerk is
*lower* than DIP (smoother everyday motion, not jitter), and its accel sits between CMU and BML —
in-family with what we already train on.

## 4. Experimental design

Two-stage, single variable. **Nymeria goes in PRETRAIN, not fine-tune** — it is synthetic-IMU motion
like AMASS, so its value is motion *distribution*; the FT stage exists to adapt to real IMU noise
(injecting synthetic data there is part of why WHIP lost). Both bases trained here rather than reusing
the old exp21 base, whose `TRAIN_DATASETS` was never recorded — reusing it would confound "Nymeria
added" with "data changed".

| stage | control | treatment |
|---|---|---|
| 1. pretrain (AvatarPoser, 60 ep, AUG_CALIB_RAD=0.122) | curated-12 AMASS | curated-12 **+ Nymeria** |
| 2. finetune (60 ep on real DIP) | ftrain | ftrain |
| eval | dip_test (s09/s10, held out) | dip_test |

`dip_test` (s09/s10) held out throughout. Scripts: `run_nymeria_base.sh` (arms: curated / nym80 /
nymeria=full), `chain_nym_ft.sh` (stage-2 FT + eval), `offline_fit.py --iters 0` (the eval).

## 5. Results — dose-response (dip_test SIP, lower = better)

| Nymeria dose | seeds | mean SIP | note |
|---|---|---|---|
| **0 h** (control, curated-12) | 19.03, 19.15, 18.90, 19.29 | **19.09 ± 0.17** | n=4; strong control (old DIP-only was 19.20) |
| **80.9 h** (nym80) | 18.91, 19.07 | **18.99** | n=2; straddles the control mean → NEUTRAL |
| **242.6 h** (nymfull, all 1100) | 18.83, 18.76, 18.66 | **18.75 ± 0.085** | n=3; **all below control min**, tight |
| **242.6 h, 3-seed ENSEMBLE** | — | **18.27** | **NEW BEST-EVER** (prior 18.37 mixed ensemble) |

- **Monotonic in the means: 19.09 → 18.99 → 18.75.** Opposite sign to WHIP (19.20 → 20.68 → 21.15).
- **Full dose is significant:** gap **0.34°**, **Welch p = 0.019** (n=3 vs n=4); all three full-dose
  seeds fall below the control **minimum** (18.90), and they cluster tightly (σ=0.085 vs control 0.17).
- **80.9 h is not enough** — the mid dose is neutral. The effect requires the full 242.6 h. This is a
  proper dose-response, and it explains why the very first 80.9 h single-seed point (18.91) looked
  only marginal.

### Controls that make this believable
- **Compute is not the cause.** A compute-matched control (curated-12 × 174 epochs = same 21k gradient
  steps as nymfull × 60) selects the *same* ep-49 checkpoint (val 0.03580); ep70/ep83 are worse.
  curated-12 converges by ~ep49, so the treatment's extra steps buy nothing.
- **Pipeline is deterministic** (same seed+data → identical checkpoint → identical 19.03), and the
  reproduced control base matches the original exp21 (val 0.03580 vs 0.03638).

## 6. Interpretation

Modest (~0.34° SIP) but **real, reproducible, and dose-dependent**. It confirms the WHIP diagnosis:
real-world motion helps the DIP deliverable *when its distribution matches DIP and there is enough of
it*. WHIP had the wrong distribution; Nymeria has the right one at scale.

Caveats kept honest: the effect is small; a single Nymeria model (mean 18.75) does not by itself beat
the prior best **ensemble** (18.37 — curated 4 avatar + 2 LSTM); Nymeria's GT is XSens (IMU-based
mocap with its own drift), not optical. The natural next step is to ensemble the Nymeria seeds and/or
fold Nymeria pretraining into the full ensemble recipe to try to push past 18.37.

## 6b. Ensemble distillation — can 18.27 fit in ONE model? NO.

The 18.27 needs 3 transformers at inference (defeats real-time). Tried to compress it into a single
model. **It does not distill** — the ensemble gain is inference-time output-variance reduction
(member disagreement on unseen inputs), which a single forward pass cannot reproduce on held-out data.

| approach | dip_test SIP | |
|---|---|---|
| 3-model ensemble (target) | **18.27** | needs 3x inference |
| weight-average ("model soup") | 35.84 | BROKEN — seeds in different basins (rel weight dist ~1.5) |
| distill: blend (ensemble + GT) | 18.81 | = single-model |
| distill: pure imitation | 18.83 | = single-model |
| distill: heavy aug (0.2) | 19.05 | worse (over-perturbed inputs) |
| *single Nymeria model (ref)* | *18.75* | deployable at 1x |

**Deployment-cost curve (ensemble gain saturates fast):**
| inference cost | dip_test SIP |
|---|---|
| 1 model (real-time) | 18.75 |
| 2 models | 18.37–18.44 |
| 3 models | 18.27 |

Bottom line: the **Nymeria pretraining gain (19.09 → 18.75) survives fully in one model at 1x cost**
— that is the deployable win. The extra ensemble bump (18.75 → 18.27) needs N models and is for
offline use only. `DISTILL_ENSEMBLE`/`DISTILL_ONLY` hook added to AvatarPoserModel.

## 7. Next steps

- [x] Ensemble the 3 nymfull seeds → dip_test = **18.27 SIP, a NEW BEST** (beats prior 18.37; vs
      18.75 single-model mean). Just 3 same-arch AvatarPoser seeds, all Nymeria-pretrained.
- [ ] nymfull seed 4–5 to tighten p, and one more nym80 seed to confirm mid-dose neutrality.
- [x] Beat 18.37 already (3-seed nymfull ensemble 18.27). Distillation to 1 model: does NOT recover it (see 6b).
- [ ] Optional: LSTM (not just AvatarPoser) Nymeria arm, since the 18.37 ensemble mixed both.
- [ ] (Backlog, user idea) joint-position IMUPoser trained on Nymeria's native SMPL joints.
