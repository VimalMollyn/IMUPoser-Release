# Additional / follow-up experiments

Running list of experiments to do later, with rationale. Add results inline as they're run.

## TODO

### Random combo dropout (vs. full combo enumeration)
**Why:** Both IMUPoser and MobilePoser train the generalist by **materializing every window once per
IMU combo** (windows × N_combos). We do the same, just lazily — `GlobalModelDataset.__getitem__`
maps `idx -> (window_idx, combo_idx)` over *all* 25 combos, so each epoch iterates the full
windows×25 product. An alternative is **combination dropout**: per window (per epoch), sample a
*single* random combo (mask out the rest) instead of enumerating all of them.

Potential benefits:
- **~25× smaller epochs / less compute & memory** for the same coverage in expectation (each window
  still sees every combo, just spread across epochs rather than all at once).
- **More stochasticity → regularization.** Could reduce the train≪val overfitting gap we observed
  (train 0.011 vs val 0.028), and might generalize as well or better than deterministic enumeration.
- Neither MobilePoser nor IMUPoser does this — it's an untested-but-natural variant.

**Experiment:** add a `config.combo_dropout` flag; when set, `__getitem__` indexes by window only and
draws a random combo each access (optionally weighted toward realistic configs). Keep val/test
deterministic (still eval per-combo as now). Train the generalist this way, compare per-combo DIP error
(esp. `lw_rp_h`, `lw_rw_lp_rp_h`) to the enumerated generalist. Also compare wall-clock/epoch.

**Setup notes:** small change to `GlobalModelDataset` (length = num_windows, random combo per
`__getitem__` via a per-epoch RNG); reuse `1. Train Global Model.py`. Watch that epochs are now ~25×
shorter, so scale `EPOCHS` up accordingly for matched total updates.

### Staged / cascaded prediction with intermediate supervision (TransPose / MobilePoser style)
**Why:** Our current model is a *single* LSTM that regresses r6d pose **directly** from IMU (with only
an auxiliary FK joint-position *loss* — the joints are never fed forward). TransPose instead decomposes
the mapping into easier sub-problems with explicit intermediate stages and supervision:
`IMU → leaf-joint positions → all-joint positions → joint rotations` (+ a separate translation branch).
PIP and MobilePoser use the same idea (MobilePoser's separate "modules" combined via `combiner.py`).
Each stage is a better-conditioned sub-task with its own gradient signal, and the explicit intermediate
*inputs* (predicted joint positions) make the final rotation regression much easier than going straight
from sparse IMU → 24×6D pose. This is one of the biggest architectural levers in the literature and we
haven't tried it — worth testing whether it closes part of our error, especially for sparse combos
where the direct IMU→pose mapping is most ambiguous.

**Experiment:** build a cascaded model (e.g. `StagedIMUPoser`):
- **S1:** IMU (acc+ori) → joint positions. (Targets already exist — `fdata['joint']` in the 25fps files.)
- **S2:** IMU + S1 joint positions → SMPL pose (r6d).
- Optionally a TransPose-style leaf→full split (S0: leaf joints, S1: all joints, S2: rotations).
- Supervise every stage; feed predicted intermediates forward (consider teacher-forcing early in
  training, then anneal to predicted).
Compare per-combo DIP error to the single-stage generalist (`lw_rp_h`, `lw_rw_lp_rp_h`).

**Setup notes:** new model class in `models/` + register in `get_model`. The FK/joint-loss machinery
already exists. Main data change: `GlobalModelDataset.__getitem__` currently returns `(input, pose)`
only — it would need to also return the **joint-position target** (load `joint.pt` / the `'joint'` key,
which the 25fps files already contain, and window it the same way as pose). Keep the same canonical
split + eval so it's comparable to all the runs above.

### Physics-simulated IMU synthesis (MuJoCo humanoid) — plausible motion + realistic IMU
**Why:** This directly targets the ceiling we identified — the **sim-to-real gap**, not data quantity.
Our synthetic IMU is purely *kinematic*: accelerations are finite-differences of SMPL vertex positions
(`acc = (v[i] + v[i+2] - 2 v[i+1]) * fps²`) and orientations come from FK. That ignores physics —
no gravity in the accelerometer signal in a body-consistent way, no contact dynamics, and it faithfully
reproduces mocap/video noise (foot sliding, jitter, penetrations), which is especially bad for the
video-recovered Motion-X data. Real IMUs measure actual rigid-body dynamics + gravity. Driving the
motion through a physics sim would (a) **synthesize more realistic IMU readings** and (b) **filter to
physically plausible motion only** — both of which should shrink the synthetic→real gap that more data,
more epochs, and specialization all failed to close.

**Experiment (two tiers):**
- *Lightweight:* load the SMPL motions onto a MuJoCo humanoid, attach `accelerometer` + `gyro` sensor
  sites at the 5 IMU body locations, and read sensor readings during **kinematic playback** (proper
  body-frame linear acceleration incl. gravity + angular velocity) instead of vertex finite-differences.
  Even without full dynamics this gives a more physically-grounded accelerometer signal.
- *Full:* use a **physics-based imitation controller** (PD/torque or an RL tracking policy à la DeepMimic
  / PHC / UHC) so the humanoid actually *tracks* the reference under gravity + contact. Motions the
  controller can't track within tolerance are physically implausible → drop or keep only the corrected
  plausible trajectory. Read IMU from the simulated bodies.

**Tooling notes:** needs a SMPL→sim humanoid + retargeting.
- *Lightweight tier:* MuJoCo with a SMPL humanoid (SMPLSim / `loco-mujoco` / custom humanoid.xml) and
  `accelerometer`/`gyro`/`framelinacc`/`frameangvel` sensor sites at the 5 IMU locations; kinematic
  playback.
- *Full tier:* **don't build the imitation controller from scratch — reuse PHC / SimXR**
  (https://github.com/ZhengyiLuo/SimXR, built on Zhengyi Luo's *Perpetual Humanoid Control* (PHC)).
  PHC already provides a **SMPL humanoid in Isaac Gym with a trained imitation policy that physically
  tracks arbitrary AMASS motion** — exactly the plausibility/dynamics step we want. Plan: run our
  AMASS+Motion-X clips through PHC (it tracks them under physics, and clips it *fails* to track are the
  physically-implausible ones → natural plausibility filter), attach IMU sensors at the 5 body sites,
  and read acc/gyro from the simulated bodies. SimXR is the same stack extended to *sparse head-sensor*
  control, so it's a close template for our sparse-IMU setting. This makes the "full tier" mostly an
  integration job rather than RL-from-scratch.
- Isaac Gym (PHC/SimXR) handles the ~40k-sequence throughput well; Genesis is a newer alternative.

**Compare:** retrain the generalist on physics-simulated IMU, eval per-combo DIP. If DIP drops while
the synthetic val barely moves, that's direct evidence the bottleneck was IMU *realism*, not motion data.

### GlobalPose-style realistic IMU synthesis (orientation is the key)
**Why:** Our baseline shows the real-DIP val is *lowest at epoch 0* and rises every epoch — the
model overfits to our **perfect FK orientation**, which real IMUs never have. GlobalPose
(Xinyu Yi, `imu_synthesis.py`) makes synthetic IMU realistic by **not** using ground-truth
orientation; instead it:
- **Calibration / mounting rotation error** per sensor: `~ randn(N,6,3) * 0.1*sqrt(pi/8)` (≈0.063
  rad ≈ 3.6°/axis), constant per sequence — the dominant real-IMU orientation imperfection.
- **Orientation via noisy gyro integration** (fast) or a full **ESKF** (`an=5e-2, wn=5e-3,
  aw=1e-4, ww=1e-5, mn=5e-3`) fusing noisy accel/gyro/mag → orientation that *drifts* like a real IMU.
- **Accel** = kinematic accel + gravity `(0,-9.8,0)` + noise (`std 5e-2`) + small random walk.
- A **T-pose calibration** pipeline (sensor↔body `RBS`) matching the real DIP calibration.
6 IMUs, 60 fps.

**Experiments (tiers):**
- *Train-time aug (in progress):* per-sensor **calibration rotation error** on orientation
  (`AUG_CALIB_RAD`, exp2), then add orientation drift + accel gravity/noise.
- *Full re-synthesis:* regenerate the dataset GlobalPose-style — gyro-integrate noisy angular
  velocity (derive ω from the pose sequence) for a drifting orientation, add gravity+noise to accel,
  inject calibration error, run the T-pose calibration. This is the most faithful "valid IMU" and
  the most likely big win, but it's a synthesis-stage rewrite (cf. `scripts/1. Preprocessing`).

Ref: https://github.com/Xinyu-Yi/GlobalPose (built on TransPose/PIP/PNP).

### Missing-IMU reconstruction as an auxiliary intermediate task
**Why:** For a sparse combo (e.g. `lw_rp_h` = 3 of 5 sensors), we currently feed **zeros** for the
absent sensors. Instead: a first stage **reconstructs the missing IMU signals** (acc+ori of the 2
absent sensors) from the present ones, and the pose model then consumes the **completed 5-IMU set**
(real present + reconstructed absent). Benefits: (a) the pose head always solves the easier
*full-sensor* problem regardless of which combo is present; (b) reconstruction is a free auxiliary
supervision — we have the full synthetic IMU for all 5 sensors, so the absent channels have ground
truth; (c) it decouples "which sensors are present" from pose regression and may regularize.

**Experiment:** two-stage model — Stage A: present IMUs → all-5 IMUs (acc+ori), supervised by the
full synthetic IMU (MSE on the held-out channels); Stage B: completed 5-IMU set → pose. Train jointly
(reconstruction loss + pose loss). Compare per-combo DIP val SIP to the direct (zeros-for-absent)
baseline. Targets exist in the data (the dataset already has all 6/5 sensors before combo-masking —
expose the unmasked sensors as the reconstruction target). Pairs naturally with the staged-pose idea
above (IMU-completion → joint positions → pose).

### Curate training datasets to DIP-relevant daily activities
**Why:** Our 29 training datasets include a lot of motion that's **off-distribution from DIP**, which
is mostly everyday/locomotion + arm motion (walking, jogging, jumping jacks, arm raises, reaching,
sitting, etc.). Datasets like Motion-X dance/`kungfu`/`music`/`perform`, MOYO (yoga), GRAB (object
grasping), DanceDB, SOMA are exotic motions DIP never contains. We already saw extra (orthogonal)
data didn't help — actively *removing* off-distribution data may help (less distribution shift) and
also trains far faster (drops Motion-X `idea400`'s 12k seqs etc.).

**Experiment:** train on a curated **everyday/locomotion** subset and compare DIP val SIP to the
full 29-set run. Implemented via `TRAIN_DATASETS=<comma list>` (restricts train to those datasets;
val/test unchanged). Proposed keep-list (classic everyday/locomotion mocap):
`CMU, BioMotionLab_NTroje, BMLmovi, KIT, EKUT, Transitions_mocap, HumanEva, SFU, HUMAN4D, SSM_synced,
MPI_mosh, MPI_Limits` — dropping Motion-X (all), MOYO, GRAB, DanceDB, SOMA, WEIZMANN, LARa. Sweep
variants (e.g. + each questionable set) to see which motion families actually help DIP.

## Parked (lower priority)

### Learning curve on the original data (data-saturation check)
_Lower priority — the saturation conclusion is already well-supported by the existing runs (more data
moved neither DIP nor sim-to-sim val), so this would mostly confirm what we expect._
**Why:** Adding the 5 newer AMASS datasets + Motion-X (~30k extra sequences) did **not** improve
DIP test error *or* the sim-to-sim validation error (held-out ACCAD/MPI_HDM05). Combined with the
loss signal (train ≈ 0.011 ≪ val ≈ 0.028, i.e. low train loss → not capacity-limited / not underfit),
this suggests the model is **not in a data-limited regime**. **Experiment:** retrain on 25/50/100% of
the original 17 datasets (sequence-level subsample, fixed seed), same canonical val/test, plot val + DIP
error vs training-set size; flat-by-50% = direct proof of saturation. Needs a `TRAIN_FRACTION` knob.

---

## Reference: results so far (DIP test, official IMUPoserEvaluator methodology)

MEAN over 24 sparse combos, best-val checkpoint, all on the same canonical split:

| Model | SIP° | Angle° | Joint cm | Vert cm |
|---|---|---|---|---|
| original IMUPoser (`results.pkl`, different test-set version — indicative) | 31.3 | 27.5 | 11.7 | 14.6 |
| canonical (29 sets: +5 new AMASS +Motion-X), ~10 epochs | 31.8 | 28.0 | 12.1 | 15.3 |
| canonical continued to ~20 epochs (overfit) | 33.2 | 29.4 | 12.9 | 16.3 |

`lw_rp_h` combo only (watch + phone-pocket + earbuds):

| Model | SIP° | Angle° | Joint cm | Vert cm |
|---|---|---|---|---|
| baseline (17 original AMASS) | 28.74 | 24.46 | 10.57 | 13.09 |
| canonical (29 sets) | 28.64 | 24.36 | 10.59 | 13.03 |

Key findings:
- More **epochs** hurt DIP (overfitting; train ≪ val).
- More **datasets** (new AMASS + Motion-X) did ~nothing on DIP *and* on sim-to-sim val.
- Early stopping (~epoch 5–10) was appropriate; fixed-50-epochs overfits.
</content>

## AutoResearch findings (lw_rp_h, AMASS-only → DIP val, fixed-epoch SIP)

- **Confirmed lever: GlobalPose-style calibration/mounting rotation error → −4.3° SIP** (32.5 → 28.2).
  Far above noise; the model was overfitting to *perfect* synthetic orientation. Magnitude saturates ~7–10°.
- **Noise floor ≈ 1.5° SIP (CORRECTED — was thought to be ~0.5°).** The earlier 0.5° estimate only
  measured GPU0-vs-GPU1 at a fixed seed. Re-running the *identical* best recipe across seeds gives SIP
  **26.79 / 26.93 / 28.11 / 28.29 — a 1.5° spread** (exp10/11/12/14). Two sources: (1) the random seed,
  and (2) the **non-deterministic CuDNN bidirectional-LSTM backward** (trainer runs `deterministic="warn"`,
  so the LSTM backward is not reproducible even at fixed seed+GPU). => treat **sub-~1.5° diffs as noise**;
  to claim a win, **seed-replicate** (≥2–3 seeds) and compare distributions, not single runs.
- Curation (DIP-relevant datasets) and orientation drift each looked like ~0.3° gains — **within the
  ~1.5° noise floor, NOT established.** Likewise the apparent "27.48 best" (exp6) was a lucky single run;
  the same recipe reaches anywhere in 26.8–28.3 depending on seed. Input-augmentation realism plateaus
  ~27–28° and the only robust lever remains calibration error.
- Key protocol fix: select on SIP-on-val (MSE val loss is anti-correlated with SIP over training); use
  fixed-epoch final model. SIP-on-val tracks DIP-test only coarsely (~+2-3° offset).

### Architecture experiments (GPU-matched controls, best recipe = calib7 + drift0.05 + curated12, 30 ep)

Each architecture was run against a **fresh GlobalModel baseline on the same GPU + same seed**, so the
only difference is the architecture (the GPU confound above makes raw comparison to historical numbers
unsafe — our fresh baselines land at 28.1–28.3, not exp6's recorded 27.48).

| arch | SIP | Angle | Joint cm | Vert cm | matched baseline SIP | ΔSIP |
|---|---|---|---|---|---|---|
| **ReconIMUPoser** (exp8, GPU0): reconstruct full 5-IMU → pose | 28.37 | 24.14 | 10.72 | 13.13 | 28.11 (exp10) | **+0.26 (worse)** |
| **StagedIMUPoser** (exp9, GPU1): IMU → joint pos → pose | 27.80 | 23.04 | 10.36 | 12.37 | 28.29 (exp11) | **−0.49** |

- **Missing-IMU reconstruction does NOT help** (+0.26° SIP, worse on every metric). Reconstructing the 2
  absent sensors from the 3 present ones injects error that propagates into the pose head; the auxiliary
  target adds no signal the pose head wasn't already extracting.
- **Staged prediction (IMU→joints→pose): NOT confirmed.** Looked like a −0.49° win at seed 42, but
  seed-replication (exp12–15, GPU-matched baseline→staged pairs) shows the staged−baseline ΔSIP is
  **{−0.49 (seed42), −0.05 (seed1), +0.85 (seed2)} → mean ≈ +0.1° ≈ 0** with a ~0.7° spread. The seed-42
  signal was noise. Intermediate joint-position supervision does not reliably help this model/sensor set.

  | seed | baseline SIP | staged SIP | Δ |
  |---|---|---|---|
  | 42 | 28.29 (exp11) | 27.80 (exp9) | −0.49 |
  | 1  | 26.79 (exp12) | 26.74 (exp13) | −0.05 |
  | 2  | 26.93 (exp14) | 27.78 (exp15) | +0.85 |

**Net (exp8–15):** neither architecture beats the plain LSTM once you control for the true ~1.5° noise
floor. The decisive methodological lesson is that **single-run comparisons below ~1.5° are unreliable
here** — the field's habit of reporting one run hides this. Calibration-error augmentation (−4.3°) remains
the only established lever; further gains likely need a different data/realism axis, not architecture.

### Transformer (TIP-style) and Diffusion (EgoEgo-style) — bigger model-family swaps

| model | seed1 SIP | seed2 SIP | LSTM seed-matched | verdict |
|---|---|---|---|---|
| **TransformerIMUPoser** (exp16/17, 60 ep) | 28.11 | 26.98 | 26.79 / 26.93 | **≈ LSTM, worse at seed1** |

- **A transformer does NOT beat the LSTM here**, even given 2× the epochs (60 vs 30). seed1 is +1.32°
  worse, seed2 is +0.05° (a tie); the 28.11↔26.98 transformer spread is itself ~1.1° (noise). With only
  ~31k short (≤125-frame) training windows, the LSTM's recurrent inductive bias wins; attention has too
  little data/sequence-length to pay off.
- **Length-generalization gotcha (important):** a transformer trained on ≤125-frame windows scores
  SIP **40.3** when eval feeds the whole 3000+-frame DIP take (full self-attention + sinusoidal PE don't
  extrapolate), but **28.1** with a sliding 125-frame window at inference. Its *windowed val_loss was
  healthy the whole time* — the failure was purely train/eval sequence-length mismatch. The LSTM is
  immune (recurrence is length-agnostic), which is itself a practical argument for it here. Fix is
  model-side sliding-window inference (`TF_EVAL_WINDOW`), like TIP / real-time IMU transformers; the
  protected evaluator is unchanged.
| **DiffusionIMUPoser** (exp18/19, 60 ep) | 28.05 | 28.50 | 26.79 / 26.93 | **≈ transformer (+~1.4°)** |

- **Diffusion is competitive once sampled correctly — the model was never the problem, the SAMPLER was.**
  As first run (DDIM 50 steps) it scored 32.8/32.4 (~6° worse, ≈ no-aug baseline), which looked like a
  flat failure. But the **denoising val_loss was lower than the LSTM's** (0.020 vs 0.037) — a real tell.
  Breaking the denoising MSE down by noise level shows it is fine even at the hard end (pure-noise t=T
  MSE 0.029 < LSTM 0.037). The issue is the **iterative DDIM trajectory**: SIP degrades *monotonically*
  with steps — **1→28.05, 20→31.90, 50→32.84**. The one-shot x0 prediction (`DIFF_SAMPLE_STEPS=1`) is
  best, giving 28.05/28.50 ≈ the transformer. (Why: with a near-deterministic conditional mapping
  IMU→pose, the one-shot x0 is already near-optimal; re-noising it and iterating drifts off the training
  distribution and compounds error. Strongly-conditioned diffusion favors few-step sampling.)
- Caveat on `val_loss` comparisons across objectives: the diffusion `val_loss` is the *denoising* MSE
  averaged over random noise levels (dominated by easy low-noise cases), NOT pose-regression error — it
  is not directly comparable to the LSTM/Transformer `val_loss`. SIP after sampling is the real metric.

**Overall (exp8–19): NO architecture or model family beats the well-tuned bidirectional LSTM — but
several MATCH it.** recon ≈ +0.3, staged ≈ 0, transformer ≈ 0-to-+1.3, diffusion (1-step) ≈ +1.4 (all
ΔSIP vs seed-matched LSTM; all within ~1–2× the 1.5° noise floor). The LSTM's recurrent bias fits
short-window sparse-IMU regression with ~31k windows and is length-agnostic (attention models needed a
sliding-window-inference fix; the LSTM did not), so it remains the best *default*, but the gaps are
small. The single lever that clears the noise floor remains **calibration-error augmentation (−4.3°)**.
Remaining levers are the data/realism axis (more faithful synthetic-IMU generation) or shrinking the
~1.5° noise floor (deterministic training + seed-averaging) so finer effects become detectable — not
bigger models. **Methodological lesson (this round): never trust a generative model's training-objective
loss as a proxy for the task metric, and always sweep sampling steps before declaring it dead.**

### Deeper models + AvatarPoser (exp20–23) — FIRST model to beat the LSTM

| model | seed1 SIP | seed2 SIP | vs LSTM (26.79/26.93) | verdict |
|---|---|---|---|---|
| Transformer **8-layer** (exp20/22) | 27.29 | 27.52 | +0.50 / +0.59 | depth helps vs 4-layer (28.1) but still > LSTM |
| **AvatarPoser** (exp21/23, transformer + IK) | **26.48** | **26.56** | **−0.31 / −0.37** | **beats LSTM at both seeds, on ALL metrics** |

- **Going deeper (4→8 transformer layers) helps a little** (seed1 28.11→27.29) but does NOT close the gap
  to the LSTM — consistent with not being data-limited (~31k short windows; more params ≈ mild overfit).
- **AvatarPoser is the first model to consistently beat the LSTM:** −0.31 / −0.37° SIP, and better on
  *every* one of the 5 metrics at *both* seeds (10/10; seed2 Angle 21.22 vs 22.76 = −1.54°). Pure-noise
  probability of 10/10 ≈ 0.1%, so despite the small SIP magnitude this is very unlikely to be noise.
- **The win is the IK loss, not attention.** Plain (28.1/27.0) and deeper (27.3/27.5) transformers are
  both *worse* than the LSTM, so the only thing that flips AvatarPoser to a win is its
  orientation-consistency term (predicted FK global orientation at the IMU joints == observed sensor
  orientation). That is a *transferable* auxiliary loss → **next: add it to the LSTM (the best base
  model)** to confirm the IK term is the lever and stack it on the strongest backbone.

**Revised overall (exp8–23): the IK orientation-consistency loss is a real (small) lever; bigger/fancier
backbones are not.** Ranking now: AvatarPoser (transformer+IK) < LSTM < deeper-transformer < transformer
≈ diffusion(1-step). Two established levers: **calibration-error augmentation (−4.3°, large)** and the
**IK consistency loss (−0.34°, small but seed- and metric-consistent)**. Both are about *geometric/sensor
realism*, not model capacity.

### IK loss is transformer-specific — it does NOT transfer to the LSTM (exp26/27)

Added the same IK orientation-consistency loss to the LSTM (IK_LOSS=1), GPU+seed+epoch-matched to the
plain-LSTM baselines. LSTM+IK ΔSIP vs plain LSTM = **+0.88 (seed1: 27.67 vs 26.79), −0.47 (seed2: 26.46
vs 26.93)** → mean ~+0.2, a 1.35° swing = **inconsistent / within noise**. Contrast AvatarPoser
(transformer+IK), which was consistently better at both seeds. Interpretation: the IK term supplies
geometric grounding the *attention* model lacks, but the *LSTM's recurrence already encodes that
consistency*, so the extra constraint just over-regularizes / adds noise. **The lever is "weak backbone
+ IK", not "IK universally".** Net: nothing has cleanly beaten the plain LSTM except AvatarPoser, and
that only ties-to-slightly-beats it. The LSTM remains the best backbone.

### Physics: acceleration-consistency loss HURTS the LSTM (exp30/31)

Physics-refinement idea (TransPose/PIP/PNP) adapted to our root-relative metric: instead of a
rigid-body simulator (impractical; mostly fixes global translation/foot-skate we don't measure),
enforce a Newtonian consistency — predicted motion's synthetic acceleration at the IMU joints must
match the OBSERVED accelerometer (the signal half the IK term ignores). Result: ACC_W=1.0 → 28.76,
ACC_W=0.1 → 28.61 vs plain LSTM seed1 26.79, i.e. **+1.8–2.0° WORSE, ~weight-independent**. The
joint-proxy (vs mounting vertex) + coarse 25fps second-difference is too noisy/biased a target, and
its large early gradient derails training. Not a useful lever as formulated. (A faithful version would
need vertex-accurate FK + the true 60fps synthesis, i.e. mesh FK every step — too costly here.)
