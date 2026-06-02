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
