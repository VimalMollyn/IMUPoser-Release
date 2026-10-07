# New motion data for lw_rw_rp @ 25 Hz — BONES-SEED + form-hoi + MotionMillion

**Status: RUNNING (started 2026-10-02; campaign extended 2026-10-03 to a 48 h scaling + SOTA push).** The live
results page (scaling laws, leaderboard, ablations, levers) is rendered by `scripts/3. Evaluation/newdata_report.py`
into `autoresearch/newdata_report.html`; every finished run is also appended to `autoresearch/results.jsonl`.
Section 5 below is a snapshot of the findings as of 2026-10-04 10:00 ET.

## 1. Question and design

Does adding ~400 h of new motion data to pretraining improve the current best lw_rw_rp (left watch, right watch,
right pocket) 25 Hz deliverable (dip_test SIP 17.32, curated-12 AMASS + Nymeria -> DIP fine-tune)?

Single variable, same two-stage protocol as the Nymeria study (`NYMERIA_RESULTS.md`):

| stage | control | treatment |
|---|---|---|
| 1. pretrain (AvatarPoser, 60 ep, AUG_CALIB_RAD=0.122, TRAIN_COMBO=lw_rw_rp, select on dip_train) | curated-12 + Nymeria (242.6 h) | control + BONES-SEED + form-hoi + MotionMillion non-GV + Motion-X |
| 2. fine-tune (60 ep on real DIP ftrain, lr 1e-4, select on fval) | ftrain | ftrain |
| eval | dip_test (s09/s10, held out), last FT ckpt | dip_test |

User decisions: MotionGV (video-estimated, 114 GB) skipped for now; Nymeria kept in both arms; BONES-SEED
originals only (no mirrors); the TCS raw-IMU zip ignored; Nymeria 25 fps rebuilt by stride-2 from the 50 fps
copy (the original 25 fps files had been reclaimed) — both arms use these same files so it cancels.

Scripts: `scripts/2. Train/run_newdata.sh` (base -> FT -> eval), `chain_treatment.sh` (auto-launch after the
control run + conversions), converters in `scripts/1. Preprocessing/{bones_seed,formhoi,motionmillion}_convert.py`.

## 2. Infrastructure that had to change

**Streaming loader** (`imuposer/datasets/shards.py`, on by default via `IMUPOSER_STREAM=1`). The old loader
held every window in RAM (~100 MB per hour of motion); curated-12 + Nymeria already took ~32 GB of the 62 GB box.
Each dataset is now packed once into flat float32 memmaps (acc, ori, pose as axis-angle, joint, tran + per-sequence
lengths) and windows are sliced lazily in the DataLoader workers; the page cache does the rest. Verified
bit-equivalent to the in-RAM loader (max |d output| 5e-7 over 400 random samples, all aux targets). Training is
GPU-bound anyway (97 % util at ~5.4 it/s on the TITAN V), so throughput is unchanged.

**Speed-ups tested.** `PRECISION=16-mixed` is 2x *slower* on the TITAN X Pascal (no fp16 tensor cores); on the
Volta it was not tested so as not to disturb the control run. Both arms run fp32 with the identical recipe.

## 3. Converting the new datasets

None of the three are IMU datasets; all are motion, so IMU is synthesized exactly like AMASS/Nymeria (SMPL FK,
global orientation of the 5 worn bones, 2nd-difference accel of the sensor vertices, 60 -> 25 fps + 5-tap accel
average). Two things made this tractable overnight:

- **Mesh-free IMU synthesis** (`soma_retarget.imu_from_pose`): only the 6 sensor vertices are skinned, with their
  own LBS weights over all 24 joints. Identical to the full 6890-vertex ParametricModel mesh to ~1e-4 relative
  (checked on CMU and BMLmovi), 70x faster (1.6 M frames/s).
- **Closed-form SOMA -> SMPL retarget** (`soma_retarget.SOMAtoSMPL`). SOMA-X's PoseInversion mesh fit ran at
  ~270 frames/s on the Pascal (16 h for BONES-SEED alone) and left 7–13° of bone-direction error. Instead each
  SMPL joint copies its SOMA counterpart's *global* rotation (relative to the rig's reference T-pose) with a fixed
  per-joint offset. Direction-only offsets left 20–30° of *twist* error on forearms/pelvis; the offsets are
  therefore calibrated from the two T-pose **meshes** (SOMA-X bridges the SOMA mesh into SMPL topology; Kabsch per
  SMPL body part). Posed per-part rotation error vs the bridged SOMA mesh on form-hoi:

  | part | closed-form (mesh offsets) | SOMA-X PoseInversion fit |
  |---|---|---|
  | thighs | 1–3° | 4–6° |
  | head | 1° | 6° |
  | pelvis | 0–1° | 1–3° |
  | shins | 1–3° | 10–11° |
  | forearms | 6–10° | 9–14° |

  Forearm residual is SOMA's twist joints (ForeArmTwist1-4) that SMPL cannot represent. BVH forward kinematics
  of the BONES-SEED files lands in SOMA-X's joint frames (idle clips: legs/spine within 2–11° of the rest frames),
  so the same offsets apply. Sensor-frame sanity: the forearm sensor x-axis is 3.2° from the forearm direction on
  BONES-SEED, 3.2° on Nymeria and 3.9° on real DIP.

Per dataset:

| dataset | source format | conversion | notes |
|---|---|---|---|
| BONES-SEED | SOMA-uniform BVH, 120 fps, 71k originals (144 h) | stream tar -> parse -> BVH FK -> retarget -> IMU | mirrors (`*_M.bvh`) skipped; 120 -> 60 fps by stride 2; reader thread + bounded parse pool (Pool.imap read the whole 45 GB tar ahead -> OOM) |
| form-hoi | SOMA/MHR params, 30 fps, 4135 episodes (29 h) | SOMA-X FK (fk_only) -> retarget -> 30 -> 60 fps lerp -> IMU | world frame tilted 3.4° (camera rig): realigned with the ground-plane normal; pose_valid_mask + human-pose QC categories cut (object-only Chamfer/silhouette checks kept); segments < 2 s dropped; MHR identity prepared with batch 1 (SOMA-X caches per-batch buffers) |
| MotionMillion | 272-dim MotionStreamer rep, 30 fps | closed-form recovery (6D local rots, cumulative heading, integrated xz velocity, root height) -> 30 -> 60 lerp -> IMU | MotionGV excluded; the 7 Motion-X subsets reused from the SMPL-X originals already in the pipeline; their world is already y-up |

### 3e. StayStill (added 2026-10-06 21:30, user request)

[StayStill](https://enekoassets.github.io/staystill.html) (SCA 2026, Zenodo 18741736, MIT): 1634 BVH clips on the
LaFAN1 rig at 30 fps, 50 subjects, 6.0 h of *idle* motion (2-min standing idle, idle with a phone, 18 idle actions:
looking around, checking watch/phone, scratching, stretching, yawning, shifting weight). Markerless capture
(FreeMoCap) retargeted by the authors to LaFAN1; we use the `lafan/` folder. `scripts/1. Preprocessing/staystill_convert.py`
retargets to SMPL in closed form (`G_smpl[k] = G_rig[m] @ A[k]`). The LaFAN rig has no T-pose reference (its zero pose is
a straight line along +X), so the 22 constant joint frames `A[k]` are calibrated from the data: bone-direction alignment
fixes each frame up to a twist about the bone; the pelvis and chest twist are pinned by the hip-to-hip and
collar-to-collar axes, the upper arms by the elbow hinge plane (mean elbow bend 46–51°, a reliable hinge), and every
other joint by a zero-mean-twist rule (exact for hinge joints, an assumption for ball joints). The knee hinge planes
were tried and rejected: in idle data the shin motion is sideways sway, not flexion (mean bend 6°), and they put the
thighs 55–95° off. Checks on 6229 calibration frames: pelvis, thigh and chest +z face the rig's feet to within 1°
(head −4°), hip rotations 10–13°, elbows 67–95°, joint positions 4–11 cm from the rig (bone-length and hip-geometry
differences, not rotation errors). Output 1633 clips, 6.04 h, 2 shards; wrist |acc| p95 5.2 m/s² (actions) /
2.3 m/s² (idle) — the calmest data in the pool. Runs: `abl_staystill_s20` (control + StayStill) and
`abl_staystill4_s20` (repeated 4×) on local GPU0.

## 4. Data (hours at 25 fps)

*filled by the report script — see the web page.*

## 5a. Primary benchmark from 2026-10-06: zero-shot on the collected IMUPoser dataset

The CHI'23 IMUPoser dataset (Samsung T5: `CHI23/IMUPoser/CameraReady/dataset-release/imuposer_dataset`; 10
participants, 167 recordings, 1.15 h of real phone/watch/head IMU with MoSh'd SMPL GT) is closer to deployment than
DIP. Converted with `scripts/3. Evaluation/imuposer_dataset_to_eval.py`; every pretrained checkpoint is evaluated on it
with NO fine-tune (`base_<tag>/eval_imuposer_zs.log`). Finding that changed the protocol: **the DIP fine-tune improves
dip_test but costs 1–5° SIP on this set** (L60 control: 16.19 base vs 20.92 after FT), and the data recipes that helped
dip_test (DIP-like selection) transfer worse than plain control (base SIP 17.7–18.8 vs 16.2–17.3). All models beat the
paper's own DIP-fine-tuned LSTM on this set (22.21° angular / 8.56 cm / 10.12 cm without end effectors). The web page is
organised around this metric.

State on 2026-10-06 16:00 ET:

- **Leader: the XL60-control base with its three best checkpoints averaged** (`base_swa_xl60_ctrl/avg.ckpt`): SIP 15.64°,
  MPJRE 18.11°, MPJPE 7.35 cm, mesh 8.83 cm, MPJVE 24.8 cm/s. With overlapping evaluation windows (stride 31, a setting fixed
  on the DIP validation split): 15.48° / 8.77 cm. Plain XL60 best checkpoint: 16.13° / 9.14 cm.
- **Checkpoint averaging helps transfer on all nine runs tried** (−0.21 to −0.91° SIP, −0.03 to −0.38 cm mesh) although it was
  neutral on dip_test after the fine-tune. Averaging top-3 + last (4 members) ties top-3 (15.60 on all, 13.73 vs 13.72 on the
  selection split): not adopted.
- **What transfers**: longer schedules (S120 16.47 vs S60 17.33), bigger models (XL60 16.13 < L60 16.19 < M60 17.31 — the
  opposite order to dip_test), weight decay 1e-2 (M60 16.67) and cosine (M60 16.33) over the plain M60 (17.31). The new-data
  recipes transfer worse than plain control at every size tried (dlall_l20 ~17.9, dlallraw_l20 18.50 despite the lowest val
  loss of any run). Val loss does not predict this benchmark (Spearman −0.23 over the bases).
- **Choices made on the dataset use a split** (`imuposer_split.py`: participants 1–2 select, 3–10 report). Ensembles of the
  averaged bases: XL+L+L20 selects best (13.54 vs XL alone 13.72) and reports 16.04 vs 16.17; with stride 31 the XL+L
  ensemble reports 15.87. Gains of 0.1–0.3°, nothing larger.
- **Post-processing does not improve pose** (all on the averaged L60, 15.85 / 8.94): PIP/MobilePoser's physics optimizer
  (`run_pip_physics.py`, 50 Hz, soft foot contacts, 0 QP failures) 15.89 with PIP's gains and 15.96 / jitter 107 with softer
  gains; orientation fit 16.08 (smoothed 15.96, jitter 42); orientation injection 15.85–16.09; reduced rigid-body refiner
  16.00. MobilePoser's physics gain comes with its own velocity/contact heads and translation, which our pose-only model
  lacks. Thread closed.
- Running for this metric: XL60 continued +60 epochs from `last.ckpt` with snapshots for averaging (fig2 GPU1), M120 control
  with snapshots (local GPU1), the L/XL data-recipe runs the user asked for (local GPU0).

## 5b. Results on dip_test (snapshot 2026-10-05 14:00 ET; dip_test SIP, lower is better; seed noise ~0.17)

**Update 2026-10-07 04:00.** The user's "L and XL on the best data recipe" ask paid off on this benchmark: **XL (57 M)
for 20 epochs on control + DIP-like mocap + DIP-like filtered GV (`dlall_xl20`) = 15.86, the best single model**, 15.67
with overlapping evaluation windows (stride 31); its epoch-10 snapshot 16.21; L on the same data 16.10, L on the raw-GV
version 16.07, M 16.04. The bigger model profits from the DIP-like data where it did not from control data (XL60 on
control was the worst 60-epoch model, 16.60). Ensembles rebuilt with it (stride 31, selected on fval): the fval pick is now
the 8-member data-diverse set + dlall_xl20 + dlall_l20 + dlallraw_l20 = **15.48** (previous pick 15.65; 2026-09-01
deliverable 17.32); the 7-member set without dlall_l20 reads 15.47 and the 3-model set L60 + dlall_xl20 + dlallraw_l20
15.52. None of this transfers to the real-device set: dlall_xl20 zero-shot 18.04 (plain XL60 control 16.13). XL on the
raw-GV recipe (`dlallraw_xl20`) is training on local GPU0 (fig2 cannot stream the 711 h sets: I/O-bound).

**Headline (2026-10-05).** Selecting data by closeness to DIP beats adding hours. On the S model at 20 epochs,
control + ALL of MotionGV filtered (636 h) = 17.06, control + a RANDOM 256 h of it = 16.80, control + the 256 h
within 20° of DIP's mean pose (moderate accel, non-static) = **16.04**; threshold sweep 15° (61 h) 16.96, 20° 16.04,
25° (347 h) 16.20. Unfiltered MotionGV (724 h, no smoothing) = 16.57 beats the filtered set: the smoothing hurt.
The M model at 20 epochs on control + DIP-like mocap (186 h) + DIP-like GV (256 h) = **16.04 (15.94 with
overlapping inference windows)**, the best single model; M on the full 676 h treatment = 16.30 at 20 AND 60 epochs
(the model converges by epoch 15 on that much data). Best overall: fval-selected 7-member ensemble = **15.65**.

**Scaling (control data = curated-12 + Nymeria, 267 h).** Model size helps and then saturates: S (3.3 M) 60 ep
16.55, M (10.9 M) 60 ep **16.32**, L (25.6 M) 60 ep **16.31**; at a 20-ep budget S 17.50/17.32, M 16.73, L 16.99
(L needs the full schedule). Longer schedules do not help S (120 ep 16.73). Data helps: curated-12 only (35 h) is
~1.3 worse than control at every size.

**New data.** The first conversion of form-hoi / MotionMillion / MotionGV lerped axis-angle poses across the ±π wrap
(accel spikes in 2–11 % of frames); every treatment result from it is invalid and was rerun after re-conversion
in matrix space. With clean data: BONES-SEED alone helps the S model at 20 ep (control+BONES 16.73 vs 17.50/17.32)
but is neutral once trained to convergence (S60 16.72 vs 16.55; M60 16.46 vs 16.32). **The FULL new-data mix does
help the M model: M 20 ep on treatment (676 h) = 16.30 vs 16.73 on control (−0.43, 2.5× the noise), tying the best
60-ep models at a third of the epochs** (2026-10-04 13:50). M 60 ep on treatment is running (SOTA candidate; the
40-ep snapshot gives a second point). Still queued: S60 treatment, mixing ratio (new data ×0.25/×0.5/×2), a
DIP-like subset of the new data (186 of 409 h), the 13 leftover AMASS sets, a treatment→control curriculum,
MotionGV filtered / unfiltered / DIP-like (256 of 636 h).

**Recipe levers on the best base.** Checkpoint averaging before FT: neutral (M 16.40, L 16.36). Inference windows
longer than the 125-frame training window: much worse (no length extrapolation). Overlapping windows with averaged
predictions (stride 31): free −0.1…−0.2 (L60 16.31 → 16.13, M60 16.32 → 16.21), selected on fval. Cosine LR in
pretraining and an FT-recipe sweep (seed noise, lr, schedule, length) are running.

**Best overall so far: 15.78** = fval-selected ensemble of five existing fine-tuned checkpoints (M60, L60,
M60+BONES, S60, M20-treatment; r6d-averaged) with stride-31 inference. −0.53 vs the best single model, −1.54 vs
the previous deliverable (17.32), no new training. The 3-member M60 + L60 + M20-treatment ensemble gets 15.79.

| model / data | 20 ep | 60 ep |
|---|---|---|
| S control | 17.50 / 17.32 (2 seeds) | 16.55 (120 ep: 16.73) |
| M control | 16.73 | 16.32 |
| L control | 16.99 | 16.31 |
| S control + BONES-SEED | 16.73 | running |
| M control + BONES-SEED | – | 16.46 |
| S / M / L curated-12 | – | see page |

## 6. Caveats to keep honest

- Seed-to-seed std on this recipe is ~0.17° SIP; one seed per arm cannot resolve a difference under ~0.3°.
- The treatment epoch is ~2.5x more steps; the Nymeria compute-matched control (curated x174 ep = no gain)
  argues this is not a compute confound, but it was measured on a different data mix.
- Shape: all new data uses the neutral SMPL body (betas = 0), like Nymeria.
- 30 fps sources (form-hoi, MotionMillion, MotionGV) are upsampled to 60 fps by interpolating rotations in
  matrix space (`synth_imu.resample_pose_aa`); the original axis-angle lerp produced spikes across the ±π wrap and
  invalidated the first treatment results. QC rule: under 0.2 % of frames with any sensor |acc| > 50 m/s².
- Ensembles and stride-31 inference are reported separately from single-model numbers; all single-model
  comparisons use the original 125-frame tiling.
