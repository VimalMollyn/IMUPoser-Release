# New motion data for lw_rw_rp @ 25 Hz — BONES-SEED + form-hoi + MotionMillion

**Status: RUNNING (2026-10-02).** Results table at the bottom is filled in by `scripts/3. Evaluation/newdata_report.py`
(the same script renders the results web page). Numbers below marked *pending* are not in yet.

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

## 4. Data (hours at 25 fps)

*filled by the report script — see the web page.*

## 5. Results

*pending — see the web page / `checkpoints/newdata/ft_*/eval_dip_test.log`.*

| arm | seed | dip_test SIP | MPJRE | MPJPE |
|---|---|---|---|---|
| control | 1 | pending | | |
| treatment | 1 | pending | | |

## 6. Caveats to keep honest

- Seed-to-seed std on this recipe is ~0.17° SIP; one seed per arm cannot resolve a difference under ~0.3°.
- The treatment epoch is ~2.5x more steps; the Nymeria compute-matched control (curated x174 ep = no gain)
  argues this is not a compute confound, but it was measured on a different data mix.
- Shape: all new data uses the neutral SMPL body (betas = 0), like Nymeria.
- MotionMillion's 30 fps -> 60 fps linear upsample (same as the existing Motion-X path) makes its synthetic
  accel spikier than mocap-rate sources; accel magnitudes are ~3x Nymeria's on the dynamic subsets.
