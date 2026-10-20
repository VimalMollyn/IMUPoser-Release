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

**fig2 and the 700-hour-plus recipes (fixed 2026-10-11 16:10).** Shards cost 828 bytes per frame, so a
treatment+MGVRAW_ recipe (~1,400 h) is ~110 GB of memmaps, far more than fig2's 62 GB of RAM can cache; every
random window became a USB-exFAT read and fig2 ran those recipes several times slower than the local box
(control-sized recipes, ~20 GB, were fine). Fix: fig2's root drive is an NVMe (Force MP600) with 139 GB free, so
all shards except the filtered MGV_* sets (110 GB; MGV_* are symlinks back to the USB drive) were copied to
`/home/vimal/imuposer_fast/shards_processed_imuposer_25fps` (29 GB left on that drive). The loader already honours
`IMUPOSER_SHARD_DIR` (`shards.py: shard_root_for`), and the queue runner forwards every queue-line variable, so a
fig2 queue line just appends `IMUPOSER_SHARD_DIR=/home/vimal/imuposer_fast/shards_processed_imuposer_25fps`.
First user: `allraw_l20` (treatment + raw MotionGV, no selection, L 20 ep) moved from the local GPU0 queue to fig2
GPU1 (after the XXL continuation). Its first-epoch it/s is the verification; a stand-alone loader benchmark was
not run.

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

**Update 2026-10-08 22:00 ET — leader 14.66° / 8.43 cm; the spread between averaged stages is the knob.** A third XL
stage (+60 epochs from the continuation; alone 16.67 best / 16.28 averaged, dip_test 16.32) gives: stages 1+3 (120 epochs
apart) **14.66 / 8.43**, all three stages (9 checkpoints) 14.87 / 8.50, stages 2+3 15.71, stages 1+2 15.08. The farther
apart the averaged checkpoints sit on the trajectory, the better the transfer; nearby checkpoints (snapshots, soups of
adjacent stages) add nothing. The recipe holds at every size: M60 + M60c 15.60 (M60 averaged 16.40), L60 + L60c 15.46
(15.85), XL 14.66 (15.64). A fourth XL stage and a third L stage are queued to test whether the trend continues.

**Update 2026-10-07 19:15 ET — leader 15.08° / 8.57 cm (superseded).** XL60 control was continued for 60 more epochs from its
`last.ckpt` (fresh AdamW with warmup; fig2, 22 h). On its own the continuation is no better (best checkpoint 16.24, top-3
average 15.95, dip_test 16.50 vs 16.60), but the **uniform average of the six best checkpoints across both stages**
(XL60 epochs 28/29/51 + continuation epochs 9/20/29, i.e. combined epochs 28–89) scores **SIP 15.08°, MPJRE 17.48°,
MPJPE 7.16 cm, mesh 8.57 cm** (`base_swa_xl60_xl60c_ctrl/avg.ckpt`), −0.56° SIP and −0.26 cm mesh over the previous
leader. It is one weight set, no ensemble, and no choice was made on the dataset (all saved top-3 checkpoints of both
stages, equal weights). The spread of the averaged checkpoints along the trajectory is what matters: adding nearby
snapshots never helped (XL60 +last 15.60, L60 wd+cos +40/50 15.82, M120 +60/80/100/110 15.95), averaging across
60-epoch-apart stages did. Earlier averaged bases for reference: XL60 15.64, L60 wd+cos 15.77, M120 15.84, L60 15.85.
**2026-10-08 08:00 — the recipe generalises to L:** L60 continued +60 epochs (local, 10 h; alone: best checkpoint 16.07,
top-3 average 15.88, dip_test 16.26) and averaged with L60's top-3 = **15.46 / 8.68 cm** (L60 averaged alone 15.85), the
second-best zero-shot model at 25.6 M parameters. Averaging across two different runs is broken (L60 + L60 wd+cos 23.5,
M60 + M120 24.3), and adding more nearby snapshots always dilutes, so this is weight averaging along one trajectory with
60-epoch spread, not a model soup. XL60 with wd + cosine (fig2): dip_test 16.63 (= XL60), zero-shot 16.19 best / 16.25
averaged — the cosine tail makes the top-3 checkpoints too similar to gain from averaging; the levers that help at L do
not help at XL. Running: a third XL stage (fig2 GPU0, 9-checkpoint 180-epoch span), M60 continuation (local GPU1),
continuation of the XL wd+cosine run (fig2 GPU1).

State on 2026-10-06 16:00 ET (superseded leader):

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
15.52. None of this transfers to the real-device set: dlall_xl20 zero-shot 18.04 (plain XL60 control 16.13).
**2026-10-08:** XL on the raw-GV recipe (`dlallraw_xl20`) = 15.97 (15.77 with stride 31): raw and filtered GV tie at XL as
they did at L; zero-shot 18.37 / averaged 18.17. The fval-selected ensemble with it is now 9 members = **15.44**
(fval 12.65); the two XL data-recipe models alone reach 15.55 and the data-diverse 5 + both XL 15.44.

**Update 2026-10-20 18:30 — first token model (TokenHMR) scores 15.44 single, but does not break the ensemble plateau.**
`gen_token_s60_ctrl` (S, d256, IMU → TokenHMR pose tokens, CE + soft-decode) finished 60 epochs at **SIP 15.44 / 7.49 cm**
zero-shot — mid-pack as a single model (better than regression-S 16.53 and flow-S ~15.9, behind the 120-ep flows 14.6–14.8),
but it is a **different output space** from every current member, so the test was whether it decorrelates. It does on the
**selection split** — H + token-S = **12.07** vs H's 12.29 (−0.22), the first member to beat that baseline — **but the gain is
selection-split overfit** (2 participants): on the **report split it is neutral-to-worse (13.88 vs 13.83)** and **overall a tie
(13.49 vs 13.50)**. Leaner token combos were worse (leader+L120+TS 14.09 report, 3ldr+TS 14.11). So the ~13.8 plateau holds and
token-S does not join the leader ensemble — the S-size token model is simply too weak. This is a reminder to trust the report
split, not selection, for ensemble decisions. The remaining hope for decorrelation is the **larger token models** (L local, M/XL
on fig2) and the flow-token ladder, all still training; a stronger token model may help where S does not.

**Update 2026-10-18 16:00 — XXL flow continued to 120 epochs: 15.26; flow does not scale past L.** The XXL flow (m = -2) continued
60 more epochs on fig2 goes from 15.85 to **15.26 / 7.47 cm MPJPE** at 8 steps — the same ~0.6° the continuation buys, but it is the
**worst** of the 120-epoch flow family, which now ranks L (14.64) < M (14.73) ≈ L m=-1 (14.74) < XL (14.83) < XXL (15.26). Bigger flow
is worse here: the L flow at 25.6 M params beats every larger one, so flow saturates in both training length (past ~120 ep) and model
size (past L). As an ensemble member XXL-120 does not help — it is both weaker and correlated with the other flows. The accuracy lever
that is still open is **decorrelation**: the token and flow-token models (TokenHMR-latent targets, a different output space) now training
on both GPUs. Best ensemble remains **13.83 report / 13.50 overall** (3 regression leaders + L120 + XL120 flow); the newer flow members
have not broken that ~13.8 plateau.

**Update 2026-10-15 18:00 — M flow continued to 120 epochs: 14.73, the second-best single model at 10 M params; the IMU → TokenHMR-token
model is ready to train.** The M flow (m = -1) continued 60 more epochs on fig2 (fresh AdamW) goes from 15.72 to **14.73 / 7.29 cm MPJPE**
at 8 steps, the same −1.0° the L flow gained from its continuation (15.74 → 14.64): training the flow models longer pays at both sizes,
and the M at 120 epochs is within 0.1° of the L at a third of the parameters. As a fourth member of the leader ensemble (tiled windows)
it changes nothing: 13.96 report / 12.50 selection against 13.97 / 12.60 for the three-way, so the ensemble is saturated near 13.8–14.0
on the report split whatever is added (a second L regression average, a different-size flow model). Other arrivals: the S regression
with 16 attention heads reads 16.53 against 16.72 with 8 heads (−0.19, about seed noise; the 16-head flow S and the two
mixture-of-experts S runs are still training), and the XL flow at 60 epochs (m = -2, batch 32 × 8, lr 1e-4) reads 16.15, worse than the
L flow at 60 (15.74); its continuation is at epoch 18/60. TokenHMR (user ask): the official tokenizer (160 tokens × 2048 codes over the
21 body joints, non-commercial research licence, vendored under `src/imuposer/thirdparty/tokenhmr`) reconstructs the IMUPoser ground
truth at 0.58° / 0.30 cm, so the representation loses nothing for this benchmark. The IMU → token model (`TokenIMUPoser`: the S
backbone over the full window, a 64-dim factorised head giving 160 × 2048 logits per frame on 32 sampled frames per window,
cross-entropy plus the geometric loss on the softmax-weighted decode of 8 of them, the tokenizer frozen) costs 245 ms per 16-window
step, four times a regression step, in the token head and the decoder; encoding the targets on the fly was three times slower again,
so the tokens are precomputed per shard (`scripts/1. Preprocessing/precompute_tokens.py`, int16, 50 of 104 shards done). The S60
run launches when the precompute ends, an M-backbone variant follows, both zero-shot only. The page headline now ranks the
ensembles by the report split: the gated ensembles' whole-set numbers (13.21) include the two participants their gate was fitted on
and were leading the headline for the wrong reason (13.94 on the report split, behind the plain stride-31 ensemble's 13.84).

**Update 2026-10-15 14:00 — spatial inpainting: worse both ways; four-way ensemble: no gain.** The flow model trained in
global-rotation space with the three sensed joints as known columns (re-imposed at every sampling step, noised in 20% of training
windows) scores 17.39 with the measured orientations written into the output and 17.12 with the model's own prediction kept for
those joints, against 15.89 for the plain S60 flow. Hard-conditioning on the measured orientations hurts on real data for the same
reason the post-hoc injection did: the measured orientation carries the calibration error that is the domain gap. Together with the
history-conditioned result, both halves of the spatio-temporal inpainting formulation are now tested and set aside. Adding the
120-epoch single-run L average as a fourth member of the leader ensemble changes nothing (13.95 report / 13.52 overall with
overlapping windows, identical to the three-way), since the two L regression averages are not diverse enough.

**Update 2026-10-15 12:00 — one 120-epoch run vs 60 + 60: about equal; knob sweeps closed.** The L control trained 120 epochs in
a single run (`scale_l120_ctrl_snap`, snapshots every 10-20 epochs; dip_test 16.43) scores 15.93 best, **14.94 with its three best
checkpoints averaged** (the 60-epoch run's top-3 average was 15.85), and 14.39 with the early snapshots 10/20/30 added, against
14.26 for the same recipe on the 60-epoch run; adding mid or late snapshots does not help (14.56-14.77). So a longer single run
mostly moves the gain from the early + late trick into the top-3 average itself and ends at the same place. It is a differently
trained regression member, and is being tried as a fourth member of the leader ensemble. Flow knob sweeps closed at S60:
time shift m = -1 15.89 (best), m = -2 16.07, uniform t 16.27, default 16.89, m = -3 16.87; pose / FK / IK loss weight 0 -> 17.11,
1 -> 15.89, 3 -> 16.57, so the single-model recipe is fixed and the remaining levers are size, 120-epoch continuations at L and
above, and ensembling.

**Update 2026-10-14 21:00 — mixture of experts, two readings.** (1) A learned gate over the three frozen leader members
(72 per-joint weights, or a per-frame temporal gate on the IMU, fitted on participants 1-2 with the metric's own root-relative
loss) improves the fitted split by 1-2 degrees (12.37 -> 10.5-11.4) and is slightly worse on the held-out participants (13.83 ->
13.94-13.99): fifteen minutes of real data is enough to memorise, not to learn a transferable routing rule, so the uniform
average stays. (A first version of this study fitted absolute global rotations and positions and got nonsense; the evaluator sets
the root and the hand / foot joints to identity before scoring, so any loss used for selection must do the same.) (2) The LLM-style
mixture, routed expert feed-forwards trained jointly from scratch (4 experts, top-2, load-balancing loss), is training at S60 for
both the regression control and the shifted-time flow, against their dense references (16.72 / 15.89). With overlapping windows the
120-epoch L flow alone scores 14.17 (14.64 with tiled windows), so it benefits from overlapping evaluation even more than the
regression models. Also queued on request: 16 attention heads instead of 8 at S60 for both objectives.

**Update 2026-10-14 18:30 — NEW ZERO-SHOT LEADER, 13.82 / 8.28 cm: the regression leader ensembled with the 120-epoch L
flow model.** The 120-epoch L flow (`gen_flow_l60c_tlow_ctrl`, noise-end t) scores **14.64** at 8 Euler steps on its own, the best
single checkpoint of any kind (the regression leader is an average of six checkpoints). Averaged in output space 50/50 with that
regression leader (14.26) it gives **13.82 / 8.28 cm** on the whole dataset, and under the ensemble protocol 12.79 on the selection
split (participants 1-2) and 14.10 on the report split (3-10), against 14.40 for the previous leader on the report split. With
overlapping evaluation windows (stride 31, a setting fixed on the DIP validation split): 13.62 whole set, 13.92 report split.
A three-way ensemble adding the XL stages 1+4+snap20 average is the best on the selection split (12.60) and reports 13.97
(13.68 whole set); weighting the regression leader 2:1 changes nothing (13.82). So flow matching, which could not use the
regression models' weight averaging, contributes what the averaged regression models lack: the two families are complementary,
and the 60 -> 120 epoch continuation that hurt the S and M flow models was the step that made the L flow worth ensembling.
The three-way ensemble with overlapping windows reads **13.52 / 8.00 cm** on the whole set and 13.84 on the report split, the
best numbers of the study so far. In flight: a third L flow stage (120 -> 180), continuations chained onto the XL and XXL flow
runs, and the m = -1 L flow.

**Update 2026-10-14 16:30 — leader trajectory, third stage: saturated; flow L at 120 epochs reaches 14.99.** The third
60-epoch stage of the leader L trajectory (`scale_l60_ctrl_snap_c2`: dip_test 16.52, zero-shot 16.27 best / 16.04 top-3) gives
14.69 averaged with stage 1 (the far-apart-stage recipe that rescued the XL), 14.76 with all three stages and 14.89 with both stages'
early snapshots: every variant stays behind the leader's own 14.26. On the generative side the L flow continued to 120 epochs
(`gen_flow_l60c_tlow_ctrl`, fig2) scores **14.99** at 32 steps, 0.75 better than its 60-epoch parent, while the S and M
continuations had got worse: the larger flow models were the under-trained ones. Continuations are therefore chained onto the XL
and XXL flow runs, and the flow ensemble will be re-scored with the 120-epoch L.

**Update 2026-10-14 15:30 — flow training knobs at S60: the geometric loss matters, guidance hurts, m = -1 is the best time
shift.** Time-shift sweep (S60, 8 steps): none 16.89, m = -1 **15.89**, m = -2 16.07, m = -3 16.87. Dropping the FK / IK loss on
the x1 estimate (velocity objective only) costs 1.2 degrees (17.11). Condition dropout 0.1 costs 0.9 without guidance (16.82) and
classifier-free guidance then makes it worse monotonically (18.52 / 20.17 / 22.87 at w = 1.5 / 2 / 3): sharpening the conditional
distribution moves samples away from its mean, which is what a mean-error metric scores. Ensembles keep paying: four shifted-time
flow models (S m = -1, S m = -2, M, L) 15.05. The flow recipe is therefore fixed: 60 epochs, logit-normal t shifted by -1 to -2,
FK / IK loss on, no dropout, 8 Euler steps, whole-window sampling, and output-space ensembles across sizes. Still training: L with
m = -1 (local), M with m = -1, XL and XXL with m = -2 (fig2).

**Update 2026-10-14 08:30 — history-conditioned (autoregressive) flow, the DiffusionPoser / EgoEgo way: worse than
whole-window sampling for the offline metric.** The flow model was given per-frame time levels, trained with a random clean (or
slightly noised) pose-history prefix as context (`GEN_HIST=1`, 70% of windows; the rest whole-window), and sampled chunk by chunk
with the generated history fixed (`GEN_AR=1`). Same S60 checkpoint, zero-shot: whole-window 16.58; autoregressive 17.49 (62-frame
chunks), 17.63 (100-frame chunks), 18.41 (25-frame chunks). Re-noising the context to t = 0.8-0.9 and refining it together with
the new frames (SDEdit-style) is worse again (22.7-22.9 at 25 frames, 18.3 at 62), because overlapping frames get re-noised by
every later chunk. Two conclusions: conditioning on generated poses locks their errors in, so for an offline score the
bidirectional whole-window sampler (which still sees the whole window's IMU) wins, and the history training itself costs about
half a degree at whole-window evaluation (16.58 vs 16.07). The autoregressive form stays the right shape for a real-time system,
but it is not a lever for the zero-shot number.

**Update 2026-10-14 03:00 — L on all the raw data with no selection: dip_test 16.75, zero-shot 18.12.** The L trained 20 epochs on
treatment + raw MotionGV without the DIP-like selection (1,400 h; the first fig2 run to stream a 700-hour-plus set, from the NVMe
shard copy at 2.1 it/s) is worse on dip_test than both the selected recipe and the control-only L60 (16.31), and does not
transfer zero-shot (18.12). Unselected video-estimated motion hurts; the DIP-likeness selection is what makes the raw data usable.

**Update 2026-10-14 01:00 — XXL on all the raw data (user ask): dip_test 15.86, zero-shot 18.45.** The XXL (102 M) trained
20 epochs on the DIP-like selection of treatment + raw MotionGV (710.6 h kept of 1,400 h) gives dip_test **15.86**, the second-best
single model after the cosine XL on the same recipe (15.75) and ahead of the XXL60 control (15.91). Zero-shot it is 18.45 best /
18.37 top-3, the same non-transfer every data-recipe model has shown (XL on this recipe: 18.04). The data recipe buys dip_test only;
the zero-shot leaders stay the control-only runs.

**Update 2026-10-13 10:40 — flow ensembles: 15.17 for S + M + L; no gain on top of the regression leader.** Output-space
ensembling (r6d outputs averaged across members, the way every regression ensemble is scored) does work for flow models even though
weight averaging does not: S + L shifted-time flow 15.38, **S + M + L 15.17** (members 16.07 / 16.32 / 15.74), the best generative
result. Mixed with the averaged L60 regression leader (14.26): leader + L flow 50/50 14.31, leader + S/M/L flow 14.56, i.e. the flow
models carry no information the leader lacks. Where this leaves the campaign: flow matching, properly implemented and trained, beats
regression single checkpoints at equal size (15.74 vs 16.19 at L), prefers the noise-end time shift and few sampling steps, gets worse
with continuation and with weight averaging, improves with output ensembles, and is still 0.9 degrees behind the averaged regression
leader. Still running: the time-shift magnitude sweep at S (m = -1 / -3), the shifted-time M continuation, the L flow continuation
and shifted-time diffusion on fig2, and the XL flow on fig2.

**Update 2026-10-13 09:10 — flow matching at L with the noise-end time shift: 15.74 zero-shot, the best flow model.** The L
flow (`gen_flow_l60_tlow_ctrl`, d512 / 8 layers, 60 epochs on control, `GEN_T_M=-2`) scores **15.74 / 9.49 cm** at 8 Euler steps
(16.04 at 32: the shifted-time models prefer few steps), below the production L60 regression's single checkpoint (16.19) and the
L20 (16.47). The shifted-time S (16.07) and M (16.87 at 32 steps, 8-step pending) sit above it, and the S continuation to 120 epochs
again got worse (16.60), so for flow the recipe is "60 epochs, time sampling shifted to the noise end, few steps", and model size
helps a little at L. What flow still lacks is the regression leaders' averaging (14.26): weight averaging hurts it, so the only
averaging route left is an output-space ensemble of flow models (S + M + L shifted-time, being scored). A time-shift magnitude
sweep (m = -1 / -2 / -3) at S60 and the XL flow on fig2 are in flight.

**Update 2026-10-13 01:10 — the leader trajectory continued: no gain, 14.26 stands.** The L60 snapshot run that holds the
zero-shot lead (early 10/20/30 + top-3 = 14.26 / 8.33) was continued for 60 epochs (`scale_l60_ctrl_snap_c`: dip_test 16.46,
zero-shot 15.99 best / 15.78 top-3). Extending the leader recipe with the continuation gives 14.41 (continuation top-3 added,
nine), 14.49 (both stages early + late, twelve) and 14.60 (all continuation snapshots, fourteen): every variant is slightly worse
than the two-stage-free leader. The continuation rescued the weak XL snapshot trajectory (15.30 -> 14.68) but has nothing to add
to a trajectory whose early + late average is already good.

**Update 2026-10-12 22:00 — generative round 2 (60 epochs): flow matching edges regression at S; longer is not better; the
noise-end time shift is the lever.** Zero-shot SIP, all S size (d256 / 4 layers) on control unless noted, no fine-tune:
regression control S60 16.72; flow S60 default t 16.89 (8 steps) / 16.39 (32 steps); **flow S60 with t shifted to the noise end
(`GEN_T_M=-2`) 16.07** (8 steps; 16.24 at 32, so more steps do not help that variant); diffusion v2 S60 19.61; flow M60 default t
17.40 (8 steps), i.e. the bigger model did not help flow at 60 epochs with the default time distribution. Continuing the default-t
S60 flow for 60 more epochs made zero-shot worse (17.96 at 8 steps, 17.82 at 32) while its validation pose error kept improving,
the same overfitting-to-synthetic pattern the regression models show, and checkpoint averaging along the flow trajectory, the lever
that produced the regression leaders, hurts flow models (top-3 + snapshots 17.74; with the continuation's checkpoints 21.28).
Verdict so far: flow matching beats the regression control at S size by 0.3-0.65 degrees (16.07-16.39 vs 16.72) and the
production S20 (17.03) by about a degree, but it is still far behind the averaged regression leaders (14.26) and cannot use their
averaging trick. Running: flow M60 and S60 continuation with the shifted t (local), flow L60 and XL60 with the shifted t (fig2),
diffusion S60 with the shifted t (fig2).

**Update 2026-10-12 10:45 — XL snapshot re-run + continuation: 14.68 / 8.27 cm, the best mesh error so far.** The XL snapshot
re-run (`scale_xl60_ctrl_snap`, whose own early + late average was only 15.30) continued for 60 epochs on fig2
(`scale_xl60_ctrl_snap_c`: dip_test 16.49, zero-shot 16.31 best / 16.10 top-3). Averaging the stage-1 snapshots 10/20/30 + its
top-3 with the continuation's top-3 (nine checkpoints) gives **14.68 / 8.27 cm**: third on SIP behind the L leader (14.26 / 8.33)
and the original XL trajectory (14.43 / 8.36), but the lowest mesh error of any model. The one-trajectory early + late rule keeps
holding, and the continuation is what rescued this trajectory (15.30 -> 14.68).

**Update 2026-10-12 10:00 — flow matching beats regression once it is trained long enough: S60 flow 16.89 zero-shot.** The
60-epoch flow model (`gen_flow_s60_ctrl`, same S backbone, 8 Euler steps, one sample) scores **16.89 / 10.51 cm**, ahead of the
regression control at S20 (17.28), the production S20 (17.03) and the production M60 single checkpoint (17.31), and close to the
L20 regression (16.47). The user's call ("just train for longer") was right: the S20 flow model was simply under-trained. A second
lever surfaced in the same round: sampling t from a logit-normal shifted to the noise end (`GEN_T_M=-2`) gives 18.10 at 20 epochs
against 23.65 with the default distribution, i.e. most of the slow start was the time distribution starving the few-step regime of
training signal. Once trained, the sampler matters, which it did not at 20 epochs: the same S60 checkpoint scores 18.85 with one
step, 16.89 with 8, 16.55 with 16 and **16.39 with 32 Euler steps** (flattening), i.e. the multi-step flow integration is worth 2.5
degrees over the model's own one-step (regression-like) estimate; at S size that already beats the L20 regression (16.47). Running: the S60 flow continues to 120 epochs (chained), a 60-epoch flow run with the
shifted t (local), flow M60 / diffusion S60 / regression S60 on fig2 (each with a chained continuation to 120).

**Update 2026-10-12 04:30 — generative pose models re-implemented (user ask); first round at S20: behind regression, undertrained.**
The user judged the June diffusion test suspect and asked for flow matching as its own method, zero-shot only, no budget limit.
`src/imuposer/models/Transformers/GenIMUPoser.py` now holds three objectives on one DiT-style backbone (pre-LN transformer, adaLN-zero
time modulation, same TF_* size knobs): a regression control, flow matching (rectified flow, velocity target, logit-normal t,
Euler / midpoint) and diffusion v2 (VP cosine schedule, v-prediction, DDIM). Targets are standardised per r6d dimension (median std
0.105: the June model added unit noise to data a tenth that size), the FK / IK loss is applied to the implied x1 estimate, checkpoints
are selected on the pose error of a deterministic estimate, and `run_newdata.sh SKIP_FT=1` scores the base checkpoint zero-shot and
skips the DIP fine-tune. First round, S (d256 / 4 layers, 5.15 M with the modulation), 20 epochs on control, zero-shot SIP:
regression control 17.28 (production S20: 17.03, so the backbone is on par); diffusion v2 19.07 best (16 steps; 1 step 19.37,
8 steps 19.65); flow matching 23.05 best (8 midpoint steps; 1 step 23.84, 8 Euler 23.65). Sampling knobs move either model by
under a degree, so sampling is not the bottleneck: the velocity networks' validation pose error at epoch 19 was 0.076 (flow) and
0.067 (diffusion) against 0.035 for regression and still falling steeply, while regression had plateaued by epoch 10. Single
samples also swing with the noise seed (diffusion, 8 steps: 19.65 with seed 0, 24.19 with seed 1), i.e. the one-step estimate has not
learned to cancel its noise input, which is the signature of an under-trained generative model. Averaging four samples was worse
on the full dataset (22.00) although it beats every single seed on a 6-sequence check (15.70 vs 16.1-17.9) with the built-in,
r6d and chordal means agreeing, so the averaging code is right and the full-dataset number is being re-derived. Running now
(all zero-shot only): flow S60, flow S20 with t shifted to the noise end (local), regression S60 + diffusion S60 (fig2 GPU0),
flow M60 (fig2 GPU1).

**Update 2026-10-11 18:10 — XXL continuation: early + late works at XXL too, but XXL stays behind L and XL.** The
XXL60 control continued for 60 more epochs on fig2 (`scale_xxl60c_ctrl`, best val checkpoints at epochs 21/52/59):
dip_test 16.19 (stage 1: 15.91), zero-shot 16.69 best / 16.35 top-3, i.e. the continuation alone is worse than stage 1 on
both metrics, as it was for XL and L. Averaging across the stages recovers it: stages 1 + 2 (six checkpoints) 15.45 /
8.76 cm, and adding the stage-1 epoch-30 snapshot (seven) **15.13 / 8.63 cm**, the best XXL so far (stage-1 top-3 +
snapshot: 15.58). So the one-trajectory early + late rule holds at 102 M as well, but the XXL still trails the XL
(14.43) and the L leader (14.26) at a far higher cost (fp32 on fig2, 33 h per 60 epochs). XXL is not worth it for the
zero-shot target; its only lead is dip_test with the control data (15.91). Infrastructure: the fig2 NVMe shard copy is
verified in production (`allraw_l20` streams 1,400 h at 2.11 it/s, GPU 97 %, zero I/O wait), and 19.1 GB of finished
checkpoints were archived off the T7 drive (27 GB free) ahead of the XXL-on-raw run's checkpoints.

**Update 2026-10-11 17:05 — the curriculum does not pay at XL.** The DIP-like XL (`dlall_xl20`, dip_test 15.86,
zero-shot 18.04) was narrowed on control for 20 epochs (`curr_dlallctrl_xl20`: dip_test 15.77, zero-shot 16.32 best /
16.36 top-3, its best epochs being 0-2) and then for 40 more (`curr_dlallctrl_xl20c`: dip_test 16.01, zero-shot 16.00
best / 15.73 top-3). The two control stages averaged together (six checkpoints) read 15.96 / 8.64 cm, i.e. worse than the
same recipe at L (15.32 / 8.60) and far behind the plain XL control averages (14.43-14.89). At XL the data recipe's
head start is not recovered by narrowing; the leaders stay the control-only runs with early + late averaging.
dip_test is held (15.77 / 16.01 vs 15.75 best single), so the curriculum remains a dip_test-neutral, zero-shot-negative
lever at this size. The XXL on the DIP-like selection of the raw data (`dlallraw_xxl20`, 710.6 h kept) started on the
local GPU at 16:45 (3.4 it/s, ~95 min per epoch, ~32 h + FT).

**Update 2026-10-11 15:40 — leader 14.26 / 8.33 cm, and it is an L (26 M) from a single 60-epoch run.** The L60 control
re-run with 10-epoch snapshots (dip_test 16.55; top-3 at epochs 39/40/49) gives, zero-shot: best checkpoint 16.05; top-3
15.55; top-3 + late snapshots 40/50 15.35; **top-3 + early snapshots 10/20/30 14.26 / 8.33 cm** (angular 16.74, positional
6.95, velocity 22.0 cm/s, jitter 117: the best on every metric); all eight 14.42; early snapshots alone 14.73. The early +
late effect is much larger at L (−1.3 vs the top-3) than at XL (−0.6), and it needs no continuation stages: one run with
three extra snapshots beats the four-stage XL series (14.43). Queued: the same at M, and a far-apart second stage of this
L run for early + farther-late.

**Update 2026-10-11 11:00 — leader 14.43 / 8.36 cm (superseded).** Applying the early + late rule to the original XL trajectory: stages
1 + 4 (six best checkpoints) plus stage 1's epoch-20 snapshot, seven checkpoints averaged uniformly, zero-shot **SIP 14.43,
angular 16.58, positional 7.03 cm, mesh 8.36 cm** (previous leader 14.59 / 8.42). Within the first run alone, the top-3
plus that one snapshot reads 15.45 against 15.64 without it.

**Update 2026-10-11 10:00 — early + late, not spread: the averaging recipe resolved within one run.** An XL60 control
re-run saving a snapshot every 10 epochs (fig2; dip_test 16.36, its best-by-validation checkpoints are late: epochs 39/45/56)
gives, zero-shot: best checkpoint 16.71; top-3 15.94; top-3 + the LATE snapshots 40/50 15.73; top-3 + the EARLY snapshots
10/20/30 **15.30**; all eight 15.39; early snapshots alone 16.60. So what the cross-stage averages were buying is the
combination of early checkpoints (epochs 10–30) with late ones; late-only spread adds little and early-only is bad. A
single 60-epoch run with early snapshots recovers most of the two-stage gain (15.30 vs 15.08) at no extra training cost.
Next: the original XL trajectory's epoch-20 snapshot added to the 14.59 leader; the same test at L (local).

**Update 2026-10-10 23:00 — the curriculum L with two control stages: 15.32 / 8.60, best L-size model.** The narrowed
DIP-like L (`curr_dlallctrl_l20`) continued for 40 more epochs on control (`curr_dlallctrl_l20c`: dip_test 16.21, zero-shot
15.79 best / 15.53 averaged) and the two control stages averaged together read **15.32 / 8.60 cm**, ahead of the plain L
four-stage average (15.37 / 8.62) with a dip_test of 16.06–16.21 behind it. The cross-stage recipe therefore works on the
curriculum trajectory too, and this is the model to pick at 26 M parameters for both benchmarks.

**Update 2026-10-10 22:30 — cosine on the XL data recipe: best single model 15.75 (15.58 with stride 31).** `dlall_xl20_cos`
beats the constant-LR XL (15.86 / 15.67), as cosine did at M. Ensembles: the fval pick stays the 11-member set (12.62 →
15.42; adding this model reads 12.63 → 15.40); a 6-model set of the three XL data models' best two + M cosine + L treatment
+ L raw + the fine-tuned XXL control reads **15.31**, the lowest dip_test number so far, and the XXL is the first
control-data member that helps a data-recipe ensemble. Zero-shot 17.71 / 18.01: a dip_test-only win like every data recipe.

**Update 2026-10-10 14:30 — XXL (102 M, d1024 / 8 layers / FF 4096), 60 epochs on control (user ask).** dip_test
**15.91**: the best control-data model on the DIP benchmark (XL60 16.60, L60 16.31, M60 16.32), level with the best
data-recipe singles (15.86 / 15.92). Zero-shot: epoch-30 snapshot 15.83 / 9.12, best-by-validation checkpoint (epoch 54)
16.01 / 9.02, top-3 average (epochs 42/54/56) 15.91 / 8.94, top-3 + the epoch-30 snapshot **15.58 / 8.80**. The XXL's
three best checkpoints are all late, so its plain average gains less than the XL's (whose top-3 spans epochs 28–51);
adding the one early snapshot recovers the gain, the same early-plus-late pattern as the XL stages. Its 60-epoch
continuation is running on fig2 for the cross-stage pair (expected ~2026-10-11 evening).

**Update 2026-10-10 12:30 — the S20 data-mix and ablation block (fig2, re-converted data).** dip_test: new mocap at 0.5×
its natural share 17.06, 2× 16.87 (natural 16.62); control + form-hoi 17.32, + MotionMillion mocap + Motion-X 16.56,
+ BONES-SEED + form-hoi 16.91; reweighted treatment (form-hoi ×3, Nymeria ×2) 16.50; control + the 13 leftover AMASS-style
sets (~45 h) 17.35; control S20 17.50 / 17.32. So at S every new-mocap arm helps dip_test by 0.2–1.0 and the mix ratio is
flat; the DIP-like selected set (16.13) remains the best S recipe. Zero-shot all 17.4–19.8 (control S20 17.03): none
transfers, the same as every other new-data arm. Ensembles of the best trajectory averages (XL 1+4, L all-four, M, curriculum
L) add only ~0.1 on the split (14.73–14.82 on P3–10 vs 14.85 for the leader alone; 14.49 on all).

**Update 2026-10-10 11:30 — fourth L stage closes the L series at ~15.4.** Stage 4 alone 16.43 best / 16.14 averaged
(dip_test 16.33). Cross-stage: all four stages (12 checkpoints) **15.37 / 8.62**, 2+4 15.44, 1+3 15.44, 1+2 15.46, 1+3+4
15.47, 1+4 15.62, 2+3 15.79. At L every reasonable combination lands between 15.4 and 15.6; the XL's extra gain from the
far-apart pair (14.59) does not appear at L. Final picture of the recipe: top-3 averaging always helps; averaging across
continuation stages helps a lot more at XL (−1.0 vs the single average) than at L (−0.4) or M (−0.8 from 16.40 to 15.60).

**Update 2026-10-09 23:30 — fourth XL stage: leader 14.59 / 8.42, and the trend saturates.** Stages 1+4 (180 epochs apart)
average to **14.59 / 8.42 cm**, 0.07 better than 1+3; 1+3+4 14.84, all four stages (12 checkpoints) 14.89, 2+4 15.47,
3+4 16.19, stage 4 alone 16.44 best / 16.30 averaged (dip_test 16.41). Pair spacing 60 / 120 / 180 epochs gives 15.08 /
14.66 / 14.59, so the gain flattens near 14.6. Every good pair contains stage 1: its epoch-28..51 checkpoints carry the
transfer, and the recipe is early checkpoints plus one far-away late stage rather than spread by itself. No further XL
stages are planned; the XXL (102 M) is the remaining size experiment.

**Update 2026-10-09 19:30 — third L stage.** L60 → +60 → +60 (dip_test 16.31 / 16.26 / 16.34; the third stage alone
16.20 best / 15.95 averaged zero-shot). Cross-stage averages: 1+3 **15.44 / 8.62**, 1+2+3 15.45, 1+2 15.46, 2+3 15.79. At
L the recipe saturates near 15.45 once two stages are averaged; the extra spread that took XL from 15.08 to 14.66 adds
nothing here. A fourth L stage is running anyway (local GPU1) to close the question; the fourth XL stage lands tonight.

**Update 2026-10-09 18:20 — the curriculum works at L: one model good on both benchmarks.** The DIP-like L20
(`dlall_l20`: dip_test 16.10, zero-shot 18.67) continued for 20 epochs on control only (`curr_dlallctrl_l20`) reads
**dip_test 16.06 and zero-shot 16.12 / 8.76 cm** as a single checkpoint, i.e. the dip_test of the data recipe with the
transfer of a control model (XL60 control: 16.60 / 16.13 / 9.14). Its mesh error is the lowest of any single checkpoint.
Its top-3 average reads **15.63 / 8.72 cm**, equal to the averaged XL60 control (15.64 / 8.83) at 25.6 M parameters, with
a dip_test of 16.06 behind it. Averaging the broad stage's checkpoints with the narrowed stage's does not work (16.61):
stages trained on different data do not share a basin. The XL version and a second narrowing stage are queued.

**Update 2026-10-09 18:00 — the M data recipe catches the XL; ensemble 15.42.** On the DIP-like recipe at M (20 epochs):
cosine schedule 15.92, grad-clip + accel-clamp 15.88 (constant-LR baseline 16.04, XL 15.86). L on the full treatment set
(676 h, no selection) 16.02 vs 16.99 for L on control. The fval-selected ensemble is now 11 members = **15.42** (fval
12.62); a 5-model set (both XL data models + M cosine + L treatment + L raw) reads 15.36, and three models (both XL + M
cosine) 15.43. None of these transfer: zero-shot 17.8–19.1. The combined cosine + stability M recipe and a cosine XL
recipe are queued.

**Update 2026-10-09 06:00 — the original question, re-converted data, S at 60 epochs:** control + ALL new mocap (676 h,
natural mix, `treatment2_s1`) = **16.43** vs 17.32 / 17.50 for control S60: the new data now helps the small model by
about a degree on dip_test (the first-conversion arms had been neutral; the matrix-space interpolation fix is what
changed). The DIP-like recipe at S (`dlall_s20`, 20 epochs) reads 16.13, close to M (16.04) and L (16.10). A control-only
second stage after the broad pretraining (`curr_trt2ctrl_s20`) gives 16.47, no gain on dip_test, but it restores the
zero-shot transfer that the new data costs (treatment2_s1 19.61 → 16.84, vs 17.33 for control S60): the narrowing stage is
the first thing that makes new data and real-device transfer compatible, and it is worth trying at L/XL.

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
