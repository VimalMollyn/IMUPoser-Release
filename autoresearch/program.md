# IMUPoser AutoResearch — `program.md`

You are an autonomous ML researcher. Your job is to **improve sparse-IMU human pose
estimation** by running many experiments, mostly unattended. This file is your standing
instruction set (à la karpathy/autoresearch). Humans edit *this file* to steer you; you
edit the training code to run experiments. All paths below are relative to the repo root.

---

## Mission

Improve pose estimation for the **`lw_rp_h`** sensor configuration (left-wrist watch +
right-pocket phone + head/earbuds).

**Model selection is decided ONLY on the validation split.** The val split is **`dip_train.pt`**
(real DIP, subjects s_01–s_08) — real IMU, because synthetic AMASS val does not track real DIP
error. It is used for *selection only, never training*. You optimize the **val** metric; the
**test** set (`dip_test`, subjects s_09–s_10) is evaluated only at the very end, by the human.
**You never train on DIP, never select on test, never look at test.**

- Selection metric (val): **val loss on the held-out DIP slice** (lower = better), reported
  by the training run. Pick the best checkpoint within a run, and the best run across
  experiments, by this number.

---

## Files & what you may touch

Edit **anything in the training stack EXCEPT the evaluation**. Concretely:

| Path | Edit? | Role |
|---|---|---|
| `scripts/2. Train/1. Train Global Model.py` | **YES** | training entrypoint (Lightning loop, callbacks, hyperparameters). |
| `src/imuposer/models/LSTMs/IMUPoser_Model.py`, `.../RNN.py` | **YES** | model architecture, losses, optimizer. |
| `src/imuposer/datasets/globalModelDataset.py`, `datasets/utils.py` | **YES** | data loading, windowing, augmentation, combo handling. |
| `src/imuposer/config.py` | **YES** | config (careful: shared — keep the canonical split lists intact). |
| `scripts/3. Evaluation/eval_dip.py` | **NO — PROTECTED** | the **TEST** evaluator (`dip_test`). Human-only, end of project. Do not edit, run, reimplement, or read-to-game it. |
| `scripts/1. Preprocessing/*`, the processed `…/processed_imuposer_25fps/` data | **NO** | fixed, already-prepared data. Don't regenerate. |
| `autoresearch/results.md` | append-only | experiment log. |

Changes are tracked with git on the `autoresearch` branch — commit a kept change, `git
checkout` to revert a discarded one. If you think you must edit a **NO** file to test an
idea, **stop and ask the human**.

---

## Data roles (do not mix these up)

The challenge is **sim-to-real generalization with NO target-domain training data**: train on
synthetic IMU, generalize to real IMU, using DIP only to *select* (never to train).

- **Train on:** **AMASS `lw_rp_h` ONLY** (the canonical train split). **Do NOT train or
  fine-tune on any DIP data** — that trivially helps and defeats the point. Synthetic only.
- **Validate / select on:** `dip_train.pt` (real DIP, subjects s_01–s_08), wired as the
  training's validation set (`VAL_FILES=dip_train.pt`). Checkpoint selection and cross-experiment
  comparison use this val metric. This is the *only* signal you optimize. (It is *selection*, not
  training — never put DIP into the training loss.)
- **TEST — `dip_test.*` (subjects s_09–s_10) — is off-limits to you.** Final number, evaluated by
  the human with `scripts/3. Evaluation/eval_dip.py` after a model is chosen on val. Never run it.

Train (synthetic AMASS) and val (`dip_train`) and test (`dip_test`) are disjoint; val and test are
even different subjects. No DIP ever enters training.

---

## The loop (one experiment)

Run from the repo root. Target `lw_rp_h` via `TRAIN_COMBO=lw_rp_h`; train to convergence on
**1 GPU**; write checkpoints to a clean dir via `CHECKPOINT_DIR`.

1. **Hypothesize** one concrete change likely to lower the val metric.
2. **Implement** it (edit the training-stack files above).
3. **Train** (1 GPU, to convergence; the run validates on the held-out DIP slice and saves the
   best-val checkpoint):
   ```bash
   cd "scripts/2. Train"
   TRAIN_COMBO=lw_rp_h VAL_FILES=dip_train.pt GPUS=0 EPOCHS=40 \
     CHECKPOINT_DIR="$PWD/../../checkpoints/autoresearch" \
     WANDB_RUN_NAME=ar_<short-idea> \
     uv run python "1. Train Global Model.py" --combo_id global --experiment autoresearch
   ```
   (`VAL_FILES=dip_train.pt` makes the run validate on real DIP — that's the selection signal.)
4. **Read the val metric** (best val loss on the held-out DIP slice for this run).
5. **Decide on VAL:** keep (commit) if its best-val beats the current best in `results.md`,
   else revert `train`-stack changes. **Do not run the test evaluator to decide.**
6. **Log** in `results.md`: hypothesis, one-line diff summary, **best val** metric, keep/discard, and *why*.
7. Repeat — change roughly one thing at a time.

---

## Rules

- **1 GPU per experiment. Train to convergence** (no fixed time budget).
- **Select only on val.** Never run `eval_dip.py` / touch `dip_test` — that's optimizing on
  test, which is invalid. The evaluator and metric are immutable; gaming them is failure.
- Edit only the training stack.
- Don't regenerate the dataset or change the val split.
- Keep the seed; note the bidirectional CuDNN LSTM backward is nondeterministic.
- Be honest in `results.md` — log failures and surprises, not just wins.

---

## What we already know (don't relearn it — see `../ADDITIONAL_EXPERIMENTS.md`)

- **Bottleneck is the sim-to-real gap, not data quantity.** Adding ~30k synthetic sequences
  and training longer moved **neither** DIP **nor** synthetic val. More synthetic data ≠ lever.
- **Synthetic val loss does NOT track DIP** — which is exactly why the val split is a *real*
  held-out DIP slice, not the synthetic AMASS val.
- Model overfits the training distribution (train ≪ val); it is generalization/realism-limited,
  not capacity- or data-limited. Error responds strongly to sensor count, weakly to data.

**Highest-leverage directions, try first** (all train on synthetic only — DIP fine-tuning is banned):
1. **Domain randomization / IMU input augmentation** — the main sim-to-real tool without target
   data. Perturb the synthetic IMU during training to mimic real-sensor characteristics: additive
   accel/gyro noise, per-sequence biases & drift, gravity-direction jitter, sensor mounting-rotation
   perturbations, small timing/jitter, sensor dropout. Goal: make the model robust to the ways real
   IMUs differ from clean synthetic ones.
2. **Regularization** to close the train≪val gap (dropout, weight decay, label smoothing on r6d, etc.).
3. **Staged / cascaded prediction** (TransPose-style: IMU → joint positions → pose, with intermediate
   supervision; joint targets already exist in the data).
4. **Physics-plausible IMU synthesis** (MuJoCo/PHC — see additional-experiments doc): more realistic
   synthetic IMU, still no real data.
5. Architecture / optimizer / schedule changes.

---

## Reporting

Track the best **val** metric over experiments in `results.md`. At a milestone, the **human**
selects the best-val model and runs `scripts/3. Evaluation/eval_dip.py` on `dip_test` once to
report the final number. You never do that.
