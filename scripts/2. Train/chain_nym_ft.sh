#!/usr/bin/env bash
# Wait for a base-training launcher PID to exit, then FT that base on real DIP (ftrain) and eval on
# dip_test. Used to chain stage 2 behind stage 1 without babysitting.
#   chain_nym_ft.sh <launcher_pid> <base_tag> <ft_tag> <gpu>
set -u
cd "$(dirname "$0")"
REPO="$PWD/../.."
PID="${1:?launcher pid}"; BASE_TAG="${2:?base tag}"; FT_TAG="${3:?ft tag}"; GPU="${4:?gpu}"

while kill -0 "$PID" 2>/dev/null; do sleep 30; done
sleep 20

BDIR="$REPO/checkpoints/nymeria/$BASE_TAG"
# best_model.txt line 1 = best-val checkpoint path (written by the trainer)
BEST="$(head -1 "$BDIR/best_model.txt" 2>/dev/null)"
[ -f "$BEST" ] || BEST="$BDIR/last.ckpt"
echo "[chain] base=$BASE_TAG best=$BEST"

export OUT_DIR="$REPO/checkpoints/nymeria" BASE_CKPT="$BEST"
bash run_whip_ft.sh "$GPU" "$FT_TAG|ftrain|fval.pt|SEED=1"

echo "[chain] FT done, evaluating $FT_TAG on dip_test"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" uv run python "scripts/3. Evaluation/offline_fit.py" \
  --data dip_test.pt --members "checkpoints/nymeria/$FT_TAG/last.ckpt" --iters 0 2>&1 | grep -E "^  SIP"
echo "[chain] COMPLETE $FT_TAG"
