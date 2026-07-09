r"""
Package the per-clip retargeted WHIP tensors (whip_25fps/*.pt) into our DIP-style .pt files:
  whip.pt       - training clips (everything EXCEPT the held-out WHIP test)
  whip_test.pt  - held-out real WHIP benchmark: actor00 (unseen actor) + 'test_*' actions (unseen actions)
Each file: dict of lists {acc:[ (T,6,3) ], ori:[ (T,6,3,3) ], pose:[ (T,24,3,3) ]} matching GlobalModelDataset.
Written into the canonical 25fps data dir so TRAIN_DATASETS=whip / VAL_FILES=whip_test.pt work.
Name has no 'dip' substring so it is auto-discoverable.

  uv run python "scripts/1. Preprocessing/whip_package.py"
"""
import glob
from pathlib import Path
import numpy as np
import torch

SRC = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/whip_25fps")
DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")


def is_test(seq, act):
    return seq.startswith("actor00_") or act.startswith("test_")


def main():
    files = sorted(glob.glob(str(SRC / "*.pt")))
    train, test = {k: [] for k in ("acc", "ori", "pose")}, {k: [] for k in ("acc", "ori", "pose")}
    qc = []
    for f in files:
        d = torch.load(f, weights_only=False)
        seq, act = d["seq"], d["action"]
        bucket = test if is_test(seq, act) else train
        for k in ("acc", "ori", "pose"):
            bucket[k].append(d[k].float())
        qc.append((seq, act, d["pos_err_cm"], d["ori_resid_deg"]))
    for name, data in (("whip.pt", train), ("whip_test.pt", test)):
        n = len(data["pose"]); frames = sum(p.shape[0] for p in data["pose"])
        torch.save(data, DD / name)
        print(f"{name}: {n} clips, {frames} frames ({frames/25/3600:.2f} h) -> {DD/name}")
    pes = np.array([q[2] for q in qc]); rs = np.array([q[3] for q in qc])
    print(f"QC over {len(qc)} clips: pos {pes.mean():.2f}cm (p90 {np.percentile(pes,90):.2f}) | "
          f"ori resid mean (lw,rw,lp,rp,h) = {rs.mean(0).round(1)}  p90 {np.percentile(rs,90,axis=0).round(1)}")


if __name__ == "__main__":
    main()
