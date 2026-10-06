r"""Selection / report split of the collected IMUPoser dataset for any choice that is made ON this dataset (ensembles,
post-processing settings): participants 1-2 select (imuposer_sel.pt), participants 3-10 report (imuposer_test8.pt).
The plain zero-shot numbers (all 10 participants) involve no such choice and stay as they are.

  uv run python imuposer_split.py
"""
import os, torch
from pathlib import Path
DD = Path(os.environ.get("IMUPOSER_25FPS_DIR", "/home/vimal/imuposer_data/processed_imuposer_25fps"))
def cat(parts, name):
    out = {"acc": [], "ori": [], "pose": [], "names": []}
    for p in parts:
        d = torch.load(DD / f"imuposer_P{p}.pt", weights_only=False)
        for k in out: out[k] += list(d[k])
    torch.save(out, DD / name)
    print(name, len(out["pose"]), "recordings,", round(sum(x.shape[0] for x in out["pose"]) / 25 / 60, 1), "min")
cat([1, 2], "imuposer_sel.pt")
cat(list(range(3, 11)), "imuposer_test8.pt")
