r"""
Plot the autonomous-research progress curve (à la karpathy/autoresearch):
each experiment's validation metric vs experiment index, with a best-so-far step line.

Reads `autoresearch/results.jsonl` (one JSON object per line), e.g.:
  {"exp": 0, "name": "baseline", "val": 0.0231, "kept": true, "note": "lw_rp_h, AMASS-only"}

`val` = the selection metric = val loss on the DIP-train val split (lower is better).
Writes `autoresearch/progress.png`.

  uv run python autoresearch/plot_progress.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
OUT = HERE / "progress.png"


def load():
    if not RESULTS.exists():
        return []
    rows = []
    for line in RESULTS.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["exp"])
    return rows


def main():
    rows = load()
    if not rows:
        print(f"no experiments logged yet in {RESULTS}")
        return

    x = [r["exp"] for r in rows]
    y = [r["val"] for r in rows]
    kept = [r.get("kept", False) for r in rows]

    # best-so-far (cumulative min)
    best, cur = [], float("inf")
    for v in y:
        cur = min(cur, v)
        best.append(cur)

    fig, ax = plt.subplots(figsize=(9, 5))
    # each experiment: green if kept (new best / accepted), gray if discarded
    ax.scatter([xi for xi, k in zip(x, kept) if k], [yi for yi, k in zip(y, kept) if k],
               c="tab:green", s=36, zorder=3, label="kept")
    ax.scatter([xi for xi, k in zip(x, kept) if not k], [yi for yi, k in zip(y, kept) if not k],
               c="0.6", s=24, zorder=2, label="discarded")
    ax.step(x, best, where="post", color="tab:blue", lw=2, zorder=4, label="best so far")

    b0, bN = best[0], best[-1]
    ax.set_title(f"IMUPoser AutoResearch — lw_rp_h, AMASS-only → DIP val\n"
                 f"{len(rows)} experiments | best val {bN:.4f} (from {b0:.4f}, "
                 f"-{100*(b0-bN)/b0:.1f}%)", fontsize=11)
    ax.set_xlabel("experiment #")
    ax.set_ylabel("validation loss (DIP-train, lw_rp_h)  ↓")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}  ({len(rows)} experiments, best={bN:.4f})")


if __name__ == "__main__":
    main()
