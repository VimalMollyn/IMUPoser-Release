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
from datetime import datetime
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

    # headline metric = SIP (deg) on the DIP-train val split (fall back to val_loss)
    rows = [r for r in rows if ("sip" in r) or ("val" in r) or ("val_loss" in r)]
    def metric(r): return r.get("sip", r.get("val_loss", r.get("val")))
    x = [r["exp"] for r in rows]
    y = [metric(r) for r in rows]
    kept = [r.get("kept", False) for r in rows]
    names = [r.get("name", "") for r in rows]
    unit = "SIP error (deg)" if all("sip" in r for r in rows) else "val metric"

    # best-so-far (cumulative min)
    best, cur = [], float("inf")
    for v in y:
        cur = min(cur, v)
        best.append(cur)

    fig, ax = plt.subplots(figsize=(9, 5))
    # NOISE BAND: identical best-recipe GlobalModel baselines re-run across seeds/GPUs (name "base*").
    # Their spread IS the empirical run-to-run noise floor (~1.5° SIP, from seed + non-deterministic
    # CuDNN bidirectional-LSTM backward). Anything inside this band is indistinguishable from noise.
    base_y = [metric(r) for r in rows if r.get("name", "").startswith("base_")]
    if len(base_y) >= 2:
        lo, hi = min(base_y), max(base_y)
        ax.axhspan(lo, hi, color="tab:orange", alpha=0.12, zorder=0,
                   label=f"same-recipe noise band ({len(base_y)} runs, {hi - lo:.1f}° spread)")
        ax.axhline(sum(base_y) / len(base_y), color="tab:orange", alpha=0.55, lw=1, ls="--", zorder=1)
    # each experiment: green if kept (new best / accepted), gray if discarded
    ax.scatter([xi for xi, k in zip(x, kept) if k], [yi for yi, k in zip(y, kept) if k],
               c="tab:green", s=36, zorder=3, label="kept")
    ax.scatter([xi for xi, k in zip(x, kept) if not k], [yi for yi, k in zip(y, kept) if not k],
               c="0.6", s=24, zorder=2, label="discarded")
    ax.step(x, best, where="post", color="tab:blue", lw=2, zorder=4, label="best so far")

    # label each experiment with its name (exp#: name), alternating above/below to reduce overlap
    yr = (max(y) - min(y)) or 1.0
    for i, (xi, yi, nm) in enumerate(zip(x, y, names)):
        dy = 0.04 * yr if i % 2 == 0 else -0.06 * yr
        ax.annotate(f"{xi}:{nm}", (xi, yi), xytext=(xi, yi + dy), fontsize=7,
                    ha="center", color="0.25", rotation=20)

    b0, bN = best[0], best[-1]
    ax.set_title(f"IMUPoser AutoResearch — lw_rp_h, AMASS-only → DIP-train val\n"
                 f"{len(rows)} experiments | best {unit} {bN:.3f} (from {b0:.3f}, "
                 f"-{100*(b0-bN)/b0:.1f}%)", fontsize=11)
    ax.set_xlabel("experiment #")
    ax.set_ylabel(f"{unit} on DIP-train val (lw_rp_h)  ↓")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    last_ts = rows[-1].get("ts", "")
    fig.text(0.99, 0.01, f"generated {datetime.now():%Y-%m-%d %H:%M}"
             + (f" · last run {last_ts}" if last_ts else ""),
             ha="right", va="bottom", fontsize=7, color="0.5")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}  ({len(rows)} experiments, best={bN:.4f})")


if __name__ == "__main__":
    main()
