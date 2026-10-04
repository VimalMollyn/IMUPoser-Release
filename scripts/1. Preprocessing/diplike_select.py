r"""Per-sequence "DIP-likeness" of packed datasets, and a DATASET_KEEP selection file.

Reuses the distribution diagnostics that explained WHIP (hurt) vs Nymeria (helped): how far a dataset's poses sit
from DIP's, and how explosive its accelerations are. Here per SEQUENCE, straight from the memmap shards:

  dist    angular distance (deg, mean over the 23 non-root joints) between the sequence's mean LOCAL joint rotation
          (chordal mean, heading-invariant by construction) and DIP-train's mean local pose
  div     own-mean diversity: mean angular distance of the sequence's frames from its own mean pose (deg)
  acc95   95th percentile of the sensor |acc| (m/s^2) over the 5 worn sensors
  stat    fraction of near-static frames (all sensors |acc| < 0.5 m/s^2)

Outputs <out>.csv (one row per sequence), prints an hours-by-distance table per dataset group, and writes the
DATASET_KEEP json for the chosen rule (--max_dist / --max_acc95 / --min_div), consumed by the streaming loader.

  uv run python diplike_select.py --prefixes BONES_,FORMHOI_,MM_,MotionX_ --out /path/diplike  [--max_dist 45 --max_acc95 15]
"""
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np
import torch

from imuposer.math.angular import axis_angle_to_rotation_matrix

SH = Path(os.environ.get("IMUPOSER_SHARD_DIR", "/home/vimal/imuposer_data/shards_processed_imuposer_25fps"))
FPS = 25


def _chordal_mean(R):          # (N,J,3,3) -> (J,3,3) rotation closest to the arithmetic mean
    M = R.mean(0)
    U, _, Vt = torch.linalg.svd(M)
    d = torch.sign(torch.det(U @ Vt))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return U @ D @ Vt


def _ang(Ra, Rb):              # (...,3,3) x (...,3,3) -> degrees
    tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))


def seq_stats(sd, dev, ref_mean=None, chunk=200_000):
    """Yield (seq_idx, L, dist, div, acc95, stat) for every sequence of one shard dir."""
    meta = json.loads((sd / "meta.json").read_text())
    F = meta["n_frames"]; lens = meta["seq_lens"]
    pose = np.memmap(sd / "pose_aa.f32", dtype=np.float32, mode="r", shape=(F, 24, 3))
    acc = np.memmap(sd / "acc.f32", dtype=np.float32, mode="r", shape=(F, 5, 3))
    b = 0
    for si, L in enumerate(lens):
        p = torch.from_numpy(np.array(pose[b:b + L, 1:], copy=True)).to(dev)          # (L,23,3) local, non-root
        R = axis_angle_to_rotation_matrix(p.reshape(-1, 3)).view(L, 23, 3, 3)
        mean = _chordal_mean(R)
        div = _ang(R, mean.unsqueeze(0)).mean().item()
        dist = _ang(mean, ref_mean).mean().item() if ref_mean is not None else float("nan")
        a = np.linalg.norm(np.array(acc[b:b + L], copy=True), axis=-1)                # (L,5)
        acc95 = float(np.quantile(a.max(1), 0.95)) if L else 0.0
        stat = float((a.max(1) < 0.5).mean()) if L else 0.0
        yield si, L, dist, div, acc95, stat, mean
        b += L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="BONES_,FORMHOI_,MM_,MotionX_")
    ap.add_argument("--ref", default="dip_train", help="shard dir of the reference distribution")
    ap.add_argument("--out", required=True, help="output stem: <out>.csv, <out>_keep.json")
    ap.add_argument("--max_dist", type=float, default=None)
    ap.add_argument("--max_acc95", type=float, default=None)
    ap.add_argument("--min_div", type=float, default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # reference: DIP-train mean local pose (chordal mean of the per-sequence means weighted by length)
    ref_rows = list(seq_stats(SH / a.ref, dev))
    w = torch.tensor([r[1] for r in ref_rows], dtype=torch.float32, device=dev)
    Ms = torch.stack([r[6] for r in ref_rows])                                           # (S,23,3,3)
    ref_mean = _chordal_mean((Ms * (w / w.sum()).view(-1, 1, 1, 1)) * len(ref_rows))     # weighted mean -> SVD
    ref_d = [_ang(r[6], ref_mean).mean().item() for r in ref_rows]
    print(f"[ref {a.ref}] {len(ref_rows)} seqs; per-sequence distance to the DIP mean: "
          f"median {np.median(ref_d):.1f}, p90 {np.quantile(ref_d, .9):.1f}, max {max(ref_d):.1f} deg; "
          f"div median {np.median([r[3] for r in ref_rows]):.1f}; acc95 median {np.median([r[4] for r in ref_rows]):.1f}", flush=True)

    prefixes = [p for p in a.prefixes.split(",") if p]
    dirs = sorted(d for d in SH.iterdir() if d.is_dir() and any(d.name.startswith(p) for p in prefixes)
                  and (d / "meta.json").exists())
    # the reference's own sequences go into the csv too (distance to the pooled DIP mean), for the report's first row
    rows = [(a.ref, r[0], r[1], d, r[3], r[4], r[5]) for r, d in zip(ref_rows, ref_d)]
    keep = {}
    for d in dirs:
        kept = []
        for si, L, dist, div, acc95, stat, _ in seq_stats(d, dev, ref_mean):
            rows.append((d.name, si, L, dist, div, acc95, stat))
            ok = ((a.max_dist is None or dist <= a.max_dist) and (a.max_acc95 is None or acc95 <= a.max_acc95)
                  and (a.min_div is None or div >= a.min_div))
            if ok:
                kept.append(si)
        keep[d.name] = kept
    with open(a.out + ".csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["dataset", "seq", "frames", "dist_deg", "div_deg", "acc95", "static_frac"])
        wr.writerows(rows)
    if any(v is not None for v in (a.max_dist, a.max_acc95, a.min_div)):
        Path(a.out + "_keep.json").write_text(json.dumps(keep))

    # summary per group: hours by distance bin + kept hours
    def grp(n):
        for p in prefixes:
            if n.startswith(p): return p
        return n
    bins = [0, 35, 40, 45, 50, 60, 1e9]
    print(f"\n{'group':12s} {'hours':>7s} | " + " ".join(f"{'<' + str(b):>7s}" for b in bins[1:]) + f" | {'acc95 med':>9s} {'div med':>8s} | kept h")
    for g in sorted({grp(r[0]) for r in rows}):
        rs = [r for r in rows if grp(r[0]) == g]
        H = sum(r[2] for r in rs) / FPS / 3600
        hb = [sum(r[2] for r in rs if bins[i] <= r[3] < bins[i + 1]) / FPS / 3600 for i in range(len(bins) - 1)]
        kh = sum(r[2] for r in rs if r[1] in set(keep.get(r[0], []))) / FPS / 3600
        print(f"{g:12s} {H:7.1f} | " + " ".join(f"{h:7.1f}" for h in hb) +
              f" | {np.median([r[5] for r in rs]):9.2f} {np.median([r[4] for r in rs]):8.1f} | {kh:6.1f}")
    print(f"\n{len(rows)} sequences in {time.time() - t0:.0f}s -> {a.out}.csv" + (f", {a.out}_keep.json" if keep and any(v is not None for v in (a.max_dist, a.max_acc95, a.min_div)) else ""))


if __name__ == "__main__":
    main()
