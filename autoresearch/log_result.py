r"""
Compute the FULL metric suite (SIP / Angle / Joint cm / Vertex cm / LocalAngle) on the
VALIDATION split (dip_train, lw_rp_h) for an experiment's best-val checkpoint, and append
one record to results.jsonl. Uses the protected evaluator code, pointed at the *val* file
(never dip_test). Then regenerates the progress graph.

  uv run python autoresearch/log_result.py --exp 2 --name calib_rot \
      --dir checkpoints/autoresearch/exp2_calib_rot --kept --commit 0338d2d
"""
import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "scripts" / "3. Evaluation" / "eval_dip.py"
RESULTS = REPO / "autoresearch" / "results.jsonl"


def best_ckpt(d: Path):
    cks = list(d.glob("*val_loss=*.ckpt"))
    if not cks:
        raise SystemExit(f"no val checkpoints in {d}")
    def vof(p):
        return float(str(p).split("validation_step_loss=")[1][:-len(".ckpt")])
    return min(cks, key=vof), vof(min(cks, key=vof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True, help="experiment checkpoint dir")
    ap.add_argument("--commit", default="")
    ap.add_argument("--kept", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    ckpt, valloss = best_ckpt(Path(args.dir) if Path(args.dir).is_absolute() else REPO / args.dir)
    # full metrics on the VAL split (dip_train), lw_rp_h
    out = subprocess.run(
        ["uv", "run", "python", str(EVAL), "--checkpoint", str(ckpt),
         "--combos", "lw_rp_h", "--test-file", "dip_train.pt"],
        cwd=REPO, capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("lw_rp_h"))
    _, sip, angle, joint, vert, localang = line.split()
    rec = {"exp": args.exp, "name": args.name, "commit": args.commit, "kept": args.kept,
           "val_loss": round(valloss, 5), "sip": float(sip), "angle": float(angle),
           "joint_cm": float(joint), "vert_cm": float(vert), "localang": float(localang),
           "note": args.note}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("logged:", json.dumps(rec))
    subprocess.run(["uv", "run", "python", str(REPO / "autoresearch" / "plot_progress.py")], cwd=REPO)


if __name__ == "__main__":
    main()
