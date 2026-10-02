r"""
Rebuild the 25 fps Nymeria training files from the 50 fps copy (the 25 fps set was reclaimed for disk).

50 -> 25 fps is an exact stride-2 decimation for pose / ori / joint / tran. The 50 fps accel was already
box-smoothed with 5 taps at 50 fps (+-40 ms); the original 25 fps pipeline used 5 taps at 25 fps
(+-80 ms), so after striding we add a 3-tap average at 25 fps to land on a comparable bandwidth.
Both experimental arms (control and treatment) use these same files, so any small difference from the
original 25 fps files cancels in the comparison.

  uv run python "scripts/1. Preprocessing/nymeria_50_to_25.py" --src .../processed_imuposer_50fps --dst .../processed_imuposer_25fps
"""
import argparse, importlib.util, os
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_50fps")
ap.add_argument("--dst", default="/home/vimal/imuposer_data/processed_imuposer_25fps")
ap.add_argument("--pattern", default="Nymeria_*.pt")
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p2", os.path.join(here, "2. preprocess_all_to_imuposer_at_25fps.py"))
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
smooth_avg = p2.smooth_avg

src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob(a.pattern))
print(f"{len(files)} files {src} -> {dst}", flush=True)
for f in files:
    out = dst / f.name
    if out.exists():
        continue
    d = torch.load(f, weights_only=False)
    o = {
        "joint": [x[::2].contiguous() for x in d["joint"]],
        "pose":  [x[::2].contiguous() for x in d["pose"]],
        "shape": d["shape"],
        "tran":  [x[::2].contiguous() for x in d["tran"]],
        "acc":   [smooth_avg(x[::2].contiguous(), s=3) for x in d["acc"]],
        "ori":   [x[::2].contiguous() for x in d["ori"]],
    }
    tmp = out.with_suffix(".pt.tmp")
    torch.save(o, tmp); os.replace(tmp, out)
    print(f"  {f.name}: {len(o['pose'])} seqs, {sum(x.shape[0] for x in o['pose'])/25/3600:.2f} h", flush=True)
print("DONE", flush=True)
