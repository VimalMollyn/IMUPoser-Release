"""Uniform weight averaging of the top-k saved checkpoints of one base run (checkpoint-SWA).

The trainer keeps the 3 best epochs by dip_train val loss; late in a constant-LR run they sit in one basin,
so averaging their weights is a free variance-reduction step (Izmailov et al. 2018). Writes
<out_dir>/avg.ckpt (state_dict averaged, everything else copied from the first member) plus a
best_model.txt pointing at it, so `run_newdata.sh` skips the base stage and goes straight to DIP fine-tuning:

    python average_checkpoints.py <src_base_dir> <out_base_dir> [--last]      (--last also includes last.ckpt)
"""
import sys
import shutil
from pathlib import Path
import torch

src = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
members = sorted(p for p in src.glob("epoch=*.ckpt"))
if "--last" in sys.argv and (src / "last.ckpt").exists():
    members.append(src / "last.ckpt")
assert members, f"no epoch=*.ckpt in {src}"
print(f"averaging {len(members)} checkpoints:")
for m in members:
    print("  ", m.name)
base = torch.load(members[0], map_location="cpu", weights_only=False)
sds = [torch.load(m, map_location="cpu", weights_only=False)["state_dict"] for m in members]
avg = {}
for k, v in sds[0].items():
    if v.is_floating_point():
        avg[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v.dtype)
    else:
        avg[k] = v.clone()  # buffers such as counters/int indices: take the first member's
base["state_dict"] = avg
ck = out / "avg.ckpt"
torch.save(base, ck)
(out / "best_model.txt").write_text(f"{ck.resolve()}\n\naveraged: {[m.name for m in members]}\n")
(out / "train.log").write_text(f"[avg] uniform average of {len(members)} checkpoints from {src}\n"
                               + "".join(f"  {m.name}\n" for m in members))
shutil.copy(ck, out / "last.ckpt")
print("wrote", ck)
