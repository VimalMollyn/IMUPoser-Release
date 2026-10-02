r"""
Memory-mapped window shards: stream training windows from disk instead of holding every dataset in RAM.

WHY: `GlobalModelDataset` used to materialise every window of every training file as CPU tensors
(~100 MB of RAM per hour of 25 fps motion). curated-12 + Nymeria already needed ~32 GB; adding
BONES-SEED / form-hoi / MotionMillion (hundreds more hours) does not fit a 62 GB box. Here each
dataset is packed ONCE into flat float32 arrays on disk and the dataset slices frame ranges lazily
through `np.memmap` inside each DataLoader worker. The OS page cache keeps hot data in RAM, cold data
costs one contiguous read (~50-70 KB per 125-frame window), and process RSS stays flat regardless of
how many hours are on disk. Think "webdataset", but random-access (true shuffling, no shard-level
locality tricks needed because the reads are small and the NVMe is fast).

Layout: <shard_root>/<dataset-name>/{meta.json, acc.f32, ori.f32, pose_aa.f32, joint.f32, tran.f32}
  acc     (F,5,3)   raw synthetic/real accel for the 5 worn sensors [lw,rw,lp,rp,h] (m/s^2, un-scaled)
  ori     (F,5,3,3) global sensor orientation
  pose_aa (F,24,3)  SMPL local pose as axis-angle (round-trips the .pt rotation matrices to float eps;
                    3x smaller than storing matrices)
  joint   (F,24,3)  joint positions (aux target "joint")
  tran    (F,3)     root translation (aux target "tran")
meta.json records the per-sequence lengths so the loader can rebuild the exact same windowing as the
in-RAM path (torch.split per sequence: 125-frame windows + a shorter tail).

Packing is automatic and idempotent (`ensure_packed`), guarded by a lock dir so two training runs that
start together do not both pack the same file; a shard is re-packed if its source .pt is newer.
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from imuposer import math as M

VERSION = 1
# field -> per-frame shape
FIELDS = {
    "acc": (5, 3),
    "ori": (5, 3, 3),
    "pose_aa": (24, 3),
    "joint": (24, 3),
    "tran": (3,),
}
_LOCK_STALE_S = 15 * 60      # a pack takes seconds; a lock older than this was left by a killed process


def _as_axis_angle(pose, L):
    """(L,24,3,3) | (L,216) | (L,72) | (L,24,3)  ->  (L,24,3) axis-angle."""
    if pose.dim() == 2 and pose.shape[1] == 72:
        return pose.view(L, 24, 3)
    if pose.dim() == 3 and pose.shape[-1] == 3 and pose.shape[1] == 24:
        return pose
    return M.rotation_matrix_to_axis_angle(pose.reshape(-1, 3, 3)).view(L, 24, 3)


def pack_pt(pt_path, out_dir, verbose=True):
    """Pack one 25 fps .pt training file (dict of lists of per-sequence tensors) into a shard dir."""
    pt_path, out_dir = Path(pt_path), Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".packing")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    d = torch.load(pt_path, weights_only=False)
    seq_lens = [int(x.shape[0]) for x in d["acc"]]
    F = int(sum(seq_lens))
    mm = {k: np.memmap(tmp / f"{k}.f32", dtype=np.float32, mode="w+", shape=(F,) + shp)
          for k, shp in FIELDS.items()}
    t0 = time.time()
    # batch across sequences: one big conversion/copy per field (per-sequence calls on files with
    # thousands of short clips were dominated by per-call overhead: 30 s vs 6 s for a 3 h file)
    keep = [i for i, L in enumerate(seq_lens) if L > 0]
    mm["acc"][:] = torch.cat([d["acc"][i].reshape(seq_lens[i], -1, 3)[:, :5] for i in keep]).float().numpy()
    mm["ori"][:] = torch.cat([d["ori"][i].reshape(seq_lens[i], -1, 3, 3)[:, :5] for i in keep]).float().numpy()
    pose_all = torch.cat([d["pose"][i].reshape(seq_lens[i], -1) for i in keep])          # (F, 216|72)
    mm["pose_aa"][:] = _as_axis_angle(pose_all, F).float().numpy()
    mm["joint"][:] = torch.cat([d["joint"][i].reshape(seq_lens[i], -1, 3)[:, :24] for i in keep]).float().numpy()
    mm["tran"][:] = torch.cat([d["tran"][i].reshape(seq_lens[i], 3) for i in keep]).float().numpy()
    for m in mm.values():
        m.flush()
    del mm
    meta = {
        "version": VERSION,
        "n_frames": F,
        "n_seqs": len(seq_lens),
        "seq_lens": seq_lens,
        "fields": {k: list(shp) for k, shp in FIELDS.items()},
        "src": str(pt_path),
        "src_mtime": pt_path.stat().st_mtime,
        "src_size": pt_path.stat().st_size,
    }
    (tmp / "meta.json").write_text(json.dumps(meta))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    if verbose:
        print(f"[shards] packed {pt_path.name}: {len(seq_lens)} seqs, {F} frames "
              f"({F / 25 / 3600:.2f} h @25fps) in {time.time() - t0:.1f}s -> {out_dir}", flush=True)
    return meta


def _meta_current(out_dir, pt_path):
    meta_p = out_dir / "meta.json"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return None
    st = pt_path.stat()
    if meta.get("version") != VERSION:
        return None
    if abs(meta.get("src_mtime", -1) - st.st_mtime) > 1e-3 or meta.get("src_size") != st.st_size:
        return None
    for k in FIELDS:
        if not (out_dir / f"{k}.f32").exists():
            return None
    return meta


def ensure_packed(pt_path, shard_root, verbose=True):
    """Return the shard dir for `pt_path`, packing it first if missing or stale."""
    pt_path, shard_root = Path(pt_path), Path(shard_root)
    out_dir = shard_root / pt_path.stem
    if not pt_path.exists():
        # shard-only dataset (the .pt was deleted after packing to save disk): use the shards as they are
        if (out_dir / "meta.json").exists():
            return out_dir
        raise FileNotFoundError(f"{pt_path} missing and no shards at {out_dir}")
    if _meta_current(out_dir, pt_path) is not None:
        return out_dir
    shard_root.mkdir(parents=True, exist_ok=True)
    lock = shard_root / (pt_path.stem + ".lock")
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > _LOCK_STALE_S:
                    os.rmdir(lock)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(5)
            if _meta_current(out_dir, pt_path) is not None:
                return out_dir
    try:
        if _meta_current(out_dir, pt_path) is None:
            pack_pt(pt_path, out_dir, verbose=verbose)
    finally:
        try:
            os.rmdir(lock)
        except FileNotFoundError:
            pass
    return out_dir


class ShardStore:
    """Lazy memmap view of one packed dataset. Opens its files on first use in each process."""

    def __init__(self, shard_dir):
        self.dir = Path(shard_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.n_frames = int(self.meta["n_frames"])
        self.seq_lens = [int(x) for x in self.meta["seq_lens"]]
        self._mm = None

    def _open(self):
        if self._mm is None:
            self._mm = {k: np.memmap(self.dir / f"{k}.f32", dtype=np.float32, mode="r",
                                     shape=(self.n_frames,) + tuple(shp))
                        for k, shp in FIELDS.items()}
        return self._mm

    def read(self, field, start, length):
        """Contiguous frame slice as a (copied, owned) torch tensor."""
        arr = self._open()[field][start:start + length]
        return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_mm"] = None          # never pickle open memmaps (spawned workers reopen lazily)
        return s


def shard_root_for(data_dir):
    """Default shard location: env IMUPOSER_SHARD_DIR, else <data_dir>/../shards_<data_dir name>."""
    env = os.environ.get("IMUPOSER_SHARD_DIR")
    if env:
        return Path(env)
    data_dir = Path(data_dir)
    return data_dir.parent / f"shards_{data_dir.name}"
