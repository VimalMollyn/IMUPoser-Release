r"""
NVIDIA form-hoi (SOMA/MHR params @30 fps, 4135 episodes, ~29 h) -> SMPL -> synthetic IMU -> 25 fps files.

Per episode: soma_params.npz (77-joint rotvec, MHR identity + bone scales) -> SOMA-X forward kinematics
(fk_only, no skinning) -> world joint rotations/positions -> closed-form SOMA->SMPL retarget
(imuposer/datasets/soma_retarget.py) -> 30 -> 60 fps linear resample -> exact 6-vertex skinning IMU
-> 25 fps. World frame: the params are y-up but tilted ~3 deg (camera-rig frame); the ground-plane normal
is used to make gravity exactly -y. Frames flagged by the human-pose QC categories (drift, desync,
penetration, jolts, jitter) and frames with pose_valid_mask=0 are cut out; the object-only checks
(Chamfer, silhouette) are kept. Remaining segments shorter than 2 s are dropped.

  uv run python "scripts/1. Preprocessing/formhoi_convert.py" --gpu 1 [--limit 50]
"""
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
import torch

from imuposer.config import Config
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.datasets.synth_imu import synthesize_sequences, amass_dir_to_25fps, resample_linear, resample_pose_aa
from imuposer.datasets.soma_retarget import SMPLSkeleton, SOMAtoSMPL, align_rotation, soma_x_reference, mesh_calibrated_offsets

ROOT = Path(os.environ.get("FORMHOI_DIR", "/home/vimal/Downloads/form-hoi/data"))
OUT = Path(os.environ.get("IMUPOSER_OUT_DIR", "/home/vimal/imuposer_data"))
SRC_FPS = 30.0
BAD_CATS = ("misalignment", "drift", "desynchron", "penetration", "jolt", "jitter")


def segments(valid, min_len):
    """valid (N,) bool -> list of (start, end) runs of True with length >= min_len."""
    out, s = [], None
    for i, v in enumerate(list(valid) + [False]):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s >= min_len: out.append((s, i))
            s = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--chunk_hours", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="FORMHOI")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(f"cuda:{a.gpu}")
    from soma import SOMALayer
    from soma.assets import get_assets_dir
    from soma.smpl.transfer import _prepare_layer_identity, _pose_layer
    from soma.io import load_soma_npz
    from soma.units import Unit

    cfg = Config(project_root_dir=str(Path(__file__).resolve().parents[2]), device=a.gpu, mkdir=False)
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    skel = SMPLSkeleton(bm, dev)
    names, ref_rot, ref_pos, soma = soma_x_reference(dev)
    conv = SOMAtoSMPL(skel, names, ref_pos, ref_rot)
    conv.off = mesh_calibrated_offsets(skel, bm, soma, dev, cfg.og_smpl_model_path)   # direction + twist from the meshes
    out25 = OUT / "processed_imuposer_25fps"; out25.mkdir(parents=True, exist_ok=True)
    if sorted(out25.glob(f"{a.tag}_*.pt")) and not a.limit:
        print(f"chunks already exist for {a.tag}; delete them to redo", flush=True); return

    eps = sorted(glob.glob(str(ROOT / "*" / "soma_params.npz")))
    if a.limit: eps = eps[:a.limit]
    print(f"{len(eps)} episodes", flush=True)
    buf, cid, hours, nseg, nskip, cut, t0 = [], 0, 0.0, 0, 0, 0, time.time()
    chunk_frames = a.chunk_hours * 3600 * 60
    tilt_log = []

    def flush():
        nonlocal buf, cid, hours
        if not buf: return
        out = synthesize_sequences(buf, bm, dev)
        fdata = amass_dir_to_25fps(out, device=dev)
        p = out25 / f"{a.tag}_{cid:03d}.pt"
        torch.save(fdata, p.with_suffix(".pt.tmp")); os.replace(p.with_suffix(".pt.tmp"), p)
        h = sum(x.shape[0] for x in fdata["pose"]) / 25 / 3600; hours += h
        print(f"  wrote {p.name}: {len(buf)} segs, {h:.2f} h | total {hours:.1f} h, {nseg} segs, {(time.time()-t0)/60:.1f} min", flush=True)
        buf = []; cid += 1

    with torch.no_grad():
        for ei, npz in enumerate(eps):
            d = Path(npz).parent
            try:
                data = load_soma_npz(npz)
                poses = torch.from_numpy(np.asarray(data["poses"])).float().to(dev)
                transl = torch.from_numpy(np.asarray(data["transl"])).float().to(dev)
                # identity / scales are per-person constants stored per frame; prepare them with a SINGLE frame
                # (SOMA-X's MHR identity backend caches per-batch buffers and breaks when the batch size changes
                # between episodes) and let pose() broadcast.
                ident_all = np.asarray(data["identity_coeffs"], np.float32)
                assert np.abs(ident_all - ident_all[:1]).max() < 1e-4, "identity varies within the episode"
                ident = torch.from_numpy(ident_all[:1]).to(dev)
                prep = {}
                if "scale_params" in data: prep["scale_params"] = torch.from_numpy(np.asarray(data["scale_params"], np.float32)[:1]).to(dev)
                if "bone_length_flexibles" in data:
                    prep["kwargs"] = {"bone_length_flexibles": torch.from_numpy(np.asarray(data["bone_length_flexibles"], np.float32)[:1]).to(dev)}
                N = poses.shape[0]
                assert list(data["joint_names"]) == names[1:], "joint order changed"
                _prepare_layer_identity(soma, ident, prep)
                so = _pose_layer(soma, poses, transl, pose2rot=True, absolute_pose=bool(data["absolute_pose"]),
                                 extra_kwargs={"apply_correctives": False, "fk_only": True})
                tr = so["transforms"]
                G_soma, P_soma = tr[:, :, :3, :3], tr[:, :, :3, 3]
                # gravity: ground-plane normal (sign so it points from feet to head) -> exactly +y
                n = np.asarray(json.load(open(d / "ground_plane.json"))["plane"][:3], np.float64)
                n = torch.tensor(n / np.linalg.norm(n), dtype=torch.float32, device=dev)
                body_up = (P_soma[:, names.index("Head")] - P_soma[:, names.index("Hips")]).mean(0)
                if torch.dot(n, body_up) < 0: n = -n
                tilt_log.append(float(torch.rad2deg(torch.acos(n[1].clamp(-1, 1)))))
                Rw = align_rotation(n, torch.tensor([0., 1., 0.], device=dev))
                aa, trn, G2, P2 = conv(G_soma, P_soma, world_rot=Rw)
                # validity: pose_valid_mask + human-pose QC categories
                valid = np.load(d / "pose_valid_mask.npy").astype(bool)
                if valid.shape[0] != N: valid = np.ones(N, bool)
                for s in json.load(open(d / "failure_segments.json")):
                    if any(b in str(s.get("failure_category", "")).lower() for b in BAD_CATS):
                        valid[int(s["start_frame"]):int(s["end_frame"])] = False
                valid &= torch.isfinite(aa).all(-1).all(-1).cpu().numpy() & torch.isfinite(trn).all(-1).cpu().numpy()
                cut += int((~valid).sum())
                for s, e in segments(valid, int(2 * SRC_FPS)):
                    aa60 = resample_pose_aa(aa[s:e], SRC_FPS, 60.0).cpu(); tr60 = resample_linear(trn[s:e].cpu(), SRC_FPS, 60.0)   # rotations: matrix-space interp (GPU)
                    buf.append((aa60, tr60, torch.zeros(10))); nseg += 1
                if sum(x[0].shape[0] for x in buf) >= chunk_frames:
                    flush()
            except Exception as ex:
                nskip += 1; print(f"  ERR {d.name}: {type(ex).__name__} {ex}", flush=True)
            if (ei + 1) % 500 == 0:
                print(f"  [{ei+1}/{len(eps)}] segs {nseg}, cut frames {cut}, tilt mean {np.mean(tilt_log):.2f} deg, {(time.time()-t0)/60:.1f} min", flush=True)
        flush()
    print(f"DONE {a.tag}: {nseg} segments from {len(eps)-nskip} episodes ({nskip} errors), {cut} frames cut, "
          f"tilt mean/max {np.mean(tilt_log):.2f}/{np.max(tilt_log):.2f} deg, {hours:.1f} h in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
