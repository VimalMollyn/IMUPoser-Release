r"""
Render DIP-test pose predictions as an MP4: ground truth + one or more trained models, side by side,
3D skeletons. The UN-SENSED limbs for lw_rp_h (right arm + left leg — no IMU) are drawn in RED so you can
see directly how each model handles the limbs it can't observe. Root-centred per frame (root-relative,
matching the metric). Headless (matplotlib Agg -> cv2 mp4).

  uv run python "scripts/3. Evaluation/render_dip.py" --seq 0 --frames 0:200 \
      --models "GT,clean=exp75_spec_clean_s1,ignore-uns=exp112_ignore_unsensed_s4" --out out.mp4
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix

REPO = Path(__file__).resolve().parents[2]
DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
# un-sensed limb joints for lw_rp_h: right arm + left leg
UNSENSED = {1, 4, 7, 10, 14, 17, 19, 21, 23}
GRAFT = [1, 4, 7, 10, 14, 17, 19, 21, 23]                 # un-sensed limbs to replace via retrieval
SENSF = [2, 5, 8, 16, 18, 20, 13, 15, 12, 3, 6, 9]        # sensed joints used as the retrieval key


def _retrieve_db(bm_dd, dev):
    """Subsampled AMASS pose DB for nearest-neighbour un-sensed-limb completion."""
    DB = []
    for ds in ["CMU", "KIT", "BMLmovi", "HumanEva", "SFU"]:
        f = bm_dd / f"{ds}.pt"
        if f.exists():
            for p in torch.load(f, weights_only=False)["pose"]:
                DB.append(p.view(-1, 24, 3, 3).float()[::8])
    DB = torch.cat(DB).to(dev)
    DBf = DB[:, SENSF].reshape(DB.shape[0], -1)
    return DB, DBf / DBf.norm(dim=1, keepdim=True)


def bestval(name):
    fs = glob.glob(f"checkpoints/autoresearch/{name}/*val_loss*.ckpt")
    return min(fs, key=lambda f: float(f.split("=")[-1][:-5])) if fs else f"checkpoints/autoresearch/{name}/last.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--frames", default="0:200")
    ap.add_argument("--models", required=True, help="comma list: 'GT,label=ckptdir,...'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    s, e = (int(x) for x in a.frames.split(":"))
    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device="0", mkdir=False)
    cfg.processed_imu_poser_25fps = DD
    dev = cfg.device
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    parent = bm.parent
    data = torch.load(DD / "dip_test.pt", weights_only=False)
    acc = data["acc"][a.seq].view(-1, 6, 3)[:, :5].float()[s:e]
    ori = data["ori"][a.seq].view(-1, 6, 3, 3)[:, :5].float()[s:e]
    gtp = data["pose"][a.seq].view(-1, 24, 3, 3).float()[s:e]
    n = acc.shape[0]
    idx = amass_combos["lw_rp_h"]

    def joints_from_pose(pose):                       # (N,24,3,3) local -> (N,24,3) global joints
        out = []
        for i in range(0, pose.shape[0], 500):
            _, j, _ = bm.forward_kinematics(pose[i:i + 500].to(dev), calc_mesh=True)
            out.append(j.cpu())
        j = torch.cat(out)
        return (j - j[:, :1]).numpy()                 # root-centred

    panels = []
    for spec in a.models.split(","):
        if spec.strip() == "GT":
            panels.append(("GT", joints_from_pose(gtp)))
            continue
        label, name = spec.rsplit("=", 1)
        retr = name.endswith(":retrieve")                 # graft NN-retrieved plausible un-sensed limbs
        name = name.replace(":retrieve", "")
        m = get_model(cfg)
        m.load_state_dict(torch.load(bestval(name), map_location=dev, weights_only=False)["state_dict"], strict=False)
        m = m.eval().to(dev)
        ca, co = torch.zeros_like(acc), torch.zeros_like(ori)
        ca[:, idx] = acc[:, idx] / cfg.acc_scale
        co[:, idx] = ori[:, idx]
        inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
        with torch.no_grad():
            pr = r6d_to_rotation_matrix(m(inp.unsqueeze(0), [n])[0, :, :144]).view(n, 24, 3, 3)
        if retr:
            DB, DBf = _retrieve_db(DD, dev)
            qf = pr[:, SENSF].reshape(n, -1)
            qf = qf / qf.norm(dim=1, keepdim=True)
            nbr = torch.cat([(qf[s:s + 1000] @ DBf.T).argmax(1) for s in range(0, n, 1000)])
            pr[:, GRAFT] = DB[nbr][:, GRAFT]
        panels.append((label, joints_from_pose(pr.cpu())))
        print(f"  rendered preds for {label}", flush=True)

    # fixed cube limits from GT
    allj = np.concatenate([p[1] for p in panels])
    rng = np.abs(allj).max() * 1.05
    W, H = 420, 480
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W * len(panels), H))
    for f in range(n):
        frames = []
        for label, J in panels:
            fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            j = J[f]
            for ji, pa in enumerate(parent):
                if pa is None:
                    continue
                red = ji in UNSENSED
                ax.plot([j[pa, 0], j[ji, 0]], [j[pa, 2], j[ji, 2]], [j[pa, 1], j[ji, 1]],
                        c=("red" if red else "steelblue"), lw=(3 if red else 2))
            ax.scatter(j[:, 0], j[:, 2], j[:, 1], c="k", s=6)
            ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng); ax.set_zlim(-rng, rng)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=8, azim=70)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(label, fontsize=11)
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR))
            plt.close(fig)
        row = cv2.hconcat([cv2.resize(fr, (W, H)) for fr in frames])
        cv2.putText(row, "red = un-sensed limbs (R arm, L leg)", (8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        vw.write(row)
    vw.release()
    print("wrote", a.out, f"({n} frames)", flush=True)


if __name__ == "__main__":
    main()
