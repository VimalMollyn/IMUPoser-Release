r"""Dump a model's raw predictions on an eval file (pose matrices + FK joints + GT) for external post-processing, e.g.
PIP / MobilePoser's physics optimizer, which runs in its own environment (rbdl).

  uv run python dump_predictions.py --data imuposer_all.pt --members <ckpt[,ckpt]> --out preds.pt [--stride 31]

Output: {"pred": [(T,24,3,3)], "gt": [(T,24,3,3)], "joints": [(T,24,3) FK joint positions of the prediction, root at 0],
         "acc": [(T,6,3)], "ori": [(T,6,3,3)], "names": [...]} (torch, CPU).
"""
import argparse, os, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix

DD = Path(os.environ.get("IMUPOSER_25FPS_DIR", "/home/vimal/imuposer_data/processed_imuposer_25fps"))
JI5 = [18, 19, 1, 2, 15]


def _detect(sd):
    if any(k.startswith("net.enc.") for k in sd): return "TransformerIMUPoser"
    return "GlobalModelIMUPoser"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="imuposer_all.pt"); ap.add_argument("--members", required=True)
    ap.add_argument("--combo", default="lw_rw_rp"); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=0, help="TF_EVAL_STRIDE for overlapping windows (0 = protocol tiling)")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    if a.stride: os.environ["TF_EVAL_STRIDE"] = str(a.stride)
    cfg = Config(model="GlobalModelIMUPoser", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device=a.device, mkdir=False)
    cfg.processed_imu_poser_25fps = DD
    dev = cfg.device
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)

    def build(path):
        sd = torch.load(path, map_location=dev, weights_only=False)["state_dict"]
        cfg.model = _detect(sd)
        if "net.inp.weight" in sd:
            os.environ["TF_DMODEL"] = str(sd["net.inp.weight"].shape[0])
            os.environ["TF_LAYERS"] = str(len({k.split(".")[3] for k in sd if k.startswith("net.enc.layers.")}))
            os.environ["TF_FF"] = str(sd["net.enc.layers.0.linear1.weight"].shape[0])
        m = get_model(cfg); m.load_state_dict(sd, strict=False); return m.eval().to(dev)
    members = [build(p) for p in a.members.split(",")]
    combo = amass_combos[a.combo]
    data = torch.load(DD / a.data, weights_only=False)
    out = {"pred": [], "gt": [], "joints": [], "acc": [], "ori": [], "names": data.get("names", [])}
    for ai, oi, gp in zip(data["acc"], data["ori"], data["pose"]):
        ai = ai.view(-1, 6, 3).float(); oi = oi.view(-1, 6, 3, 3).float(); gp = gp.view(-1, 24, 3, 3).float()
        n = ai.shape[0]
        ca, co = torch.zeros_like(ai[:, :5]), torch.zeros_like(oi[:, :5])
        ca[:, combo] = ai[:, :5][:, combo] / cfg.acc_scale; co[:, combo] = oi[:, :5][:, combo]
        inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
        with torch.no_grad():
            r6 = sum(m(inp.unsqueeze(0), [n])[0, :, :144] for m in members) / len(members)
            R = r6d_to_rotation_matrix(r6).view(n, 24, 3, 3)
            J = torch.cat([bm.forward_kinematics(R[s:s + 512])[1] for s in range(0, n, 512)])    # (n,24,3)
        out["pred"].append(R.cpu()); out["gt"].append(gp); out["joints"].append(J.cpu()); out["acc"].append(ai); out["ori"].append(oi)
    torch.save(out, a.out)
    print(f"dumped {len(out['pred'])} sequences, {sum(p.shape[0] for p in out['pred'])} frames -> {a.out}")


if __name__ == "__main__":
    main()
