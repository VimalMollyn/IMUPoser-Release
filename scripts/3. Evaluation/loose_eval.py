r"""
Synthetic loose-pocket robustness eval.

We have NO real loose-pocket-phone data (DIP's IMUs are strap-mounted), so to measure whether the
loose-augmentation training actually helps a shifting pocket phone, we CORRUPT the pocket sensor at
test time (a random calibration offset + slow drift + a mid-sequence re-seat + accel jostle -- the
same perturbations the training injects, but with fresh draws) and compare how much each model
degrades. A model trained with loose aug should degrade LESS.

Compare baseline vs loose-trained under identical corruption (fixed seed):
  uv run python "scripts/3. Evaluation/loose_eval.py" \
      --members checkpoints/nymeria/ft_lwrwrp_s1/last.ckpt,checkpoints/nymeria/ft_lwrwrp_loose_s1/last.ckpt \
      --combo lw_rw_rp --loose-slot 3 --sev 0.30
"""
import argparse, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math as M
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree

DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
SIP = [1, 2, 16, 17]
IGN = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
ACC_SCALE = 30.0


def _detect(sd):
    return "TransformerIMUPoser" if any(k.startswith("net.enc.") for k in sd) else "GlobalModelIMUPoser"


def build(path, cfg, dev):
    sd = torch.load(path if path.endswith(".ckpt") else path + "/last.ckpt",
                    map_location=dev, weights_only=False)["state_dict"]
    cfg.model = "AvatarPoserModel"; m = get_model(cfg); m.load_state_dict(sd, strict=False)
    return m.eval().to(dev)


def corrupt_pocket(ori5, acc5, slot, sev, g):
    """Apply a loose perturbation to one sensor slot: calib offset + drift + mid-seq re-seat + accel jostle.
    Deterministic given generator g, so every model sees the SAME corruption."""
    n = ori5.shape[0]
    o = ori5.clone(); a = acc5.clone()
    # constant calibration offset
    aa = torch.randn(3, generator=g) * sev
    Rc = M.axis_angle_to_rotation_matrix(aa.unsqueeze(0))[0]
    o[:, slot] = torch.matmul(Rc, o[:, slot])
    # slow drift (rad/s integrated)
    t = (torch.arange(n, dtype=torch.float32) / 25.0).unsqueeze(1)
    bias = torch.randn(3, generator=g) * (sev * 0.13)
    Rd = M.axis_angle_to_rotation_matrix(bias.unsqueeze(0) * t)
    o[:, slot] = torch.matmul(Rd, o[:, slot])
    # mid-sequence re-seat
    if n > 2:
        t0 = int(torch.randint(1, n, (1,), generator=g).item())
        ar = torch.randn(3, generator=g) * sev
        Rr = M.axis_angle_to_rotation_matrix(ar.unsqueeze(0))[0]
        o[t0:, slot] = torch.matmul(Rr, o[t0:, slot])
    # accel jostle (m/s^2)
    a[:, slot] = a[:, slot] + torch.randn(n, 3, generator=g) * (sev * 15.0)
    return o, a


def eval_model(model, data, combo, dev, bm, corrupt=False, slot=3, sev=0.30, seed=0):
    I3 = torch.eye(3, device=dev)
    sip_sum, n_sum = 0.0, 0
    for ai, oi, gp in zip([d.view(-1, 6, 3)[:, :5].float() for d in data["acc"]],
                          [d.view(-1, 6, 3, 3)[:, :5].float() for d in data["ori"]],
                          [d.view(-1, 24, 3, 3).float() for d in data["pose"]]):
        n = ai.shape[0]
        if corrupt:
            g = torch.Generator().manual_seed(seed + n_sum)   # per-seq but reproducible across models
            oi, ai = corrupt_pocket(oi, ai, slot, sev, g)
        ca = torch.zeros_like(ai); co = torch.zeros_like(oi)
        ca[:, combo] = ai[:, combo] / ACC_SCALE; co[:, combo] = oi[:, combo]
        inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
        with torch.no_grad():
            r6 = model(inp.unsqueeze(0), [n])[0, :, :144]
        pr = r6d_to_rotation_matrix(r6).view(n, 24, 3, 3); gt = gp.to(dev)
        p, t = pr.clone(), gt.clone(); p[:, IGN] = I3; t[:, IGN] = I3
        gpred = bm.forward_kinematics(p)[0]; ggt = bm.forward_kinematics(t)[0]
        gdeg = radian_to_degree(angle_between(gpred.reshape(-1, 3, 3), ggt.reshape(-1, 3, 3)).view(n, 24))
        sip_sum += gdeg[:, SIP].mean(1).sum().item(); n_sum += n
    return sip_sum / n_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma-separated ckpt paths to compare")
    ap.add_argument("--combo", default="lw_rw_rp")
    ap.add_argument("--loose-slot", type=int, default=3, help="sensor slot to corrupt (3=rp)")
    ap.add_argument("--sev", type=float, default=0.30, help="corruption severity (rad std of calib/reseat)")
    ap.add_argument("--data", default="dip_test.pt"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg = Config(model="AvatarPoserModel", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True,
                 device=str(dev).replace("cuda:", ""), mkdir=False)
    cfg.processed_imu_poser_25fps = DD
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)
    combo = amass_combos[a.combo]
    data = torch.load(DD / a.data, weights_only=False)

    print(f"[loose_eval] combo={a.combo} corrupt slot={a.loose_slot} sev={a.sev}  data={a.data}")
    print(f"{'model':40s} {'clean SIP':>10s} {'loose SIP':>10s} {'degradation':>12s}")
    for path in a.members.split(","):
        m = build(path, cfg, dev)
        clean = eval_model(m, data, combo, dev, bm, corrupt=False)
        loose = eval_model(m, data, combo, dev, bm, corrupt=True, slot=a.loose_slot, sev=a.sev)
        name = Path(path).parent.name if path.endswith(".ckpt") else Path(path).name
        print(f"{name:40s} {clean:10.2f} {loose:10.2f} {loose-clean:+11.2f}")
    print("(lower loose SIP and smaller degradation = more robust to a shifting pocket phone)")


if __name__ == "__main__":
    main()
