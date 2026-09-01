r"""
Evaluate a DECOUPLED / part-based model on dip_test: each single-sensor specialist predicts only its
own body region, and the regions are STITCHED into a full-body pose (torso/neck/head, which have no
sensor in the lw_rw_rp config, are filled with rest pose). Compare against the JOINT lw_rw_rp model
(all 3 sensors -> whole body) with --joint.

  uv run python "scripts/3. Evaluation/decoupled_eval.py" \
      --lw checkpoints/decoupled/lw_arm/last.ckpt \
      --rw checkpoints/decoupled/rw_arm/last.ckpt \
      --rp checkpoints/decoupled/rp_legs/last.ckpt \
      --joint checkpoints/decoupled/joint_lwrwrp_cur/last.ckpt --data dip_test.pt
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from imuposer.config import Config, amass_combos
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.models.utils import get_model
from imuposer.math.angular import r6d_to_rotation_matrix, angle_between, radian_to_degree

DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
SIP = [1, 2, 16, 17]
IGN = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
FPS = 25.0
# sensor slot -> SMPL joint (for feeding ori into the 5-slot input); combo indexes these slots
JI5 = [18, 19, 1, 2, 15]
# region each single sensor is responsible for (its own limb). Torso/neck/head have no sensor.
REGION = {"lw": [13, 16, 18, 20], "rw": [14, 17, 19, 21], "rp": [0, 1, 2, 4, 5, 7, 8, 10, 11]}
COMBO = {"lw": amass_combos["lw"], "rw": amass_combos["rw"], "rp": amass_combos["rp"]}


def _detect(sd):
    if any(k.startswith("net.enc.") for k in sd): return "TransformerIMUPoser"
    return "GlobalModelIMUPoser"


def seq_metrics(pr, gt, bm, dev):
    I3 = torch.eye(3, device=dev); n = pr.shape[0]
    p, t = pr.clone(), gt.clone(); p[:, IGN] = I3; t[:, IGN] = I3
    gp, jp, _ = bm.forward_kinematics(p, calc_mesh=True)
    gg, jg, _ = bm.forward_kinematics(t, calc_mesh=True)
    off = (jg[:, :1] - jp[:, :1])
    g = radian_to_degree(angle_between(gp.reshape(-1, 3, 3), gg.reshape(-1, 3, 3)).view(n, 24))
    sip = g[:, SIP].mean(1).sum().item(); mpjre = g.mean(1).sum().item()
    mpjpe = (((jp + off) - jg).norm(dim=2).mean(1) * 100).sum().item()
    return dict(sip=sip, mpjre=mpjre, mpjpe=mpjpe, n=n, perjoint=g.sum(0).cpu())


def build(path, cfg, dev):
    sd = torch.load(path, map_location=dev, weights_only=False)["state_dict"]
    cfg.model = _detect(sd); m = get_model(cfg); m.load_state_dict(sd, strict=False)
    return m.eval().to(dev)


def feed(model, acc, ori, combo, cfg, dev):
    """Run a model on one sequence with only `combo` sensors active -> (n,24,3,3) rotations."""
    n = acc.shape[0]
    ca, co = torch.zeros_like(acc), torch.zeros_like(ori)
    ca[:, combo] = acc[:, combo] / cfg.acc_scale; co[:, combo] = ori[:, combo]
    inp = torch.cat([ca.reshape(n, -1), co.reshape(n, -1)], 1).to(dev)
    with torch.no_grad():
        r6 = model(inp.unsqueeze(0), [n])[0, :, :144]
    return r6d_to_rotation_matrix(r6).view(n, 24, 3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lw", required=True); ap.add_argument("--rw", required=True); ap.add_argument("--rp", required=True)
    ap.add_argument("--joint", default=None, help="joint lw_rw_rp model to compare against")
    ap.add_argument("--data", default="dip_test.pt"); ap.add_argument("--device", default="0")
    a = ap.parse_args()

    cfg = Config(model="AvatarPoserModel", project_root_dir=str(REPO), joints_set=amass_combos["global"],
                 normalize="no_translation", r6d=True, loss_type="mse", use_joint_loss=True, device=a.device, mkdir=False)
    cfg.processed_imu_poser_25fps = DD
    dev = cfg.device
    bm = ParametricModel(cfg.og_smpl_model_path, device=dev)

    parts = {k: build(getattr(a, k), cfg, dev) for k in ("lw", "rw", "rp")}
    joint_m = build(a.joint, cfg, dev) if a.joint else None
    data = torch.load(DD / a.data, weights_only=False)

    acc = dict(dec={k: 0.0 for k in ("sip", "mpjre", "mpjpe", "n")}, joint={k: 0.0 for k in ("sip", "mpjre", "mpjpe", "n")})
    pj_dec = torch.zeros(24); pj_joint = torch.zeros(24)
    lw_rw_rp = amass_combos["lw_rw_rp"]
    t0 = time.time()
    for ai, oi, gp in zip([d.view(-1, 6, 3)[:, :5].float() for d in data["acc"]],
                          [d.view(-1, 6, 3, 3)[:, :5].float() for d in data["ori"]],
                          [d.view(-1, 24, 3, 3).float() for d in data["pose"]]):
        n = ai.shape[0]; gt = gp.to(dev)
        # DECOUPLED: assemble each region from its own single-sensor model; torso/head -> identity.
        asm = torch.eye(3, device=dev).repeat(n, 24, 1, 1)
        for k, model in parts.items():
            R = feed(model, ai, oi, COMBO[k], cfg, dev)
            asm[:, REGION[k]] = R[:, REGION[k]]
        md = seq_metrics(asm, gt, bm, dev)
        for kk in ("sip", "mpjre", "mpjpe", "n"): acc["dec"][kk] += md[kk]
        pj_dec += md["perjoint"]
        # JOINT baseline
        if joint_m is not None:
            Rj = feed(joint_m, ai, oi, lw_rw_rp, cfg, dev)
            mj = seq_metrics(Rj, gt, bm, dev)
            for kk in ("sip", "mpjre", "mpjpe", "n"): acc["joint"][kk] += mj[kk]
            pj_joint += mj["perjoint"]

    def show(tag, d, pj):
        N = d["n"]
        if N == 0: return
        print(f"== {tag} (N={int(N)}) ==")
        print(f"  SIP {d['sip']/N:6.2f}   MPJRE {d['mpjre']/N:6.2f}   MPJPE {d['mpjpe']/N:6.2f} cm")
        JN = ["pelv","lhip","rhip","sp1","lkne","rkne","sp2","lank","rank","sp3","lfoot","rfoot",
              "neck","lcol","rcol","head","lsho","rsho","lelb","relb","lwri","rwri","lhnd","rhnd"]
        pjm = pj / N
        for grp, js in (("Larm", REGION["lw"]), ("Rarm", REGION["rw"]), ("legs", REGION["rp"]),
                        ("torso/head", [3, 6, 9, 12, 15])):
            print(f"    {grp:11s} " + "  ".join(f"{JN[j]}:{pjm[j]:4.1f}" for j in js))

    print(f"[{a.data}]  ({time.time()-t0:.0f}s)")
    show("DECOUPLED (per-sensor part models, torso/head=rest)", acc["dec"], pj_dec)
    if joint_m is not None:
        show("JOINT (lw_rw_rp, all sensors -> whole body)", acc["joint"], pj_joint)
        d, j = acc["dec"], acc["joint"]
        print(f"  delta SIP {d['sip']/d['n'] - j['sip']/j['n']:+.2f}  (decoupled - joint; negative = decoupled better)")


if __name__ == "__main__":
    main()
