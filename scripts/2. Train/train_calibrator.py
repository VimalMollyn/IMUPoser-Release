r"""
TIC-style IMU calibrator (Stage 2 of pocket-phone robustness).

Adapts "Transformer IMU Calibrator" (TIC, SIGGRAPH'25) to our SPARSE setup. A small transformer sees
the (corrupted) IMU window of the active sensors and predicts a per-frame CORRECTION for the loose
pocket sensor -- a rotation R_corr applied to its orientation and an additive delta to its
acceleration -- to recover the clean signal. It sits BEFORE the (frozen) pose model, so it corrects
the input without retraining the poser.

Self-supervised: we synthesize IMU from mocap, so the CLEAN pocket signal is known. We apply a random
loose corruption (calib offset + drift + mid-window re-seat + accel jostle -- the real failure mode)
and train the calibrator to invert it. TIC trains on synthetic AMASS+DIP the same way; their real
5-subject set is only their eval (we have no real loose data, so eval is synthetic -- see loose_eval).

  CUDA_VISIBLE_DEVICES=1 uv run python "scripts/2. Train/train_calibrator.py" \
      --combo lw_rw_rp --loose-slot 3 --epochs 40 --out checkpoints/calibrator/cal_lwrwrp
"""
import argparse, glob, math as pymath, os, sys, time
from pathlib import Path
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from imuposer.config import Config, amass_combos, original_amass_datasets, val_datasets, test_datasets
from imuposer import math as M
from imuposer.math.angular import r6d_to_rotation_matrix

DD = Path("/media/vimal/T7_2TB/CHI23/processed_imuposer_data/processed_imuposer_25fps")
ACC_SCALE = 30.0
WIN = 125
EYE6 = torch.tensor([1., 0, 0, 0, 1, 0])


class Calibrator(nn.Module):
    r"""60-dim corrupted IMU window -> per-frame correction (r6d rotation + accel delta) for one slot."""
    def __init__(self, d_model=128, nhead=4, layers=3, ff=512, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(60, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d_model, 9)      # 6 r6d correction + 3 accel delta
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start at identity/zero correction

    def forward(self, x):                     # x: (B,T,60)
        h = self.enc(self.inp(x))
        o = self.out(h)                        # (B,T,9)
        r6 = o[..., :6] + EYE6.to(o.device)    # residual around identity rotation
        return r6, o[..., 6:]                  # (B,T,6), (B,T,3)


def load_windows(combo, files, cap_frames=None):
    """Return clean windows: acc (K,WIN,5,3) SCALED, ori (K,WIN,5,3,3). Only combo sensors kept nonzero."""
    accs, oris, tot = [], [], 0
    keep = torch.tensor(combo)
    for f in files:
        d = torch.load(DD / f, weights_only=False)
        for a, o in zip(d["acc"], d["ori"]):
            a = a.view(-1, 6, 3)[:, :5].float() / ACC_SCALE
            o = o.view(-1, 6, 3, 3)[:, :5].float()
            n = a.shape[0]; k = n // WIN
            if k == 0:
                continue
            # mask absent sensors, then reshape into (k, WIN, 5, ...) in one shot
            am = torch.zeros(n, 5, 3); om = torch.zeros(n, 5, 3, 3)
            am[:, keep] = a[:, keep]; om[:, keep] = o[:, keep]
            accs.append(am[:k*WIN].view(k, WIN, 5, 3)); oris.append(om[:k*WIN].view(k, WIN, 5, 3, 3))
            tot += k * WIN
        if cap_frames and tot >= cap_frames:
            break
    return torch.cat(accs), torch.cat(oris)


def corrupt(acc, ori, slot, dev):
    """Apply a random loose corruption to `slot` for a whole batch. Returns corrupted (acc,ori)."""
    B, T = acc.shape[:2]
    a, o = acc.clone(), ori.clone()
    # constant calib offset (per sample), ~N(0, 0.30 rad) plus a uniform severity scale for range coverage
    sev = (0.1 + 0.35 * torch.rand(B, 1, device=dev))                     # 0.1..0.45 rad
    aa = torch.randn(B, 3, device=dev) * sev
    Rc = M.axis_angle_to_rotation_matrix(aa)
    o[:, :, slot] = torch.matmul(Rc.unsqueeze(1), o[:, :, slot])
    # drift (rad/s integrated)
    t = (torch.arange(T, device=dev, dtype=torch.float32) / 25.0).view(1, T, 1)
    bias = torch.randn(B, 3, device=dev) * (sev * 0.13)
    d = M.axis_angle_to_rotation_matrix((bias.unsqueeze(1) * t).reshape(-1, 3)).view(B, T, 3, 3)
    o[:, :, slot] = torch.matmul(d, o[:, :, slot])
    # mid-window re-seat (step change from t0)
    for i in range(B):
        t0 = int(torch.randint(1, T, (1,)).item())
        ar = torch.randn(3, device=dev) * sev[i]
        Rr = M.axis_angle_to_rotation_matrix(ar.unsqueeze(0))[0]
        o[i, t0:, slot] = torch.matmul(Rr, o[i, t0:, slot])
    # accel jostle (scaled units; sev*15 m/s^2 / 30)
    a[:, :, slot] = a[:, :, slot] + torch.randn(B, T, 3, device=dev) * (sev.unsqueeze(1) * 0.5)
    return a, o


def train_files(combo):
    all_f = sorted(x.name for x in DD.iterdir() if x.name.endswith(".pt") and "dip" not in x.name)
    val, test = set(val_datasets), set(test_datasets)
    curated = [f for f in all_f if f[:-3] in set(original_amass_datasets) and f[:-3] not in val and f[:-3] not in test]
    _nn = int(os.environ.get("CAL_NYM_CHUNKS", "3"))                  # keep light to avoid disk/RAM contention
    nym = [f for f in all_f if f.startswith("Nymeria_00_")][:_nn]     # a slice of Nymeria for motion diversity
    return curated + nym + ["dip_train.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", default="lw_rw_rp"); ap.add_argument("--loose-slot", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--out", required=True)
    ap.add_argument("--files", default="", help="comma-separated file override (for quick tests)")
    a = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    combo = amass_combos[a.combo]; slot = a.loose_slot
    files = a.files.split(",") if a.files else train_files(a.combo)
    print(f"[cal] loading {len(files)} files (combo {a.combo}, loose slot {slot}) ...", flush=True)
    acc, ori = load_windows(combo, files)
    print(f"[cal] {acc.shape[0]} windows", flush=True)

    # optional wandb (guarded: WANDB_MODE=disabled or a missing package just skips it)
    wb = None
    if os.environ.get("WANDB_MODE") != "disabled":
        try:
            import wandb as wb
            wb.init(project=os.environ.get("WANDB_PROJECT", "imu_calibrator"),
                    name=os.environ.get("WANDB_RUN_NAME", Path(a.out).name),
                    config={"combo": a.combo, "loose_slot": slot, "epochs": a.epochs,
                            "bs": a.bs, "lr": a.lr, "n_windows": int(acc.shape[0]), "n_files": len(files)})
        except Exception as e:
            print(f"[cal] wandb disabled ({e})", flush=True); wb = None

    net = Calibrator().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = torch.arange(acc.shape[0])
    best = 1e9
    for ep in range(a.epochs):
        net.train(); perm = idx[torch.randperm(idx.numel())]
        tot, to_, ta_, tbase, nb = 0.0, 0.0, 0.0, 0.0, 0
        for b in range(0, perm.numel(), a.bs):
            sel = perm[b:b+a.bs]
            ca, co = acc[sel].to(dev), ori[sel].to(dev)
            ka, ko = corrupt(ca, co, slot, dev)                          # corrupted inputs
            inp = torch.cat([ka.reshape(ka.shape[0], WIN, -1), ko.reshape(ko.shape[0], WIN, -1)], -1)
            r6, dacc = net(inp)
            Rcorr = r6d_to_rotation_matrix(r6.reshape(-1, 6)).view(-1, WIN, 3, 3)
            rec_ori = torch.matmul(Rcorr, ko[:, :, slot])               # corrected pocket ori
            rec_acc = ka[:, :, slot] + dacc                             # corrected pocket acc
            loss_o = ((rec_ori - co[:, :, slot]) ** 2).sum(dim=(-1, -2)).mean()
            loss_a = ((rec_acc - ca[:, :, slot]) ** 2).sum(-1).mean()
            loss = loss_o + loss_a
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():   # reference: uncorrected orientation error (how bad the corruption is)
                base_o = ((ko[:, :, slot] - co[:, :, slot]) ** 2).sum(dim=(-1, -2)).mean().item()
            tot += loss.item(); to_ += loss_o.item(); ta_ += loss_a.item(); tbase += base_o; nb += 1
        nb = max(nb, 1)
        avg, avg_o, avg_a, avg_base = tot/nb, to_/nb, ta_/nb, tbase/nb
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"[cal] ep {ep:2d}  loss {avg:.4f}  (ori {avg_o:.4f} vs uncorrected {avg_base:.4f}, acc {avg_a:.4f})", flush=True)
        if wb is not None:
            wb.log({"epoch": ep, "loss": avg, "loss_ori": avg_o, "loss_acc": avg_a,
                    "uncorrected_ori": avg_base, "ori_recovered_frac": 1.0 - avg_o/max(avg_base, 1e-9)})
        if avg < best:
            best = avg; torch.save({"state_dict": net.state_dict(), "combo": a.combo, "slot": slot}, out / "best.ckpt")
    torch.save({"state_dict": net.state_dict(), "combo": a.combo, "slot": slot}, out / "last.ckpt")
    print(f"[cal] DONE best_loss {best:.4f} -> {out}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
