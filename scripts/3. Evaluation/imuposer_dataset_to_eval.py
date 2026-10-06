r"""Convert the collected IMUPoser dataset (CHI'23 release: 10 participants x ~16 activities of real phone/watch/head
IMU with MoSh'd SMPL ground truth) into the evaluator's format, for ZERO-SHOT evaluation (no fine-tuning on it).

Each release .pkl holds {"imu": (T, 60) = [5 sensors x 3 accel | 5 sensors x 9 orientation], "pose": (T, 72) SMPL
axis-angle}, sensors ordered left wrist, right wrist, left pocket, right pocket, head, everything in one global frame,
accel in m/s^2 at 25 fps. The pipeline's files carry 6 sensor slots (the 6th is the pelvis, never used by the
lw_rw_rp models) and rotation-matrix poses, so: pad slot 6 with zeros / identity, convert axis-angle to matrices.

Writes <out_dir>/imuposer_all.pt (every recording) and imuposer_P<k>.pt per participant, plus imuposer_index.json.

  uv run python imuposer_dataset_to_eval.py [--src DIR] [--out_dir DIR]
"""
import argparse, glob, json, os, pickle
from pathlib import Path
import torch
from imuposer.math.angular import axis_angle_to_rotation_matrix

SRC = "/media/vimal/Samsung_T5_2TB/CHI23/IMUPoser/CameraReady/dataset-release/imuposer_dataset"
OUT = os.environ.get("IMUPOSER_25FPS_DIR", "/home/vimal/imuposer_data/processed_imuposer_25fps")


def convert(files):
    acc, ori, pose, names = [], [], [], []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        imu, aa = d["imu"].float(), d["pose"].float()
        T = imu.shape[0]
        a5 = imu[:, :15].view(T, 5, 3); o5 = imu[:, 15:].view(T, 5, 3, 3)
        a6 = torch.zeros(T, 6, 3); a6[:, :5] = a5
        o6 = torch.eye(3).expand(T, 6, 3, 3).clone(); o6[:, :5] = o5
        R = axis_angle_to_rotation_matrix(aa.reshape(-1, 3)).view(T, 24, 3, 3)
        acc.append(a6); ori.append(o6); pose.append(R)
        names.append(f"{Path(f).parent.name}/{Path(f).stem}")
    return {"acc": acc, "ori": ori, "pose": pose, "names": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC); ap.add_argument("--out_dir", default=OUT)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    parts = sorted({Path(f).parent.name for f in glob.glob(f"{a.src}/P*/*.pkl")}, key=lambda p: int(p[1:]))
    index = {}
    allf = []
    for p in parts:
        files = sorted(glob.glob(f"{a.src}/{p}/*.pkl"), key=lambda f: int(Path(f).name.split(".")[0]))
        d = convert(files)
        torch.save(d, out / f"imuposer_{p}.pt")
        index[p] = [(n, int(x.shape[0])) for n, x in zip(d["names"], d["pose"])]
        allf += files
        print(f"{p}: {len(files)} recordings, {sum(x.shape[0] for x in d['pose']) / 25 / 60:.1f} min", flush=True)
    d = convert(allf)
    torch.save(d, out / "imuposer_all.pt")
    (out / "imuposer_index.json").write_text(json.dumps(index, indent=1))
    tot = sum(x.shape[0] for x in d["pose"])
    print(f"all: {len(allf)} recordings, {tot} frames = {tot / 25 / 3600:.2f} h at 25 fps -> {out / 'imuposer_all.pt'}")


if __name__ == "__main__":
    main()
