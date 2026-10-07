r"""
Build the results / scaling-laws page for the new-motion-data campaign (lw_rw_rp @ 25 Hz).

Every run is a dir pair checkpoints/newdata/base_<tag>/ (pretrain) + ft_<tag>/ (DIP fine-tune + eval_dip_test.log).
Per run we derive: training hours (the biggest "[stream] ..." line of the base log), parameter count (from
last.ckpt, cached), epoch budget (tag / log), status, validation curve, dip_test metrics.

  uv run python "scripts/3. Evaluation/newdata_report.py" [--out path.html]
"""
import argparse, glob, html, json, math, os, re, time
from datetime import datetime
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CK = REPO / "checkpoints" / "newdata"
LOGS = Path("/home/vimal/imuposer_data/logs")
DATA = Path("/home/vimal/imuposer_data/processed_imuposer_25fps")
SHARDS = Path("/home/vimal/imuposer_data/shards_processed_imuposer_25fps")
PARAM_CACHE = CK / "_params_cache.json"
PREV_DELIVERABLE = 17.32       # lw_rw_rp 25 Hz, 2026-09-01 (curated-12 + Nymeria -> DIP FT, in-RAM loader)
# Runs trained on the first conversion of form-hoi / MotionMillion / MotionGV, which upsampled 30 -> 60 fps by
# linearly interpolating AXIS-ANGLE poses: across the +-pi wrap that collapses a limb to rest for one frame and
# the synthetic accel explodes (1.7-10.8 % of frames > 50 m/s^2). Fixed 2026-10-03 16:40 (matrix-space interp);
# these results are kept for the record but excluded from every chart and the leaderboard.
INVALID = {"treatment_s1": "pre-fix data", "treatment_s2": "pre-fix data", "scale_s20_trt": "pre-fix data",
           "scale_m20_trt": "pre-fix data", "abl_formhoi_s20": "pre-fix data"}
NOISE = 0.17                   # seed-to-seed std of dip_test SIP on this recipe (Nymeria study, n=4)
CURATED = "CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits".split(",")
GROUPS = [("curated-12 AMASS", lambda n: n in CURATED), ("Nymeria", lambda n: n.startswith("Nymeria_")),
          ("BONES-SEED", lambda n: n.startswith("BONES_")), ("form-hoi", lambda n: n.startswith("FORMHOI_")),
          ("StayStill (idle motion, LaFAN rig)", lambda n: n.startswith("STAYSTILL_")),
          ("MotionMillion mocap subsets (272-dim)", lambda n: n.startswith("MM_")), ("Motion-X (existing SMPL-X)", lambda n: n.startswith("MotionX_")),
          ("MotionGV filtered (5-frame avg, >=2 s, |acc|<=120)", lambda n: n.startswith("MGV_")),
          ("MotionGV unfiltered (>=1 s)", lambda n: n.startswith("MGVRAW_"))]
COL = {"S": "#5b6ee1", "M": "#2f9e7a", "L": "#8e6ad1", "XL": "#d98f3b"}
ARMCOL = {"curated-12": "#6b8fa3", "control": "#5b6ee1", "treatment": "#d98f3b", "treatment+GV": "#c45c8a"}


# ---------------------------------------------------------------- parsing
def read(p):
    try:
        return Path(p).read_text(errors="replace")
    except Exception:
        return ""


def parse_train_log(p):
    txt = read(p).replace("\r", "\n")
    if not txt:
        return None
    r = {"epochs_done": 0, "steps_per_epoch": None, "it_s": None, "val": {}, "status": "pending", "train_files": None, "hours": None, "windows": None}
    best = None
    for m in re.finditer(r"\[stream\] (\d+) files, (\d+) windows, ([\d.]+) h", txt):
        h = float(m.group(3))
        if best is None or h > best[2]:
            best = (int(m.group(1)), int(m.group(2)), h)
    if best:
        r["train_files"], r["windows"], r["hours"] = best
    first = {}
    for m in re.finditer(r"Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[[^\]]*?,\s*([\d.]+)it/s[^\n]*?val_loss=([\d.]+)", txt):
        e = int(m.group(1))
        r["steps_per_epoch"] = int(m.group(3)); r["it_s"] = float(m.group(4))
        if e not in first:
            first[e] = float(m.group(5))
    for e, v in first.items():
        if e >= 1:
            r["val"][e - 1] = v
    done = re.findall(r"Epoch (\d+):\s+100%", txt)
    r["epochs_done"] = (max(int(x) for x in done) + 1) if done else 0
    for f in glob.glob(str(Path(p).parent / "epoch=epoch=*-val_loss=*.ckpt")):
        m = re.search(r"epoch=epoch=(\d+)-val_loss=validation_step_loss=(\d+\.\d+)", f)
        if m:
            r["val"][int(m.group(1))] = float(m.group(2))
    if (Path(p).parent / "best_model.txt").exists():
        r["status"] = "done"
        b = read(Path(p).parent / "best_model.txt").split("\n")[0]
        m = re.search(r"epoch=(\d+)-val_loss=validation_step_loss=(\d+\.\d+)", b)
        if m:
            r["best_epoch"], r["best_val"] = int(m.group(1)), float(m.group(2))
    elif "Traceback" in txt or "CUDA out of memory" in txt:
        r["status"] = "error"
    elif r["epochs_done"] or "Epoch 0" in txt:
        r["status"] = "running"
    return r


_EVAL_RE = r"SIP\s+([\d.]+) deg\s+MPJRE\s+([\d.]+) deg\s+MPJPE\s+([\d.]+) cm\s+MPVPE\s+([\d.]+) cm\s+MPJVE\s+([\d.]+) cm/s\s+Jit\s+(\d+)"


def _eval_dict(m):
    return {"sip": float(m.group(1)), "mpjre": float(m.group(2)), "mpjpe": float(m.group(3)), "mpvpe": float(m.group(4)),
            "mpjve": float(m.group(5)), "jitter": int(m.group(6))}


def parse_eval(p, section=None):
    """Metrics line of an offline_fit.py log. section=None: the first line (network output); section='fit': the line
    under '== OFFLINE FIT' (after test-time optimisation)."""
    txt = read(p)
    if section == "fit":
        i = txt.find("== OFFLINE FIT")
        if i < 0: return None
        txt = txt[i:]
    m = re.search(_EVAL_RE, txt)
    return _eval_dict(m) if m else None


def params_of(base_dir):
    """Parameter count from last.ckpt (cached by dir name)."""
    cache = json.loads(read(PARAM_CACHE) or "{}")
    k = Path(base_dir).name
    if k in cache:
        return cache[k]
    ck = Path(base_dir) / "last.ckpt"
    if not ck.exists():
        cks = sorted(Path(base_dir).glob("*.ckpt"))
        if not cks:
            return None
        ck = cks[0]
    try:
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        n = int(sum(v.numel() for v in sd.values()))
    except Exception:
        return None
    cache[k] = n
    PARAM_CACHE.write_text(json.dumps(cache))
    return n


def size_label(n):
    if n is None: return "?"
    m = n / 1e6
    return "S" if m < 6 else "M" if m < 16 else "L" if m < 40 else "XL"


def arm_label(tag):
    t = tag.lower()
    if "+gv" in t or "_gv_" in t or t.startswith("gv") and "trt" in t: return "treatment+GV"
    if t.startswith(("gvfilt", "gvraw", "gvdl", "gvrand", "abl_", "dlall")): return "control+"
    if t.startswith("trtgvdl"): return "treatment+"
    if "_cur" in t: return "curated-12"
    if "_ctrl" in t or t.startswith("control"): return "control"
    if "_trt" in t or t.startswith("treatment"): return "treatment"
    return "other"


def budget_of(tag, base):
    m = re.search(r"(?:^|_)(?:xl|l|m|s)(\d{2,3})(?:_|$)", tag)
    if m: return int(m.group(1))
    return 60


EXTRA_LABELS = [("abl_staystill4", "+ StayStill ×4 (6 h of idle motion, repeated 4×)"), ("abl_staystill", "+ StayStill (6 h of idle motion)"),
                ("abl_formhoi", "+ form-hoi"), ("abl_bones", "+ BONES-SEED"), ("abl_mm", "+ MotionMillion mocap + Motion-X"),
                ("gvfilt", "+ MotionGV filtered (all 636 h)"), ("gvraw", "+ MotionGV unfiltered (724 h)"), ("gvdl20", "+ MotionGV, DIP-like subset (within 20°, 256 h)"),
                ("gvdl15", "+ MotionGV, DIP-like subset (within 15°)"), ("gvdl25", "+ MotionGV, DIP-like subset (within 25°)"),
                ("gvrand", "+ MotionGV, RANDOM subset of 256 h (control for the selection)"),
                ("dlallraw", "+ DIP-like new mocap data + DIP-like MotionGV raw"),
                ("dlall", "+ DIP-like new mocap data (186 h) + DIP-like MotionGV (256 h)"),
                ("trtgvdl", "+ all new mocap data + DIP-like MotionGV (256 h)"),
                ("gvrawdl20", "+ MotionGV raw, DIP-like subset (within 20°)"),
                ("treatment_rew", "reweighted: form-hoi ×3, Nymeria ×2, BONES/MM/Motion-X ×0.5"),
                ("curr_trt2ctrl", "curriculum: treatment (676 h, 60 ep) -> 20 ep on control"),
                ("mix025", "treatment, new data sampled ×0.25 per epoch"), ("mix050", "treatment, new data sampled ×0.5 per epoch"),
                ("mix200", "treatment, new data sampled ×2 per epoch"),
                ("abl_amassrest", "+ the 13 AMASS sets outside curated-12"),
                ("dl_", "DIP-like subset of the new data (pose-distance / accel rule)")]
# SOTA levers: variations of the training / fine-tuning recipe on top of a finished base. They are single models
# (so they may lead the leaderboard) but are kept out of the scaling charts and the data ablations.
LEVER_LABELS = [(r"^swa_xl60_xl60c", "XL60 + its 60-epoch continuation, six best checkpoints averaged (epochs 28–89)"),
                (r"^swa_", "three best checkpoints averaged"), (r"_cos$", "cosine LR schedule in pretraining"),
                (r"_wd1e2$", "AdamW weight decay 1e-2 in pretraining (default 1e-4)"),
                (r"_stab$", "grad-norm clip 1.0 + synthetic accel clamped to ±160 m/s² (the L/XL stability settings)"),
                (r"^cal_.*_bf16$", "fig2 calibration: same recipe in bf16 mixed precision on an RTX 4060 Ti"),
                (r"^cal_.*_fp32$", "fig2 calibration: same recipe in fp32 on an RTX 4060 Ti"),
                (r"^ftseed(\d+)_", "FT seed {0} (FT-stage noise)"), (r"^ftlr5e5_", "FT lr 5e-5"), (r"^ftlr2e4_", "FT lr 2e-4"),
                (r"^ftcos120_", "FT cosine LR, 120 ep"), (r"^ftcos_", "FT cosine LR"), (r"^ft120_", "FT 120 ep")]


def lever_label(tag):
    for pat, lab in LEVER_LABELS:
        m = re.search(pat, tag)
        if m: return lab.format(*m.groups())
    return ""


def lever_ref(tag):
    """The plain run a lever run should be compared with."""
    if tag.endswith("_cos"): return tag[:-4]
    if tag.endswith("_wd1e2"): return tag[:-6]
    if tag.endswith("_stab"): return tag[:-5]
    if tag.startswith("cal_") and (tag.endswith("_bf16") or tag.endswith("_fp32")): return "scale_" + tag[4:-5]
    if tag == "swa_xl60_xl60c_ctrl": return "scale_xl60_ctrl"
    if tag.startswith("swa_"):   # swa_<run>: the run is scale_<run> for the size grid, else the bare tag (control_s1, ...)
        return "scale_" + tag[4:] if (CK / f"base_scale_{tag[4:]}").exists() else tag[4:]
    m = re.match(r"^ft[a-z0-9]*_(.+)$", tag)
    if m: return "scale_" + m.group(1)
    return None


DIPLIKE_CSV = LOGS / "diplike_stats.csv"
DIPLIKE_GROUPS = [("DIP train (reference)", "dip_train"), ("curated AMASS: CMU", "CMU"), ("curated AMASS: BMLrub", "BioMotionLab"),
                  ("curated AMASS: KIT", "KIT"), ("Nymeria", "Nymeria_"), ("BONES-SEED", "BONES_"), ("form-hoi", "FORMHOI_"), ("StayStill", "STAYSTILL_"),
                  ("MotionMillion mocap", "MM_"), ("Motion-X", "MotionX_"), ("MotionGV filtered (sample)", "MGV_"),
                  ("MotionGV unfiltered (sample)", "MGVRAW_"), ("leftover AMASS", None)]


def diplike_table():
    """Per dataset group: hours, hours-weighted distance of sequences from DIP's mean pose, accel p95, diversity."""
    import csv as _csv
    txt = read(DIPLIKE_CSV)
    if not txt:
        return []
    rows = list(_csv.DictReader(txt.splitlines()))
    ref = read(LOGS / "diplike_ref.csv")            # the reference's own sequences (written by a separate run)
    if ref:
        rows += [r for r in _csv.DictReader(ref.splitlines()) if r["dataset"] == "dip_train"]
    out = []
    for label, pre in DIPLIKE_GROUPS:
        rs = [r for r in rows if (pre and r["dataset"].startswith(pre))]
        if not rs:
            continue
        fr = [float(r["frames"]) for r in rs]; H = sum(fr) / 25 / 3600
        def wq(key, q):
            v = sorted(zip([float(r[key]) for r in rs], fr))
            c, tot = 0.0, sum(fr)
            for x, w in v:
                c += w
                if c >= q * tot: return x
            return v[-1][0]
        out.append({"group": label, "hours": H, "seqs": len(rs), "dist50": wq("dist_deg", .5), "dist90": wq("dist_deg", .9),
                    "acc95": wq("acc95", .5), "div": wq("div_deg", .5), "static": sum(float(r["static_frac"]) * w for r, w in zip(rs, fr)) / sum(fr),
                    "near": sum(w for r, w in zip(rs, fr) if float(r["dist_deg"]) <= 20) / sum(fr)})
    return out


def results_rows(kind):
    out = []
    for line in read(REPO / "autoresearch" / "results.jsonl").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == kind: out.append(r)
    return out


def run_times(tag):
    out = {}
    for lp in list(LOGS.glob("run_*.log")) + list(LOGS.glob("chain_*.log")) + list(LOGS.glob("queue_gpu*.log")) + list(LOGS.glob("queue_fig2_gpu*.log")):
        t = read(lp)
        m = re.search(rf"\[(\S+)\] START base_{re.escape(tag)} ", t)
        if m: out["start"] = m.group(1)
        m = re.search(rf"\[(\S+)\] DONE {re.escape(tag)}\b", t)
        if m: out["done"] = m.group(1)
    return out


def collect_runs():
    runs = []

    def mk_run(tag, d, base):
        ft = parse_train_log(CK / f"ft_{tag}" / "train.log")
        ev = parse_eval(CK / f"ft_{tag}" / "eval_dip_test.log")
        # zero-shot on the collected IMUPoser dataset: the PRETRAINED checkpoint, no DIP fine-tune (the primary metric
        # from 2026-10-06 on), optionally after test-time optimisation (offline_fit --iters 300)
        zs = parse_eval(d / "eval_imuposer_zs.log")
        zs_tto = parse_eval(d / "eval_imuposer_tto300.log", section="fit")
        n = params_of(d)
        seed = int(re.search(r"_s(\d+)$", tag).group(1)) if re.search(r"_s(\d+)$", tag) else 1
        lever = lever_label(tag)
        extra = lever or next((lab for pre, lab in EXTRA_LABELS if tag.startswith(pre)), "")
        return {"tag": tag, "arm": arm_label(tag), "extra": extra, "lever": lever, "seed": seed, "base": base, "ft": ft, "eval": ev,
                "zs": zs, "zs_tto": zs_tto,
                "times": run_times(tag), "params": n, "size": size_label(n), "budget": budget_of(tag, base),
                "hours": base.get("hours"), "steps": (base.get("steps_per_epoch") or 0) * budget_of(tag, base)}

    for d in sorted(CK.glob("base_*")):
        tag = d.name[5:]
        base = parse_train_log(d / "train.log") or {"status": "pending", "val": {}, "epochs_done": 0}
        if tag.startswith("swa_"):      # averaged checkpoints: hours / steps / status come from the source base run
            sb = parse_train_log(CK / f"base_{lever_ref(tag)}" / "train.log") or {}
            base = dict(base, hours=sb.get("hours"), steps_per_epoch=sb.get("steps_per_epoch"), status="done",
                        epochs_done=sb.get("epochs_done", 0), val=sb.get("val", {}))
        runs.append(mk_run(tag, d, base))
    # FT-only variants (BASE_FROM=<tag> in run_newdata.sh): no base_<tag> dir, ft_<tag>/ft_meta.json names the base
    for fd in sorted(CK.glob("ft_*")):
        tag = fd.name[3:]
        if (CK / f"base_{tag}").exists() or not (fd / "ft_meta.json").exists(): continue
        try:
            meta = json.loads(read(fd / "ft_meta.json"))
        except Exception:
            continue
        d = CK / f"base_{meta['base_from']}"
        base = parse_train_log(d / "train.log") or {"status": "pending", "val": {}, "epochs_done": 0}
        snap = meta.get("snapshot_epoch")
        if snap:   # the epoch-N snapshot of a longer run = that run's N-epoch budget point (a distinct model, not an FT variant)
            m = re.search(r"epoch=(\d+)-val_loss=validation_step_loss=(\d+\.\d+)", meta.get("snapshot_src") or "")
            base = dict(base, epochs_done=min(base.get("epochs_done", 0), snap), status="done",
                        val={e: v for e, v in base.get("val", {}).items() if e < snap},
                        **({"best_epoch": int(m.group(1)), "best_val": float(m.group(2))} if m else {}))
            runs.append(mk_run(tag, d, base))
        else:
            runs.append(dict(mk_run(tag, d, base), ft_variant=True))   # same pretrained model, FT recipe varied
    for d in sorted(CK.glob("base_*")):
        tag = d.name[5:]
        base = parse_train_log(d / "train.log") or {"status": "pending", "val": {}, "epochs_done": 0}
        n = params_of(d)
        seed = int(re.search(r"_s(\d+)$", tag).group(1)) if re.search(r"_s(\d+)$", tag) else 1
        for idir in sorted(CK.glob(f"ft_{tag}_interim_ep*")):
            ep = int(re.search(r"interim_ep(\d+)", idir.name).group(1))
            iev = parse_eval(idir / "eval_dip_test.log")
            if iev:
                runs.append({"tag": f"{tag} (interim, ep {ep})", "arm": arm_label(tag), "extra": "", "seed": seed, "interim": True,
                             "base": {"status": "done", "val": {}, "epochs_done": ep + 1, "hours": base.get("hours"), "best_epoch": ep, "best_val": base["val"].get(ep)},
                             "ft": None, "eval": iev, "times": {}, "params": n, "size": size_label(n), "budget": ep + 1, "hours": base.get("hours"),
                             "steps": (base.get("steps_per_epoch") or 0) * (ep + 1)})
    return runs


def data_hours():
    rows = []
    shard_only = {d.name for d in SHARDS.iterdir() if (d / "meta.json").exists()} if SHARDS.exists() else set()
    for gname, pred in GROUPS:
        names = sorted({p.stem for p in DATA.glob("*.pt") if pred(p.stem)} | {n for n in shard_only if pred(n)})
        h, nseq = 0.0, 0
        for n in names:
            meta = SHARDS / n / "meta.json"
            if meta.exists():
                m = json.loads(meta.read_text()); h += m["n_frames"] / 25 / 3600; nseq += m["n_seqs"]
            elif (DATA / f"{n}.pt").exists():
                h += (DATA / f"{n}.pt").stat().st_size / 1452 / 25 / 3600
        rows.append({"group": gname, "files": len(names), "hours": h, "seqs": nseq})
    return rows


# ---------------------------------------------------------------- svg helpers
def _ticks_log(lo, hi):
    out = []
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi * 1.01:
        for k in (1, 2, 5):
            v = k * 10 ** e
            if lo * 0.99 <= v <= hi * 1.01: out.append(v)
        e += 1
    return out


def fmt_num(v):
    if v >= 1e6: return f"{v/1e6:g}M"
    if v >= 1e3: return f"{v/1e3:g}k"
    return f"{v:g}"


def svg_xy(series, w=560, h=320, xlabel="", ylabel="dip_test SIP (deg)", logx=True, ymin=None, ymax=None, xfmt=fmt_num):
    """series: list of dicts {label, color, points:[(x,y,label)], dash, marker}"""
    pts = [p for s in series for p in s["points"]]
    if not pts:
        return "<p class='muted'>no points yet</p>"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    if logx:
        x0, x1 = x0 / 1.4, x1 * 1.4
        X_ = lambda x: math.log10(x)
    else:
        pad = (x1 - x0) * 0.08 or 1; x0, x1 = x0 - pad, x1 + pad; X_ = lambda x: x
    y0 = (ymin if ymin is not None else min(ys)) - 0.25; y1 = (ymax if ymax is not None else max(ys)) + 0.25
    L, R, T, B = 52, 16, 14, 44
    def X(x): return L + (X_(x) - X_(x0)) / (X_(x1) - X_(x0)) * (w - L - R)
    def Y(y): return T + (y1 - y) / (y1 - y0) * (h - T - B)
    g = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img' aria-label='{html.escape(ylabel)} vs {html.escape(xlabel)}'>"]
    ny = 5
    for i in range(ny + 1):
        yv = y0 + (y1 - y0) * i / ny
        g.append(f"<line x1='{L}' x2='{w-R}' y1='{Y(yv):.1f}' y2='{Y(yv):.1f}' class='grid'/>")
        g.append(f"<text x='{L-6}' y='{Y(yv)+4:.1f}' class='tick' text-anchor='end'>{yv:.1f}</text>")
    xt = _ticks_log(x0, x1) if logx else [x0 + (x1 - x0) * i / 5 for i in range(6)]
    for xv in xt:
        g.append(f"<line x1='{X(xv):.1f}' x2='{X(xv):.1f}' y1='{T}' y2='{h-B}' class='grid'/>")
        g.append(f"<text x='{X(xv):.1f}' y='{h-B+16}' class='tick' text-anchor='middle'>{xfmt(xv)}</text>")
    g.append(f"<text x='{(L+w-R)/2:.0f}' y='{h-6}' class='axis' text-anchor='middle'>{html.escape(xlabel)}</text>")
    g.append(f"<text transform='translate(13 {(T+h-B)/2:.0f}) rotate(-90)' class='axis' text-anchor='middle'>{html.escape(ylabel)}</text>")
    for s in series:
        p = sorted(s["points"])
        if len(p) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y, *_) in enumerate(p))
            g.append(f"<path d='{d}' fill='none' stroke='{s['color']}' stroke-width='2' {'stroke-dasharray=\"6 4\"' if s.get('dash') else ''} opacity='0.9'/>")
        for x, y, *lab in p:
            if s.get("marker") == "square":
                g.append(f"<rect x='{X(x)-4:.1f}' y='{Y(y)-4:.1f}' width='8' height='8' fill='{s['color']}'/>")
            else:
                g.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='4.5' fill='{s['color']}'/>")
            if lab and lab[0]:
                g.append(f"<text x='{X(x)+7:.1f}' y='{Y(y)-6:.1f}' class='tick'>{html.escape(str(lab[0]))}</text>")
    g.append("</svg>")
    leg = "".join(f"<span class='leg'><i style='background:{s['color']}'></i>{html.escape(s['label'])}</span>" for s in series)
    return "".join(g) + f"<div class='legend'>{leg}</div>"


def svg_lines(series, w=720, h=300, xlabel="epoch", ylabel="val loss (dip_train)"):
    pts = [p for s in series for p in s[3]]
    if not pts:
        return "<p class='muted'>no training curves yet</p>"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = 0, max(max(xs), 1); y0 = min(ys) * 0.97; y1 = min(max(ys), min(ys) * 1.6) * 1.03
    L, R, T, B = 56, 16, 14, 40
    def X(x): return L + (x - x0) / (x1 - x0) * (w - L - R)
    def Y(y): return T + (y1 - min(y, y1)) / (y1 - y0) * (h - T - B)
    g = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img' aria-label='{html.escape(ylabel)} vs {xlabel}'>"]
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        g.append(f"<line x1='{L}' x2='{w-R}' y1='{Y(yv):.1f}' y2='{Y(yv):.1f}' class='grid'/>")
        g.append(f"<text x='{L-6}' y='{Y(yv)+4:.1f}' class='tick' text-anchor='end'>{yv:.4f}</text>")
    for xv in range(0, int(x1) + 1, max(1, int(x1) // 6 or 1)):
        g.append(f"<text x='{X(xv):.1f}' y='{h-B+18}' class='tick' text-anchor='middle'>{xv}</text>")
    g.append(f"<text x='{(L+w-R)/2:.0f}' y='{h-6}' class='axis' text-anchor='middle'>{xlabel}</text>")
    g.append(f"<text transform='translate(14 {(T+h-B)/2:.0f}) rotate(-90)' class='axis' text-anchor='middle'>{ylabel}</text>")
    for label, color, dash, p in series:
        if not p: continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(sorted(p)))
        g.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2' {'stroke-dasharray=\"6 4\"' if dash else ''}/>")
        lx, ly = sorted(p)[-1]
        g.append(f"<circle cx='{X(lx):.1f}' cy='{Y(ly):.1f}' r='3.5' fill='{color}'/>")
    g.append("</svg>")
    leg = "".join(f"<span class='leg'><i style='background:{c}'></i>{html.escape(l)}</span>" for l, c, _, p in series if p)
    return "".join(g) + f"<div class='legend'>{leg}</div>"


def svg_stack(rows, w=720, h=70):
    tot = sum(r["hours"] for r in rows) or 1
    cols = ["#5b6ee1", "#d98f3b", "#2f9e7a", "#c45c8a", "#8e6ad1", "#6b8fa3", "#b5873a", "#7a7a7a"]
    g = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img' aria-label='hours by dataset group'>"]
    x = 0
    for i, r in enumerate(rows):
        ww = r["hours"] / tot * w
        g.append(f"<rect x='{x:.1f}' y='8' width='{max(ww,0):.1f}' height='34' fill='{cols[i%len(cols)]}'/>")
        if ww > 50:
            g.append(f"<text x='{x+ww/2:.1f}' y='30' class='barlab' text-anchor='middle'>{r['hours']:.0f} h</text>")
        x += ww
    g.append("</svg>")
    leg = "".join(f"<span class='leg'><i style='background:{cols[i%len(cols)]}'></i>{html.escape(r['group'])} {r['hours']:.0f} h</span>" for i, r in enumerate(rows))
    return "".join(g) + f"<div class='legend'>{leg}</div>"


# ---------------------------------------------------------------- page
def fmt(v, nd=2): return "–" if v is None else f"{v:.{nd}f}"
def mean(xs): return sum(xs) / len(xs) if xs else None


def chip(st):
    c = {"done": "ok", "running": "run", "error": "bad", "pending": "wait"}.get(st, "wait")
    return f"<span class='chip {c}'>{st}</span>"


def row_status(r):
    b = r["base"]; st = b.get("status", "pending")
    if r.get("interim"): return f"{chip('done')} read-out at base epoch {b.get('best_epoch')}"
    if st == "done": return f"{chip(st)} best ep {b.get('best_epoch','?')}, val {fmt(b.get('best_val'),5)}"
    prog = f"{b.get('epochs_done',0)}/{r['budget']} ep" + (f" @ {b['it_s']:.1f} it/s" if b.get("it_s") else "")
    return f"{chip(st)} {prog}"


def build(out_path):
    runs_all = collect_runs()
    invalid = [r for r in runs_all if r["tag"].split(" ")[0] in INVALID]
    runs = [r for r in runs_all if r["tag"].split(" ")[0] not in INVALID]
    rows = data_hours()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fin = [r for r in runs if r["eval"] and not r.get("interim")]
    # headline = best run under the FIXED protocol. Lever variants are reported in their own section: picking the
    # best of several recipe variants by its dip_test score would be selection on the test set.
    proto = [r for r in fin if not r.get("lever")]
    best = min(proto, key=lambda r: r["eval"]["sip"]) if proto else None

    # ---- headline. PRIMARY metric (since 2026-10-06): zero-shot SIP on the collected IMUPoser dataset of the PRETRAINED
    # checkpoint, no DIP fine-tune (the fine-tune helps dip_test and hurts this real-device set). dip_test is secondary.
    zfin = [r for r in runs if r.get("zs") and not r.get("interim") and not r.get("ft_variant")]
    zbest = min(zfin, key=lambda r: r["zs"]["sip"]) if zfin else None
    if zbest:
        z = zbest["zs"]
        head = (f"Best zero-shot on the IMUPoser dataset, no fine-tune: SIP {z['sip']:.2f}°, mesh error {z['mpvpe']:.2f} cm "
                f"({zbest['size']} model, {zbest['params']/1e6:.1f} M params, pretrained on {zbest['arm'].rstrip('+')}{' ' + zbest['extra'] if zbest['extra'] else ''}, "
                f"{zbest['hours']:.0f} h, {zbest['budget']} epochs); angular {z['mpjre']:.2f}°, positional {z['mpjpe']:.2f} cm, "
                f"velocity {z['mpjve']:.1f} cm/s, jitter {z['jitter']}.")
        zmesh_best = min(zfin, key=lambda r: r["zs"]["mpvpe"])
        if zmesh_best is not zbest:
            head += (f" Lowest mesh error: {zmesh_best['zs']['mpvpe']:.2f} cm ({zmesh_best['size']}, {zmesh_best['arm'].rstrip('+')}"
                     f"{' ' + zmesh_best['extra'] if zmesh_best['extra'] else ''}, SIP {zmesh_best['zs']['sip']:.2f}°).")
        # whole-dataset score of the same model with overlapping evaluation windows (stride fixed on the DIP validation split, not here)
        st = next((r for r in results_rows("zs_split") if r.get("name") == "swa_xl60_stride31" and r.get("all_sip")), None)
        if st and zbest["tag"] == "swa_xl60_ctrl":
            head += f" Same model with overlapping evaluation windows (stride 31): SIP {st['all_sip']:.2f}°, mesh {st['all_mesh']:.2f} cm."
    else:
        head = "No zero-shot evaluation yet."
    head2 = (f"The paper's own DIP-fine-tuned model on this set, same sensors, without end effectors: 22.21°, 8.56 cm, 10.12 cm. "
             + (f"Secondary benchmark, dip_test after the DIP fine-tune: best single model {best['eval']['sip']:.2f} ({best['size']}, {best['arm'].rstrip('+')}{' ' + best['extra'] if best['extra'] else ''}), previous deliverable {PREV_DELIVERABLE:.2f}." if best else ""))
    sub = f"{len(fin)} finished runs ({len(zfin)} with zero-shot scores), {sum(1 for r in runs if r['base'].get('status')=='running')} running, page generated {now} ET. Seed noise on dip_test is about ±{NOISE} deg."

    # ---- scaling charts for a metric: model axis per data arm, data axis per size, compute proxy
    def charts(metric, ylab, pool):
        fm = [r for r in pool if metric(r) is not None]
        ms = []
        for arm, col in (("curated-12", ARMCOL["curated-12"]), ("control", ARMCOL["control"]), ("treatment", ARMCOL["treatment"])):
            for budget, dash, mk in ((20, False, None), (60, True, "square")):
                p = [(r["params"], metric(r), r["size"]) for r in fm if r["params"] and r["arm"] == arm and r["budget"] == budget and not r["extra"] and r["seed"] == 1]
                if p: ms.append({"label": f"{arm} · {budget} ep", "color": col, "points": p, "dash": dash, "marker": mk})
        ds = []
        for size in ("S", "M", "L", "XL"):
            for budget, dash, mk in ((20, False, None), (60, True, "square")):
                p = [(r["hours"], metric(r), r["arm"][:9]) for r in fm if r["size"] == size and r["budget"] == budget and r["hours"] and not r["extra"] and r["seed"] == 1 and r["arm"] in ("curated-12", "control", "treatment")]
                if p: ds.append({"label": f"{size} · {budget} ep", "color": COL[size], "points": p, "dash": dash, "marker": mk})
        cs = []
        for size in ("S", "M", "L", "XL"):
            p = [(r["params"] * r["steps"], metric(r), f"{r['arm'][:4]} {r['budget']}ep") for r in fm if r["size"] == size and r["steps"] and not r["extra"] and r["arm"] in ("curated-12", "control", "treatment")]
            if p: cs.append({"label": size, "color": COL[size], "points": p})
        return (svg_xy(ms, xlabel="parameters (log)", ylabel=ylab),
                svg_xy(ds, xlabel="pretraining hours (log)", ylabel=ylab),
                svg_xy(cs, xlabel="compute proxy: parameters × optimizer steps (log)", ylabel=ylab, xfmt=lambda v: f"{v:.0e}".replace("e+", "e")))
    model_chart, data_chart, comp_chart = charts(lambda r: r["eval"]["sip"] if r.get("eval") else None, "dip_test SIP (deg, lower is better)", fin)
    zmodel_chart, zdata_chart, zcomp_chart = charts(lambda r: r["zs"]["sip"] if r.get("zs") else None, "IMUPoser zero-shot SIP (deg, lower is better)", zfin)
    vmodel_chart, vdata_chart, vcomp_chart = charts(lambda r: r["zs"]["mpvpe"] if r.get("zs") else None, "IMUPoser zero-shot mesh error (cm, lower is better)", zfin)

    # grid table (both benchmarks)
    grid = sorted([r for r in runs if r["arm"] in ("curated-12", "control", "treatment") and not r["extra"] and not r.get("interim")],
                  key=lambda r: ({"curated-12": 0, "control": 1, "treatment": 2}[r["arm"]], r["budget"], r["params"] or 0, r["seed"]))
    grows = "".join(f"<tr><td>{r['size']} <span class='muted'>{fmt((r['params'] or 0)/1e6,1)} M</span></td><td>{r['arm']}</td><td class='num'>{fmt(r['hours'],0)}</td>"
                    f"<td class='num'>{r['budget']}</td><td class='num'>{r['seed']}</td><td>{row_status(r)}</td>"
                    f"<td class='num'><b>{fmt(r.get('zs') and r['zs']['sip'])}</b></td><td class='num'>{fmt(r.get('zs') and r['zs']['mpjre'])}</td><td class='num'>{fmt(r.get('zs') and r['zs']['mpjpe'])}</td>"
                    f"<td class='num'>{fmt(r['eval'] and r['eval']['sip'])}</td></tr>" for r in grid)

    # ---- zero-shot leaderboard: every run with a zero-shot score (pretrained checkpoint), best first; TTO column where run
    zft = {r["tag"]: r for r in results_rows("imuposer_zs") if r.get("stage") == "ft"}
    zlb = sorted(zfin, key=lambda r: r["zs"]["sip"])
    zrows = "".join(f"<tr><td class='num'>{i+1}</td><td class='mono'>{html.escape(r['tag'])}</td><td>{r['size']} {fmt((r['params'] or 0)/1e6,1)} M</td><td>{r['arm'].rstrip('+')} {html.escape(r['extra'])}</td><td class='num'>{fmt(r['hours'],0)}</td><td class='num'>{r['budget']}</td>"
                    f"<td class='num'><b>{r['zs']['sip']:.2f}</b></td><td class='num'>{fmt(r.get('zs_tto') and r['zs_tto']['sip'])}</td><td class='num'><b>{r['zs']['mpvpe']:.2f}</b></td><td class='num'>{fmt(r.get('zs_tto') and r['zs_tto']['mpvpe'])}</td><td class='num'>{r['zs']['mpjre']:.2f}</td><td class='num'>{r['zs']['mpjpe']:.2f}</td><td class='num'>{r['zs']['mpjve']:.1f}</td><td class='num'>{r['zs']['jitter']}</td>"
                    f"<td class='num muted'>{fmt(zft.get(r['tag'], {}).get('sip'))}</td><td class='num muted'>{fmt(r['eval'] and r['eval']['sip'])}</td></tr>" for i, r in enumerate(zlb))

    # ---- treatment vs control (original question): S at 60 ep, seeds
    c60 = [r for r in fin if r["arm"] == "control" and r["size"] == "S" and r["budget"] == 60 and not r["extra"]]
    t60 = [r for r in fin if r["arm"] == "treatment" and r["size"] == "S" and r["budget"] == 60 and not r["extra"]]
    interim = [r for r in runs if r.get("interim")]
    tc_rows = "".join(f"<tr><td>{(html.escape(r['tag']) + ' <span class=\"chip wait\">interim read-out</span>') if r.get('interim') else r['arm']}</td><td class='num'>{r['seed']}</td><td class='num'>{fmt(r['hours'],0)}</td><td>{row_status(r)}</td>"
                      f"<td class='num'>{fmt(r['eval'] and r['eval']['sip'])}</td><td class='num'>{fmt(r['eval'] and r['eval']['mpjre'])}</td><td class='num'>{fmt(r['eval'] and r['eval']['mpjpe'])}</td></tr>"
                      for r in sorted([r for r in runs if r["size"] == "S" and r["budget"] in (60,) and r["arm"] in ("control", "treatment") and not r["extra"]] + interim, key=lambda r: (r["arm"], r["seed"], bool(r.get("interim")))))
    curve_series = []
    for r in sorted([r for r in runs if r["size"] == "S" and r["arm"] in ("control", "treatment") and not r["extra"] and not r.get("interim") and r["budget"] == 60], key=lambda r: (r["arm"], r["seed"])):
        curve_series.append((f"{r['arm']} s{r['seed']}", ARMCOL[r["arm"]], r["seed"] != 1, sorted(r["base"]["val"].items())))
    curve = svg_lines(curve_series)

    # ---- ablations and GV
    abl = sorted([r for r in runs if (r["extra"] or (r["arm"] == "control" and r["size"] == "S" and r["budget"] == 20)) and not r.get("interim")], key=lambda r: (r["extra"], r["seed"]))
    arows = "".join(f"<tr><td>{r['arm'].rstrip('+')} {html.escape(r['extra']) if r['extra'] else '(reference)'}</td><td class='num'>{r['seed']}</td><td class='num'>{fmt(r['hours'],0)}</td><td>{row_status(r)}</td>"
                    f"<td class='num'>{fmt(r['eval'] and r['eval']['sip'])}</td><td class='num'>{fmt(r['eval'] and r['eval']['mpjre'])}</td><td class='num'>{fmt(r['eval'] and r['eval']['mpjpe'])}</td></tr>" for r in abl)

    # ---- distance from DIP per dataset group (diplike_select.py)
    dl = diplike_table()
    dl_rows = "".join(f"<tr><td>{html.escape(r['group'])}</td><td class='num'>{r['hours']:.0f}</td><td class='num'>{r['seqs']}</td>"
                      f"<td class='num'>{r['dist50']:.1f}</td><td class='num'>{r['dist90']:.1f}</td><td class='num'>{r['near']*100:.0f} %</td>"
                      f"<td class='num'>{r['div']:.1f}</td><td class='num'>{r['acc95']:.1f}</td><td class='num'>{r['static']*100:.0f} %</td></tr>" for r in dl)

    # ---- data table
    drows = "".join(f"<tr><td>{html.escape(r['group'])}</td><td class='num'>{r['files']}</td><td class='num'>{r['hours']:.1f}</td><td class='num'>{r['seqs'] or '–'}</td></tr>" for r in rows)

    # ---- leaderboard
    lb = sorted([r for r in fin if not r.get("ft_variant")], key=lambda r: r["eval"]["sip"])[:10]   # one row per pretrained model

    # ---- SOTA levers (recipe variations on a finished base) and ensembles of existing checkpoints
    by_tag = {r["tag"]: r for r in fin}
    def delta(v, ref):
        if v is None or ref is None: return "–"
        d = v - ref
        cls = "ok" if d < -NOISE else "bad" if d > NOISE else "muted"
        return f"<span class='{cls}'>{d:+.2f}</span>"
    levers = sorted([r for r in runs if r.get("lever") and not r.get("interim")], key=lambda r: (r["eval"]["sip"] if r["eval"] else 99, r["tag"]))
    lever_rows = ""
    for r in levers:
        ref = by_tag.get(lever_ref(r["tag"]) or "")
        lever_rows += (f"<tr><td class='mono'>{html.escape(r['tag'])}</td><td>{html.escape(r['extra'])}</td><td>{r['size']} {r['arm']}</td>"
                       f"<td class='mono'>{html.escape(lever_ref(r['tag']) or '–')} <span class='muted'>{fmt(ref and ref['eval']['sip'])}</span></td><td>{row_status(r)}</td>"
                       f"<td class='num'>{fmt(r['eval'] and r['eval']['sip'])}</td><td class='num'>{delta(r['eval'] and r['eval']['sip'], ref and ref['eval']['sip'])}</td></tr>")
    ens = sorted(results_rows("ensemble"), key=lambda r: r.get("fval_sip") or 99)
    best_single = best["eval"]["sip"] if best else None
    ens_rows = "".join(f"<tr><td class='mono'>{html.escape(' + '.join(m.replace('ft_', '') for m in r.get('members', [])))}</td><td class='num'>{len(r.get('members', []))}</td><td class='num'>{r.get('stride', 125)}</td>"
                       f"<td class='num'>{fmt(r.get('fval_sip'))}</td><td class='num'><b>{fmt(r.get('sip_dip_test'))}</b></td><td class='num'>{fmt(r.get('mpjre'))}</td><td class='num'>{fmt(r.get('mpjpe_cm'))}</td>"
                       f"<td class='num'>{delta(r.get('sip_dip_test'), best_single)}</td></tr>" for r in ens)
    best_ens = min(ens, key=lambda r: r.get("fval_sip") or 99) if ens else None
    win = next((r for r in results_rows("lever") if r.get("name") == "lever_eval_window"), None)
    win_rows = ""
    if win:
        ws = sorted({w for d in win["fval"].values() for w in d}, key=int)
        win_rows = "".join(f"<tr><td>{m}</td>" + "".join(f"<td class='num'>{fmt(win['fval'][m].get(w))}</td>" for w in ws) + "</tr>" for m in win["fval"])
        win_head = "".join(f"<th class='num'>{w} frames</th>" for w in ws)
    else:
        win_head = ""
    stride = next((r for r in results_rows("lever") if r.get("name") == "lever_eval_stride"), None)
    stride_rows, stride_head = "", ""
    if stride:
        ss = sorted({s for d in stride["fval"].values() for s in d}, key=int, reverse=True)
        stride_head = "".join(f"<th class='num'>stride {s}</th>" for s in ss)
        for m in stride["fval"]:
            stride_rows += f"<tr><td>{m} · fval</td>" + "".join(f"<td class='num'>{fmt(stride['fval'][m].get(s))}</td>" for s in ss) + "</tr>"
            stride_rows += f"<tr><td>{m} · dip_test</td>" + "".join(f"<td class='num'><b>{fmt(stride['dip_test'][m].get(s))}</b></td>" if stride['dip_test'][m].get(s) is not None else "<td class='num muted'>–</td>" for s in ss) + "</tr>"
    # ---- checkpoint averaging, zero-shot: the averaged run against its plain source run (same training run, no extra compute)
    def dcell(v):
        return "<td class='num muted'>–</td>" if v is None else f"<td class='num' style='color:{'#1a7f37' if v < 0 else '#b42318'}'>{v:+.2f}</td>"
    avg_rows = ""
    for r in sorted((x for x in zfin if x["tag"].startswith("swa_")), key=lambda x: x["zs"]["sip"]):
        ref = by_tag.get(lever_ref(r["tag"]) or "")
        rz = ref.get("zs") if ref else None
        avg_rows += (f"<tr><td class='mono'>{html.escape(lever_ref(r['tag']) or '–')}</td><td>{r['size']} {fmt((r['params'] or 0)/1e6, 1)} M</td><td class='num'>{r['budget']}</td>"
                     f"<td class='num'>{fmt(rz and rz['sip'])}</td><td class='num'><b>{r['zs']['sip']:.2f}</b></td>{dcell(rz and r['zs']['sip'] - rz['sip'])}"
                     f"<td class='num'>{fmt(rz and rz['mpvpe'])}</td><td class='num'><b>{r['zs']['mpvpe']:.2f}</b></td>{dcell(rz and r['zs']['mpvpe'] - rz['mpvpe'])}</tr>")
    # ---- choices made ON this dataset (ensembles, evaluation settings): selected on participants 1-2, reported on participants 3-10
    split_rows = ""
    for r in sorted(results_rows("zs_split"), key=lambda x: x.get("test8_sip") or 99):
        split_rows += (f"<tr><td>{html.escape(r.get('label', r.get('name', '')))}</td><td class='num'>{fmt(r.get('sel_sip'))}</td><td class='num'>{fmt(r.get('sel_mesh'))}</td>"
                       f"<td class='num'><b>{fmt(r.get('test8_sip'))}</b></td><td class='num'>{fmt(r.get('test8_mpjre'))}</td><td class='num'>{fmt(r.get('test8_mpjpe'))}</td>"
                       f"<td class='num'><b>{fmt(r.get('test8_mesh'))}</b></td><td class='num'>{fmt(r.get('test8_mpjve'), 1)}</td><td class='num'>{fmt(r.get('test8_jitter'), 0)}</td></tr>")
    # ---- zero-shot on the collected IMUPoser dataset (real phone/watch/head IMU, 10 participants)
    zs = results_rows("imuposer_zs")
    zs_by = {}
    for r in zs:
        zs_by.setdefault(r["tag"], {})[r["stage"]] = r
    def zcell(r, k, nd=2):
        return fmt(r.get(k), nd) if r else "–"
    zs_rows = ""
    for tag, d in sorted(zs_by.items(), key=lambda kv: (kv[1].get("ft") or kv[1].get("base") or kv[1].get("ensemble") or {}).get("sip", 99)):
        b, f_, e = d.get("base"), d.get("ft"), d.get("ensemble")
        ref = by_tag.get(tag)
        dip = fmt(ref["eval"]["sip"]) if ref else (fmt(next((x.get("sip_dip_test") for x in results_rows("ensemble") if x.get("stride") == 31 and len(x.get("members", [])) == 7), None)) if e else "–")
        row = e or f_
        zs_rows += (f"<tr><td class='mono'>{html.escape(tag)}</td><td class='num'>{dip}</td>"
                    f"<td class='num'>{zcell(b, 'sip')}</td><td class='num'><b>{zcell(row, 'sip')}</b></td>"
                    f"<td class='num'>{zcell(b, 'mpjre')} / {zcell(row, 'mpjre')}</td><td class='num'>{zcell(b, 'mpjpe_cm')} / {zcell(row, 'mpjpe_cm')}</td>"
                    f"<td class='num'>{zcell(b, 'mpvpe_cm')} / {zcell(row, 'mpvpe_cm')}</td><td class='num'>{zcell(b, 'mpjve', 1)} / {zcell(row, 'mpjve', 1)}</td>"
                    f"<td class='num'>{zcell(b, 'jitter', 0)} / {zcell(row, 'jitter', 0)}</td></tr>")
    zs_gt_jit = next((r.get("jitter_gt") for r in zs), None)

    ens_head = (f"Best ensemble of existing fine-tuned checkpoints, selected on fval: dip_test SIP {best_ens['sip_dip_test']:.2f} "
                f"({len(best_ens['members'])} members{', inference stride ' + str(best_ens['stride']) if best_ens.get('stride') else ''}, no new training)." if best_ens else "")
    lrows = "".join(f"<tr><td class='num'>{i+1}</td><td>{html.escape(r['tag'])}</td><td>{r['size']} {fmt((r['params'] or 0)/1e6,1)} M</td><td>{r['arm']} {html.escape(r['extra'])}</td><td class='num'>{fmt(r['hours'],0)}</td><td class='num'>{r['budget']}</td>"
                    f"<td class='num'><b>{r['eval']['sip']:.2f}</b></td><td class='num'>{r['eval']['mpjre']:.2f}</td><td class='num'>{r['eval']['mpjpe']:.2f}</td></tr>" for i, r in enumerate(lb))

    # ---- timeline
    tl = "".join(f"<li><b>{html.escape(r['tag'])}</b>: start {r['times'].get('start','–')[5:16]}, done {r['times'].get('done','–')[5:16]}</li>"
                 for r in sorted([r for r in runs if not r.get("interim")], key=lambda r: r["times"].get("start", "9")) if r["times"].get("start"))

    page = f"""<title>IMUPoser Scaling Study</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: measured prose column, full-width panels for charts and tables */
:root{{--bg:#f6f4ef;--fg:#1d1c19;--muted:#6a665c;--line:#dcd7cc;--panel:#fffdf8;--accent:#1f5f7a;--ok:#2f9e7a;--warn:#d98f3b;--bad:#c4453c;--grid:#e6e1d6;
--display:"Source Serif 4",Georgia,"Times New Roman",serif;--body:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#171715;--fg:#ece8df;--muted:#a39e92;--line:#34322d;--panel:#1f1e1b;--accent:#7cc2dc;--ok:#5bc69e;--warn:#e8a85a;--bad:#e36a60;--grid:#2c2a26;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#171715;--fg:#ece8df;--muted:#a39e92;--line:#34322d;--panel:#1f1e1b;--accent:#7cc2dc;--ok:#5bc69e;--warn:#e8a85a;--bad:#e36a60;--grid:#2c2a26;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.55;margin:0}}
.wrap{{max-width:1040px;margin:0 auto;padding-block:28px 56px;padding-inline:16px}}
h1,h2,h3{{font-family:var(--display);text-wrap:balance;margin:0 0 .35em}} h1{{font-size:2rem;font-weight:700}} h2{{font-size:1.35rem;margin-top:2.2em}} h3{{font-size:1.05rem;margin-top:1.4em}}
p{{max-width:76ch}} .muted{{color:var(--muted)}} .eyebrow{{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}}
.headline{{background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--accent);padding:16px 20px;margin:18px 0 8px}} .headline .big{{font-family:var(--display);font-size:1.4rem;font-weight:700;margin:0 0 6px}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:10px 0}} th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}} th{{font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}} .tablewrap{{overflow-x:auto}} .mono{{font-family:var(--mono);font-size:.82rem}}
.chip{{display:inline-block;font-size:.7rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:2px 7px;border-radius:999px;margin-right:6px}}
.chip.ok{{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}} .chip.run{{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}} .chip.bad{{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}} .chip.wait{{background:color-mix(in srgb,var(--muted) 18%,transparent);color:var(--muted)}}
.chart{{width:100%;height:auto;display:block;background:var(--panel);border:1px solid var(--line);border-radius:4px}} .chart .grid{{stroke:var(--grid);stroke-width:1}} .chart .tick{{fill:var(--muted);font-size:11px;font-family:var(--mono)}} .chart .axis{{fill:var(--fg);font-size:12px}} .chart .barlab{{fill:#fff;font-size:12px;font-weight:600}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.84rem;margin:8px 0 0}} .leg i{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:6px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}} .grid3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}} .panel{{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:14px 16px;min-width:0}}
ul{{padding-left:1.2em}} li{{margin:.25em 0}} code{{font-family:var(--mono);font-size:.86em}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:8px 0}} dt{{color:var(--muted)}} dd{{margin:0}}
</style>
<div class="wrap">
<div class="eyebrow">IMUPoser · left watch + right watch + right pocket · 25 Hz · page generated {html.escape(now)} ET</div>
<h1>How far do data and model size take sparse-IMU pose estimation?</h1>
<p>One benchmark throughout: pretrain on synthetic IMU from motion capture, fine-tune on real DIP-IMU training subjects, report on the held-out DIP test subjects (s09, s10). SIP is the mean angular error of hips and shoulders; lower is better. Every number on this page is that single protocol with one thing varied at a time.</p>
<div class="headline"><div class="eyebrow">headline · primary benchmark: zero-shot on real device IMU</div><p class="big">{html.escape(head)}</p><p style="margin:0 0 6px">{html.escape(head2)}</p><p class="muted" style="margin:0">{html.escape(sub)}</p></div>

<h2>Zero-shot on the collected IMUPoser dataset (real phone, watch and head IMU)</h2>
<p>The CHI'23 IMUPoser dataset: 10 participants, 167 recordings, 1.15 h of real device IMU (left and right wrist, left and right pocket, head) with MoSh'd SMPL ground truth; arm raises, walking, boxing, kicking, push-ups, basketball, tennis swings, jumping jacks, hopping, jogging, head movements, sitting. It is closer to the deployment setting than DIP, and nothing is trained or tuned on it. Every pretrained checkpoint is evaluated on it directly, with no DIP fine-tune: the fine-tune improves dip_test but costs 1 to 5° SIP here (last two columns), so from 2026-10-06 this zero-shot score is the primary number and dip_test is secondary. Metrics follow the paper: SIP error, angular error (MPJRE), positional error (MPJPE), mesh error (MPVPE), velocity error (MPJVE), jitter (ground-truth jitter on this set is {zs_gt_jit if zs_gt_jit is not None else 54}; on dip_test 222). "TTO" is the physics-style test-time optimisation (300 Adam steps per sequence pulling the predicted pose's forward-kinematics sensor orientations onto the measured ones, anchored to the network output), where it has been run.</p>
<h3>SIP error (hips and shoulders, °)</h3>
<div class="grid3">
<div><h4 class="muted" style="margin:.2em 0">Model size, at fixed data</h4>{zmodel_chart}</div>
<div><h4 class="muted" style="margin:.2em 0">Data, at fixed model size</h4>{zdata_chart}</div>
<div><h4 class="muted" style="margin:.2em 0">Compute</h4>{zcomp_chart}</div>
</div>
<h3>Mesh error (mean vertex position error, cm)</h3>
<div class="grid3">
<div><h4 class="muted" style="margin:.2em 0">Model size, at fixed data</h4>{vmodel_chart}</div>
<div><h4 class="muted" style="margin:.2em 0">Data, at fixed model size</h4>{vdata_chart}</div>
<div><h4 class="muted" style="margin:.2em 0">Compute</h4>{vcomp_chart}</div>
</div>
<div class="tablewrap"><table><thead><tr><th class="num">#</th><th>run</th><th>model</th><th>pretraining data</th><th class="num">hours</th><th class="num">ep</th><th class="num">SIP °</th><th class="num">SIP ° + TTO</th><th class="num">mesh cm</th><th class="num">mesh cm + TTO</th><th class="num">MPJRE °</th><th class="num">MPJPE cm</th><th class="num">MPJVE cm/s</th><th class="num">jitter</th><th class="num">SIP ° after DIP FT</th><th class="num">dip_test SIP ° (FT)</th></tr></thead><tbody>{zrows}</tbody></table></div>
<p class="muted">Reference from the paper's own camera-ready results on this dataset (its DIP-fine-tuned LSTM, left wrist + right wrist + right pocket, averaged without end effectors, which is also what the evaluator here ignores): global angular error 22.21°, positional error 8.56 cm, mesh error 10.12 cm (with end effectors: 21.72°, 9.46 cm, 11.39 cm). The paper did not report SIP or a velocity error on this set.</p>
<h3>Checkpoint averaging (zero-shot)</h3>
<p>Uniform average of the three best-by-validation checkpoints a run saved, scored zero-shot against the run's single best checkpoint. Same training run, no extra compute. It was neutral on the DIP benchmark after fine-tuning, and it helps transfer to the real-device dataset at every model size and schedule tried.</p>
<div class="tablewrap"><table><thead><tr><th>source run</th><th>model</th><th class="num">ep</th><th class="num">SIP ° best ckpt</th><th class="num">SIP ° averaged</th><th class="num">Δ</th><th class="num">mesh cm best ckpt</th><th class="num">mesh cm averaged</th><th class="num">Δ</th></tr></thead><tbody>{avg_rows}</tbody></table></div>
<h3>Ensembles and evaluation settings (selection / report split)</h3>
<p>Choices made on this dataset are selected on participants 1–2 (14.8 min) and reported on participants 3–10 (54.3 min), so the report column is untouched by the choice. The whole-dataset numbers above involve no such choice. Ensembles average the predicted joint rotations of the listed averaged bases; stride 31 evaluates overlapping windows instead of disjoint ones.</p>
<div class="tablewrap"><table><thead><tr><th>configuration</th><th class="num">select SIP °</th><th class="num">select mesh cm</th><th class="num">report SIP °</th><th class="num">report MPJRE °</th><th class="num">report MPJPE cm</th><th class="num">report mesh cm</th><th class="num">report MPJVE</th><th class="num">report jitter</th></tr></thead><tbody>{split_rows}</tbody></table></div>
<h3>Post-processing on the zero-shot leader (averaged L60 control)</h3>
<p>Every test-time refinement tried on top of the best pretrained model at the time, scored on the whole dataset. None improves pose accuracy. The smoothed orientation fit and the physics optimizer with softer gains improve temporal quality (jitter, velocity error) at a small SIP cost; with PIP's own gains the physics stage tracks the network and changes nothing. MobilePoser's gain from this stage comes with its own joint-velocity and foot-contact heads and a translation estimate, which our pose-only model does not predict, so the simulator has nothing to correct toward.</p>
<div class="tablewrap"><table><thead><tr><th>post-process</th><th class="num">SIP °</th><th class="num">MPJRE °</th><th class="num">MPJPE cm</th><th class="num">mesh cm</th><th class="num">MPJVE cm/s</th><th class="num">jitter</th></tr></thead><tbody>
<tr><td>none (network output)</td><td class="num"><b>15.85</b></td><td class="num">18.52</td><td class="num">7.41</td><td class="num"><b>8.94</b></td><td class="num">23.6</td><td class="num">148</td></tr>
<tr><td>PIP / MobilePoser physics optimizer (rigid-body PD + contact QP, 50 Hz, PIP's gains)</td><td class="num">15.89</td><td class="num">18.56</td><td class="num">7.45</td><td class="num">8.98</td><td class="num">23.7</td><td class="num">145</td></tr>
<tr><td>PIP / MobilePoser physics optimizer, softer angular gains (kp 600, kd 30)</td><td class="num">15.96</td><td class="num">18.64</td><td class="num">7.51</td><td class="num">9.04</td><td class="num">23.2</td><td class="num">107</td></tr>
<tr><td>orientation fit, 300 steps (default offline fit)</td><td class="num">16.08</td><td class="num">18.68</td><td class="num">7.59</td><td class="num">9.19</td><td class="num">27.6</td><td class="num">264</td></tr>
<tr><td>orientation fit + smoothness 1.0</td><td class="num">15.96</td><td class="num">18.58</td><td class="num">7.59</td><td class="num">9.18</td><td class="num"><b>21.9</b></td><td class="num"><b>42</b></td></tr>
<tr><td>orientation fit, gentle (100 steps, anchor 0.1, smooth)</td><td class="num">16.10</td><td class="num">18.61</td><td class="num">7.57</td><td class="num">9.16</td><td class="num">21.6</td><td class="num">50</td></tr>
<tr><td>inject measured elbow + head orientations</td><td class="num">15.85</td><td class="num">18.60</td><td class="num">7.45</td><td class="num">9.00</td><td class="num">23.6</td><td class="num">147</td></tr>
<tr><td>inject all five measured orientations</td><td class="num">16.09</td><td class="num">18.59</td><td class="num">7.52</td><td class="num">9.09</td><td class="num">23.6</td><td class="num">146</td></tr>
<tr><td>reduced rigid-body PD refiner (no contacts), stiffness 40</td><td class="num">16.00</td><td class="num">19.21</td><td class="num">7.69</td><td class="num">9.52</td><td class="num">23.8</td><td class="num">78</td></tr>
</tbody></table></div>
<p class="muted">PIP's optimizer was fed what our pose-only model does not predict: joint velocities from finite differences of the predicted joints, foot contacts from foot speed, the root stood on PIP's floor; the simulator runs at 50 Hz on 2× upsampled predictions because PIP's 60 Hz gains diverge at 25 Hz. With those gains the character tracks the network pose almost exactly, so the pose does not move; the optimizer's documented benefits (translation, foot sliding, jitter of jittery inputs) are not what root-relative pose metrics measure.</p>

<h2>The DIP benchmark (after the DIP fine-tune): scaling laws</h2>
<p>Secondary benchmark. Pretrain, fine-tune on real DIP training subjects, test on the held-out DIP subjects. Model size: S = d256/4 layers (3.3 M), M = d384/6 (10.9 M), L = d512/8 (25.6 M), XL = d768/8 (~58 M); same optimizer (AdamW 3e-4), effective batch 256, dropout 0.1. XL is the exception: at 3e-4 it diverged, so XL runs use 1.5e-4 with a 1000-step warmup and beta2 0.95 (L and XL also use gradient clipping and the acceleration clamp), and the XL-on-control run was trained on fig2 in bf16, which the calibration runs put at about +0.1. Data: curated-12 AMASS (35 h), control = curated-12 + Nymeria (267 h), treatment = control + BONES-SEED + form-hoi + MotionMillion mocap + Motion-X (676 h). Round markers are a fixed 20-epoch budget (each run sees every window 20 times); squares are the full 60-epoch schedule.</p>
<div class="grid3">
<div><h3>Model size, at fixed data</h3>{model_chart}</div>
<div><h3>Data, at fixed model size</h3>{data_chart}</div>
<div><h3>Compute</h3>{comp_chart}</div>
</div>
<div class="tablewrap"><table><thead><tr><th>model</th><th>data</th><th class="num">hours</th><th class="num">epochs</th><th class="num">seed</th><th>pretrain</th><th class="num">zero-shot SIP °</th><th class="num">zero-shot MPJRE °</th><th class="num">zero-shot MPJPE cm</th><th class="num">dip_test SIP ° (FT)</th></tr></thead><tbody>{grows}</tbody></table></div>

<h2>dip_test leaderboard (one row per pretrained model, after the DIP fine-tune)</h2>
<div class="tablewrap"><table><thead><tr><th class="num">#</th><th>run</th><th>model</th><th>data</th><th class="num">hours</th><th class="num">ep</th><th class="num">SIP °</th><th class="num">MPJRE °</th><th class="num">MPJPE cm</th></tr></thead><tbody>{lrows}</tbody></table></div>
<p class="muted">Previous deliverable (2026-09-01, same S model and control data, in-RAM loader, original Nymeria files): {PREV_DELIVERABLE:.2f}.</p>

<h2>Pushing the best model: recipe levers and ensembles</h2>
<p>Model size saturates near 11 M parameters on 267 h and the new data does not move the converged M model, so the remaining room is in the recipe. Each lever below changes one thing on top of a finished base run and is compared with that run; green means better than the ±{NOISE} seed noise, red worse. Fine-tune-only variants reuse the pretrained checkpoint, so they also measure how much of the run-to-run noise comes from the fine-tuning stage alone.</p>
<div class="tablewrap"><table><thead><tr><th>run</th><th>lever</th><th>model</th><th>compared with</th><th>status</th><th class="num">SIP °</th><th class="num">Δ</th></tr></thead><tbody>{lever_rows}</tbody></table></div>
<h3>Ensembles of existing fine-tuned checkpoints</h3>
<p>{html.escape(ens_head)} Members are averaged in the 6-D rotation representation before orthonormalisation. Candidate ensembles were ranked on fval (the fine-tuning validation subjects) and dip_test is reported for every candidate; the fval order and the dip_test order agree. No new seeds were trained for this.</p>
<div class="tablewrap"><table><thead><tr><th>members</th><th class="num">n</th><th class="num">stride</th><th class="num">fval SIP °</th><th class="num">dip_test SIP °</th><th class="num">MPJRE °</th><th class="num">MPJPE cm</th><th class="num">Δ vs best single</th></tr></thead><tbody>{ens_rows}</tbody></table></div>
<h3>Inference window</h3>
<p>The transformer is trained on 125-frame windows and evaluated by tiling each sequence with that window. Longer windows at inference degrade sharply (fval SIP below): the model does not extrapolate to longer contexts.</p>
<div class="tablewrap"><table><thead><tr><th>model</th>{win_head}</tr></thead><tbody>{win_rows}</tbody></table></div>
<p>Keeping the 125-frame window but sliding it with a smaller stride and averaging the overlapping predictions removes the error at window boundaries. The gain is small but monotone in both models and both splits (stride 125 is the original tiling; stride 31 costs 4× the inference compute, stride 12 10×). Every other number on this page uses the original tiling so runs stay comparable.</p>
<div class="tablewrap"><table><thead><tr><th>model · split</th>{stride_head}</tr></thead><tbody>{stride_rows}</tbody></table></div>

<h2>Does the new motion data help the S model? (the original question)</h2>
<div class="tablewrap"><table><thead><tr><th>arm</th><th class="num">seed</th><th class="num">hours</th><th>pretrain</th><th class="num">SIP °</th><th class="num">MPJRE °</th><th class="num">MPJPE cm</th></tr></thead><tbody>{tc_rows}</tbody></table></div>
<div class="grid2"><div><h3>Validation loss during pretraining</h3>{curve}</div><div class="panel"><h3 style="margin-top:0">Reading it</h3><p>Validation loss is on real DIP training windows and selects the pretrain checkpoint; it is not the test metric. With the corrected conversion, the clean picture so far is budget-dependent: BONES-SEED helps the S model at a 20-epoch budget but not at 60 epochs, and not the M model at 60 epochs. Extra motion acts as a regulariser while the model is under-trained and has not yet moved the converged asymptote. The arms below (full corrected treatment, mixing ratio, DIP-like selection, the leftover AMASS sets, a broad-to-narrow curriculum, MotionGV filtered / raw / DIP-like) test whether a different way of adding data changes that.</p></div></div>

<h2>Which data, and how to add it? (S model, 20 epochs unless noted)</h2>
<p>Each row changes one thing about the pretraining data relative to the reference (control, S, 20 epochs). "Sampled ×k per epoch" keeps every sequence but changes how often the new data is drawn (×1 is the natural ratio, about 60 percent of windows); the DIP-like subset keeps only sequences within 20° of DIP's mean pose with moderate accelerations and some motion (186 of the 409 new hours; 256 of the 636 MotionGV hours); the curriculum pretrains on everything and then narrows back to control before the DIP fine-tune.</p>
<div class="tablewrap"><table><thead><tr><th>pretraining data</th><th class="num">seed</th><th class="num">hours</th><th>pretrain</th><th class="num">SIP °</th><th class="num">MPJRE °</th><th class="num">MPJPE cm</th></tr></thead><tbody>{arows}</tbody></table></div>
<p class="muted">MotionGV (MotionMillion's video-estimated part) is tested two ways: filtered (5-frame moving average at 30 fps, clips of at least 2 s, clip dropped if any sensor acceleration exceeds 120 m/s²; {next((f"{r['hours']:.0f} h" for r in rows if r['group'].startswith('MotionGV filtered')), '–')} kept) and unfiltered (clips of at least 1 s, no smoothing, no cap; {next((f"{r['hours']:.0f} h" for r in rows if r['group'].startswith('MotionGV unfiltered')), '–')}). After the interpolation fix both have under 0.1 percent of frames above 50 m/s².</p>

<h3>How far is each dataset from DIP?</h3>
<p>Per sequence, from the packed shards: the angular distance between the sequence's mean local joint rotation and DIP-train's mean pose (heading-invariant; median and 90th percentile, weighted by hours), the share of hours within 20° of DIP (DIP's own sequences sit at a median of 10° and a maximum of 21°), the sequence's own pose diversity, the 95th-percentile sensor acceleration and the share of near-static frames. Earlier, WHIP (sports) hurt and Nymeria (everyday motion) helped, and distribution rather than quality was the explanation; this table applies the same lens to the new sources and to the leftover AMASS sets, and is the basis of the DIP-like selection rule tested in the ablations.</p>
<div class="tablewrap"><table><thead><tr><th>dataset</th><th class="num">hours</th><th class="num">sequences</th><th class="num">dist. to DIP, median °</th><th class="num">p90 °</th><th class="num">within 20°</th><th class="num">diversity °</th><th class="num">acc p95 m/s²</th><th class="num">static</th></tr></thead><tbody>{dl_rows}</tbody></table></div>

<h2>Does the DIP fine-tune transfer? (base vs after DIP FT on the IMUPoser dataset)</h2>
<p>For eight pretrained models the DIP-fine-tuned version was also evaluated on the collected dataset. Cells read base / after DIP FT. The fine-tune adapts the model to DIP's sensors and calibration: it improves dip_test and costs 1 to 5° SIP on this real-device set, while angular and positional errors move much less.</p>
<div class="tablewrap"><table><thead><tr><th>pretrained model</th><th class="num">dip_test SIP ° (after FT)</th><th class="num">SIP ° base</th><th class="num">SIP ° after DIP FT</th><th class="num">MPJRE °</th><th class="num">MPJPE cm</th><th class="num">MPVPE cm</th><th class="num">MPJVE cm/s</th><th class="num">jitter</th></tr></thead><tbody>{zs_rows}</tbody></table></div>

<h2>What was trained on</h2>
{svg_stack(rows)}
<div class="tablewrap"><table><thead><tr><th>group</th><th class="num">files</th><th class="num">hours @25 fps</th><th class="num">sequences</th></tr></thead><tbody>{drows}</tbody></table></div>

<h2>Method notes</h2>
<div class="grid2">
<div class="panel"><h3 style="margin-top:0">Protocol</h3><dl><dt>pretrain</dt><dd>AvatarPoser transformer, 125-frame windows, lw_rw_rp specialist, calibration-error augmentation 0.122 rad, select on dip_train loss</dd><dt>fine-tune</dt><dd>real DIP ftrain (32 seqs), 60 ep, lr 1e-4, select on fval (9 seqs)</dd><dt>test</dt><dd>dip_test s09/s10, last fine-tuned checkpoint</dd><dt>loader</dt><dd>streaming memory-mapped shards (verified identical to the in-RAM loader); RAM flat at any data size</dd><dt>data selection</dt><dd>DIP-like subsets keep sequences whose mean pose is within 20° of DIP's, with acceleration p95 ≤ 22 m/s² and pose diversity ≥ 4° (DATASET_KEEP; shards untouched)</dd><dt>large models</dt><dd>L and XL train with gradient-norm clipping at 1.0 and synthetic acceleration clamped to ±160 m/s² (a 16 g sensor's range) after L showed 25–30× loss spikes that M never did; an M run with the same settings is the control</dd><dt>machines</dt><dd>TITAN V + TITAN X Pascal in fp32 (16-mixed: 2× slower on Pascal, +6 % on Volta); from 2026-10-06 also fig2, 2× RTX 4060 Ti in bf16 mixed precision (1.74× faster there), with an S20 control calibration run against the fp32 numbers</dd></dl></div>
<div class="panel"><h3 style="margin-top:0">Converting SOMA and 272-dim motion to SMPL IMU</h3><p>SOMA-rig data (BONES-SEED BVH, form-hoi params) is retargeted closed-form: each SMPL joint copies its SOMA counterpart's global rotation with a per-joint offset calibrated from the two T-pose meshes (SOMA-X topology bridge + Kabsch). Posed per-part error 1–3° on thighs, head, pelvis and 6–10° on forearms, better than SOMA-X's own mesh fit and ~5000× faster. MotionMillion's 272-dim representation is inverted in closed form (no IK). IMU is synthesized by skinning only the six sensor vertices (exact to 1e-4 vs the full mesh, 70× faster). Mixed precision was not adopted; the real speedups were vectorizing a per-element rotation conversion (45–750×) and the mesh-free synthesis.</p></div>
</div>

<h2>Invalidated runs (first conversion of the new data)</h2>
<p>The first conversion of form-hoi, MotionMillion and MotionGV upsampled 30 to 60 fps by linearly interpolating axis-angle poses. Across the ±π wrap that collapses a limb to rest for one frame and the second-difference accelerometer explodes: 1.7 to 10.8 percent of frames above 50 m/s² with peaks in the thousands, against 0.00 percent for BONES-SEED (stride-decimated) and 0.13 percent for Nymeria. Rotations are now interpolated in matrix space and the affected datasets re-converted; every arm that used them is rerun. The numbers below are kept for the record only.</p>
<div class="tablewrap"><table><thead><tr><th>run</th><th>model</th><th>data</th><th class="num">SIP °</th><th>why invalid</th></tr></thead><tbody>
{''.join(f"<tr><td>{html.escape(r['tag'])}</td><td>{r['size']}</td><td>{r['arm']} {html.escape(r['extra'])}</td><td class='num'>{fmt(r['eval'] and r['eval']['sip'])}</td><td>{INVALID[r['tag'].split(' ')[0]]}</td></tr>" for r in invalid)}
</tbody></table></div>

<h2>Timeline</h2>
<ul>{tl}</ul>
</div>
"""
    Path(out_path).write_text(page)
    return head


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(REPO / "autoresearch" / "newdata_report.html"))
    a = ap.parse_args()
    print(build(a.out))
