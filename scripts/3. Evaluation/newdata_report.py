r"""
Build the results page for the new-motion-data experiment (control vs treatment, lw_rw_rp @ 25 Hz).

Reads: checkpoints/newdata/{base,ft}_<tag>/ (train.log progress lines, best_model.txt, eval_dip_test.log),
       /home/vimal/imuposer_data/logs/ (run + chain + conversion logs), shard meta.json (hours per dataset).
Writes: autoresearch/newdata_report.html (self-contained, inline SVG charts) -- published as the artifact.

  uv run python "scripts/3. Evaluation/newdata_report.py" [--out path.html]
"""
import argparse, glob, html, json, os, re, time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CK = REPO / "checkpoints" / "newdata"
LOGS = Path("/home/vimal/imuposer_data/logs")
DATA = Path("/home/vimal/imuposer_data/processed_imuposer_25fps")
SHARDS = Path("/home/vimal/imuposer_data/shards_processed_imuposer_25fps")
REF_SIP = 17.32           # previous lw_rw_rp 25 Hz deliverable (curated-12 + Nymeria -> DIP FT), results.jsonl 2026-09-01
CURATED = "CMU,BioMotionLab_NTroje,BMLmovi,KIT,EKUT,Transitions_mocap,HumanEva,SFU,HUMAN4D,SSM_synced,MPI_mosh,MPI_Limits".split(",")
GROUPS = [("curated-12 AMASS", lambda n: n in CURATED), ("Nymeria", lambda n: n.startswith("Nymeria_")),
          ("BONES-SEED", lambda n: n.startswith("BONES_")), ("form-hoi", lambda n: n.startswith("FORMHOI_")),
          ("MotionMillion (272-dim)", lambda n: n.startswith("MM_")), ("Motion-X (existing SMPL-X)", lambda n: n.startswith("MotionX_"))]


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
    r = {"epochs_done": 0, "steps_per_epoch": None, "it_s": None, "val": {}, "status": "pending", "train_files": None, "hours": None}
    m = re.search(r"\[stream\] (\d+) files, (\d+) windows, ([\d.]+) h", txt)
    if m:
        r["train_files"], r["hours"] = int(m.group(1)), float(m.group(3))
    # first progress line of each epoch carries the val loss of the previous epoch
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
    # exact values for the checkpointed epochs
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
    elif "Traceback" in txt or "Error" in txt:
        r["status"] = "error"
    elif r["epochs_done"] or "Epoch 0" in txt:
        r["status"] = "running"
    return r


def parse_eval(p):
    txt = read(p)
    m = re.search(r"SIP\s+([\d.]+) deg\s+MPJRE\s+([\d.]+) deg\s+MPJPE\s+([\d.]+) cm\s+MPVPE\s+([\d.]+) cm\s+MPJVE\s+([\d.]+)", txt)
    if not m:
        return None
    return {"sip": float(m.group(1)), "mpjre": float(m.group(2)), "mpjpe": float(m.group(3)), "mpvpe": float(m.group(4)), "mpjve": float(m.group(5))}


def run_times(tag):
    out = {}
    for lp in list(LOGS.glob("run_*.log")) + list(LOGS.glob("chain_treatment_*.log")):
        t = read(lp)
        m = re.search(rf"\[(\S+)\] START base_{tag} ", t)
        if m: out["start"] = m.group(1)
        m = re.search(rf"\[(\S+)\] base done", t) if f"START base_{tag}" in t else None
        if m: out["base_done"] = m.group(1)
        m = re.search(rf"\[(\S+)\] DONE {tag}", t)
        if m: out["done"] = m.group(1)
    return out


def collect_runs():
    runs = []
    for d in sorted(CK.glob("base_*")):
        tag = d.name[5:]
        arm = "control" if tag.startswith("control") else "treatment"
        seed = int(re.search(r"_s(\d+)", tag).group(1)) if re.search(r"_s(\d+)", tag) else 0
        base = parse_train_log(d / "train.log") or {"status": "pending", "val": {}, "epochs_done": 0}
        ft = parse_train_log(CK / f"ft_{tag}" / "train.log")
        ev = parse_eval(CK / f"ft_{tag}" / "eval_dip_test.log")
        runs.append({"tag": tag, "arm": arm, "seed": seed, "base": base, "ft": ft, "eval": ev, "times": run_times(tag)})
        # interim read-outs (fine-tune of a best-so-far checkpoint while the base was still training)
        for idir in sorted(CK.glob(f"ft_{tag}_interim_ep*")):
            ep = int(re.search(r"interim_ep(\d+)", idir.name).group(1))
            iev = parse_eval(idir / "eval_dip_test.log")
            if iev:
                runs.append({"tag": f"{tag} (interim, base ep {ep})", "arm": arm, "seed": seed, "interim": True,
                             "base": {"status": "done", "val": {}, "epochs_done": ep + 1, "hours": base.get("hours"), "train_files": base.get("train_files"),
                                      "best_epoch": ep, "best_val": base["val"].get(ep)},
                             "ft": parse_train_log(idir / "train.log"), "eval": iev, "times": {}})
    return runs


def data_hours():
    rows = []
    for gname, pred in GROUPS:
        names = sorted(p.stem for p in DATA.glob("*.pt") if pred(p.stem))
        h, nseq, packed = 0.0, 0, 0
        for n in names:
            meta = SHARDS / n / "meta.json"
            if meta.exists():
                m = json.loads(meta.read_text()); h += m["n_frames"] / 25 / 3600; nseq += m["n_seqs"]; packed += 1
            else:
                h += (DATA / f"{n}.pt").stat().st_size / 1452 / 25 / 3600      # bytes per 25fps frame in the .pt layout
        rows.append({"group": gname, "files": len(names), "hours": h, "seqs": nseq, "packed": packed})
    return rows


def conversion_status():
    out = []
    specs = [("MotionMillion", ["motionmillion_convert.log", "motionmillion_convert_B.log", "motionmillion_convert_C.log"]),
             ("BONES-SEED", ["bones_seed_convert.log"]), ("form-hoi", ["formhoi_convert.log"])]
    for name, files in specs:
        done, hours, last = [], 0.0, ""
        for f in files:
            t = read(LOGS / f)
            for m in re.finditer(r"^DONE (\S+): .*?([\d.]+) h in ([\d.]+) min", t, re.M):
                done.append(m.group(1).rstrip(":")); hours += float(m.group(2))
            ls = [l for l in t.strip().split("\n") if l.startswith("  wrote") or l.startswith("DONE") or "/4135]" in l]
            if ls: last = ls[-1].strip()
        out.append({"name": name, "done": done, "hours": hours, "last": last})
    return out


# ---------------------------------------------------------------- svg helpers
def svg_lines(series, w=720, h=300, xlabel="epoch", ylabel="val loss (dip_train)", ymax=None):
    """series: list of (label, color, dash, [(x,y)...])"""
    pts = [p for s in series for p in s[3]]
    if not pts:
        return "<p class='muted'>no training curves yet</p>"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = 0, max(max(xs), 1); y0 = min(ys) * 0.97; y1 = (ymax or max(ys)) * 1.03
    L, R, T, B = 56, 16, 14, 40
    def X(x): return L + (x - x0) / (x1 - x0) * (w - L - R)
    def Y(y): return T + (y1 - y) / (y1 - y0) * (h - T - B)
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
    return "".join(g)


def svg_bars(groups, w=720, h=280, ref=None, ylabel="dip_test SIP (deg, lower is better)"):
    """groups: list of (arm, color, [(seed, value)...], mean)"""
    vals = [v for g in groups for _, v in g[2]] + ([ref] if ref else [])
    if not vals:
        return "<p class='muted'>no dip_test results yet</p>"
    y0 = min(vals) - 0.6; y1 = max(vals) + 0.6
    L, R, T, B = 56, 16, 16, 44
    def Y(y): return T + (y1 - y) / (y1 - y0) * (h - T - B)
    n = sum(max(len(g[2]), 1) for g in groups) + len(groups)
    bw = (w - L - R) / n
    g = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img' aria-label='{html.escape(ylabel)} per seed'>"]
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        g.append(f"<line x1='{L}' x2='{w-R}' y1='{Y(yv):.1f}' y2='{Y(yv):.1f}' class='grid'/>")
        g.append(f"<text x='{L-6}' y='{Y(yv)+4:.1f}' class='tick' text-anchor='end'>{yv:.2f}</text>")
    if ref:
        g.append(f"<line x1='{L}' x2='{w-R}' y1='{Y(ref):.1f}' y2='{Y(ref):.1f}' class='ref'/>")
        g.append(f"<text x='{w-R-4}' y='{Y(ref)-5:.1f}' class='tick' text-anchor='end'>previous best {ref:.2f}</text>")
    x = L + bw / 2
    for arm, color, items, mean in groups:
        xs0 = x
        for seed, v in items:
            g.append(f"<rect x='{x:.1f}' y='{Y(v):.1f}' width='{bw*0.8:.1f}' height='{Y(y0)-Y(v):.1f}' fill='{color}' rx='2'/>")
            g.append(f"<text x='{x+bw*0.4:.1f}' y='{Y(v)-5:.1f}' class='tick' text-anchor='middle'>{v:.2f}</text>")
            g.append(f"<text x='{x+bw*0.4:.1f}' y='{h-B+16}' class='tick' text-anchor='middle'>s{seed}</text>")
            x += bw
        if not items:
            g.append(f"<text x='{x+bw*0.4:.1f}' y='{(T+h-B)/2:.0f}' class='tick' text-anchor='middle'>pending</text>"); x += bw
        if mean is not None and len(items) > 1:
            g.append(f"<line x1='{xs0:.1f}' x2='{x-bw*0.2:.1f}' y1='{Y(mean):.1f}' y2='{Y(mean):.1f}' stroke='{color}' stroke-width='2' stroke-dasharray='3 3'/>")
        g.append(f"<text x='{(xs0+x-bw*0.2)/2:.1f}' y='{h-B+34}' class='axis' text-anchor='middle'>{arm}{'' if mean is None else f' (mean {mean:.2f})'}</text>")
        x += bw
    g.append("</svg>")
    return "".join(g)


def svg_stack(rows, w=720, h=70):
    tot = sum(r["hours"] for r in rows) or 1
    cols = ["#5b6ee1", "#d98f3b", "#2f9e7a", "#c45c8a", "#8e6ad1", "#6b8fa3"]
    g = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img' aria-label='training hours by dataset group'>"]
    x = 0
    for i, r in enumerate(rows):
        ww = r["hours"] / tot * w
        g.append(f"<rect x='{x:.1f}' y='8' width='{max(ww,0):.1f}' height='34' fill='{cols[i%len(cols)]}'/>")
        if ww > 60:
            g.append(f"<text x='{x+ww/2:.1f}' y='30' class='barlab' text-anchor='middle'>{r['hours']:.0f} h</text>")
        x += ww
    g.append("</svg>")
    leg = "".join(f"<span class='leg'><i style='background:{cols[i%len(cols)]}'></i>{html.escape(r['group'])} {r['hours']:.0f} h</span>" for i, r in enumerate(rows))
    return "".join(g) + f"<div class='legend'>{leg}</div>"


# ---------------------------------------------------------------- page
def mean(xs):
    return sum(xs) / len(xs) if xs else None


def welch(a, b):
    """Welch t-test p-value (two-sided) via a normal approximation when n is tiny; returns None if n<2."""
    import math
    if len(a) < 2 or len(b) < 2: return None
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1); vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0: return None
    t = (ma - mb) / se
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    # Student t survival via incomplete beta (simple numeric integration)
    import math as m_
    def tcdf(t, df):
        x = df / (df + t * t)
        # regularized incomplete beta I_x(df/2, 1/2) by numeric integration
        a, b = df / 2, 0.5
        n = 4000; s = 0.0
        for i in range(n):
            u = (i + 0.5) / n * x
            s += u ** (a - 1) * (1 - u) ** (b - 1)
        s *= x / n
        B = m_.gamma(a) * m_.gamma(b) / m_.gamma(a + b)
        return 1 - 0.5 * s / B
    p = 2 * (1 - tcdf(abs(t), df))
    return max(min(p, 1.0), 0.0)


def build(out_path):
    runs = collect_runs()
    rows = data_hours()
    conv = conversion_status()
    now = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    final = [r for r in runs if not r.get("interim")]
    ctrl = [r for r in final if r["arm"] == "control"]; trt = [r for r in final if r["arm"] == "treatment"]
    interim = [r for r in runs if r.get("interim")]
    c_sip = [r["eval"]["sip"] for r in ctrl if r["eval"]]; t_sip = [r["eval"]["sip"] for r in trt if r["eval"]]
    cm, tm = mean(c_sip), mean(t_sip)
    p = welch(c_sip, t_sip)

    # headline
    if cm is not None and tm is not None:
        delta = tm - cm
        verdict = ("treatment better" if delta < 0 else "treatment worse") + f" by {abs(delta):.2f} deg SIP"
        head = f"{verdict} ({len(t_sip)} vs {len(c_sip)} seeds{'' if p is None else f', Welch p = {p:.3f}'})"
        sub = "Noise floor from the Nymeria study: about 0.17 deg seed-to-seed std; a difference under ~0.3 deg is not a result."
    elif cm is not None:
        head = f"Control arm so far: dip_test SIP {cm:.2f} ({len(c_sip)} seed{'s' if len(c_sip)>1 else ''}). Treatment arm still training."
        sub = "The treatment number lands when its 60-epoch base run plus DIP fine-tune finishes; this page is regenerated then."
        if interim:
            r = interim[-1]
            head = f"Interim: treatment {r['eval']['sip']:.2f} vs control {cm:.2f} dip_test SIP (treatment read out at base epoch {r['base']['best_epoch']} of 60, not final)."
            sub = ("An interim checkpoint is not a finished run, but the direction matches the WHIP lesson: lower validation loss "
                   "on DIP-train windows while dip_test gets worse. Final treatment numbers replace this when the runs finish.")
    else:
        head = "Both arms are still training."; sub = "Results appear here as each seed's fine-tune and dip_test evaluation complete."

    # results table
    def fmt(v, nd=2): return "–" if v is None else f"{v:.{nd}f}"
    trows = []
    for r in sorted(runs, key=lambda r: (r["arm"], r["seed"], bool(r.get("interim")))):
        b = r["base"]; e = r["eval"]
        st = b.get("status", "pending")
        if r.get("interim"):
            trows.append(f"<tr class='interim'><td>{r['arm']} <span class='chip wait'>interim</span></td><td>s{r['seed']}</td><td>{fmt(b.get('hours'),0)} h</td>"
                         f"<td>read-out at base epoch {b.get('best_epoch')} (val {fmt(b.get('best_val'),5)})</td><td>done</td>"
                         f"<td class='num'>{fmt(e['sip'])}</td><td class='num'>{fmt(e['mpjre'])}</td><td class='num'>{fmt(e['mpjpe'])}</td></tr>")
            continue
        prog = f"{b.get('epochs_done',0)}/60 ep" + (f" @ {b['it_s']:.1f} it/s" if b.get("it_s") else "")
        if st == "done": prog = f"done (best ep {b.get('best_epoch','?')}, val {fmt(b.get('best_val'),5)})"
        ftst = "–" if r["ft"] is None else ("done" if r["ft"]["status"] == "done" else r["ft"]["status"])
        chip = {"done": "ok", "running": "run", "error": "bad", "pending": "wait"}[st]
        trows.append(f"<tr><td>{r['arm']}</td><td>s{r['seed']}</td><td>{fmt(b.get('hours'),0)} h / {b.get('train_files') or '–'} files</td>"
                     f"<td><span class='chip {chip}'>{st}</span> {html.escape(prog)}</td><td>{ftst}</td>"
                     f"<td class='num'>{fmt(e and e['sip'])}</td><td class='num'>{fmt(e and e['mpjre'])}</td><td class='num'>{fmt(e and e['mpjpe'])}</td></tr>")
    # curves
    cols = {"control": "#5b6ee1", "treatment": "#d98f3b"}
    series = []
    for r in sorted(final, key=lambda r: (r["arm"], r["seed"])):
        pts = sorted(r["base"]["val"].items())
        series.append((f"{r['arm']} s{r['seed']}", cols[r["arm"]], r["seed"] != 1, pts))
    curve = svg_lines(series)
    legend = "".join(f"<span class='leg'><i style='background:{cols[a]}'></i>{a}</span>" for a in cols) + "<span class='leg muted'>dashed = seeds 2+</span>"
    bars = svg_bars([("control", cols["control"], [(r["seed"], r["eval"]["sip"]) for r in ctrl if r["eval"]], cm),
                     ("treatment", cols["treatment"], [(r["seed"], r["eval"]["sip"]) for r in trt if r["eval"]], tm)], ref=REF_SIP)
    # data table
    drows = "".join(f"<tr><td>{html.escape(r['group'])}</td><td class='num'>{r['files']}</td><td class='num'>{r['hours']:.1f}</td>"
                    f"<td class='num'>{r['seqs'] or '–'}</td><td>{'control + treatment' if r['group'] in ('curated-12 AMASS','Nymeria') else 'treatment only'}</td></tr>" for r in rows)
    ctrl_h = sum(r["hours"] for r in rows if r["group"] in ("curated-12 AMASS", "Nymeria")); all_h = sum(r["hours"] for r in rows)
    crow = "".join(f"<tr><td>{html.escape(c['name'])}</td><td class='num'>{c['hours']:.1f}</td><td>{len(c['done'])} subsets done</td><td class='mono'>{html.escape(c['last'][:90])}</td></tr>" for c in conv)
    # timeline
    tl = []
    for r in sorted(final, key=lambda r: (r["arm"], r["seed"])):
        t = r["times"]
        tl.append(f"<li><b>{r['arm']} s{r['seed']}</b>: start {t.get('start','–')[11:16] if t.get('start') else '–'}, base done {t.get('base_done','–')[11:16] if t.get('base_done') else '–'}, eval done {t.get('done','–')[11:16] if t.get('done') else '–'}</li>")

    page = f"""<title>IMUPoser New-Data Study</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one measured column (72ch) for prose, full-width panels for tables and charts */
:root{{--bg:#f6f4ef;--fg:#1d1c19;--muted:#6a665c;--line:#dcd7cc;--panel:#fffdf8;--accent:#1f5f7a;--ok:#2f9e7a;--warn:#d98f3b;--bad:#c4453c;--grid:#e6e1d6;
--display:"Source Serif 4",Georgia,"Times New Roman",serif;--body:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#171715;--fg:#ece8df;--muted:#a39e92;--line:#34322d;--panel:#1f1e1b;--accent:#7cc2dc;--ok:#5bc69e;--warn:#e8a85a;--bad:#e36a60;--grid:#2c2a26;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#171715;--fg:#ece8df;--muted:#a39e92;--line:#34322d;--panel:#1f1e1b;--accent:#7cc2dc;--ok:#5bc69e;--warn:#e8a85a;--bad:#e36a60;--grid:#2c2a26;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.55;margin:0}}
.wrap{{max-width:980px;margin:0 auto;padding-block:28px 56px;padding-inline:16px}}
h1,h2,h3{{font-family:var(--display);text-wrap:balance;margin:0 0 .35em}} h1{{font-size:2rem;font-weight:700}} h2{{font-size:1.35rem;margin-top:2.2em}} h3{{font-size:1.05rem;margin-top:1.4em}}
p{{max-width:72ch}} .muted{{color:var(--muted)}} .eyebrow{{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}}
.headline{{background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--accent);padding:16px 20px;margin:18px 0 8px}} .headline .big{{font-family:var(--display);font-size:1.45rem;font-weight:700;margin:0 0 6px}}
table{{border-collapse:collapse;width:100%;font-size:.92rem;margin:10px 0}} th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}} th{{font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}} .tablewrap{{overflow-x:auto}} .mono{{font-family:var(--mono);font-size:.82rem}}
.chip{{display:inline-block;font-size:.72rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:2px 7px;border-radius:999px;margin-right:6px}}
.chip.ok{{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}} .chip.run{{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}} .chip.bad{{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}} .chip.wait{{background:color-mix(in srgb,var(--muted) 18%,transparent);color:var(--muted)}}
.chart{{width:100%;height:auto;display:block;background:var(--panel);border:1px solid var(--line);border-radius:4px}} .chart .grid{{stroke:var(--grid);stroke-width:1}} .chart .tick{{fill:var(--muted);font-size:11px;font-family:var(--mono)}} .chart .axis{{fill:var(--fg);font-size:12px}} .chart .ref{{stroke:var(--bad);stroke-width:1.5;stroke-dasharray:4 4}} .chart .barlab{{fill:#fff;font-size:12px;font-weight:600}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.85rem;margin:8px 0 0}} .leg i{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:6px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}} .panel{{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:14px 16px;min-width:0}}
ul{{padding-left:1.2em}} li{{margin:.25em 0}} code{{font-family:var(--mono);font-size:.86em}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:8px 0}} dt{{color:var(--muted)}} dd{{margin:0}}
</style>
<div class="wrap">
<div class="eyebrow">IMUPoser · lw_rw_rp · 25 Hz · generated {html.escape(now)}</div>
<h1>Does new motion data help the left-watch, right-watch, right-pocket model?</h1>
<p>Pretraining data added: BONES-SEED (originals), NVIDIA form-hoi, and MotionMillion's mocap subsets. Everything else is held fixed: same architecture, schedule, augmentation, DIP fine-tune and the held-out dip_test subjects. Both arms keep curated-12 AMASS + Nymeria, so the question is whether the new data helps <em>on top of the current best recipe</em>.</p>
<div class="headline"><div class="eyebrow">headline</div><p class="big">{html.escape(head)}</p><p class="muted" style="margin:0">{html.escape(sub)}</p></div>

<h2>Results on dip_test (subjects 9 and 10, never trained on)</h2>
<div class="tablewrap"><table><thead><tr><th>arm</th><th>seed</th><th>pretrain data</th><th>base (60 ep)</th><th>DIP FT</th><th class="num">SIP °</th><th class="num">MPJRE °</th><th class="num">MPJPE cm</th></tr></thead><tbody>{''.join(trows)}</tbody></table></div>
<div class="grid2"><div><h3>dip_test SIP per seed</h3>{bars}</div><div><h3>Validation loss during pretraining</h3>{curve}<div class="legend">{legend}</div></div></div>
<p class="muted">SIP = mean angular error of hips and shoulders (the standard sparse-IMU metric). The reference line is the 2026-09-01 deliverable (same recipe, original 25 fps Nymeria files, in-RAM loader). Validation loss is on real DIP training subjects and selects the base checkpoint; it is not the test metric.</p>

<h2>What was trained on</h2>
{svg_stack(rows)}
<div class="tablewrap"><table><thead><tr><th>group</th><th class="num">files</th><th class="num">hours @25 fps</th><th class="num">sequences</th><th>used by</th></tr></thead><tbody>{drows}</tbody></table></div>
<p>Control arm: {ctrl_h:.0f} h. Treatment arm: {all_h:.0f} h. Hours are counted from the packed shards where available, otherwise estimated from file size.</p>
<div class="tablewrap"><table><thead><tr><th>conversion</th><th class="num">hours written</th><th>progress</th><th>last log line</th></tr></thead><tbody>{crow}</tbody></table></div>

<h2>Protocol</h2>
<div class="grid2">
<div class="panel"><h3 style="margin-top:0">Stage 1 · pretrain</h3><dl><dt>model</dt><dd>AvatarPoser transformer, 125-frame windows</dd><dt>combo</dt><dd>lw_rw_rp specialist</dd><dt>epochs</dt><dd>60, fixed; select best dip_train loss</dd><dt>augment</dt><dd>calibration error 0.122 rad</dd><dt>loader</dt><dd>streaming memmap shards (new, verified identical to the in-RAM loader)</dd><dt>precision</dt><dd>fp32 both arms</dd></dl></div>
<div class="panel"><h3 style="margin-top:0">Stage 2 · fine-tune and test</h3><dl><dt>fine-tune</dt><dd>real DIP ftrain (32 seqs), 60 ep, lr 1e-4, select on fval (9 seqs)</dd><dt>test</dt><dd>dip_test s09/s10, last fine-tuned checkpoint, offline_fit --iters 0</dd><dt>seeds</dt><dd>3 per arm planned; noise floor ~0.17° std</dd><dt>confound check</dt><dd>the Nymeria study showed extra steps alone (curated ×174 ep) buy nothing, so the larger treatment epoch is not a compute confound</dd></dl></div>
</div>

<h2>How the new datasets became IMU training data</h2>
<p>None of the three are IMU datasets. They are motion datasets, so sensors are synthesized exactly as for AMASS and Nymeria: SMPL forward kinematics, global orientation of the five worn bones, and second-difference acceleration of the sensor vertices, then the standard 60 to 25 fps conversion.</p>
<div class="grid2">
<div class="panel"><h3 style="margin-top:0">BONES-SEED and form-hoi (SOMA rig)</h3><p>SOMA-X's mesh fit (PoseInversion) was both slow (270 frames/s, 16 h for BONES-SEED) and only 7 to 13° accurate on bone directions. Replaced by a closed-form retarget: copy each SOMA joint's global rotation onto its SMPL counterpart with a per-joint rest offset calibrated from the two T-pose meshes (SOMA-X bridges the SOMA mesh into SMPL topology, Kabsch per body part).</p>
<dl><dt>posed per-part error</dt><dd>thighs 1 to 3°, head 1°, pelvis 1°, shins 1 to 3°, forearms 6 to 10° (SOMA-X's own fit: 4 to 6°, 6°, 1 to 3°, 10 to 11°, 9 to 14°)</dd><dt>BVH frames</dt><dd>BVH forward kinematics lands in SOMA-X's joint frames (idle clips within 2 to 11° of the rest frames on legs and spine)</dd><dt>sensor frame check</dt><dd>forearm sensor x-axis vs forearm direction: BONES 3.2°, Nymeria 3.2°, real DIP 3.9°</dd><dt>form-hoi frame</dt><dd>camera-rig frame tilted 3.4°; realigned to gravity with the ground-plane normal; QC-flagged human-pose frames and invalid frames cut (object-only checks kept)</dd></dl></div>
<div class="panel"><h3 style="margin-top:0">MotionMillion (272-dim)</h3><p>The MotionStreamer representation stores 22 local joint rotations as 6D plus heading-free root velocity and height. Recovery is closed-form (undo the accumulated heading on the root, integrate xz velocity, take the root height), no IK. Their world is already y-up. MotionGV (video-estimated, 114 GB) is excluded for now; the seven Motion-X subsets already in the pipeline are reused from the SMPL-X originals.</p>
<h3>IMU synthesis without the mesh</h3><p>Only the six sensor vertices are skinned (exact linear blend skinning with their own weights), which reproduces the full 6890-vertex mesh to 1e-4 relative and runs 70× faster. Validated against the full mesh on CMU and BMLmovi.</p></div>
</div>
<h3>Data quality of the sources (25 fps, sensors 1 to 5)</h3>
<div class="tablewrap"><table><thead><tr><th>source</th><th>mean clip</th><th class="num">|acc| mean m/s²</th><th class="num">|acc| p95</th><th class="num">frames &gt;50 m/s²</th><th class="num">joint jerk median</th><th class="num">jerk p95</th></tr></thead><tbody>
<tr><td>MotionGV folder9 sample (video-estimated, raw)</td><td>2.6 s</td><td class="num">20.2</td><td class="num">37.2</td><td class="num">4.0 %</td><td class="num">100</td><td class="num">1145</td></tr>
<tr><td>MotionMillion finedance (mocap)</td><td>4.0 s</td><td class="num">10.2</td><td class="num">18.4</td><td class="num">1.3 %</td><td class="num">68</td><td class="num">273</td></tr>
<tr><td>MotionMillion interx (mocap)</td><td>6.1 s</td><td class="num">6.5</td><td class="num">6.9</td><td class="num">1.1 %</td><td class="num">29</td><td class="num">161</td></tr>
<tr><td>BONES-SEED</td><td>6.9 s</td><td class="num">3.1</td><td class="num">11.0</td><td class="num">0.01 %</td><td class="num">42</td><td class="num">285</td></tr>
<tr><td>form-hoi</td><td>15.8 s</td><td class="num">5.9</td><td class="num">2.8</td><td class="num">1.1 %</td><td class="num">31</td><td class="num">86</td></tr>
<tr><td>Nymeria</td><td>15 min</td><td class="num">1.0</td><td class="num">2.7</td><td class="num">0.07 %</td><td class="num">42</td><td class="num">89</td></tr>
<tr><td>CMU (AMASS)</td><td>15.9 s</td><td class="num">2.5</td><td class="num">8.8</td><td class="num">0.02 %</td><td class="num">38</td><td class="num">174</td></tr>
</tbody></table></div>
<p class="muted">Jerk is the third difference of joint positions (m/s³, median over frames and joints). Raw MotionGV is 4 to 10× jitterier than any mocap source, its clips are short (median 1.3 s, 35 % under one second) and 4 % of its frames carry accelerations above 50 m/s², so it is converted with a 5-frame moving average at 30 fps and a 2 s minimum clip length before synthesis (see the MotionGV arm).</p>

<h2>Timeline</h2>
<ul>{''.join(tl)}</ul>
<p class="muted">Both GPUs are kept busy: TITAN V runs seed 1 of each arm, the slower TITAN X Pascal runs seed 2. Conversions ran at the lowest CPU priority alongside training. Mixed precision was tested and rejected for now (2× slower on the Pascal, which lacks fp16 tensor cores; untested on Volta to keep the control run clean).</p>
</div>
"""
    Path(out_path).write_text(page)
    return head


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(REPO / "autoresearch" / "newdata_report.html"))
    a = ap.parse_args()
    print(build(a.out))
