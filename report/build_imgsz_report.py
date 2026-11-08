"""Build report/Imgsz_Tuning_Report.pdf: Ben's imgsz tuning compared with the baseline (Group 6 gesture detector).

Every number is read from the saved results (results/tuning_imgsz_summary.csv and results/<run>/), so after
re-running Gesture_YOLO_Tuning_imgsz.ipynb just run this again:

    python report/build_imgsz_report.py

Shares the look and helpers of build_baseline_report.py; figures and the HTML source go to report/build/.
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_baseline_report import (BROWSERS, BUILD, CLASSES, GRID, INK, INK_2, LOOSE_BOXES, ROOT, SERIES,
                                   UNLABELLED, bootstrap, table)

PDF = Path(__file__).resolve().parent / "Imgsz_Tuning_Report.pdf"
RESULTS = ROOT / "results"
SIZES = [480, 640, 800, 960]
RUN = {480: "imgsz_480", 640: "baseline", 800: "imgsz_800", 960: "imgsz_960"}
LABEL = {s: f"{s} (baseline)" if s == 640 else str(s) for s in SIZES}

# Control from Gesture_YOLO_Tuning_imgsz.ipynb section 3.4 (not saved to a file): the baseline weights,
# trained at 640, run on our photos at each inference size. Team-only mAP50 / mAP50-95.
CONTROL = {480: (0.837, 0.418), 640: (0.845, 0.419), 800: (0.823, 0.392), 960: (0.785, 0.372)}
# The baseline trained in 8.5 min on an idle GPU; the summary holds the final re-run on a shared GPU.
BASELINE_IDLE_MIN = 8.5


# --------------------------------------------------------------------------- data

def load():
    s = pd.read_csv(RESULTS / "tuning_imgsz_summary.csv", index_col="imgsz").sort_index()
    d = {"summary": s, "metrics": {}, "ea": {}, "size": {}, "cls": {}, "hands": {}, "preds": {}}
    for size, run in RUN.items():
        r = RESULTS / run
        d["metrics"][size] = json.loads((r / "metrics.json").read_text())
        d["ea"][size] = json.loads((r / "error_analysis.json").read_text())
        d["size"][size] = pd.read_csv(r / "team_by_size.csv", index_col=0)
        d["cls"][size] = pd.read_csv(r / "team_by_class.csv", index_col=0)
        d["hands"][size] = pd.read_csv(r / "team_hands.csv")
        d["preds"][size] = pd.read_csv(r / "team_preds.csv")
    return d


def fixed_broken(hands):
    """Hands each run gets right that the baseline got wrong (fixed), and the reverse (broken)."""
    key = ["file", "gt_index"]
    base = hands[640].set_index(key)["correct"]
    out = {}
    for s in SIZES:
        if s == 640:
            continue
        r = hands[s].set_index(key)["correct"].reindex(base.index, fill_value=False)
        out[s] = (int((~base & r).sum()), int((base & ~r).sum()))
    return out


def train_hand_sizes():
    """Share of training hands under 1% of the image and the median, from dataset/labels/train."""
    files = list((ROOT / "dataset" / "labels" / "train").glob("*.txt"))
    if not files:   # dataset/ is not in git; fall back to the values measured in the notebook
        return 386, 1280, 0.016
    area = np.concatenate([np.loadtxt(f, ndmin=2)[:, 3] * np.loadtxt(f, ndmin=2)[:, 4] for f in files])
    return int((area < 0.01).sum()), len(area), float(np.median(area))


# --------------------------------------------------------------------------- figures

def fig_overview(s, noise):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    a = axes[0]
    base = s.loc[640, "F1 @ conf 0.25"]
    a.axhspan(base - noise, base + noise, color=GRID, alpha=0.6, lw=0)
    for col, lbl, c in [("team mAP50", "team mAP50", SERIES[0]), ("F1 @ conf 0.25", "F1 @ conf 0.25", SERIES[1])]:
        a.plot(s.index, s[col], color=c, marker="o", ms=5, label=lbl)
        for x, v in zip(s.index, s[col]):
            a.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    a.set_xticks(SIZES); a.set_ylim(0.74, 0.90); a.set_xlabel("imgsz")
    a.set_title("Accuracy on our photos"); a.legend(loc="lower left", fontsize=8.5)
    a.text(970, 0.745, f"grey band: baseline F1 ± {noise:.2f}\n(noise of one model)", fontsize=7.5, color=INK_2,
           ha="right", va="bottom")
    for ax, col, title, fmt in [(axes[1], "train_minutes", "Training time (min)", "{:.1f}"),
                                (axes[2], "inference ms (batch 16)", "Inference (ms / image)", "{:.1f}")]:
        vals = s[col].copy()
        if col == "train_minutes":
            vals.loc[640] = BASELINE_IDLE_MIN
        ax.bar(range(len(SIZES)), vals, 0.6, color=[INK_2 if x == 640 else SERIES[0] for x in SIZES])
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02 + 0.1, fmt.format(v), ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(SIZES)), SIZES); ax.set_xlabel("imgsz"); ax.set_title(title)
        ax.set_ylim(0, vals.max() * 1.18); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(BUILD / "imgsz_overview.png"); plt.close(fig)


def fig_breakdown(sizes_tbl, cls_tbl):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
    w = 0.2
    for ax, tbl, groups, title in [(a, sizes_tbl, list(sizes_tbl[640].index), "Recall by hand size (conf 0.25)"),
                                   (b, cls_tbl, CLASSES, "Recall by class (conf 0.25)")]:
        x = np.arange(len(groups))
        for i, s in enumerate(SIZES):
            ax.bar(x + (i - 1.5) * w, tbl[s].loc[groups, "recall"], w - 0.02, color=SERIES[i], label=LABEL[s])
        n = tbl[640].loc[groups, "hands"].astype(int)
        ax.set_xticks(x, [f"{g.replace(' (', chr(10) + '(')}\nn={k}" if "(" in g else f"{g}\nn={k}"
                          for g, k in zip(groups, n)])
        ax.set_ylim(0, 1.05); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, [f"imgsz {l}" for l in labels], ncol=4, fontsize=8.5, loc="upper center",
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(BUILD / "imgsz_breakdown.png"); plt.close(fig)


# --------------------------------------------------------------------------- report

def signed(v, digits=3):
    return f"{v:+.{digits}f}".replace("+0.000", "±0.000").replace("-", "−")


def build_html(d):
    s, ea, m = d["summary"], d["ea"], d["metrics"]
    hp = m[640]["hyperparameters"]
    ci = bootstrap(d["hands"][640], d["preds"][640])
    noise = (ci["F1"][1] - ci["F1"][0]) / 2
    fb = fixed_broken(d["hands"])
    small = {k: v.iloc[0] for k, v in d["size"].items()}
    large = {k: v.iloc[2] for k, v in d["size"].items()}
    n_small = int(small[640]["hands"])
    small_found = {k: int(round(v["recall"] * n_small)) for k, v in small.items()}
    n_hands, n_imgs = ea[640]["hands"], ea[640]["images"]
    span = (s["team mAP50"] - s.loc[640, "team mAP50"]).abs().max()
    best = s["team mAP50"].idxmax()
    tr_small, tr_n, tr_med = train_hand_sizes()
    our_med = d["hands"][640]["area_frac"].median()
    px = np.sqrt(d["hands"][640]["area_frac"]) * 640
    small_px = np.sqrt(d["hands"][640].loc[d["hands"][640]["area_frac"] < 0.01, "area_frac"]) * 640
    mins = s["train_minutes"].copy(); mins.loc[640] = BASELINE_IDLE_MIN

    runs = pd.DataFrame({
        "imgsz": [str(x) for x in SIZES],
        "epochs run": [m[x]["epochs_run"] for x in SIZES],
        "best epoch": [m[x]["best_epoch"] for x in SIZES],
        "train min": [f"{mins[x]:.1f}" + ("*" if x == 640 else "") for x in SIZES]},
        index=["baseline" if x == 640 else RUN[x] for x in SIZES])

    res = pd.DataFrame(index=[LABEL[x] for x in SIZES])
    for col, name in [("val mAP50", "val mAP50"), ("team mAP50", "team mAP50"), ("team mAP50-95", "team mAP50-95"),
                      ("team fist mAP50", "fist mAP50"), ("team P", "precision"), ("team R", "recall"),
                      ("F1 @ conf 0.25", "F1 @ 0.25")]:
        res[name] = [f"{s.loc[x, col]:.3f}" for x in SIZES]
    res["Δ team mAP50"] = ["—" if x == 640 else signed(s.loc[x, "team mAP50"] - s.loc[640, "team mAP50"]) for x in SIZES]
    res["Δ F1 @ 0.25"] = ["—" if x == 640 else signed(s.loc[x, "F1 @ conf 0.25"] - s.loc[640, "F1 @ conf 0.25"]) for x in SIZES]
    res["small hands"] = [f"{small_found[x]} / {n_small}" for x in SIZES]
    res["fixed / broken"] = ["—" if x == 640 else f"{fb[x][0]} / {fb[x][1]}" for x in SIZES]

    ctrl = pd.DataFrame({"baseline weights run at this size": [f"{CONTROL[x][0]:.3f}" for x in SIZES],
                         "model trained at this size": [f"{s.loc[x, 'team mAP50']:.3f}" for x in SIZES]},
                        index=[LABEL[x] for x in SIZES])

    loose_missed = int((d["hands"][640]["file"].isin(LOOSE_BOXES) & ~d["hands"][640]["correct"]).sum())
    p = d["preds"][640]
    label_bg = int((p["file"].isin(LOOSE_BOXES + list(UNLABELLED)) & (p["true_class"] == "background")).sum())

    css = (Path(__file__).parent / "report.css").read_text()
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Image Size Tuning</title>
<style>{css}
h1 {{ font-size: 20pt; }}
.small-table table {{ font-size: 8.3pt; }}
.small-table th, .small-table td {{ padding: 3px 4px; }}
</style></head><body>

<header>
  <p class="kicker">IAT 360 Computer Vision Project · Group 6 · Hyperparameter tuning</p>
  <h1>Image size (<code style="font-size:18pt">imgsz</code>) compared with the baseline</h1>
  <p class="byline">Tuned by Ben Nichiporik · YOLOv8s gesture detector · imgsz 480 / 800 / 960 vs baseline 640 · {date.today():%B %Y}</p>
</header>

<section class="tiles">
  <div class="tile accent"><span class="num">{s.loc[best, 'team mAP50']:.3f}</span><span class="lbl">best team mAP50 (imgsz {best})<br>vs {s.loc[640, 'team mAP50']:.3f} baseline: {signed(s.loc[best, 'team mAP50'] - s.loc[640, 'team mAP50'])}</span></div>
  <div class="tile"><span class="num">±{span:.3f}</span><span class="lbl">largest change in team mAP50<br>(single-model noise ±{noise:.2f})</span></div>
  <div class="tile"><span class="num">{" / ".join(str(small_found[x]) for x in SIZES)}</span><span class="lbl">small hands found of {n_small}<br>at imgsz {" / ".join(str(x) for x in SIZES)}</span></div>
  <div class="tile"><span class="num">{s.loc[640, 'F1 @ conf 0.25']:.3f}</span><span class="lbl">baseline F1 at conf 0.25,<br>best of all four runs</span></div>
</section>

<p class="lead"><b>Result: changing the image size did not improve the baseline.</b> Team-only mAP50 moved by at most
{span:.3f}, well inside the ±{noise:.2f} uncertainty of a single model on our {n_imgs} photos. imgsz {best} is best on paper,
but the baseline (640) keeps the best F1 at the deployment threshold and the best speed/accuracy balance. A larger input did not
help small hands, because the baseline's problem is unfamiliar <i>scenes</i>, not too few pixels.</p>

<h2>1. Setup and hypothesis</h2>
<div class="cols">
<div>
<p><code>imgsz</code> is the size (longest side, in pixels) that every image is resized to for training and inference.
The baseline evaluation showed that the model's main weakness on our own photos is <b>small hands</b>: only
{small[640]['recall']:.0%} of the {n_small} hands covering under 1% of the image were found. At imgsz 640 those hands are
only about {small_px.min():.0f}-{small_px.max():.0f} px across (median hand {px.median():.0f} px).</p>
<p><b>Hypothesis.</b> A larger <code>imgsz</code> (800, 960) keeps more pixels per hand and should raise recall on small
hands, especially fists, at the cost of slower training and inference. A smaller one (480) shows how much accuracy we lose
for speed, which matters for a real-time photo booth.</p>
</div>
<div>
{table(runs, index_name="run")}
<p class="caption">Only <code>imgsz</code> changes. Everything else is the baseline: YOLOv8s (COCO), SGD lr0 {hp['lr0']},
{hp['epochs']} epochs, patience {hp['patience']}, batch {hp['batch']}, seed {hp['seed']}, deterministic.
*{BASELINE_IDLE_MIN} min on an idle GPU ({s.loc[640, 'train_minutes']:.1f} min in the final re-run on a shared GPU, identical model).</p>
<p><b>Fair comparison.</b> Same split, seed and evaluation code (<code>gesture_eval.py</code>). Each model is scored at the
size it was trained at, and the headline is our own photos only: {n_imgs} images, {n_hands} hands, at confidence ≥ 0.25 and IoU ≥ 0.5.</p>
</div>
</div>

<h2>2. Results compared with the baseline</h2>
<div class="small-table">{table(res, index_name="imgsz")}</div>
<p class="caption">mAP, precision and recall from Ultralytics on our photos (val = HaGRID validation). F1, small hands (&lt;1% of the
image) and fixed / broken (hands the baseline got wrong that this run gets right / the reverse) at conf 0.25, IoU 0.5.</p>
<img src="imgsz_overview.png" class="full">
<p class="caption">Grey band: baseline F1 ± {noise:.2f}, the 95% bootstrap half-width of one model's F1 on our photos.
Inference is per image at batch 16 on an RTX 5060 Ti and varies by 1-2 ms between runs.</p>
<ul>
<li><b>No run beats the baseline by more than noise.</b> Validation is saturated
({s['val mAP50'].min():.3f}-{s['val mAP50'].max():.3f}) for every size, so it can't separate the runs either.</li>
<li><b>imgsz 800 is best on paper, but only just</b>: highest team mAP50 ({s.loc[800, 'team mAP50']:.3f},
{signed(s.loc[800, 'team mAP50'] - s.loc[640, 'team mAP50'])}), mAP50-95 ({s.loc[800, 'team mAP50-95']:.3f}) and fist mAP50
({s.loc[800, 'team fist mAP50']:.3f}, {signed(s.loc[800, 'team fist mAP50'] - s.loc[640, 'team fist mAP50'])}). Its precision is the
same as the baseline's and its recall is lower ({s.loc[800, 'team R']:.3f} vs {s.loc[640, 'team R']:.3f}), so F1 at the deployment
threshold is slightly <i>worse</i> ({s.loc[800, 'F1 @ conf 0.25']:.3f} vs {s.loc[640, 'F1 @ conf 0.25']:.3f}). It fixed
{fb[800][0]} hands and broke {fb[800][1]}.</li>
<li><b>imgsz 960 is the worst for its cost.</b> It early-stopped at epoch {m[960]['epochs_run']} (best epoch {m[960]['best_epoch']}),
took {mins[960] / mins[640]:.0f}× the baseline's training time, fixed {fb[960][0]} hands but broke {fb[960][1]}, and large-hand recall
fell from {large[640]['recall']:.2f} to {large[960]['recall']:.2f}.</li>
<li><b>imgsz 480</b> is the fastest and only slightly worse ({signed(s.loc[480, 'team mAP50'] - s.loc[640, 'team mAP50'])} mAP50),
but it finds the fewest small hands ({small_found[480]} / {n_small}).</li>
</ul>
<img src="imgsz_breakdown.png" class="full">
<p><b>Small hands barely moved.</b> {", ".join(f"{small_found[x]}/{n_small} at {x}" for x in SIZES)}. One extra hand at 800
is not a real improvement on a sample this small, so a larger <code>imgsz</code> did not fix the baseline's main weakness.
Per class, fist stays the weakest class at every size (recall {", ".join(f"{d['cls'][x].loc['fist', 'recall']:.2f}" for x in SIZES)}).</p>

<h2>3. Why a larger image size didn't help</h2>
<div class="cols">
<div>
<p><b>Resolution was not the real problem.</b> The training set already has plenty of small hands: {tr_small} of the
{tr_n:,} HaGRID training hands ({tr_small / tr_n:.0%}) cover under 1% of the image, and the median training hand
({tr_med * 100:.1f}% of the image) is <i>smaller</i> than ours ({our_med * 100:.1f}%). The gap is <b>scene context</b>. HaGRID shows
one person standing and facing a phone camera, mostly in a plain indoor room. Our missed hands are in crowds, outdoors, at odd
angles and against busy backgrounds. A larger input gives the model more pixels, but it doesn't teach it what those scenes look like.</p>
<p><b>imgsz 960 may be under-trained.</b> Its best checkpoint is from epoch {m[960]['best_epoch']}, so it effectively trained for
about half as long as the others. That probably explains part of its drop on large hands.</p>
</div>
<div>
{table(ctrl, index_name="imgsz")}
<p class="caption">Control: team mAP50 of the baseline weights (trained at 640) run at each size, compared with a model trained at that size.</p>
<p><b>The control fits this explanation.</b> Running the 640-trained baseline at a larger size <i>lowers</i> team mAP50
({CONTROL[640][0]:.3f} → {CONTROL[800][0]:.3f} at 800 → {CONTROL[960][0]:.3f} at 960). Our hands are already larger on average than
HaGRID's, and upscaling pushes them further from the sizes seen in training. Training at the larger size only wins back what the
mismatch cost.</p>
</div>
</div>
<p><b>Label errors.</b> A few of our test labels contain human error (loose boxes on Ben's <code>fist_380/381/384</code>,
unlabelled raised hands in <code>fist_376</code>, <code>fist_382</code> and <code>palm_388</code>, an unclear gesture in
<code>palm_376</code>, and the duplicate pair <code>dislike_388</code> = <code>dislike_393</code>). For the baseline they cause
{loose_missed} of the {ea[640]['missed_hands']} missed hands and {label_bg} of the {ea[640]['background_false_positives']}
background predictions. Every run is scored against the same labels, so the comparison between sizes is still fair, but absolute
scores (fist especially) are slightly pessimistic.</p>

<h2>4. Choice and next steps</h2>
<ol>
<li><b>For the group comparison, my best value is imgsz {best}</b> (best team mAP50, mAP50-95 and fist mAP50). The gain over the
baseline is {signed(s.loc[best, 'team mAP50'] - s.loc[640, 'team mAP50'])} mAP50, effectively a tie, and it costs about
{mins[best] / mins[640]:.0f}× the training time and {s.loc[best, 'inference ms (batch 16)'] / s.loc[640, 'inference ms (batch 16)']:.1f}×
the inference time.</li>
<li><b>For the photo booth, keep imgsz 640.</b> It has the best F1 at the deployment threshold ({s.loc[640, 'F1 @ conf 0.25']:.3f})
and the best balance of speed and accuracy.</li>
<li><b>Fix the data, not the resolution.</b> More training photos with varied scenes (crowds, outdoors, several people) and
augmentation that changes the scene (<code>mosaic</code>, <code>mixup</code>, <code>scale</code>, colour changes) target the real
weakness. If another member's hyperparameter improves scale robustness (e.g. <code>scale</code> or <code>mosaic</code>),
combining it with imgsz {best} is the combination worth testing.</li>
</ol>
</body></html>"""


def main():
    BUILD.mkdir(exist_ok=True)
    d = load()
    ci = bootstrap(d["hands"][640], d["preds"][640])
    fig_overview(d["summary"], (ci["F1"][1] - ci["F1"][0]) / 2)
    fig_breakdown(d["size"], d["cls"])
    print("fixed / broken vs baseline:", fixed_broken(d["hands"]))
    html = BUILD / "Imgsz_Tuning_Report.html"
    html.write_text(build_html(d), encoding="utf-8")

    browser = next((b for b in BROWSERS if b.exists()), None)
    if browser is None:
        sys.exit(f"No Edge/Chrome found; open {html} in a browser and print it to PDF.")
    subprocess.run([str(browser), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", html.as_uri()], check=True, capture_output=True, timeout=120)
    print("wrote", PDF)


if __name__ == "__main__":
    main()
