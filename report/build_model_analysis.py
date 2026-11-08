"""Build report/Model_Analysis_Report.pdf, the Model analysis part of the Group 6 report:

    1  detailed evaluation of the baseline model
    2  Ben's fine-tuned models (imgsz 480 / 800 / 960) compared with the baseline
    3  challenges in these steps and how they were addressed

Every number is read from the saved results (results/baseline/, results/imgsz_*/ and
results/tuning_imgsz_summary.csv), so after re-running the notebooks just run this again:

    python report/build_model_analysis.py

Figures and the HTML source go to report/build/.
"""
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_baseline_report as baseline_report
from build_baseline_report import (AMBIGUOUS, BUILD, CLASSES, GRID, INK, INK_2, LOOSE_BOXES, ROOT, SERIES,
                                   UNLABELLED, bootstrap, duplicate_pairs, print_pdf, table)

PDF = Path(__file__).resolve().parent / "Model_Analysis_Report.pdf"
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
    return f"{v:+.{digits}f}".replace("-", "−")


def fig(src, n, caption, cls="full"):
    return f'<figure><img src="{src}" class="{cls}"><p class="caption"><b>Figure {n}.</b> {caption}</p></figure>'


def tab(df, n, caption, table_class="", **kw):
    html = table(df, **kw)
    if table_class:
        html = html.replace("<table>", f'<table class="{table_class}">', 1)
    return f'<div class="tbl"><p class="caption"><b>Table {n}.</b> {caption}</p>{html}</div>'


def build_html(bd, d):
    # ---- baseline facts
    m, ea, pc, h, man = bd["metrics"], bd["ea"], bd["per_class"], bd["history"], bd["manifest"]
    hp = m["hyperparameters"]
    best_epoch = int(h.loc[h["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    best_row = h.loc[h["epoch"] == best_epoch].iloc[0]
    ov = pd.DataFrame(m["overall"]).T
    val, team = ov.iloc[0], ov.iloc[2]
    cls, size, person = bd["cls"], bd["size"], bd["person"]
    deploy, bestc = ea["conf"], ea["best_F1_conf"]
    ci = bootstrap(bd["hands"], bd["preds"])
    noise = (ci["F1"][1] - ci["F1"][0]) / 2
    dups = duplicate_pairs(man)
    fp = bd["preds"][bd["preds"]["true_class"] == "background"]["pred_class"].value_counts()
    sp = ea["predict_speed_ms_per_image"]
    total_ms = sum(sp.values())
    n_team = int((man["source"] == "team").sum())
    small_b = size.iloc[0]
    hands_b, preds_b = bd["hands"], bd["preds"]
    area = hands_b["area_frac"]
    tr_small, tr_n, tr_med = train_hand_sizes()
    small_by_person = hands_b[area < 0.01].groupby("person").size()
    small_owner = small_by_person.idxmax()
    iou_low = person["mean_IoU_correct"].idxmin()
    loose_missed = int((hands_b["file"].isin(LOOSE_BOXES) & ~hands_b["correct"]).sum())
    label_bg = int((preds_b["file"].isin(LOOSE_BOXES + list(UNLABELLED)) & (preds_b["true_class"] == "background")).sum())
    code = lambda f: f"<code>{Path(f).stem}</code>"

    counts = pd.crosstab(man["split"] + " · " + man["source"].map({"hagrid": "HaGRID", "team": "our photos"}),
                         man["class"])[CLASSES]
    counts["total"] = counts.sum(axis=1)
    counts = counts.reindex([i for i in ["train · HaGRID", "val · HaGRID", "test · HaGRID", "test · our photos"]
                             if i in counts.index])
    hp_tbl = pd.DataFrame({"value": [
        "YOLOv8s, COCO-pretrained, all layers fine-tuned", hp["imgsz"],
        f"{hp['epochs']} (early-stopping patience {hp['patience']})", hp["batch"],
        f"{hp['optimizer']}, lr0 {hp['lr0']}, momentum {hp['momentum']}, weight decay {hp['weight_decay']}",
        f"flip left-right {hp['fliplr']}, up-down {hp['flipud']}; Ultralytics defaults (mosaic, HSV, scale, translate)",
        f"{hp['seed']}, deterministic"]},
        index=["model", "image size (imgsz)", "epochs", "batch", "optimizer", "augmentation", "seed"])
    ov_tbl = ov.copy()
    ov_tbl.index = [i.replace("test, team only", "test, our photos only") for i in ov_tbl.index]
    team_cls = pc.loc["test (team only)"].loc[CLASSES, ["precision", "recall", "mAP50", "mAP50-95"]].copy()
    team_cls["hands"] = cls["hands"].astype(int)
    team_cls["missed"] = cls["missed"].astype(int)
    team_cls["wrong class"] = cls["wrong_class"].astype(int)
    pers_tbl = person[["hands", "predictions", "correct", "precision", "recall", "mean_IoU_correct"]].copy()
    for c in ["hands", "predictions", "correct"]:
        pers_tbl[c] = pers_tbl[c].astype(int)
    pers_tbl = pers_tbl.rename(columns={"mean_IoU_correct": "mean IoU"})
    size_tbl = size.copy()
    size_tbl["hands"] = size_tbl["hands"].astype(int)
    size_tbl["missed"] = size_tbl["missed"].astype(int)

    # ---- imgsz facts
    s, ms = d["summary"], d["metrics"]
    fb = fixed_broken(d["hands"])
    small = {k: v.iloc[0] for k, v in d["size"].items()}
    large = {k: v.iloc[2] for k, v in d["size"].items()}
    n_small = int(small[640]["hands"])
    found = {k: int(round(v["recall"] * n_small)) for k, v in small.items()}
    span = (s["team mAP50"] - s.loc[640, "team mAP50"]).abs().max()
    best = s["team mAP50"].idxmax()
    px = np.sqrt(area) * 640
    small_px = np.sqrt(area[area < 0.01]) * 640
    mins = s["train_minutes"].copy()
    mins.loc[640] = BASELINE_IDLE_MIN
    inf = s["inference ms (batch 16)"]

    runs = pd.DataFrame({
        "imgsz": [str(x) for x in SIZES],
        "epochs run": [ms[x]["epochs_run"] for x in SIZES],
        "best epoch": [ms[x]["best_epoch"] for x in SIZES],
        "train min": [f"{mins[x]:.1f}" + ("*" if x == 640 else "") for x in SIZES],
        "ms / image": [f"{inf[x]:.1f}" for x in SIZES]},
        index=["baseline" if x == 640 else RUN[x] for x in SIZES])
    res = pd.DataFrame(index=[LABEL[x] for x in SIZES])
    for col, name in [("val mAP50", "val mAP50"), ("team mAP50", "mAP50"), ("team mAP50-95", "mAP50-95"),
                      ("team fist mAP50", "fist mAP50"), ("team P", "precision"), ("team R", "recall"),
                      ("F1 @ conf 0.25", "F1 @ 0.25")]:
        res[name] = [f"{s.loc[x, col]:.3f}" for x in SIZES]
    res["Δ mAP50"] = ["—" if x == 640 else signed(s.loc[x, "team mAP50"] - s.loc[640, "team mAP50"]) for x in SIZES]
    res["Δ F1"] = ["—" if x == 640 else signed(s.loc[x, "F1 @ conf 0.25"] - s.loc[640, "F1 @ conf 0.25"]) for x in SIZES]
    res["small hands"] = [f"{found[x]} / {n_small}" for x in SIZES]
    res["fixed / broken"] = ["—" if x == 640 else f"{fb[x][0]} / {fb[x][1]}" for x in SIZES]
    ctrl = pd.DataFrame({"baseline weights run at this size": [f"{CONTROL[x][0]:.3f}" for x in SIZES],
                         "model trained at this size": [f"{s.loc[x, 'team mAP50']:.3f}" for x in SIZES]},
                        index=[LABEL[x] for x in SIZES])

    dup_txt = "; ".join(f"<code>{Path(a).stem}</code> ({pa}) and <code>{Path(b).stem}</code> ({pb})"
                        for a, pa, b, pb in dups)
    css = (Path(__file__).parent / "report.css").read_text()
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Model Analysis</title>
<style>{css}
h1 {{ font-size: 20pt; }}
h2 {{ font-size: 14pt; margin-top: 18px; }}
h3 {{ font-size: 11pt; margin: 12px 0 4px; break-after: avoid; }}
figure {{ margin: 6px 0 10px; break-inside: avoid; }}
figure .caption {{ margin-top: 2px; }}
.tbl {{ break-inside: avoid; margin: 6px 0 10px; }}
.tbl .caption {{ margin: 0 0 2px; }}
.small-table table {{ font-size: 8.3pt; }}
.small-table th, .small-table td {{ padding: 3px 4px; }}
table.wide-left td {{ text-align: left; }}
</style></head><body>

<header>
  <p class="kicker">IAT 360 Computer Vision Project · Group 6 · Gesture-controlled photo booth</p>
  <h1>Model analysis: baseline evaluation and image-size tuning</h1>
  <p class="byline">Baseline evaluation and <code>imgsz</code> tuning by Ben Nichiporik · {date.today():%B %Y}</p>
</header>

<p>This part of the report evaluates the baseline gesture detector in detail (Section 1), compares it with three models
fine-tuned at different image sizes (Section 2), and describes the problems we ran into in these steps and how we
handled them (Section 3). The detector finds hands in a photo and classifies each one as one of four gestures:
<b>palm</b>, <b>fist</b>, <b>like</b> (thumbs-up) or <b>dislike</b> (thumbs-down).</p>

<h2>1. Baseline model evaluation</h2>

<h3>1.1 Model and training</h3>
<p>The baseline is <b>YOLOv8s</b> (11.1 M parameters) pretrained on COCO and fine-tuned on our data with all layers
trainable (transfer learning). It was trained for {m.get('epochs_run', len(h))} epochs, and the checkpoint with the best
validation fitness (epoch {best_epoch}) is the baseline model (<code>models/baseline_best.pt</code>). Table 1 lists the
hyperparameters. They are defined once in <code>gesture_eval.py</code> (<code>BASELINE</code>), so each member's tuning
runs change exactly one of them. Table 2 shows how the data was split.</p>
<div class="cols">
{tab(hp_tbl, 1, "Baseline hyperparameters.", table_class="wide-left", index_name="setting")}
{tab(counts, 2, "Images per split and class. HaGRID images 001-368 of each class are used for training, validation and part of the test set. Our own photos (369-400, 8 per member per class) are used only for testing.", fmt="{:.0f}", index_name="split · source")}
</div>

<h3>1.2 Evaluation protocol</h3>
<p>Every model in the project is scored by the same code (<code>gesture_eval.py</code>) on three sets: HaGRID
validation (used to choose the checkpoint), the full test split, and <b>our own photos only</b> ({n_team} images,
{ea['hands']} hands). The last one is the honest estimate of how the photo booth will perform: those photos were never
seen in training and look like real use, with different people, places and lighting. We report the standard Ultralytics
metrics (precision, recall, mAP50 and mAP50-95). We also score our photos at a fixed <b>deployment threshold</b>:
a prediction counts if its confidence is at least {deploy}, and it is correct if its box overlaps a labelled hand with
IoU ≥ 0.5 and the class matches. Because the test set is small, we resample the photos (bootstrap, 2,000 resamples)
to put 95% confidence intervals on precision, recall and F1.</p>

<h3>1.3 Training behaviour</h3>
{fig("fig_training.png", 1, "Baseline training curves. Left: box and class loss on the training and validation sets. Right: validation mAP50 and mAP50-95. The dotted line marks the selected checkpoint.")}
<p>Transfer learning works quickly. Validation mAP50 is already {h['metrics/mAP50(B)'].iloc[0]:.2f} after one epoch,
because the COCO-pretrained backbone already provides general shape and edge features. The best epoch ({best_epoch})
reaches mAP50 {best_row['metrics/mAP50(B)']:.3f} and mAP50-95 {best_row['metrics/mAP50-95(B)']:.3f}. The validation losses
keep falling alongside the training losses through epoch {len(h)}, and early stopping never triggered, so the model is
not overfitting <i>to HaGRID</i>. The overfitting that matters is to HaGRID's <i>style</i> of photo, which only the test
on our own photos reveals.</p>

<h3>1.4 Overall results</h3>
{tab(ov_tbl, 3, "Baseline results on each evaluation set (Ultralytics metrics).", index_name="evaluation set")}
<p>The validation scores are inflated. HaGRID validation images come from the same collection as the training images:
one person standing and facing a phone camera, mostly indoors, with one hand per photo. On our own photos, mAP50 drops by
{val['mAP50'] - team['mAP50']:.2f} (to {team['mAP50']:.3f}) and mAP50-95 by {val['mAP50-95'] - team['mAP50-95']:.2f}. The
larger drop in mAP50-95 means the model usually <i>finds</i> the hand, but its box doesn't line up tightly with ours:
correct detections have a mean IoU of only {ea['mean_IoU_correct']:.2f} with our labels. Part of this comes from annotation
style rather than model error, because each member drew boxes in MakeSense slightly differently (some include the wrist).
At conf {deploy} on our photos, the 95% bootstrap intervals are precision {ci['precision'][0]:.2f}-{ci['precision'][1]:.2f},
recall {ci['recall'][0]:.2f}-{ci['recall'][1]:.2f} and F1 {ci['F1'][0]:.2f}-{ci['F1'][1]:.2f}. A single model's F1 is
therefore uncertain by about <b>±{noise:.2f}</b>, which is the bar a tuned model has to clear (Section 2).</p>

<h3>1.5 Per-class performance</h3>
{fig("fig_per_class.png", 2, "mAP50 and mAP50-95 per class on validation, the full test split and our photos only (labelled values).")}
{tab(team_cls, 4, f"Per-class results on our photos. Precision, recall and mAP from Ultralytics; missed and wrong-class counts at conf {deploy}, IoU 0.5.", index_name="class")}
<p><b>Palm and like transfer well</b> because their shapes are distinctive (spread fingers, upward thumb).
<b>Fist is the weakest class</b>: {int(cls.loc['fist', 'missed'])} of {int(cls.loc['fist', 'hands'])} fists are missed
entirely, {loose_missed} of them only because their labels were drawn too loosely (Section 1.9). A fist is a compact,
low-texture blob that looks like many other things at small scale, and several of our fists are raised in crowd scenes
with busy backgrounds, which HaGRID never shows. <b>Dislike is confused with fist</b> in both directions: a thumbs-down
seen from the side looks like a fist with a short thumb, and these are the highest-confidence errors.</p>

<h3>1.6 Errors at the deployment threshold</h3>
<div class="cols">
{fig("../../results/baseline/confusion_conf025.png", 3, f"Confusion matrix on our photos at conf {deploy}, IoU 0.5 (counts).", "half")}
{fig("fig_sweep.png", 4, "Precision, recall and F1 on our photos as the confidence threshold changes.", "half")}
</div>
<p>At conf {deploy} the model makes {ea['predictions']} predictions for {ea['hands']} labelled hands in our {ea['images']}
photos: <b>{ea['correct']} correct</b>, <b>{ea['missed_hands']} hands missed</b>, <b>{ea['wrong_class']} found with the
wrong class</b> and <b>{ea['background_false_positives']} predictions on background</b> (no labelled hand there; by
predicted class: {", ".join(f"{k} {v}" for k, v in fp.items())}). Missing a hand is the most common error. F1 is highest at
conf {bestc['conf']} (precision {bestc['precision']:.2f}, recall {bestc['recall']:.2f}, F1 {bestc['F1']:.2f}), compared with
precision {ea['precision']:.2f} and recall {ea['recall']:.2f} at {deploy}. For a photo booth, a false trigger (taking a
picture nobody asked for) is worse than a missed gesture (the user just holds it a moment longer). We therefore recommend
a threshold of about {bestc['conf']}, plus requiring the gesture over several consecutive frames.</p>

<h3>1.7 Where the model fails</h3>
{fig("fig_breakdown.png", 5, f"Left: recall on our photos by hand size (share of the image). Right: precision, recall and mean box IoU for each member's photos (conf {deploy}).")}
<div class="cols">
{tab(size_tbl, 5, "Recall by hand size on our photos.", fmt="{:.2f}", index_name="hand size")}
{tab(pers_tbl, 6, "Results by the member who took the photos.", fmt="{:.2f}", index_name="member")}
</div>
<p><b>Hand size is the clearest driver.</b> The median labelled hand covers {area.median() * 100:.1f}% of the image, but
{int(small_b['hands'])} hands cover under 1%, and only {small_b['recall']:.0%} of those are found. At imgsz {hp['imgsz']}
those hands are only about {small_px.min():.0f}-{small_px.max():.0f} px across after resizing, which leaves little detail.
Size alone is not the cause, though: {tr_small} of the {tr_n:,} training hands ({tr_small / tr_n:.0%}) are also under 1% of
the image, and the median training hand ({tr_med * 100:.1f}%) is smaller than ours. The difference is <b>context</b>.
HaGRID's small hands belong to one person posing in a plain room, while ours are in crowds, outdoors and at odd angles.
<b>Per member</b>, the differences follow photo choice and labelling style. {small_owner}'s photos hold
{small_by_person[small_owner]} of the {int(small_b['hands'])} small hands, which lowers recall there. {iou_low}'s photos have
the lowest mean box IoU ({person.loc[iou_low, 'mean_IoU_correct']:.2f}) even though their precision is
{person.loc[iou_low, 'precision']:.2f}: the model finds those hands, but the boxes are drawn differently.</p>
{fig("../../results/baseline/failure_cases.jpg", 6, f"Failure cases. Green = our label, red = prediction (conf ≥ {deploy}). Top row: the smallest missed hands. Bottom row: wrong-class predictions (fist/dislike and palm/like confusions).")}

<h3>1.8 Speed</h3>
<p>Per image at imgsz {hp['imgsz']} (batch 1, NVIDIA RTX 5060 Ti): preprocess {sp['preprocess']:.1f} ms, inference
{sp['inference']:.1f} ms and postprocess {sp['postprocess']:.1f} ms, <b>{total_ms:.1f} ms in total (about
{1000 / total_ms:.0f} frames per second)</b>. The model file is {(ROOT / 'models' / 'baseline_best.pt').stat().st_size / 1e6:.0f} MB
(28.4 GFLOPs). A webcam photo booth needs about 15-30 frames per second, so a GPU has plenty of headroom. CPU-only
speed was not measured.</p>

<h3>1.9 Limits of this evaluation</h3>
<ul>
<li><b>Small test set.</b> {ea['hands']} hands in {n_team} photos; the F1 interval in Section 1.4 is {ci['F1'][1] - ci['F1'][0]:.2f} wide.</li>
<li><b>Duplicate photo.</b> {dup_txt} are the same picture, so those hands count twice and the test set has
{n_team - len(dups)} distinct photos.</li>
<li><b>Labelling errors (human error).</b> Some test labels are wrong and were left as they are: boxes drawn much larger
than the fist ({", ".join(code(f) for f in LOOSE_BOXES)}), gesturing hands that were never labelled
({"; ".join(f"{code(f)}, {v}" for f, v in UNLABELLED.items())}) and an unclear gesture
({"; ".join(f"{code(f)}, {v}" for f, v in AMBIGUOUS.items())}). A correct, tight box on a loose label overlaps it by
less than 0.5 IoU, so it counts as a miss plus a false positive. These errors cause {loose_missed} of the
{ea['missed_hands']} missed hands and {label_bg} of the {ea['background_false_positives']} background predictions, so the
scores on our photos are slightly pessimistic.</li>
<li><b>Optimistic validation.</b> Validation is HaGRID-only, so choosing the checkpoint by it does not reward robustness
to our kind of photo.</li>
<li><b>One training seed.</b> Run-to-run variation was not measured beyond the bootstrap over test photos.</li>
</ul>

<h2>2. Fine-tuned models: image size (<code style="font-size:12.5pt">imgsz</code>), Ben Nichiporik</h2>

<h3>2.1 Hypothesis</h3>
<p><code>imgsz</code> is the size (longest side, in pixels) that every image is resized to for training and inference.
Section 1.7 showed that the baseline's main weakness is small hands, which are only about
{small_px.min():.0f}-{small_px.max():.0f} px across at imgsz 640 (the median hand is {px.median():.0f} px). <b>A larger
imgsz (800, 960) should keep more pixels per hand and raise recall on small hands</b>, especially fists, at the cost of
slower training and inference. A smaller imgsz (480) shows how much accuracy we would lose for speed, which matters
for a real-time photo booth.</p>

<h3>2.2 Method</h3>
<div class="cols">
<div>
<p>I trained three models that are identical to the baseline except for <code>imgsz</code> (480, 800 and 960): same
data split, seed, epochs, early stopping and augmentation. Each model was scored with the same code as the baseline,
at the size it was trained at, and the headline is again our own photos ({n_team} images, {ea['hands']} hands).
Validation is saturated for every size (mAP50 {s['val mAP50'].min():.3f}-{s['val mAP50'].max():.3f}), so it can't
separate the runs. I also compared the runs hand by hand: which of the baseline's errors each run fixed, and which of
its correct detections it broke.</p>
</div>
{tab(runs, 7, f"Training runs. *{BASELINE_IDLE_MIN} min on an idle GPU ({s.loc[640, 'train_minutes']:.1f} min in a re-run on a shared GPU, with an identical model). Inference is per image at batch 16 and varies by 1-2 ms between runs.", index_name="run")}
</div>

<h3>2.3 Results</h3>
<div class="small-table">{tab(res, 8, "Fine-tuned models compared with the baseline on our photos. Δ = change from the baseline. Small hands (under 1% of the image) found and fixed / broken hands at conf 0.25, IoU 0.5.", index_name="imgsz")}</div>
{fig("imgsz_overview.png", 7, f"Left: mAP50 and F1 on our photos for each image size; the grey band is the baseline F1 ± {noise:.2f}, the uncertainty of one model. Middle and right: training time and inference time per image.")}
<p><b>No image size beats the baseline by more than noise.</b> Team mAP50 moves by at most {span:.3f}, far inside the
±{noise:.2f} uncertainty from Section 1.4.</p>
<ul>
<li><b>imgsz 800 is the best on paper, but only just.</b> It has the highest mAP50 ({s.loc[800, 'team mAP50']:.3f},
{signed(s.loc[800, 'team mAP50'] - s.loc[640, 'team mAP50'])}), mAP50-95 ({s.loc[800, 'team mAP50-95']:.3f}) and fist mAP50
({s.loc[800, 'team fist mAP50']:.3f}, {signed(s.loc[800, 'team fist mAP50'] - s.loc[640, 'team fist mAP50'])}). Its
precision equals the baseline's and its recall is lower ({s.loc[800, 'team R']:.3f} vs {s.loc[640, 'team R']:.3f}), so
F1 at the deployment threshold is slightly <i>worse</i> ({s.loc[800, 'F1 @ conf 0.25']:.3f} vs
{s.loc[640, 'F1 @ conf 0.25']:.3f}). It fixed {fb[800][0]} of the baseline's errors and broke {fb[800][1]} of its correct
detections.</li>
<li><b>imgsz 960 is the worst for its cost.</b> It early-stopped at epoch {ms[960]['epochs_run']} (best epoch
{ms[960]['best_epoch']}), took {mins[960] / mins[640]:.0f}× the baseline's training time, fixed {fb[960][0]} hands but broke
{fb[960][1]}, and its recall on large hands fell from {large[640]['recall']:.2f} to {large[960]['recall']:.2f}.</li>
<li><b>imgsz 480</b> is the fastest and only slightly worse ({signed(s.loc[480, 'team mAP50'] - s.loc[640, 'team mAP50'])} mAP50),
but it finds the fewest small hands ({found[480]} / {n_small}).</li>
</ul>
{fig("imgsz_breakdown.png", 8, "Recall on our photos at conf 0.25 by hand size (left) and by class (right) for each image size.")}
<p><b>Small hands barely moved</b>: {", ".join(f"{found[x]}/{n_small} at {x}" for x in SIZES)}. One extra hand at 800 is not a
real improvement on a sample this small, so a larger image size did not fix the baseline's main weakness. Fist stays the
weakest class at every size (recall {", ".join(f"{d['cls'][x].loc['fist', 'recall']:.2f}" for x in SIZES)}).</p>

<h3>2.4 Why a larger image size didn't help</h3>
<div class="cols">
<div>
<p><b>Resolution was not the real problem.</b> As Section 1.7 showed, the training set already has plenty of small hands
({tr_small / tr_n:.0%} are under 1% of the image), and its median hand is smaller than ours. The gap is <b>scene
context</b>: our missed hands are in crowds, outdoors, at odd angles and against busy backgrounds. A larger input gives
the model more pixels, but it doesn't teach it what those scenes look like.</p>
<p><b>imgsz 960 may be under-trained.</b> Its best checkpoint is from epoch {ms[960]['best_epoch']}, so it effectively
trained for about half as long as the others. That probably explains part of its drop on large hands.</p>
</div>
<div>
{tab(ctrl, 9, "Control: mAP50 on our photos of the baseline weights (trained at 640) run at each size, compared with a model trained at that size.", index_name="imgsz")}
<p><b>The control supports this.</b> Running the 640-trained baseline at a larger size <i>lowers</i> mAP50
({CONTROL[640][0]:.3f} → {CONTROL[800][0]:.3f} at 800 → {CONTROL[960][0]:.3f} at 960). Our hands are already larger on average
than HaGRID's, so upscaling pushes them further from the sizes seen in training. Training at the larger size only wins
back what that mismatch costs.</p>
</div>
</div>

<h3>2.5 Conclusion</h3>
<p>The hypothesis was not supported. <b>My best value is imgsz {best}</b> (best mAP50, mAP50-95 and fist mAP50), so it is
my entry in the group comparison. However, its gain over the baseline
({signed(s.loc[best, 'team mAP50'] - s.loc[640, 'team mAP50'])} mAP50) is effectively a tie, and it costs about
{mins[best] / mins[640]:.0f}× the training time and {inf[best] / inf[640]:.1f}× the inference time. For the photo booth
itself, <b>imgsz 640 remains the better choice</b>: it has the best F1 at the deployment threshold
({s.loc[640, 'F1 @ conf 0.25']:.3f}) and the best balance of speed and accuracy. The real weakness should be addressed
through the data: more training photos with varied scenes (crowds, outdoors, several people) and augmentation that
changes the scene (<code>mosaic</code>, <code>mixup</code>, <code>scale</code>). If another member's hyperparameter
improves robustness to scale or scene, combining it with imgsz {best} is worth testing.</p>

<h2>3. Challenges and how they were addressed</h2>
<ul>
<li><b>Inconsistent class IDs.</b> Members labelled in MakeSense with the gesture list in different orders, so Joseph's
label files used different class numbers from ours. <code>prepare_dataset.py</code> checks every label file against the
gesture in its file name and remaps Joseph's class IDs, leaving files that are already in our order alone. A mix-up is
reported instead of silently corrupting the data.</li>
<li><b>Missing labels.</b> Amy's palm_369-376 photos had no label files at first. We held those images out until the
labels were added to the shared Drive, which brought our test set to {n_team} photos.</li>
<li><b>Duplicate photos.</b> A perceptual-hash check found pictures that two members had both contributed. Ben replaced
<code>dislike_380</code> with a new photo; the remaining pair ({dup_txt}) is documented in Section 1.9.</li>
<li><b>Labelling errors.</b> Reviewing every test photo against the models' predictions found loose boxes, unlabelled
hands and an unclear gesture. The group decided to keep the labels as they are and report them as human error. Every
model is scored against the same labels, so comparisons between models stay fair.</li>
<li><b>A validation set that can't tell models apart.</b> Every model scores about 0.99 mAP50 on HaGRID validation. We made
our own photos the headline metric and used bootstrap intervals to decide which differences are real.</li>
<li><b>Sharing data and compute.</b> We first tried Google Colab, but in the end everyone runs locally. The dataset lives
in a shared Google Drive folder, and <code>prepare_dataset.py</code> builds the same seeded split from it on every machine.
Training is deterministic: re-training the baseline reproduced identical metrics.</li>
<li><b>Noisy timings.</b> Training and inference times changed when the GPU was shared (the baseline took
{BASELINE_IDLE_MIN} or {s.loc[640, 'train_minutes']:.1f} min for an identical model), so speed differences of 1-2 ms, or of
a few minutes of training, are not meaningful.</li>
</ul>
</body></html>"""


def main():
    bd = baseline_report.load()
    baseline_report.make_figures(bd)
    d = load()
    ci = bootstrap(d["hands"][640], d["preds"][640])
    fig_overview(d["summary"], (ci["F1"][1] - ci["F1"][0]) / 2)
    fig_breakdown(d["size"], d["cls"])
    print("fixed / broken vs baseline:", fixed_broken(d["hands"]))
    html = BUILD / "Model_Analysis_Report.html"
    html.write_text(build_html(bd, d), encoding="utf-8")
    print_pdf(html, PDF)


if __name__ == "__main__":
    main()
