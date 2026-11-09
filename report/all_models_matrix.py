"""All-models matrix for the final report: every tuned model and the combined model on our photos.

Run from the repo root: python report/all_models_matrix.py  ->  results/all_models_matrix.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Values from the group comparison in Gesture_YOLO_Tuning_Group.ipynb
# (our photos; precision, recall, F1 and accuracy at conf 0.25, IoU 0.5).
rows = [
    ("baseline",                          0.845, 0.418, 0.838, 0.805, 0.821, 0.697),
    ("imgsz 480 (Ben)",                   0.830, 0.426, 0.836, 0.792, 0.813, 0.685),
    ("imgsz 800 (Ben)",                   0.848, 0.429, 0.810, 0.805, 0.808, 0.678),
    ("imgsz 960 (Ben)",                   0.828, 0.426, 0.812, 0.760, 0.785, 0.646),
    ("lr0 0.0025 (Thu)",                  0.856, 0.449, 0.730, 0.825, 0.774, 0.632),
    ("lr0 0.005 (Thu)",                   0.858, 0.425, 0.812, 0.844, 0.828, 0.707),
    ("lr0 0.02 (Thu)",                    0.821, 0.400, 0.801, 0.760, 0.780, 0.639),
    ("epochs 20 (Joseph)",                0.844, 0.415, 0.772, 0.838, 0.804, 0.672),
    ("epochs 35 (Joseph)",                0.846, 0.432, 0.771, 0.766, 0.769, 0.624),
    ("epochs 80 (Joseph)",                0.843, 0.414, 0.852, 0.825, 0.838, 0.722),
    ("degrees 10, final (Daffodil)",      0.863, 0.436, 0.903, 0.851, 0.876, 0.780),
    ("degrees 20 (Daffodil)",             0.856, 0.433, 0.843, 0.838, 0.840, 0.725),
    ("degrees 30 (Daffodil)",             0.890, 0.475, 0.852, 0.857, 0.854, 0.746),
    ("degrees 10 + lr0 0.005 (combined)", 0.870, 0.452, 0.854, 0.838, 0.846, 0.733),
]
cols = ["mAP50", "mAP50-95", "precision", "recall", "F1", "accuracy"]

vals = np.array([r[1:] for r in rows])
# Shade each column by its own range, so the best model in every metric is darkest.
lo, hi = vals.min(axis=0), vals.max(axis=0)
shade = (vals - lo) / np.where(hi > lo, hi - lo, 1)

fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(shade, cmap="Blues", aspect="auto", vmin=0, vmax=1.15)
for i in range(len(rows)):
    for j in range(len(cols)):
        ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=12,
                color="white" if shade[i, j] > 0.55 else "black",
                fontweight="bold" if vals[i, j] == hi[j] else "normal")
ax.set_xticks(range(len(cols)), cols, fontsize=12)
ax.xaxis.tick_top()
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=12)
# Separate the hyperparameter groups.
for y in (0.5, 3.5, 6.5, 9.5, 12.5):
    ax.axhline(y, color="white", lw=4)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("All models on team photos (conf ≥ 0.25, IoU ≥ 0.5)\nDarker = better within each column; bold = best",
             fontsize=13, pad=12)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent.parent / "results" / "all_models_matrix.png", dpi=160)
