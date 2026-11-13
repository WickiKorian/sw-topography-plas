"""
Correspondence between single-trial (pycrostates) and average-level (RAGU) maps.

Replaces cells 30-31 of run_pipeline.ipynb, which used the old RAGU result
(Microstates_maps.mat, June 2026) and the old single-trial model.

  - RAGU maps:          RAGU_FILE (revised groups, -200 to 800 ms solution, 8 maps)
  - single-trial maps:  K=4 group model of the main analysis, numbered and
                        polarised as in the manuscript (same relabelling as
                        a3_stats.py: spatial correlation with the submitted maps)
  - signed spatial correlation (4 x 8); single-trial maps are polarity-
    independent, so a match is read from |r| (threshold |r| >= .90, Methods)

RAGU classes are numbered as in Figure 3 (FIGURE_ORDER, as in
b4_ragu_microstate_table.py).

Output (in Ragu_results):
  pyc_ragu_corr_signed_new.csv   correlation matrix (Figure 3 numbering)
  Figure5_heatmap.(svg|pdf|png)  heatmap for Figure 5 (outline: |r| >= .90);
                                 the maps of Figure 5 are taken from Figure 3
                                 (RAGU classes) and Figure 4 (MS 1-4)
Output (in microstates/<FILTER_TYPE>/2. group_models):
  correspondence_k4_k8.csv       step 5: for k = 4-8, highest |r| of each RAGU
                                 map with any single-trial template (choice of
                                 k, Methods; values reported in the Results)
"""

import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from config import *

matplotlib.rcParams["pdf.fonttype"] = 42     # editable text in Illustrator
matplotlib.rcParams["ps.fonttype"]  = 42
logger = setup_logger("ragu_correspondence", "e3_figure5_correspondence.log")

K                   = 4
MATCH_THRESHOLD     = 0.90
RAGU_RESULTS = RAGU_DIR / "Ragu_results"
# Figure 3 number -> map index in the RAGU file (1-based), as in b4_ragu_microstate_table.py
FIGURE_ORDER = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 3}
RAGU_FILE = RAGU_RESULTS / "Microstates_-200_800ms_ConditionXGroup_HCCI.mat"
OUT_STEM  = RAGU_RESULTS / "pyc_ragu_corr_new"

z = lambda a: (a - a.mean(-1, keepdims=True)) / a.std(-1, keepdims=True)

# ── 1. RAGU maps ──────────────────────────────────────────────────────────────
rd          = loadmat(str(RAGU_FILE), squeeze_me=True, struct_as_record=False)["rd"]
ragu_maps   = np.asarray(rd.MSMaps)                          # (8, 129)
ragu_labels = [str(ch.Name) for ch in rd.Channel]
logger.info(f"RAGU: {RAGU_FILE.name}, maps {ragu_maps.shape}, "
            f"window {rd.TimeOnset + (rd.StartFrame - 1) * rd.DeltaX} to "
            f"{rd.TimeOnset + (rd.EndFrame - 1) * rd.DeltaX} ms")

# ── 2. Single-trial maps, manuscript numbering and polarity ──────────────────
with open(MICROSTATES_DIR / FILTER_TYPE / "2. group_models" / f"group_k{K}_model.pkl", "rb") as f:
    model = pickle.load(f)
new_maps   = model.cluster_centers_                          # (4, 128)
pyc_labels = list(model.info["ch_names"])
ref_maps   = np.load(REFERENCE_MAPS)

c_ref = z(new_maps) @ z(ref_maps).T / new_maps.shape[1]
new_idx, ref_idx = linear_sum_assignment(-np.abs(c_ref))
order = [int(n) for n, _ in sorted(zip(new_idx, ref_idx), key=lambda x: x[1])]
signs = np.sign([c_ref[n, r] for r, n in enumerate(order)])
for r, n in enumerate(order):
    logger.info(f"  new map {n} -> MS {r + 1} (reference), r = {c_ref[n, r]:+.3f}")
pyc = new_maps[order] * signs[:, None]

# ── 3. Correlation ────────────────────────────────────────────────────────────
ragu = ragu_maps[:, [ragu_labels.index(ch) for ch in pyc_labels]]   # drops E129
ragu = ragu[[FIGURE_ORDER[c] - 1 for c in range(1, 9)]]             # Figure 3 numbering
C    = z(pyc) @ z(ragu).T / pyc.shape[1]                           # (4, 8) signed

df = pd.DataFrame(C, index=[f"MS {i + 1}" for i in range(K)],
                  columns=[f"Class {j + 1}" for j in range(C.shape[1])]).round(3)
df.to_csv(f"{OUT_STEM}.csv".replace("pyc_ragu_corr_new", "pyc_ragu_corr_signed_new"))
logger.info("signed spatial correlation:\n" + df.to_string())
for i in range(K):
    matches = [f"class {j + 1} (r = {C[i, j]:+.2f})" for j in np.argsort(-np.abs(C[i]))
               if abs(C[i, j]) >= MATCH_THRESHOLD]
    logger.info(f"MS {i + 1}: " + (", ".join(matches) if matches else
                f"no match (max |r| = {np.abs(C[i]).max():.2f})"))

# ── 4. Heatmap for Figure 5 (style of the submitted figure) ─────────────────────
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"]  = "Arial"
fig, ax = plt.subplots(figsize=(9, 4.4))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for i in range(K):
    for j in range(C.shape[1]):
        ax.text(j, i, f"{C[i, j]:+.2f}".replace("-", "\u2212"), ha="center", va="center", fontsize=12,
                color="white" if abs(C[i, j]) > 0.5 else "black")
        if abs(C[i, j]) >= MATCH_THRESHOLD:
            # outline drawn fully inside the cell, so that all outlines look identical
            # (no overlap with the axes frame or with neighbouring outlines)
            ax.add_patch(plt.Rectangle((j - 0.43, i - 0.43), 0.86, 0.86, fill=False,
                                       edgecolor="#1e8c3a", lw=3, joinstyle="miter", zorder=3))
ax.set_xticks(range(C.shape[1])); ax.set_xticklabels([str(j + 1) for j in range(C.shape[1])], fontsize=12)
ax.set_yticks(range(K));          ax.set_yticklabels([f"MS {i + 1}" for i in range(K)], fontsize=14, fontweight="bold")
ax.set_xlabel("Average-level maps (class)", fontsize=16); ax.set_ylabel("Single-trial templates", fontsize=16)
for sp in ax.spines.values():
    sp.set_linewidth(1.5)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("Spatial correlation (signed r)", fontsize=16); cb.ax.tick_params(labelsize=12)
fig.tight_layout()
for ext in ("svg", "pdf", "png"):
    fig.savefig(RAGU_RESULTS / f"Figure5_heatmap.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
# ── 5. Choice of k: correspondence for k = 4-8 (Methods/Results) ─────────────
GM_DIR = MICROSTATES_DIR / FILTER_TYPE / "2. group_models"
rows = []
for k in range(4, 9):
    with open(GM_DIR / f"group_k{k}_model.pkl", "rb") as f:
        m = pickle.load(f)
    maps = m.cluster_centers_
    r_k  = ragu_maps[:, [ragu_labels.index(ch) for ch in m.info["ch_names"]]]
    r_k  = r_k[[FIGURE_ORDER[c] - 1 for c in range(1, 9)]]
    Ck   = np.abs(z(maps) @ z(r_k).T / maps.shape[1])               # (k, 8), |r|
    for j in range(8):
        rows.append(dict(k=k, ragu_class=j + 1, max_abs_r=Ck[:, j].max(),
                         template=int(Ck[:, j].argmax()) + 1, captured=Ck[:, j].max() >= MATCH_THRESHOLD))
    logger.info(f"k = {k}: captured {int((Ck.max(0) >= MATCH_THRESHOLD).sum())}/8 | max |r|: "
                + ", ".join(f"{j + 1}: {Ck[:, j].max():.2f}" for j in range(8)))
pd.DataFrame(rows).to_csv(GM_DIR / "correspondence_k4_k8.csv", index=False)
logger.info("done")
