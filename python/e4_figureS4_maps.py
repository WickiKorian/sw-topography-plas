"""
Supplementary Figure S4: group-wise maps and stimulation - sham difference maps
(Reviewer 1: groups plotted separately; Reviewer 2: difference maps with
normalisation and colour scaling specified).

Data and windows as Figure 2 (e1_figure2.py): participant averages that went into
RAGU, average reference, 200-ms windows from -200 to 1800 ms.
  A  grand-average maps per group and condition, each divided by its global field
     power (normalised amplitude, a.u.), same colour scale as Figure 2 (+/- 1.6)
  B  stimulation - sham difference of the GFP-normalised grand-average maps
     (all participants, HC, CI; a.u.): amplitude differences removed, only
     differences in map shape remain (the quantity tested by the TANOVA); one
     symmetric colour scale and fixed contour levels for all maps

Output: matlab_ragu/derived/figures/FigureS4/FigureS4.(png|pdf|svg)
"""

import h5py
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from config import *

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
mne.set_log_level("ERROR")
logger = setup_logger("figureS4", "e4_figureS4_maps.log")

VLIM      = 1.6
RAGU_ROOT = RAGU_DIR / "derived"
TXT_DIR   = RAGU_ROOT / "RAGU_TXT_RowsTime_Final"
OUT_DIR   = RAGU_ROOT / "figures" / "FigureS4"; OUT_DIR.mkdir(parents=True, exist_ok=True)
cmap = LinearSegmentedColormap.from_list(
    "soft_RdBu", ["#4c5ec4", "#a0a0e8", "#d5d5f5", "#ffffff", "#f5d5d5", "#e8a0a0", "#c44e52"])
WINDOWS = [(-200, 0), (0, 200), (200, 400), (400, 600), (600, 800),
           (800, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800)]
MINUS = "−"

with h5py.File(RAGU_ROOT / "RAGU_export_BL_vs_STIM.mat", "r") as f:
    cl = f["chanlocs"]
    labels = ["".join(chr(c) for c in f[cl["labels"][i, 0]][:].flatten()) for i in range(cl["labels"].shape[0])]
    xyz = [np.array([f[cl[k][i, 0]][:].flatten()[0] for i in range(cl[k].shape[0])]) if cl[k].dtype == object
           else cl[k][:].flatten() for k in ("X", "Y", "Z")]
    times = f["times"][:].flatten()
info = mne.create_info(labels, sfreq=200.0, ch_types="eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(labels, np.column_stack([xyz[1], xyz[0], xyz[2]]) / 100))))

def load(ids, cond):
    x = np.array([np.loadtxt(TXT_DIR / f"{pid}_{cond}.txt") for pid in ids])
    return x - x.mean(axis=2, keepdims=True)

groups = {"All": PARTICIPANT_IDS,
          "HC": [p for p in PARTICIPANT_IDS if p in HEALTHY_IDS],
          "CI": [p for p in PARTICIPANT_IDS if p in IMPAIRED_IDS]}
grand = {(g, c): load(ids, c).mean(0) for g, ids in groups.items() for c in ("STIM", "BL")}
n = {g: len(ids) for g, ids in groups.items()}
logger.info(f"n = {n}")

def wmap(g, c, t0, t1):
    idx = (times >= t0) & (times <= t1)
    return grand[(g, c)][idx].mean(0)

norm = lambda v: v / np.sqrt(np.mean(v ** 2))
diffs = {(g, w): norm(wmap(g, "STIM", *w)) - norm(wmap(g, "BL", *w)) for g in groups for w in WINDOWS}
DLIM = float(np.ceil(2 * max(np.abs(v).max() for v in diffs.values())) / 2)
LEVELS = np.linspace(-DLIM, DLIM, 9)
logger.info(f"difference scale: +/- {DLIM} a.u.")

rowsA = [("HC", "STIM"), ("HC", "BL"), ("CI", "STIM"), ("CI", "BL")]
rowsB = ["All", "HC", "CI"]
cname = {"STIM": "stimulation", "BL": "sham"}
fig = plt.figure(figsize=(17.1 / 2.54, 13.5 / 2.54))
gs = fig.add_gridspec(len(rowsA) + len(rowsB) + 1, len(WINDOWS) + 1,
                      height_ratios=[1] * len(rowsA) + [0.35] + [1] * len(rowsB),
                      width_ratios=[1] * len(WINDOWS) + [0.25], hspace=0.08, wspace=0.05,
                      left=0.13, right=0.91, top=0.93, bottom=0.03)

def topo(ax, v, vlim, contours=6):
    im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=cmap, vlim=(-vlim, vlim),
                                 contours=contours, sensors=False, sphere=(0, 0, 0, 0.18))
    for line in ax.lines: line.set_linewidth(0.4)
    return im

for r, (g, c) in enumerate(rowsA):
    for k, w in enumerate(WINDOWS):
        ax = fig.add_subplot(gs[r, k]); v = wmap(g, c, *w)
        imA = topo(ax, v / np.sqrt(np.mean(v ** 2)), VLIM)
        if r == 0:
            ax.set_title(f"{w[0]}–{w[1]}".replace("-", MINUS), fontsize=6.5, pad=2)
        if k == 0:
            ax.text(-0.15, 0.5, f"{g} {cname[c]}\n(n = {n[g]})", transform=ax.transAxes, ha="right", va="center", fontsize=7)
for r, g in enumerate(rowsB):
    for k, w in enumerate(WINDOWS):
        ax = fig.add_subplot(gs[len(rowsA) + 1 + r, k])
        imB = topo(ax, diffs[(g, w)], DLIM, contours=LEVELS)
        if k == 0:
            lab = "All" if g == "All" else g
            ax.text(-0.15, 0.5, f"{lab}\n(n = {n[g]})", transform=ax.transAxes, ha="right", va="center", fontsize=7)

caxA = fig.add_subplot(gs[:len(rowsA), -1]); cbA = fig.colorbar(imA, cax=caxA)
cbA.set_ticks([-VLIM, 0, VLIM]); cbA.ax.set_yticklabels([f"{MINUS}{VLIM}", "0", f"{VLIM}"])
cbA.ax.tick_params(labelsize=6); cbA.set_label("Normalised amplitude (a.u.)", fontsize=6.5)
caxB = fig.add_subplot(gs[len(rowsA) + 1:, -1]); cbB = fig.colorbar(imB, cax=caxB)
cbB.set_ticks([-DLIM, 0, DLIM]); cbB.ax.set_yticklabels([f"{MINUS}{DLIM:g}", "0", f"{DLIM:g}"])
cbB.ax.tick_params(labelsize=6); cbB.set_label("Stimulation " + MINUS + " sham,\nnormalised (a.u.)", fontsize=6.5)
fig.text(0.01, 0.955, "A", fontsize=10, fontweight="bold")
fig.text(0.01, 0.40, "B", fontsize=10, fontweight="bold")
fig.text(0.535, 0.975, "Time window (ms)", ha="center", fontsize=7)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT_DIR / f"FigureS4.{ext}", dpi=600 if ext == "png" else None)
plt.close(fig); logger.info("done")
