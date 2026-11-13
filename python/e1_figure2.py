"""
Figure 2: all elements produced in Python (layout assembled in Adobe Illustrator).

Replaces archive/12_figure2_addons.py and the notebook archive/13_figure2_topomaps/13_figure2_topomaps.ipynb
(same code and settings, outputs unchanged).

Data: the participant averages that went into RAGU
(matlab_ragu/derived/RAGU_TXT_RowsTime_Final/<ID>_<BL|STIM>.txt, 900 time points
x 129 channels, -1.5 to 3.0 s; N = 40, identical to RAGU_export_BL_vs_STIM.mat),
channel positions from RAGU_export_BL_vs_STIM.mat. Average reference.

Elements
  ERP_Fz_with_CI     ERP at Fz (E11), stimulation and sham, with 95% CI across
                     participants (mean +/- t(0.975, n-1) * SD / sqrt(n));
                     no baseline correction (data high-pass filtered at 0.3 Hz)
  ERP_Fz_CI_only     the same axes with only the CI bands (to lay under the
                     existing curves in Illustrator)
  topo_top / topo_bottom
                     grand-average maps, mean of each 200-ms window (-200 to
                     800 ms; 800 to 1800 ms), each divided by its global field
                     power (normalised amplitude, as in the TANOVA); one colour
                     scale (+/- 1.6) for all maps
  colorbar_norm      colour bar of the maps (normalised amplitude, a.u.)

Output: matlab_ragu/derived/figures/Figure2/
"""

import h5py
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from config import *

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"          # text stays editable in Illustrator
matplotlib.rcParams["font.family"] = "Arial"
mne.set_log_level("ERROR")
logger = setup_logger("figure2", "e1_figure2.log")

FZ, VLIM  = "E11", 1.6
RAGU_ROOT = RAGU_DIR / "derived"
TXT_DIR   = RAGU_ROOT / "RAGU_TXT_RowsTime_Final"
OUT_DIR   = RAGU_ROOT / "figures" / "Figure2"; OUT_DIR.mkdir(parents=True, exist_ok=True)
COL  = {"STIM": "#c44e52", "BL": "0.25"}
NAME = {"STIM": "Stimulation", "BL": "Sham"}
cmap = LinearSegmentedColormap.from_list(
    "soft_RdBu", ["#4c5ec4", "#a0a0e8", "#d5d5f5", "#ffffff", "#f5d5d5", "#e8a0a0", "#c44e52"])
TOP    = [(-200, 0), (0, 200), (200, 400), (400, 600), (600, 800)]
BOTTOM = [(800, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800)]


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig); logger.info(f"saved {stem}")


# ── data ───────────────────────────────────────────────────────────────────────
with h5py.File(RAGU_ROOT / "RAGU_export_BL_vs_STIM.mat", "r") as f:
    cl = f["chanlocs"]
    labels = ["".join(chr(c) for c in f[cl["labels"][i, 0]][:].flatten()) for i in range(cl["labels"].shape[0])]
    xyz = [np.array([f[cl[k][i, 0]][:].flatten()[0] for i in range(cl[k].shape[0])]) if cl[k].dtype == object
           else cl[k][:].flatten() for k in ("X", "Y", "Z")]
    times = f["times"][:].flatten()
info = mne.create_info(labels, sfreq=200.0, ch_types="eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(labels, np.column_stack([xyz[1], xyz[0], xyz[2]]) / 100))))

data = {}                                           # (n_participants, n_times, n_channels), average reference
for cond in ("BL", "STIM"):
    x = np.array([np.loadtxt(TXT_DIR / f"{pid}_{cond}.txt") for pid in PARTICIPANT_IDS])
    data[cond] = x - x.mean(axis=2, keepdims=True)
logger.info(f"N = {len(PARTICIPANT_IDS)} participants")

# ── ERP at Fz with 95% CI ──────────────────────────────────────────────────────
fz, ci = labels.index(FZ), {}
for cond in ("BL", "STIM"):
    y = data[cond][:, :, fz]; n = len(y)
    ci[cond] = (y.mean(0), stats.t.ppf(0.975, n - 1) * y.std(0, ddof=1) / np.sqrt(n), n)


def erp_panel(stem, curves):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for cond in ("BL", "STIM"):
        m, h, n = ci[cond]
        ax.fill_between(times, m - h, m + h, color=COL[cond], alpha=0.25, lw=0)
        if curves:
            ax.plot(times, m, color=COL[cond], lw=1.5, label=f"{NAME[cond]} (N = {n})")
    ax.axvline(0, color="k", ls="--", lw=0.8); ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (ms)", fontsize=8); ax.set_ylabel("Amplitude (µV)", fontsize=8)
    ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    if curves:
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, stem)


erp_panel("ERP_Fz_with_CI", curves=True)
erp_panel("ERP_Fz_CI_only", curves=False)

# ── topographies (GFP-normalised) ──────────────────────────────────────────────
grand = {c: data[c].mean(0) for c in data}          # (n_times, n_channels)


def topo_panel(windows, stem):
    fig, axes = plt.subplots(2, len(windows), figsize=(len(windows) * 1.6, 2 * 1.6))
    for col, (t0, t1) in enumerate(windows):
        idx = (times >= t0) & (times <= t1)
        for row, cond in enumerate(("STIM", "BL")):
            topo = grand[cond][idx].mean(0)
            mne.viz.plot_topomap(topo / np.sqrt(np.mean(topo ** 2)), info, axes=axes[row, col], show=False,
                                 cmap=cmap, vlim=(-VLIM, VLIM), contours=8, sensors=False, sphere=(0, 0, 0, 0.18))
            for line in axes[row, col].lines:
                line.set_linewidth(0.5)
    fig.tight_layout(); save(fig, stem)


topo_panel(TOP, "topo_top")
topo_panel(BOTTOM, "topo_bottom")

# ── colour bar ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(0.35, 2.2))
cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=matplotlib.colors.Normalize(-VLIM, VLIM))
cb.set_ticks([-VLIM, -0.8, 0, 0.8, VLIM]); cb.ax.tick_params(labelsize=7)
cb.set_label("Normalised amplitude (a.u.)", fontsize=8)
save(fig, "colorbar_norm")
logger.info("done")
