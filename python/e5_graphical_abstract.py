"""
Graphical abstract (SLEEP: landscape 5:2, sans-serif 12-16 pt, no caption).

Data as Figure 2 (e1_figure2.py): participant averages that went into RAGU,
average reference, N = 40. ERP at Fz (E11) for stimulation and sham; scalp maps
of the stimulation grand average, each divided by its global field power
(normalised), at the slow-wave peak (0-400 ms) and the second induced peak
(1200-1400 ms).

Output: Review/Submission/graphical_abstract.(tif|pdf|png)
"""

import h5py
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Arc, FancyArrowPatch
from config import *

matplotlib.rcParams["pdf.fonttype"] = 42
mne.set_log_level("ERROR")
logger = setup_logger("graphical_abstract", "e5_graphical_abstract.log")

RAGU_ROOT = RAGU_DIR / "derived"
TXT_DIR   = RAGU_ROOT / "RAGU_TXT_RowsTime_Final"
OUT_DIR   = PROJECT_ROOT.parent / "Review" / "Submission"; OUT_DIR.mkdir(parents=True, exist_ok=True)
FZ, VLIM  = "E11", 1.6
RED, GREY = "#c44e52", "0.3"
cmap = LinearSegmentedColormap.from_list(
    "soft_RdBu", ["#4c5ec4", "#a0a0e8", "#d5d5f5", "#ffffff", "#f5d5d5", "#e8a0a0", "#c44e52"])
WIN1, WIN2 = (0, 400), (1200, 1400)
MINUS = "−"

with h5py.File(RAGU_ROOT / "RAGU_export_BL_vs_STIM.mat", "r") as f:
    cl = f["chanlocs"]
    labels = ["".join(chr(c) for c in f[cl["labels"][i, 0]][:].flatten()) for i in range(cl["labels"].shape[0])]
    xyz = [np.array([f[cl[k][i, 0]][:].flatten()[0] for i in range(cl[k].shape[0])]) if cl[k].dtype == object
           else cl[k][:].flatten() for k in ("X", "Y", "Z")]
    times = f["times"][:].flatten()
info = mne.create_info(labels, sfreq=200.0, ch_types="eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(labels, np.column_stack([xyz[1], xyz[0], xyz[2]]) / 100))))
grand = {}
for cond in ("BL", "STIM"):
    x = np.array([np.loadtxt(TXT_DIR / f"{pid}_{cond}.txt") for pid in PARTICIPANT_IDS])
    grand[cond] = (x - x.mean(axis=2, keepdims=True)).mean(0)
fz = labels.index(FZ)

W, H = 18.0 / 2.54, 7.2 / 2.54                        # 5:2
fig = plt.figure(figsize=(W, H))

# ── left: stimulation icon and label ─────────────────────────────────────────
axL = fig.add_axes([0.0, 0.0, 0.22, 1.0]); axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
cx, cy = 0.42, 0.66
axL.add_patch(Polygon([[cx - 0.14, cy - 0.06], [cx - 0.06, cy - 0.06], [cx + 0.04, cy - 0.15],
                       [cx + 0.04, cy + 0.15], [cx - 0.06, cy + 0.06], [cx - 0.14, cy + 0.06]],
                      closed=True, color="0.2"))
for r in (0.10, 0.17, 0.24):
    axL.add_patch(Arc((cx + 0.04, cy), 2 * r, 2.5 * r, theta1=-50, theta2=50, lw=2, color=RED))
axL.text(0.5, 0.33, "Phase-locked\nacoustic\nstimulation", ha="center", va="center", fontsize=12)
# "N = 40 older adults" with italic N (no Arial Italic file available: the N is
# drawn as an Arial outline slanted by 12 degrees, as in Arial Italic)
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
pt2disp = Affine2D().scale(1 / 72) + fig.dpi_scale_trans
fp = FontProperties(family=matplotlib.rcParams["font.family"], size=12)
nPath = Affine2D().skew_deg(12, 0).transform_path(TextPath((0, 0), "N", prop=fp))
ext = nPath.get_extents()
fig.canvas.draw(); rend = fig.canvas.get_renderer()
rest = fig.text(0, 0, " = 40 older adults", fontsize=12, color="0.35", ha="left", va="baseline", transform=pt2disp)
w_rest = rest.get_window_extent(rend).width * 72 / fig.dpi
cx, cy = 0.11 * W * 72, 0.09 * H * 72                # centre in points
x0 = cx - (ext.x1 + w_rest) / 2; y0 = cy - ext.height / 2
fig.add_artist(PathPatch(nPath, transform=Affine2D().translate(x0, y0) + pt2disp, color="0.35", lw=0))
rest.set_position((x0 + ext.x1, y0))

# ── centre: ERP at Fz ────────────────────────────────────────────────────────
AX = [0.26, 0.15, 0.50, 0.42]
ax = fig.add_axes(AX)
m = (times >= -500) & (times <= 2000)
ax.plot(times[m] / 1000, grand["BL"][m, fz], color=GREY, lw=1.8, label="Sham")
ax.plot(times[m] / 1000, grand["STIM"][m, fz], color=RED, lw=2.2, label="Stimulation")
for (t0, t1) in (WIN1, WIN2):
    ax.axvspan(t0 / 1000, t1 / 1000, color="0.88", zorder=0, lw=0)
ax.axvline(0, color="0.5", ls=":", lw=1)
ax.set_xlim(-0.5, 2.0); ax.set_xticks([0, 1, 2]); ax.set_yticks([])
ax.set_xlabel("Time after sound (s)", fontsize=12, labelpad=1)
ax.tick_params(labelsize=12, length=3)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.legend(frameon=False, fontsize=12, loc="lower right", handlelength=1.2, borderaxespad=0)

# maps above the two windows
MH = 0.26; MW = MH * H / W
xfig = lambda t: AX[0] + AX[2] * (t + 0.5) / 2.5          # time (s) -> figure x
def topo_at(win, label):
    idx = (times >= win[0]) & (times <= win[1])
    v = grand["STIM"][idx].mean(0); v = v / np.sqrt(np.mean(v ** 2))
    c = xfig(np.mean(win) / 1000)
    axm = fig.add_axes([c - MW / 2, 0.585, MW, MH])
    fig.text(c, 0.985, label, ha="center", va="top", fontsize=12, linespacing=1.1)
    mne.viz.plot_topomap(v, info, axes=axm, show=False, cmap=cmap, vlim=(-VLIM, VLIM),
                         contours=0, sensors=False, sphere=(0, 0, 0, 0.18))
    return axm

topo_at(WIN1, "Frontal slow wave\npreserved")
topo_at(WIN2, "Second fronto-\ncentral component")

# ── right: group statement ───────────────────────────────────────────────────
fig.text(0.885, 0.52, "Present in both\nhealthy and\ncognitively impaired\nolder adults", ha="center", va="center",
         fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="0.95", ec="0.6", lw=1))

for ext in ("tif", "pdf", "png"):
    kw = {"dpi": 600}
    if ext == "tif":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(OUT_DIR / f"graphical_abstract.{ext}", **kw)
plt.close(fig); logger.info("saved graphical abstract")
