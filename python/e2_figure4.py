"""
e2_figure4.py
Figure 4 of the manuscript: time-resolved single-trial microstate coverage
(revision; replaces the 250-ms bin version of a3_stats.plot_timecourse).

Panels (one figure, vector output for final assembly in Illustrator)
  top     grand-average ERP at Fz (E11), average reference, no baseline correction;
          stimulation vs sham, healthy controls (solid) vs cognitive impairment (dashed)
  MS 1-4  coverage at full temporal resolution (fraction of epochs assigned to the
          state at each time point, averaged across participants), same four lines
  shading significant clusters of the stimulation effect across ALL participants
          (cluster-based permutation test, Bonferroni across the 4 states;
          a4_coverage_clusters.py, contrast ALL_stim_vs_sham, sig_corr)
  maps    group-level templates (K = 4), each scaled to its maximum absolute value
          (-1 to 1, a.u.), with one labelled colour bar (Reviewer 1)
  curves  displayed with a 50-ms moving average (display only; statistics unaffected)

Inputs (all computed by earlier scripts, nothing is recomputed here)
  5. coverage_clusters/coverage_timecourses_k4.npz   per-sample coverage
  5. coverage_clusters/coverage_clusters_k4.csv      cluster statistics
  2. group_models/group_k4_maps.npy                  templates
  matlab_ragu/derived/RAGU_TXT_RowsTime_Final        participant ERPs (as Figure 2)

Output: microstates/filter_03_20/17. figure4/Figure4.{svg,pdf,png}
        and FigureS2.{svg,pdf,png}: same layout with 250-ms bins (sensitivity analysis)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import h5py
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from config import *

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"          # text stays editable in Illustrator
matplotlib.rcParams["font.family"] = "Arial"
mne.set_log_level("ERROR")
logger = setup_logger("figure4", "e2_figure4.log")

K        = 4
BASE     = MICROSTATES_DIR / FILTER_TYPE
OUT      = BASE / "17. figure4"; OUT.mkdir(parents=True, exist_ok=True)
RAGU     = RAGU_DIR / "derived"
STIM_C, SHAM_C = "#E63300", "0.25"                     # as in Figures 2 and 4 (submitted)
LS       = {"HC": "-", "CI": (0, (4, 2))}
SIG_C    = "#add8e6"
MS_NAME  = {1: "MS 1", 2: "MS 2", 3: "MS 3", 4: "MS 4"}
cmap     = LinearSegmentedColormap.from_list(          # same colours as the Figure 2/3 maps
    "soft_RdBu", ["#4c5ec4", "#a0a0e8", "#d5d5f5", "#ffffff", "#f5d5d5", "#e8a0a0", "#c44e52"])
groups   = {"HC": set(HEALTHY_IDS), "CI": set(IMPAIRED_IDS)}

# ── coverage (participants x [sham, stim] x states x time) ─────────────────────
z     = np.load(BASE / "5. coverage_clusters" / f"coverage_timecourses_k{K}.npz", allow_pickle=True)
cov, t_cov, pids = z["cov"], z["times"] * 1000, list(z["pids"])
# numbering as in the manuscript (same assignment as a3_stats.py / a4_coverage_clusters.py):
# raw k-means map n -> reference MS r; maps are also sign-aligned to the reference for display
from scipy.optimize import linear_sum_assignment
raw_maps = np.load(BASE / "2. group_models" / f"group_k{K}_maps.npy")
ref_maps = np.load(REFERENCE_MAPS)
_z   = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
corr = _z(raw_maps) @ _z(ref_maps).T / raw_maps.shape[1]
n_idx, r_idx = linear_sum_assignment(-np.abs(corr))
order = [n for n, _ in sorted(zip(n_idx.tolist(), r_idx.tolist()), key=lambda x: x[1])]
sign  = [np.sign(corr[n, r]) for r, n in enumerate(order)]
cov   = cov[:, :, order]
logger.info(f"raw maps -> MS 1-4: {order}, display signs {sign}")
ok    = ~np.isnan(cov).any(axis=(0, 1, 2))           # unlabelled epoch edges excluded (as in 04)
gidx  = {g: [i for i, p in enumerate(pids) if p in ids] for g, ids in groups.items()}
logger.info(f"coverage: {cov.shape}, HC {len(gidx['HC'])}, CI {len(gidx['CI'])}")

# ── clusters across all participants ───────────────────────────────────────────
cl = pd.read_csv(BASE / "5. coverage_clusters" / f"coverage_clusters_k{K}.csv")
cl = cl[(cl.contrast == "ALL_stim_vs_sham") & cl.sig_corr]
for r in cl.itertuples():
    logger.info(f"cluster {r.ms_label}: {1000*r.t_start:.0f} to {1000*r.t_end:.0f} ms ({r.direction.split(' /')[0]}), p_corr = {r.p_corr:.4f}")

# ── ERP at Fz per group and condition (as archive/12_figure2_addons.py) ───────────────
with h5py.File(RAGU / "RAGU_export_BL_vs_STIM.mat", "r") as f:
    c = f["chanlocs"]
    labels = ["".join(chr(x) for x in f[c["labels"][i, 0]][:].flatten()) for i in range(c["labels"].shape[0])]
    t_erp  = f["times"][:].flatten()
fz  = labels.index("E11")
erp = {}
for g, ids in groups.items():
    for cond, key in (("BL", "sham"), ("STIM", "stim")):
        y = [(lambda x: (x - x.mean(1, keepdims=True))[:, fz])(np.loadtxt(RAGU / "RAGU_TXT_RowsTime_Final" / f"{p}_{cond}.txt"))
             for p in PARTICIPANT_IDS if p in ids]
        erp[(g, key)] = np.mean(y, 0)

# ── templates and channel layout ───────────────────────────────────────────────
maps = np.array([sign[r] * raw_maps[n] for r, n in enumerate(order)])
info = mne.read_epochs(str(EPOCHS_DIR / FILTER_TYPE / f"{PARTICIPANT_IDS[0]}_sham-epo.fif"), preload=False).info
assert maps.shape[1] == len(info.ch_names)

# ── figure (layout and styling as the submitted Figure 4, a3_stats.plot_timecourse) ──
SMOOTH_MS = 150                                        # display only: moving average
w = int(round(SMOOTH_MS / (t_cov[1] - t_cov[0]))) | 1
smooth = lambda y: np.convolve(y, np.ones(w) / w, mode="same")
edge = w // 2                                          # drop samples affected by the window edge
tt = t_cov[ok][edge:-edge]
cmap_maps = PASTEL_RDBU                                # colours of the submitted Figure 4 maps

lines = [(g, cond, col) for g in ("HC", "CI") for cond, col in (("stim", STIM_C), ("sham", SHAM_C))]


def draw(curves, tt, spans, sig_label, stem):
    """Figure layout of Figure 4 (and of the submitted Figure 4); spans = [(state, t0_ms, t1_ms)]."""
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(K + 1, 2, width_ratios=[7.2, 1], height_ratios=[1.6] + [1] * K,
                           hspace=0.18, wspace=0.03)
    ax0 = fig.add_subplot(gs[0, 0])
    for g, cond, col in lines:
        ax0.plot(t_erp, erp[(g, cond)], color=col, ls=LS[g], lw=2)
    ax0.axhline(0, color="0.75", lw=0.8); ax0.axvline(0, color="0.3", ls=":", lw=1.5)
    ax0.set_ylabel("Amplitude (µV)", fontsize=13); ax0.tick_params(labelsize=11, labelbottom=False)
    ax0.text(0.01, 0.93, "ERP (Fz)", transform=ax0.transAxes, fontsize=14, fontweight="bold", va="top")
    ax0.set_xlim(-1600, 3400)
    for sp in ("top", "right"): ax0.spines[sp].set_visible(False)
    ax0.spines["left"].set_linewidth(1.5); ax0.spines["bottom"].set_linewidth(1.5)

    # same y-range for all coverage panels, so effects are comparable across states
    lo = [min(curves[(s, g, c)].min() for g, c, _ in lines) for s in range(K)]
    hi = [max(curves[(s, g, c)].max() for g, c, _ in lines) for s in range(K)]
    span = max(h - l for l, h in zip(lo, hi)) + 0.02

    axes = [ax0]
    for s in range(K):
        ax = fig.add_subplot(gs[s + 1, 0], sharex=ax0); axes.append(ax)
        for st, t0, t1 in spans:
            if st == s: ax.axvspan(t0, t1, color=SIG_C, alpha=0.55, lw=0, zorder=0)
        for g, cond, col in lines:
            ax.plot(tt, curves[(s, g, cond)], color=col, ls=LS[g], lw=2, zorder=2)
        mid = (lo[s] + hi[s]) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.axvline(0, color="0.3", ls=":", lw=1.5, zorder=1)
        ax.set_ylabel("Coverage", fontsize=13); ax.tick_params(labelsize=11, labelbottom=(s == K - 1))
        ax.text(0.01, 0.5, MS_NAME[s + 1], transform=ax.transAxes, fontsize=15, fontweight="bold", va="center", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.spines["left"].set_linewidth(1.5); ax.spines["bottom"].set_linewidth(1.5)
        axm = fig.add_subplot(gs[s + 1, 1])
        m = maps[s] / np.abs(maps[s]).max()
        im, _ = mne.viz.plot_topomap(m, info, axes=axm, show=False, cmap=cmap_maps, vlim=(-1, 1),
                                     contours=8, sensors=False, outlines="head", extrapolate="head")
    axes[-1].set_xlabel("Time relative to stimulus onset (ms)", fontsize=14)
    axes[-1].set_xticks(range(-1500, 3001, 500))

    pos = axm.get_position()
    cax = fig.add_axes([pos.x0 + 0.1 * pos.width, pos.y0 - 0.035, 0.8 * pos.width, 0.012])
    cb  = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["−", "0", "+"], fontsize=11)
    cb.set_label("Norm. amplitude (a.u.)", fontsize=10)

    handles = [Line2D([], [], color=STIM_C, lw=3, label="Stim"), Line2D([], [], color=SHAM_C, lw=3, label="Sham"),
               Line2D([], [], color="0.5", ls=LS["HC"], lw=3, label="HC"), Line2D([], [], color="0.5", ls=LS["CI"], lw=3, label="CI"),
               Patch(facecolor=SIG_C, alpha=0.55, label=sig_label)]
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=13, frameon=True,
                     edgecolor="0.3", fancybox=True, bbox_to_anchor=(0.47, 0.0))
    leg.get_frame().set_linewidth(2)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.98, bottom=0.12)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    logger.info(f"saved {OUT / (stem + '.svg')}")


# ── Figure 4: full temporal resolution, cluster-based test ─────────────────────
curves = {(s, g, cond): smooth(cov[gidx[g], 1 if cond == "stim" else 0, s][:, ok].mean(axis=0))[edge:-edge]
          for s in range(K) for g, cond, _ in lines}
draw(curves, tt, [(r.state, 1000 * r.t_start, 1000 * r.t_end) for r in cl.itertuples()],
     "Sig. cluster, all participants", "Figure4")
logger.info(f"Figure 4: display smoothing {SMOOTH_MS} ms")

# ── Supplementary Figure S2: same layout, 250-ms bins (sensitivity analysis) ───
# bin coverage per participant from the backfitting table (the values tested in a3_stats.py),
# raw state n -> MS r as above; significant bins: FDR across 18 bins x 4 states (a3_stats.py)
P  = pd.read_csv(BASE / "3. backfitting" / f"parameters_k{K}.csv")
pb = pd.read_csv(BASE / "4. statistics" / f"k{K}" / f"permutation_results_k{K}.csv")
pb = pb[(pb.family == "timecov_bin") & (pb.test == "all_condition_effect")].set_index("col")
NB = 18; t_bins = -1500 + 250 * np.arange(NB) + 125
gmap = {"HC": "healthy", "CI": "impaired"}
curves_b = {}
for s in range(K):
    cols = [f"bin{b}_state{order[s]}_cov" for b in range(NB)]
    for g, cond, _ in lines:
        d = P[(P.group == gmap[g]) & (P.condition == cond)]
        assert len(d) == len(gidx[g]), (g, cond, len(d))
        curves_b[(s, g, cond)] = d[cols].mean().values
spans_b = [(s, -1500 + 250 * b, -1500 + 250 * (b + 1)) for s in range(K) for b in range(NB)
           if pb.loc[f"bin{b}_state{s}_cov", "sig_fdr"]]
draw(curves_b, t_bins, spans_b, "Sig. bin, all participants (FDR)", "FigureS2")
logger.info(f"Figure S2: {len(spans_b)} significant bins")
logger.info("done")
