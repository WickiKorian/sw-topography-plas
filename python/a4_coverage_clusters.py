"""
Time-resolved microstate coverage: cluster-based permutation tests.

Replaces the 250-ms bin analysis of a3_stats.py (reviewer concerns: bin width,
multiplicity, coverage summing to 1).

Step 1  per-sample coverage   backfit the K=4 group templates (same parameters as
                              a2_cluster_backfit.py) and compute, for every time
                              point, the fraction of epochs assigned to each state
                              -> saved once, re-used on later runs
Step 2  cluster permutation   per state and contrast (Maris & Oostenveld, 2007)
                              contrasts: ALL stim-vs-sham (all participants),
                              HC stim-vs-sham, CI stim-vs-sham,
                              interaction (HC diff vs CI diff),
                              group effect in sham, group effect in stim
                              Bonferroni correction across the 4 states,
                              separately for each contrast
Step 3  figure                stim-minus-sham difference per state with 95% CI,
                              significant clusters marked
"""

import pickle
import warnings
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from mne.stats import (permutation_cluster_1samp_test, permutation_cluster_test,
                       ttest_ind_no_p)
from config import *

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = setup_logger("coverage_clusters", "a4_coverage_clusters.log")

# ── Settings ───────────────────────────────────────────────────────────────────
K             = 4
N_PERM        = 10000
ALPHA_FORMING = 0.05      # cluster-forming threshold (two-tailed, per time point)
ALPHA_CLUSTER = 0.05      # significance of clusters after correction
SEED          = 42

BASE        = MICROSTATES_DIR / FILTER_TYPE
EPOCHS_PATH = EPOCHS_DIR / FILTER_TYPE
OUT_DIR     = BASE / "5. coverage_clusters"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COV_FILE    = OUT_DIR / f"coverage_timecourses_k{K}.npz"


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_epochs(path) -> mne.Epochs:
    """Same loading as a2_cluster_backfit.py: average reference, drop Cz."""
    ep = mne.read_epochs(str(path), preload=True, verbose=False)
    ep.set_eeg_reference("average", projection=False)
    if "Cz" in ep.ch_names:
        ep.drop_channels(["Cz"])
    return ep


def get_labels(segmentation) -> np.ndarray:
    for attr in ("labels_", "_labels", "labels"):
        if hasattr(segmentation, attr):
            v = getattr(segmentation, attr)
            return np.asarray(v() if callable(v) else v)
    raise AttributeError("no labels attribute on segmentation")


# ── Step 1: per-sample coverage ────────────────────────────────────────────────
def compute_coverage():
    with open(BASE / "2. group_models" / f"group_k{K}_model.pkl", "rb") as f:
        model = pickle.load(f)

    pids, cov, times = list(PARTICIPANT_IDS), None, None
    for i, pid in enumerate(pids, 1):
        for c, cond in enumerate(("sham", "stim")):
            ep  = load_epochs(EPOCHS_PATH / f"{pid}_{cond}-epo.fif")
            seg = model.predict(
                ep,
                factor=BACKFIT_SMOOTHING_FACTOR,
                half_window_size=BACKFIT_SMOOTHING_WINDOW,
                min_segment_length=BACKFIT_MIN_SEGMENT_SAMPLES,
                reject_edges=BACKFIT_REJECT_EDGES,
            )
            lab = get_labels(seg)                       # (n_epochs, n_times); -1 = unlabelled
            if cov is None:
                times = ep.times
                cov   = np.full((len(pids), 2, K, len(times)), np.nan)
            n_valid = (lab >= 0).sum(axis=0)
            for s in range(K):
                cov[i - 1, c, s] = np.where(
                    n_valid > 0, (lab == s).sum(axis=0) / np.maximum(n_valid, 1), np.nan
                )
            logger.info(f"[{i}/{len(pids)}] {pid} {cond}: {len(ep)} epochs")

    np.savez(COV_FILE, cov=cov, times=times, pids=np.array(pids))
    return cov, times, pids


if COV_FILE.exists():
    _d = np.load(COV_FILE, allow_pickle=True)
    cov, times, pids = _d["cov"], _d["times"], list(_d["pids"])
    logger.info(f"loaded per-sample coverage from {COV_FILE.name}")
else:
    logger.info("step 1: computing per-sample coverage (backfitting)...")
    cov, times, pids = compute_coverage()

# ── Microstate numbering as in the submitted analysis (as in a3_stats.py) ──────
# K-means numbers the maps arbitrarily. Each map is assigned to the reference map
# of the submitted analysis with the highest absolute spatial correlation
# (one-to-one assignment); the coverage array is reordered accordingly, so that
# state s in all outputs is MS s+1 as reported in the manuscript.
from scipy.optimize import linear_sum_assignment
new_maps = np.load(BASE / "2. group_models" / f"group_k{K}_maps.npy")
ref_maps = np.load(REFERENCE_MAPS)
_z   = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
corr = _z(new_maps) @ _z(ref_maps).T / new_maps.shape[1]           # (new, reference)
new_idx, ref_idx = linear_sum_assignment(-np.abs(corr))
for n, r in sorted(zip(new_idx.tolist(), ref_idx.tolist()), key=lambda x: x[1]):
    logger.info(f"  new map {n} -> MS {r + 1} (reference), r = {corr[n, r]:+.3f}")
order = [n for n, _ in sorted(zip(new_idx.tolist(), ref_idx.tolist()), key=lambda x: x[1])]
cov   = cov[:, :, order]

is_hc = np.array([p in HEALTHY_IDS for p in pids])
assert (is_hc.sum(), (~is_hc).sum()) == (25, 15), (is_hc.sum(), (~is_hc).sum())

# time points labelled in every participant/condition/state (edges excluded)
ok    = ~np.isnan(cov).any(axis=(0, 1, 2))
t_sel = times[ok]
logger.info(f"analysis window: {t_sel[0]:.3f} to {t_sel[-1]:.3f} s ({ok.sum()} samples)")

diff = cov[:, 1] - cov[:, 0]                    # stim - sham: (n_sub, K, n_times)


# ── Step 2: cluster-based permutation tests ────────────────────────────────────
def t_thr(df):
    return stats.t.ppf(1 - ALPHA_FORMING / 2, df)


def one_sample(X):
    return permutation_cluster_1samp_test(
        X, threshold=t_thr(X.shape[0] - 1), n_permutations=N_PERM,
        tail=0, seed=SEED, out_type="indices", verbose=False)


def two_sample(A, B):
    return permutation_cluster_test(
        [A, B], threshold=t_thr(A.shape[0] + B.shape[0] - 2), n_permutations=N_PERM,
        tail=0, stat_fun=ttest_ind_no_p, seed=SEED, out_type="indices", verbose=False)


rows = []
for s in range(K):
    tests = {
        "ALL_stim_vs_sham":   (one_sample, (diff[:, s][:, ok],)),
        "HC_stim_vs_sham":    (one_sample, (diff[is_hc][:, s][:, ok],)),
        "CI_stim_vs_sham":    (one_sample, (diff[~is_hc][:, s][:, ok],)),
        "interaction_HC_vs_CI": (two_sample, (diff[is_hc][:, s][:, ok],
                                              diff[~is_hc][:, s][:, ok])),
        "group_HC_vs_CI_sham": (two_sample, (cov[is_hc, 0, s][:, ok],
                                             cov[~is_hc, 0, s][:, ok])),
        "group_HC_vs_CI_stim": (two_sample, (cov[is_hc, 1, s][:, ok],
                                             cov[~is_hc, 1, s][:, ok])),
    }
    for contrast, (fun, args) in tests.items():
        T, clusters, pvals, _ = fun(*args)
        for cl, p in zip(clusters, pvals):
            idx = cl[0]
            rows.append(dict(
                contrast=contrast, state=s, ms_label=f"MS {s + 1}",
                t_start=round(float(t_sel[idx[0]]), 3),
                t_end=round(float(t_sel[idx[-1]]), 3),
                direction="stim>sham / HC>CI" if T[idx].mean() > 0 else "stim<sham / HC<CI",
                cluster_mass=float(T[idx].sum()),
                p_cluster=float(p),
            ))
        logger.info(f"MS {s + 1} {contrast}: {len(clusters)} cluster(s)")

res = pd.DataFrame(rows)
if len(res):
    # Each cluster test already controls the error rate across time.
    # Correct for the K states tested per contrast (Bonferroni); BH across all
    # clusters would be inappropriate (many small non-significant clusters).
    res["p_corr"]  = np.minimum(1.0, res["p_cluster"] * K)
    res["sig_corr"] = res["p_corr"] < ALPHA_CLUSTER
res.to_csv(OUT_DIR / f"coverage_clusters_k{K}.csv", index=False)

logger.info("significant clusters (Bonferroni across states):")
for _, r in res[res.get("sig_corr", False) == True].iterrows():
    logger.info(f"  {r.ms_label} {r.contrast}: {r.t_start:.3f}-{r.t_end:.3f} s, "
                f"{r.direction}, p={r.p_cluster:.4f}, p_corr={r.p_corr:.4f}")


# ── Step 3: figure (difference stim - sham, 95% CI) ────────────────────────────
fig, axes = plt.subplots(K, 1, figsize=(10, 2.4 * K), sharex=True)
groups = [("HC", is_hc, "#222222", GROUP_LINE_STYLES["healthy"], "healthy"),
          ("CI", ~is_hc, "#888888", GROUP_LINE_STYLES["impaired"], "impaired")]

for s, ax in enumerate(axes):
    for name, mask, color, ls, key in groups:
        X  = diff[mask][:, s][:, ok] * 100                   # percentage points
        m  = X.mean(0)
        hw = stats.t.ppf(0.975, X.shape[0] - 1) * X.std(0, ddof=1) / np.sqrt(X.shape[0])
        ax.fill_between(t_sel, m - hw, m + hw, color=color, alpha=0.18, lw=0)
        ax.plot(t_sel, m, color=color, ls=ls, lw=1.8, label=name)

        sig = res[(res.get("sig_corr", False) == True) & (res.state == s)
                  & (res.contrast == f"{name}_stim_vs_sham")] if len(res) else []
        for _, r in (sig.iterrows() if len(sig) else []):
            ax.axvspan(r.t_start, r.t_end, color=SIG_BIN_COLORS[key], alpha=0.5, lw=0, zorder=0)

    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_title(f"MS {s + 1}", fontsize=10, fontweight="bold", loc="left")
    ax.set_ylabel("Δ coverage\n(stim − sham, pp)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(fontsize=8, frameon=False, loc="upper right")
axes[-1].set_xlabel("Time relative to stimulus onset (s)")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT_DIR / f"coverage_difference_clusters_k{K}.{ext}",
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
logger.info("done")
