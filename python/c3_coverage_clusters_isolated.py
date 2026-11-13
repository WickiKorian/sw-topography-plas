"""
Cluster-based permutation test of the time-resolved coverage (as 04) in epochs
without neighbouring stimuli: which Figure 4 clusters remain?

Subsets (definitions as in c2_overlap_variants.py, fixed from the response
duration of 2.2 s, not from results):
  prev3.7          no preceding stimulus within 3.7 s (pre-stimulus window and
                   post-stimulus window free of responses to earlier stimuli)
  prev3.7_next3.0  additionally no following stimulus within 3.0 s (fully isolated)
c1_stimulus_intervals.py: stimulation does not change the probability of a
following stimulus (62% vs 61%), so the fully isolated subset is not depleted of
"successful" trials; following stimuli occur ~120 ms earlier under stimulation,
i.e. their responses fall systematically into the late part of the epoch.

Step 1  backfit the K=4 group model (same parameters as 02/04) on each subset,
        per-sample coverage per participant x night -> cached per file
        (resumable: the run stops after TIME_BUDGET_S and continues on the next call)
Step 2  ALL stim-vs-sham cluster test per state (as 04), Bonferroni across 4 states;
        participants with < MIN_EPOCHS epochs in a night excluded from that subset
Step 3  figure: stim - sham difference per state for all epochs and both subsets,
        significant clusters as bars (display smoothing 150 ms)

Step 4  EXPLORATORY test (formerly archive/c5_ms1_second_peak_isolated.py):
  EXPLORATORY test (post hoc, not reported as confirmatory): is MS 1 coverage
  higher under stimulation in the second-peak window when neighbouring stimuli
  are excluded?

  Specification (fixed before running, 2026-11-04):
    Rationale   MS 1 corresponds to the central valley of the second induced peak
                (RAGU class 2, r = -.97, inverted) and to the second-peak map
                (class 5, r = .86; 23_correspondence / Figure 5). In all epochs MS 1
                showed no increase at the second peak (archive/posthoc_ms1_secondpeak.py,
                June 2026, old pipeline: not significant). Script 26 suggested that
                following/preceding stimuli may mask it.
    Measure     MS 1 coverage averaged over 1080-1380 ms (second-peak window, as
                Supplementary Table S8), stim - sham, per participant
    Subsets     primary: prev3.7 (no preceding stimulus within 3.7 s, n = 40);
                secondary: prev3.7_next3.0 (fully isolated); all epochs for reference
    Test        paired sign-flip permutation test (10000), one-sided (stim > sham,
                direction from the RAGU result) and two-sided; dz
    Reporting   whatever the outcome; post hoc, exploratory

  Input: coverage per participant from 04 (all epochs) and 26 (cache, subsets).
  Output: microstates/filter_03_20/19. coverage_clusters_isolated/ms1_second_peak_exploratory.csv

Output: microstates/filter_03_20/19. coverage_clusters_isolated/
"""

import gc, os, pickle, time, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from mne.stats import permutation_cluster_1samp_test
from config import *

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = setup_logger("coverage_clusters_isolated", "c3_coverage_clusters_isolated.log")

K, N_PERM, SEED = 4, 10000, 42
ALPHA_FORMING = ALPHA_CLUSTER = 0.05
MIN_EPOCHS    = 20                     # as 19
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "1e9"))
BASE  = MICROSTATES_DIR / FILTER_TYPE
OUT   = BASE / "19. coverage_clusters_isolated"; CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# epoch selection: functions and variant definitions of c2_overlap_variants.py
_src = open("c2_overlap_variants.py", encoding="utf-8").read()
_ns = {}; exec(compile(_src[:_src.index("# ── Step 1-2")], "c2_overlap_variants.py", "exec"), _ns)
neighbour_intervals, keep_mask = _ns["neighbour_intervals"], _ns["keep_mask"]
SUBSETS = {v: _ns["VARIANTS"][v] for v in ("prev3.7", "prev3.7_next3.0")}


def get_labels(seg):
    for a in ("labels_", "_labels", "labels"):
        if hasattr(seg, a):
            v = getattr(seg, a); return np.asarray(v() if callable(v) else v)
    raise AttributeError("no labels attribute on segmentation")


# ── Step 1: backfitting per subset (cached, resumable) ─────────────────────────
t0, model = time.time(), None
for pid in PARTICIPANT_IDS:
    for cond in ("sham", "stim"):
        files = {v: CACHE / f"{v}_{pid}_{cond}.npz" for v in SUBSETS}
        if all(f.exists() for f in files.values()):
            continue
        if time.time() - t0 > TIME_BUDGET_S:
            logger.info("time budget reached - run again to continue"); raise SystemExit
        if model is None:
            with open(BASE / "2. group_models" / f"group_k{K}_model.pkl", "rb") as f:
                model = pickle.load(f)
        ep_all = mne.read_epochs(str(EPOCHS_DIR / FILTER_TYPE / f"{pid}_{cond}-epo.fif"), preload=False, verbose=False)
        f_int = CACHE / f"intervals_{pid}_{cond}.npz"     # reading the raw annotations needs much memory:
        if not f_int.exists():                            # done once, in a call of its own
            prev, nxt = neighbour_intervals(pid, cond, ep_all)
            np.savez(f_int, prev=prev, nxt=nxt); gc.collect()
            logger.info(f"{pid} {cond}: intervals cached - run again to continue"); raise SystemExit
        d = np.load(f_int); prev, nxt = d["prev"], d["nxt"]
        for v, crit in SUBSETS.items():
            keep = np.where(keep_mask(prev, nxt, crit))[0]
            if len(keep) < MIN_EPOCHS:
                np.savez(files[v], cov=np.array([]), times=ep_all.times, n=len(keep)); continue
            ep = ep_all[keep].load_data()
            ep.set_eeg_reference("average", projection=False)
            if "Cz" in ep.ch_names:
                ep.drop_channels(["Cz"])
            lab = get_labels(model.predict(ep, factor=BACKFIT_SMOOTHING_FACTOR,
                                           half_window_size=BACKFIT_SMOOTHING_WINDOW,
                                           min_segment_length=BACKFIT_MIN_SEGMENT_SAMPLES,
                                           reject_edges=BACKFIT_REJECT_EDGES))
            nv  = (lab >= 0).sum(0)
            cov = np.array([np.where(nv > 0, (lab == s).sum(0) / np.maximum(nv, 1), np.nan) for s in range(K)])
            np.savez(files[v], cov=cov, times=ep.times, n=len(keep))
            del ep, lab; gc.collect()
        logger.info(f"{pid} {cond}: " + ", ".join(f"{v} {int(np.load(files[v])['n'])}" for v in SUBSETS))

# ── Step 2: cluster tests ──────────────────────────────────────────────────────
main = np.load(BASE / "5. coverage_clusters" / f"coverage_timecourses_k{K}.npz", allow_pickle=True)
times = main["times"]
new_maps = np.load(BASE / "2. group_models" / f"group_k{K}_maps.npy")       # numbering as in 04
ref_maps = np.load(REFERENCE_MAPS)
_z = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
ni, ri = linear_sum_assignment(-np.abs(_z(new_maps) @ _z(ref_maps).T))
order = [n for n, _ in sorted(zip(ni.tolist(), ri.tolist()), key=lambda x: x[1])]

sets = {"all_epochs": main["cov"][:, :, order]}
n_ep = []
for v in SUBSETS:
    c = np.full((len(PARTICIPANT_IDS), 2, K, len(times)), np.nan)
    for i, pid in enumerate(PARTICIPANT_IDS):
        for j, cond in enumerate(("sham", "stim")):
            d = np.load(CACHE / f"{v}_{pid}_{cond}.npz"); n_ep.append(dict(subset=v, participant_id=pid, condition=cond, n=int(d["n"])))
            if d["cov"].size:
                c[i, j] = d["cov"][order]
    sets[v] = c
pd.DataFrame(n_ep).to_csv(OUT / "epochs_per_subset.csv", index=False)

# participants: all with >= MIN_EPOCHS epochs in both nights; time window: samples of 04
# that are labelled in every included participant (with few epochs, single edge samples
# can be unlabelled - these samples are dropped, not the participant)
rows, diffs, oks = [], {}, {}
ok_all = ~np.isnan(sets["all_epochs"]).any(axis=(0, 1, 2))             # window of 04
for name, c in sets.items():
    use = ~np.isnan(c).all(axis=(2, 3)).any(axis=1)                    # data in both nights
    ok  = ok_all & ~np.isnan(c[use]).any(axis=(0, 1, 2))
    logger.info(f"{name}: {use.sum()} participants, {ok.sum()} of {ok_all.sum()} samples")
    diff = (c[use, 1] - c[use, 0])[:, :, ok]
    diffs[name], oks[name] = diff, ok
    for s in range(K):
        X = diff[:, s]
        T, cls, pv, _ = permutation_cluster_1samp_test(X, threshold=stats.t.ppf(1 - ALPHA_FORMING / 2, X.shape[0] - 1),
                                                       n_permutations=N_PERM, tail=0, seed=SEED,
                                                       out_type="indices", verbose=False)
        for cl, p in zip(cls, pv):
            idx = cl[0]
            rows.append(dict(subset=name, n_participants=int(use.sum()), state=s, ms_label=f"MS {s + 1}",
                             t_start=round(float(times[ok][idx[0]]), 3), t_end=round(float(times[ok][idx[-1]]), 3),
                             direction="stim>sham" if T[idx].mean() > 0 else "stim<sham",
                             cluster_mass=float(T[idx].sum()), p_cluster=float(p)))
res = pd.DataFrame(rows)
res["p_corr"] = np.minimum(1, res.p_cluster * K); res["sig_corr"] = res.p_corr < ALPHA_CLUSTER
res.to_csv(OUT / f"coverage_clusters_isolated_k{K}.csv", index=False)
for _, r in res[res.sig_corr].iterrows():
    logger.info(f"  {r.subset:16s} {r.ms_label} {r.t_start:+.3f} to {r.t_end:+.3f} s {r.direction} p_corr={r.p_corr:.4f}")

# ── Step 3: figure ─────────────────────────────────────────────────────────────
common = np.logical_and.reduce(list(oks.values()))
t = times[common] * 1000
w = int(round(150 / (t[1] - t[0]))) | 1; e = w // 2
sm = lambda y: np.convolve(y, np.ones(w) / w, mode="same")[e:-e]
STY = {"all_epochs": ("0.55", "All epochs"), "prev3.7": ("#2c7fb8", "No preceding stimulus (3.7 s)"),
       "prev3.7_next3.0": ("#E63300", "Fully isolated")}
plt.rcParams.update({"font.size": 8})                              # drawn at print size (16 cm wide)
fig, axes = plt.subplots(K, 1, figsize=(6.3, 1.45 * K), sharex=True)
for s, ax in enumerate(axes):
    for k, (name, (col, lab)) in enumerate(STY.items()):
        ax.plot(t[e:-e], 100 * sm(diffs[name][:, s][:, common[oks[name]]].mean(0)), color=col, lw=1.1, label=f"{lab} (n = {diffs[name].shape[0]})")
    ylo = ax.get_ylim()[0]
    for k, (name, (col, _)) in enumerate(STY.items()):
        for _, r in res[(res.subset == name) & (res.state == s) & res.sig_corr].iterrows():
            ax.plot([1000 * r.t_start, 1000 * r.t_end], [ylo - 0.4 * (k + 1)] * 2, color=col, lw=2.5, solid_capstyle="butt")
    ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="0.3", ls=":", lw=1)
    ax.set_ylabel("Δ coverage (pp)"); ax.text(0.01, 0.92, f"MS {s + 1}", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.spines[["top", "right"]].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc="upper center", ncol=3)   # above the panels, no overlap
axes[-1].set_xlabel("Time relative to stimulus onset (ms)")
fig.tight_layout(rect=(0, 0, 1, 0.965))
for ext in ("png", "svg"):
    fig.savefig(OUT / f"coverage_clusters_isolated_k{K}.{ext}", dpi=300 if ext == "png" else None)
plt.close(fig)


# ── Step 4: EXPLORATORY test of MS 1 in the second-peak window (formerly archive/c5_ms1_second_peak_isolated.py)
#    Specification fixed before running (2026-11-04), see docstring.
STATE, WIN = 0, (1.080, 1.380)
win = (times >= WIN[0]) & (times <= WIN[1])
assert list(main["pids"]) == list(PARTICIPANT_IDS)


def window_diff(subset):
    d = []
    for i, pid in enumerate(PARTICIPANT_IDS):
        if subset == "all_epochs":
            c = main["cov"][i][:, order]                      # (2, K, T)
        else:
            c = []
            for cond in ("sham", "stim"):
                f = np.load(CACHE / f"{subset}_{pid}_{cond}.npz")
                c.append(f["cov"][order] if f["cov"].size else np.full((K, len(times)), np.nan))
            c = np.array(c)
        d.append(np.nanmean(c[1, STATE, win]) - np.nanmean(c[0, STATE, win]))
    d = np.array(d); return d[np.isfinite(d)]


rng, rows = np.random.default_rng(SEED), []
for subset in ("all_epochs", "prev3.7", "prev3.7_next3.0"):
    d = window_diff(subset)
    flips = rng.choice([-1, 1], size=(N_PERM, len(d)))
    null  = (flips * d).mean(1)
    p1 = (np.sum(null >= d.mean()) + 1) / (N_PERM + 1)
    p2 = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / (N_PERM + 1)
    rows.append(dict(subset=subset, n=len(d), mean_diff_pp=100 * d.mean(), dz=d.mean() / d.std(ddof=1),
                     p_one_sided=p1, p_two_sided=p2, n_positive=int((d > 0).sum())))
res = pd.DataFrame(rows); res.to_csv(OUT / "ms1_second_peak_exploratory.csv", index=False)
logger.info(f"MS 1 coverage {int(WIN[0]*1000)}-{int(WIN[1]*1000)} ms, stim - sham (EXPLORATORY):\n" + res.to_string(index=False))
logger.info("done")
