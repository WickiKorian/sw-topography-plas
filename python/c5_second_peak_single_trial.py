"""
Single-trial frequency of the second induced peak (Reviewer question:
"how frequently is the second induced peak seen; is it consistent on every
trial or a larger deflection seen on only a few trials?").

Approach
  1. Window: the latency of the second peak is taken from the grand average
     COLLAPSED over conditions and groups (stim + sham, HC + CI), so the window
     is not biased towards the stim-vs-sham difference ("collapsed localizer").
     Window = latency of the positive maximum between 0.8 and 1.9 s, +/- 150 ms.
  2. Per trial: mean amplitude in that window over a fronto-central ROI
     (average reference, same epochs as the main analysis).
  3. Per participant, the sham trials define the null distribution (no second
     peak in sham). A stim trial counts as showing a second peak if its
     amplitude exceeds the 95th percentile of that participant's sham trials
     (5% expected by chance).
  4. Consistency: percentage of stim trials above threshold, and the shape of
     the distribution - a consistent response shifts the whole distribution
     (median shifts, skew similar to sham); a response carried by a few large
     deflections leaves the median and adds a heavy right tail (skew increases).
  Sensitivity (overlap with neighbouring stimuli, see c2_overlap_variants.py):
    prev1.5  no preceding stimulus within 1.5 s
    prev3.7  no preceding stimulus within 3.7 s (1.5 s + 2.2 s response duration)
    next3.0  no following stimulus within the epoch (see c1_stimulus_intervals.py)
    prev3.7_next3.0  both
  Also writes the grand-average ROI time course (stim vs sham) per subset.

Output: microstates/filter_03_20/10. second_peak/second_peak_single_trial.csv

Can be run in several calls: per-recording results are cached in
10. second_peak/cache, and the environment variable C5_BUDGET_S sets the time
budget per call in seconds.
"""

import os
import mne
import numpy as np
import pandas as pd
from scipy import stats
from config import *

mne.set_log_level("ERROR")
logger = setup_logger("second_peak", "c5_second_peak_single_trial.log")

ROI        = ["E5", "E6", "E7", "E11", "E12", "E106"]     # around Fz / FCz
SEARCH     = (0.8, 1.9)          # where the second peak is searched (s)
HALF_WIN   = 0.15                # +/- around the collapsed-average latency (s)
EDIR       = EPOCHS_DIR / FILTER_TYPE
OUT_DIR    = MICROSTATES_DIR / FILTER_TYPE / "10. second_peak"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _raw_file(pid, condition):
    folder = DATA_ROOT / pid[:5] / EEG_SUBFOLDER
    night  = "02" if condition == "sham" else ("04" if pid in USE_NIGHT_E2 else "03")
    files  = list(folder.glob(f"{pid}-{night}*_scored_clean_PREP_AutoReject.set"))
    return files[0] if len(files) == 1 else None


def load(pid, condition):
    # average reference (as in 02) computed in chunks of epochs to limit memory
    ep  = mne.read_epochs(str(EDIR / f"{pid}_{condition}-epo.fif"), preload=False)
    idx = [ep.ch_names.index(ch) for ch in ROI]
    roi = []
    for start in range(0, len(ep), 300):
        x = ep[start:start + 300].get_data()                     # (n, ch, t), loads chunk
        x = x - x.mean(axis=1, keepdims=True)                     # average reference
        roi.append(x[:, idx].mean(axis=1))
    roi = np.concatenate(roi) * 1e6                               # (n_trials, n_times) µV
    # time to the preceding / following marker (own marker excluded, 20-ms tolerance)
    prev = np.full(len(ep), np.inf); nxt = np.full(len(ep), np.inf)
    f = _raw_file(pid, condition)
    if f is not None:
        marker = ("DI64" if pid in SHAM_USES_DI64 else "D128") if condition == "sham" else "DI64"
        ann    = mne.read_annotations(str(f))
        on     = np.sort(ann.onset[ann.description == marker])
        t_ep   = ep.events[:, 0] / ep.info["sfreq"]
        i      = np.searchsorted(on, t_ep - 0.02)
        prev   = t_ep - np.where(i > 0, on[np.maximum(i - 1, 0)], -np.inf)
        j      = np.searchsorted(on, t_ep + 0.02)
        nxt    = np.where(j < len(on), on[np.minimum(j, len(on) - 1)], np.inf) - t_ep
    else:
        logger.warning(f"{pid} {condition}: raw file not found, neighbour intervals unknown")
    return roi, ep.times, (prev, nxt)


# ── Load all ROI time courses once (cached per recording; resumable) ──────────
import time
CACHE = OUT_DIR / "cache"; CACHE.mkdir(exist_ok=True)
T0, BUDGET = time.time(), float(os.environ.get("C5_BUDGET_S", "inf"))   # optional time budget per call
data, times = {}, None
for i, pid in enumerate(PARTICIPANT_IDS, 1):
    for cond in ("sham", "stim"):
        f = CACHE / f"{pid}_{cond}.npz"
        if not f.exists():
            if time.time() - T0 > BUDGET:
                logger.info("time budget reached - run again to continue"); raise SystemExit
            roi, t, (prev, nxt) = load(pid, cond)
            np.savez(f, roi=roi, times=t, prev=prev, nxt=nxt)
        d = np.load(f)
        data[(pid, cond)] = (d["roi"], d["times"], (d["prev"], d["nxt"]))
        times = d["times"]
    logger.info(f"[{i}/{len(PARTICIPANT_IDS)}] {pid} loaded")

# ── 1. Window from the collapsed grand average ────────────────────────────────
collapsed = np.mean([data[k][0].mean(axis=0) for k in data], axis=0)   # equal weight per night
s = (times >= SEARCH[0]) & (times <= SEARCH[1])
t_peak = times[s][np.argmax(collapsed[s])]
TWIN   = (t_peak - HALF_WIN, t_peak + HALF_WIN)
w      = (times >= TWIN[0]) & (times <= TWIN[1])
logger.info(f"collapsed grand average: second-peak latency {t_peak * 1000:.0f} ms, "
            f"amplitude {collapsed[s].max():+.2f} µV -> window "
            f"{TWIN[0] * 1000:.0f} to {TWIN[1] * 1000:.0f} ms")

# ── 2-4. Per participant ──────────────────────────────────────────────────────
SUBSETS = {"all": (None, None), "prev1.5": (1.5, None), "prev3.7": (3.7, None),
           "next3.0": (None, 3.0), "prev3.7_next3.0": (3.7, 3.0)}
def subset_mask(nb, subset):
    prev, nxt = nb; p, n = SUBSETS[subset]
    m = np.ones(len(prev), bool)
    if p is not None: m &= prev > p
    if n is not None: m &= nxt > n
    return m
erp  = {}
rows = []
for pid in PARTICIPANT_IDS:
    sham_x, _, sham_nb = data[(pid, "sham")]
    stim_x, _, stim_nb = data[(pid, "stim")]
    for subset in SUBSETS:
        ms, mt = subset_mask(sham_nb, subset), subset_mask(stim_nb, subset)
        erp[(subset, pid, "sham")] = sham_x[ms].mean(axis=0) if ms.sum() else None
        erp[(subset, pid, "stim")] = stim_x[mt].mean(axis=0) if mt.sum() else None
        sham = sham_x[ms][:, w].mean(axis=1)
        stim = stim_x[mt][:, w].mean(axis=1)
        if len(sham) < 20 or len(stim) < 20:
            continue
        thr = np.percentile(sham, 95)
        rows.append(dict(
            participant_id=pid, group="HC" if pid in HEALTHY_IDS else "CI", epochs=subset,
            n_sham=len(sham), n_stim=len(stim),
            pct_stim_above=100 * np.mean(stim > thr),
            pct_stim_above_sham_median=100 * np.mean(stim > np.median(sham)),
            median_sham=np.median(sham), median_stim=np.median(stim),
            skew_sham=stats.skew(sham), skew_stim=stats.skew(stim),
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "second_peak_single_trial.csv", index=False)

for subset in SUBSETS:
    d = df[df.epochs == subset]
    logger.info(f"-- {subset} epochs ({d.participant_id.nunique()} participants)")
    for g in ("HC", "CI", None):
        x = d if g is None else d[d.group == g]
        label = g or "all"
        p_pct  = stats.wilcoxon(x.pct_stim_above - 5).pvalue
        p_med  = stats.wilcoxon(x.median_stim, x.median_sham).pvalue
        p_skew = stats.wilcoxon(x.skew_stim, x.skew_sham).pvalue
        logger.info(
            f"  {label:>3} (n={len(x)}): stim trials above sham 95th pct "
            f"{x.pct_stim_above.mean():.1f}% ± {x.pct_stim_above.std():.1f} "
            f"(median {x.pct_stim_above.median():.1f}%; vs 5%: p={p_pct:.4f}); "
            f"above sham median {x.pct_stim_above_sham_median.mean():.1f}% (50% by chance); "
            f"median amplitude sham {x.median_sham.mean():+.2f} -> stim {x.median_stim.mean():+.2f} µV "
            f"(p={p_med:.4f}); skew sham {x.skew_sham.mean():+.2f} -> stim {x.skew_stim.mean():+.2f} "
            f"(p={p_skew:.4f})")
    n_above = (d.pct_stim_above > 5).sum()
    logger.info(f"  participants with > 5% stim trials above threshold: {n_above}/{len(d)}")

# ── Grand-average ROI time course per subset (stim vs sham) ───────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(SUBSETS), 1, figsize=(6.5, 1.7 * len(SUBSETS)), sharex=True, sharey=True)
TITLE = {"all": "All epochs", "prev1.5": "No preceding stimulus within 1.5 s", "prev3.7": "No preceding stimulus within 3.7 s",
         "next3.0": "No following stimulus within 3 s", "prev3.7_next3.0": "Fully isolated"}
for ax, subset in zip(axes, SUBSETS):
    for cond, col in (("sham", "0.25"), ("stim", "#E63300")):
        x = [erp[(subset, p, cond)] for p in PARTICIPANT_IDS if erp.get((subset, p, cond)) is not None]
        ax.plot(times, np.mean(x, axis=0), color=col, lw=1.3, label=f"{'Baseline' if cond == 'sham' else 'Stimulation'} (n = {len(x)})")
    ax.axvspan(*TWIN, color="0.9", zorder=0); ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_title(TITLE[subset], fontsize=9, loc="left"); ax.set_ylabel("µV", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
axes[-1].set_xlabel("Time relative to stimulus onset (s)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"second_peak_roi_by_overlap_subset.{ext}", dpi=300)
plt.close(fig)
logger.info("done")
