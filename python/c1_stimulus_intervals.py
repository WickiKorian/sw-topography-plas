"""
Timing of neighbouring stimuli (overlap of responses within an epoch).

Part A  Time to the PRECEDING stimulus (formerly archive/c1_overlap_check.py)
  Overlap check: how often does an earlier stimulus (or sham marker) fall
  inside the pre-stimulus part of an epoch (-1.5 to 0 s)?

  For every epoch that was kept in a1_epochs.py, the time since the previous
  marker of the same type is computed. ALL markers of the night are used,
  including those whose epoch was dropped (boundary / amplitude), because a
  stimulus was delivered regardless of whether its epoch was kept.

  Output: microstates/filter_03_20/5. overlap_check/
    overlap_per_participant.csv   one row per participant x condition
    overlap_isi_hist.png          distribution of time since previous marker

Part B  Time to the FOLLOWING stimulus (formerly archive/c4_next_stimulus_timing.py)
  Are following stimuli triggered by the slow wave that the current stimulus induced?

  c2_overlap_variants.py originally assumed so (closed loop -> excluding following stimuli would remove
  successful trials and bias the fully isolated subset against late effects).
  This was never tested. Test, from stimulus markers only (no backfitting):

  If stimuli targeted the induced slow wave, the time from an analysed stimulus to
  the NEXT marker would show an extra peak at the latency of the induced wave
  (about 0.85-1.6 s; second peak 1080-1380 ms) in the stimulation night, but not
  in the sham night, where markers only follow natural slow waves.

  Per participant and night:
    share_early = n(next in 0.85-1.6 s) / n(next within 3.0 s)
    pct_next3   = % of analysed epochs with a following marker within 3.0 s
  Paired comparison stim vs sham (Wilcoxon signed-rank, dz), all participants.
  Histogram of the time to the next marker (50-ms bins, % of epochs with a
  following marker within 3 s), stim vs sham.

  Output: microstates/filter_03_20/18. next_stimulus_timing/
"""


import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from config import *
import importlib.util
from scipy.stats import wilcoxon


mne.set_log_level("ERROR")
logger = setup_logger("stimulus_intervals", "c1_stimulus_intervals.log")

# Tolerance for "strictly before": the epoch time is the marker time rounded to
# the 200-Hz sample grid, so it can lie up to ~3 ms AFTER its own marker. With a
# 1-ms tolerance (previous version) about 30% of epochs counted their own marker
# as the "previous" one. True inter-stimulus intervals are >= 0.85 s, so 20 ms
# safely excludes the own marker without missing any real preceding stimulus.
OWN_MARKER_TOL = 0.02   # s

OUT_DIR = MICROSTATES_DIR / FILTER_TYPE / "5. overlap_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _raw_file(pid, condition):
    folder = DATA_ROOT / pid[:5] / EEG_SUBFOLDER
    night  = "02" if condition == "sham" else (
        "04" if (STIM_NIGHT == "per_participant" and pid in USE_NIGHT_E2) else
        {"E1": "03", "E2": "04", "E3": "05"}.get(STIM_NIGHT, "03"))
    files = list(folder.glob(f"{pid}-{night}*_scored_clean_PREP_AutoReject.set"))
    if len(files) != 1:
        raise FileNotFoundError(f"{pid} {condition}: {len(files)} files found")
    return files[0]


STAGES = ["N1", "N2", "N3", "R", "W"]
rows, all_isi = [], {"sham": [], "stim": []}
for i, pid in enumerate(PARTICIPANT_IDS, 1):
    group = "HC" if pid in HEALTHY_IDS else "CI"
    for condition in ("sham", "stim"):
        marker = ("DI64" if pid in SHAM_USES_DI64 else "D128") if condition == "sham" else "DI64"
        cache = OUT_DIR / "cache" / f"{pid}_{condition}.npz"          # resumable: one file per recording
        if cache.exists():
            c = np.load(cache, allow_pickle=True)
            all_isi[condition].append(c["isi"]); rows.append(c["row"].item()); continue

        # all markers of the night (annotations only, no EEG data loaded)
        ann     = mne.read_annotations(str(_raw_file(pid, condition)))
        onsets  = np.sort(ann.onset[ann.description == marker])

        # sleep stage of every marker (last scoring annotation before the marker)
        is_stage = np.isin(ann.description, STAGES)
        st_on, st_lab = ann.onset[is_stage], ann.description[is_stage]
        order = np.argsort(st_on); st_on, st_lab = st_on[order], st_lab[order]
        j = np.searchsorted(st_on, onsets, side="right") - 1
        stage = np.where(j >= 0, st_lab[np.maximum(j, 0)], "unscored")
        stage_pct = {f"pct_markers_{s}": 100 * np.mean(stage == s) for s in STAGES + ["unscored"]}

        # times of the epochs that were kept
        ep      = mne.read_epochs(str(EPOCHS_DIR / FILTER_TYPE / f"{pid}_{condition}-epo.fif"),
                                  preload=False, verbose=False)
        t_ep    = ep.events[:, 0] / ep.info["sfreq"]     # EEGLAB: first_samp = 0

        # time since the previous marker for every kept epoch
        idx     = np.searchsorted(onsets, t_ep - OWN_MARKER_TOL)   # markers strictly before (own marker excluded)
        prev    = np.where(idx > 0, onsets[np.maximum(idx - 1, 0)], -np.inf)
        isi     = t_ep - prev
        all_isi[condition].append(isi[np.isfinite(isi)])

        in_window = isi <= -EPOCH_TMIN                    # previous marker within 1.5 s
        rows.append(dict(
            participant_id = pid, group = group, condition = condition,
            n_markers      = len(onsets),
            n_epochs       = len(t_ep),
            pct_prev_in_window = 100 * in_window.mean(),
            median_isi_s   = float(np.median(isi[np.isfinite(isi)])),
            pct_prev_lt_1s = 100 * (isi <= 1.0).mean(),
            min_isi_s      = float(np.min(isi[np.isfinite(isi)])),
            **stage_pct,
        ))
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, isi=isi[np.isfinite(isi)], row=np.array(rows[-1], dtype=object))
        logger.info(f"[{i}/{len(PARTICIPANT_IDS)}] {pid} {condition}: "
                    f"{len(t_ep)} epochs, {100 * in_window.mean():.1f}% with previous "
                    f"{marker} within 1.5 s (median ISI {rows[-1]['median_isi_s']:.2f} s)")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "overlap_per_participant.csv", index=False)

summary = df.groupby(["condition", "group"])[["pct_prev_in_window", "median_isi_s"]].agg(
    ["mean", "min", "max"]).round(2)
logger.info("summary (mean / min / max across participants):\n" + summary.to_string())
stage_cols = [c for c in df.columns if c.startswith("pct_markers_")]
logger.info("sleep stage of markers (mean % across participants):\n"
            + df.groupby("condition")[stage_cols].mean().round(1).to_string())
logger.info("minimum interval between markers (s): "
            + df.groupby("condition").min_isi_s.min().round(3).to_string())

# histogram of time since previous marker (pooled epochs) -> Supplementary Figure S3A
np.savez(OUT_DIR / "isi_pooled.npz", **{c: np.concatenate(v) for c, v in all_isi.items()})
fig, ax = plt.subplots(figsize=(7, 3.5))
bins = np.arange(0, 10.5, 0.25)                       # last bar (10-10.25) collects all intervals > 10 s
for condition, color, lab in (("sham", "0.25", "Baseline (sham)"), ("stim", "#E63300", "Stimulation")):
    x = np.concatenate(all_isi[condition])
    ax.hist(np.clip(x, 0, 10.1), bins=bins, histtype="step", lw=2, color=color,
            weights=np.full(x.size, 1 / x.size), label=lab)   # proportion of epochs per 0.25-s bar
ymax = ax.get_ylim()[1]
for xv, txt in ((-EPOCH_TMIN, "epoch start\n(1.5 s)"), (3.7, "no preceding\nstimulus (3.7 s)")):
    ax.axvline(xv, color="black", ls=":", lw=1)
    ax.text(xv + 0.1, ymax * 0.82, txt, fontsize=8, va="top")
ax.set_xticks([0, 2, 4, 6, 8, 10.125]); ax.set_xticklabels(["0", "2", "4", "6", "8", "> 10"])
ax.set_xlabel("Time since preceding (pseudo-)stimulus (s)")
ax.set_ylabel("Proportion of epochs")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "overlap_isi_hist.png", dpi=300)


# ══ Part B: time to the following stimulus ═══════════════════════════════════════
BASE  = MICROSTATES_DIR / FILTER_TYPE
OUT   = BASE / "18. next_stimulus_timing"; OUT.mkdir(parents=True, exist_ok=True)
EARLY = (0.85, 1.6)      # s, latency of the induced slow wave / second peak
POST  = 3.0              # s, epoch end
BINS  = np.arange(0.8, POST + 0.05, 0.05)

# same marker / interval code as c2_overlap_variants.py
spec = importlib.util.spec_from_file_location("ov", "c2_overlap_variants.py")
src  = open("c2_overlap_variants.py", encoding="utf-8").read()
src  = src[:src.index("# ── Step 1-2")]                       # functions only, no analysis
ns = {}; exec(compile(src, "c2_overlap_variants.py", "exec"), ns)
neighbour_intervals = ns["neighbour_intervals"]

rows, nxt_all = [], {"sham": [], "stim": []}
for pid in PARTICIPANT_IDS:
    for cond in ("sham", "stim"):
        cache = OUT / "cache" / f"{pid}_{cond}_next.npy"          # reading the raw annotations is slow
        if cache.exists():
            nxt = np.load(cache)
        else:
            ep = mne.read_epochs(str(EPOCHS_DIR / FILTER_TYPE / f"{pid}_{cond}-epo.fif"), preload=False, verbose=False)
            _, nxt = neighbour_intervals(pid, cond, ep)
            cache.parent.mkdir(exist_ok=True); np.save(cache, nxt)
        n3 = (nxt <= POST).sum()
        ne = ((nxt >= EARLY[0]) & (nxt <= EARLY[1])).sum()
        rows.append(dict(participant_id=pid, condition=cond, n_epochs=len(nxt), n_next3=int(n3), n_next_early=int(ne),
                         pct_next3=100 * n3 / len(nxt), share_early=ne / n3 if n3 else np.nan,
                         median_next_s=float(np.median(nxt[nxt <= POST])) if n3 else np.nan))
        nxt_all[cond].append(nxt[nxt <= POST])
    logger.info(f"{pid} done")

df = pd.DataFrame(rows); df.to_csv(OUT / "next_stimulus_per_participant.csv", index=False)
w = df.pivot(index="participant_id", columns="condition")
res = []
for m in ("share_early", "pct_next3", "median_next_s"):
    d = (w[(m, "stim")] - w[(m, "sham")]).dropna()
    st = wilcoxon(d)
    res.append(dict(measure=m, sham_mean=w[(m, "sham")].mean(), stim_mean=w[(m, "stim")].mean(),
                    dz=d.mean() / d.std(ddof=1), p_wilcoxon=st.pvalue, n=len(d)))
res = pd.DataFrame(res); res.to_csv(OUT / "next_stimulus_stim_vs_sham.csv", index=False)
logger.info("\n" + res.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.5))
for cond, col in (("sham", "0.25"), ("stim", "#E63300")):
    x = np.concatenate(nxt_all[cond]); h, _ = np.histogram(x, BINS)
    ax.step(BINS[:-1], 100 * h / len(x), where="post", color=col, lw=1.8, label=cond)
ax.axvspan(1.08, 1.38, color="0.9", zorder=0, label="second-peak window")
ax.set_xlabel("Time from analysed stimulus to next marker (s)"); ax.set_ylabel("% of following markers (≤ 3 s)")
ax.legend(frameon=False); [ax.spines[s].set_visible(False) for s in ("top", "right")]
fig.tight_layout(); fig.savefig(OUT / "next_stimulus_hist.png", dpi=200); plt.close(fig)
logger.info(f"saved {OUT}")
logger.info("done")
