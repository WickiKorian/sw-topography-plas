"""
Overlap control analyses: how much of the stimulation effect could come from
NEIGHBOURING stimuli rather than from the current one?

Stimuli often follow each other within a few seconds (minimum interval 0.85 s,
see c1_stimulus_intervals.py). The response to a stimulus lasts about 2.2 s (end of
the TANOVA condition effect, 2140 ms). Therefore:

  - a PRECEDING stimulus can reach into the epoch (which starts at -1.5 s) if it
    occurred less than 1.5 + 2.2 = 3.7 s before the current stimulus;
    archive/06_sensitivity_isolated.py only excluded preceding stimuli within 1.5 s,
    i.e. their onset, not their late components (e.g. the second peak at ~1.2 s);
  - a FOLLOWING stimulus within the epoch (< 3.0 s after the current one) adds
    its own response to the post-stimulus part, including the second-peak window.

Variants (criteria fixed in advance from the response duration, not from results)
  prev1.5          no preceding stimulus within 1.5 s   (= 06, for reference)
  prev3.7          no preceding stimulus within 3.7 s   -> pre-stimulus window free
                                                           of any earlier response
  next3.0          no following stimulus within 3.0 s   -> post-stimulus window free
                                                           of later stimuli
  prev3.7_next3.0  both                                  -> fully isolated epochs

Interpretation
  - Pre-stimulus effects: if they reflect earlier stimuli, they should vanish
    with prev3.7 (not necessarily with prev1.5).
  - Post-stimulus effects: should persist with prev3.7.
  - next3.0: following stimuli could in principle be triggered by the induced
    slow wave (closed loop), which would make this subset biased against late
    responses. Tested in c1_stimulus_intervals.py: under stimulation, epochs
    were not more often followed by a stimulus within 3 s (62% vs 61%), but the
    following stimulus occurred ~120 ms earlier (median 1.37 vs 1.49 s). The
    subset is therefore not depleted of "successful" trials; rather, responses
    to following stimuli fall systematically into the late part of the epoch.
    Cluster-based follow-up: c3_coverage_clusters_isolated.py.

Same backfitting and statistics as 02/03/06 (equal weighting, FDR per family x
contrast). Participants with fewer than MIN_EPOCHS epochs in a night are
excluded from that variant; n is reported.

Output: microstates/filter_03_20/13. overlap_variants/<variant>/
        + summary_overlap_variants_k4.csv (condition effects, all participants)
"""

import gc
import importlib.util
import pickle
import mne
import numpy as np
import pandas as pd
from config import *

mne.set_log_level("ERROR")
logger = setup_logger("overlap_variants", "c2_overlap_variants.log")

OWN_MARKER_TOL = 0.02          # s, see 05/06 (epoch time vs. own marker rounding)
RESP_DUR       = 2.2           # s, duration of the stimulation response (TANOVA)
PRE_WIN        = -EPOCH_TMIN   # 1.5 s
POST_WIN       = 3.0           # s, epoch end
VARIANTS = {
    "prev1.5":         dict(prev=PRE_WIN,            next=None),
    "prev3.7":         dict(prev=PRE_WIN + RESP_DUR, next=None),
    "next3.0":         dict(prev=None,               next=POST_WIN),
    "prev3.7_next3.0": dict(prev=PRE_WIN + RESP_DUR, next=POST_WIN),
}
K          = 4
MIN_EPOCHS = 20
BASE       = MICROSTATES_DIR / FILTER_TYPE
ROOT       = BASE / "13. overlap_variants"
ROOT.mkdir(parents=True, exist_ok=True)


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

backfit_mod = _load_module("backfit_mod", "a2_cluster_backfit.py")

def _load_stats_functions():
    src = open("a3_stats.py", encoding="utf-8").read()
    ns  = {}
    exec(compile(src[:src.index("\n# ── Run ──")], "a3_stats.py", "exec"), ns)
    return ns


def _raw_file(pid, condition):
    folder = DATA_ROOT / pid[:5] / EEG_SUBFOLDER
    night  = "02" if condition == "sham" else (
        "04" if (STIM_NIGHT == "per_participant" and pid in USE_NIGHT_E2) else
        {"E1": "03", "E2": "04", "E3": "05"}.get(STIM_NIGHT, "03"))
    files = list(folder.glob(f"{pid}-{night}*_scored_clean_PREP_AutoReject.set"))
    if len(files) != 1:
        raise FileNotFoundError(f"{pid} {condition}: {len(files)} files found")
    return files[0]


def neighbour_intervals(pid, condition, ep):
    """Time to the preceding and to the following marker for every kept epoch."""
    marker = ("DI64" if pid in SHAM_USES_DI64 else "D128") if condition == "sham" else "DI64"
    ann    = mne.read_annotations(str(_raw_file(pid, condition)))
    on     = np.sort(ann.onset[ann.description == marker])
    t      = ep.events[:, 0] / ep.info["sfreq"]
    i      = np.searchsorted(on, t - OWN_MARKER_TOL)
    prev   = t - np.where(i > 0, on[np.maximum(i - 1, 0)], -np.inf)
    j      = np.searchsorted(on, t + OWN_MARKER_TOL)
    nxt    = np.where(j < len(on), on[np.minimum(j, len(on) - 1)], np.inf) - t
    return prev, nxt


def keep_mask(prev, nxt, crit):
    m = np.ones(len(prev), bool)
    if crit["prev"] is not None: m &= prev > crit["prev"]
    if crit["next"] is not None: m &= nxt > crit["next"]
    return m


# ── Step 1-2: backfit every variant (one pass over the data) ──────────────────
params_csv = {v: ROOT / v / f"parameters_k{K}.csv" for v in VARIANTS}
todo = [v for v in VARIANTS if not params_csv[v].exists()]
if todo:
    with open(BASE / "2. group_models" / f"group_k{K}_model.pkl", "rb") as f:
        model = pickle.load(f)
    rows = {v: [] for v in todo}
    for i, pid in enumerate(PARTICIPANT_IDS, 1):
        for condition in ("sham", "stim"):
            ep_all = mne.read_epochs(str(EPOCHS_DIR / FILTER_TYPE / f"{pid}_{condition}-epo.fif"),
                                     preload=False, verbose=False)
            prev, nxt = neighbour_intervals(pid, condition, ep_all)
            msg = []
            for v in todo:
                keep = np.where(keep_mask(prev, nxt, VARIANTS[v]))[0]
                msg.append(f"{v} {len(keep)}")
                if len(keep) < MIN_EPOCHS:
                    continue
                ep = ep_all[keep].load_data()
                ep.set_eeg_reference("average", projection=False)
                if "Cz" in ep.ch_names:
                    ep.drop_channels(["Cz"])
                seg = model.predict(ep, factor=BACKFIT_SMOOTHING_FACTOR,
                                    half_window_size=BACKFIT_SMOOTHING_WINDOW,
                                    min_segment_length=BACKFIT_MIN_SEGMENT_SAMPLES,
                                    reject_edges=BACKFIT_REJECT_EDGES)
                params = seg.compute_parameters()
                row = dict(participant_id=pid, group="healthy" if pid in HEALTHY_IDS else "impaired",
                           condition=condition, k=K, n_epochs=len(ep), n_epochs_all=len(ep_all))
                for ms in model.cluster_names:
                    for metric in ("timecov", "meandurs", "occurrences", "gev"):
                        row[f"{ms}_{metric}"] = params.get(f"{ms}_{metric}", np.nan)
                labels = backfit_mod._extract_labels(seg)
                bins, n_bins = backfit_mod.compute_time_binned_coverage(
                    labels, ep.info["sfreq"], EPOCH_TMIN, COVERAGE_BIN_DURATION_S, K)
                row.update(bins); row["n_time_bins"] = n_bins
                rows[v].append(row)
                del ep, seg, labels; gc.collect()
            logger.info(f"[{i}/{len(PARTICIPANT_IDS)}] {pid} {condition} (of {len(ep_all)} epochs): " + ", ".join(msg))
            del ep_all; gc.collect()
    for v in todo:
        (ROOT / v).mkdir(exist_ok=True)
        pd.DataFrame(rows[v]).to_csv(params_csv[v], index=False)


# ── Step 3: statistics per variant ────────────────────────────────────────────
st   = _load_stats_functions()
main = pd.read_csv(BASE / "4. statistics" / f"k{K}" / f"permutation_results_k{K}.csv")
summary = main[main.test == "all_condition_effect"][["col", "effect_size", "p_fdr"]].rename(
    columns={"effect_size": "dz_all_epochs", "p_fdr": "pfdr_all_epochs"})

for v in VARIANTS:
    df = pd.read_csv(params_csv[v])
    complete = df.groupby("participant_id")["condition"].nunique() == 2
    df = df[df["participant_id"].isin(complete[complete].index)].copy()
    n_groups = df.drop_duplicates("participant_id")["group"].value_counts().to_dict()
    ep_share = (df.n_epochs / df.n_epochs_all).groupby(df.condition).mean() * 100
    logger.info(f"{v}: participants {n_groups}; epochs kept: sham {ep_share.get('sham', np.nan):.0f}%, "
                f"stim {ep_share.get('stim', np.nan):.0f}%")
    df["w"] = 1.0
    gm = st["load_group_model"](FILTER_TYPE, K)   # fresh copy: match_labels_to_reference reorders the model in place
    df, _ = st["match_labels_to_reference"](df, gm, K)
    rng = np.random.default_rng(42)
    scalar_cols = [f"{s}_{p}" for s in range(K) for p in ("timecov", "meandurs", "occurrences", "gev")]
    n_bins   = int(df["n_time_bins"].iloc[0])
    bin_cols = [f"bin{b}_state{s}_cov" for b in range(n_bins) for s in range(K)]
    res = pd.DataFrame(st["run_tests"](df, scalar_cols, "scalar", rng)
                       + st["run_tests"](df, bin_cols, "timecov_bin", rng))
    res["p_fdr"], res["sig_fdr"] = np.nan, False
    for _, idx in res.groupby(["family", "test"]).groups.items():
        is_sig, p_adj = st["fdr_correct"](res.loc[idx, "p_value"].values)
        res.loc[idx, "p_fdr"], res.loc[idx, "sig_fdr"] = p_adj, is_sig
    res["n_participants"] = sum(n_groups.values())
    res.to_csv(ROOT / v / f"permutation_results_k{K}.csv", index=False)
    a = res[res.test == "all_condition_effect"][["col", "effect_size", "p_fdr"]]
    summary = summary.merge(a.rename(columns={"effect_size": f"dz_{v}", "p_fdr": f"pfdr_{v}"}), on="col", how="left")

summary.to_csv(ROOT / f"summary_overlap_variants_k{K}.csv", index=False)


def _label(col):
    if col.startswith("bin"):
        b, s = col.replace("bin", "").replace("_cov", "").split("_state")
        t0 = EPOCH_TMIN + int(b) * COVERAGE_BIN_DURATION_S
        return f"MS {int(s) + 1} {t0 * 1000:+.0f}..{(t0 + COVERAGE_BIN_DURATION_S) * 1000:+.0f}"
    s, p = col.split("_", 1)
    return f"MS {int(s) + 1} {p}"

sig_any = summary[[c for c in summary.columns if c.startswith("pfdr_")]].lt(0.05).any(axis=1)
show = summary[sig_any].copy()
show.insert(0, "effect", show.col.map(_label))
logger.info("condition effect (all participants): dz / p_FDR per variant\n"
            + show.drop(columns="col").round(2).to_string(index=False))
logger.info("done")
