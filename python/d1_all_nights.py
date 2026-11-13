"""
Supplementary analysis: all stimulation nights (E1, E2, E3) vs baseline (BL).

Replaces the old b_ scripts (archive/b_config / b_01 / b_02 / b_03), which still used
the old group assignment (study prefix), the old preprocessing (IIR filter,
3.5 s epochs, no boundary handling, peak-to-peak + drift rejection) and a
separate clustering.

Everything here uses the revised main pipeline:
  - groups from participants_master.csv (25 HC / 15 CI)
  - epochs made with process_one_night() of a1_epochs.py
    (FIR filter per segment, boundary epochs dropped, -1.5 to 3.0 s,
     +/-300 uV rejection for the same 7 flagged participants)
  - labelled with the SAME K=4 group maps as the main analysis (no new
    clustering), backfitting parameters as in a2_cluster_backfit.py
  - statistics with the functions of a3_stats.py (labels as in the manuscript,
    equal weighting, FDR per family x contrast, effect sizes)

Nights: BL = night 02 (sham), E1 = 03, E2 = 04, E3 = 05.
Existing epoch files of the main analysis are re-used (BL = sham files; the
main stim file is E1, or E2 for the participants in USE_NIGHT_E2).
Participants without a given night are skipped for that night.

Steps (each step is cached, so the script can be restarted):
  1. epochs for E1/E2/E3      -> epochs/filter_03_20_nights/{pid}_{night}-epo.fif
  2. backfitting              -> microstates/filter_03_20/7. all_nights/parameters_nights_k4.csv
  3. statistics per night     -> permutation_results_nights_k4.csv
"""

import gc
import importlib.util
import pickle
import mne
import numpy as np
import pandas as pd
from config import *

mne.set_log_level("ERROR")
logger = setup_logger("all_nights", "d1_all_nights.log")

K           = 4
NIGHTS      = {"E1": "03", "E2": "04", "E3": "05"}
MIN_EPOCHS_NIGHT = 50
EPOCH_OUT   = EPOCHS_DIR / f"{FILTER_TYPE}_nights"
BASE        = MICROSTATES_DIR / FILTER_TYPE
OUT_DIR     = BASE / "7. all_nights"
PARAMS_CSV  = OUT_DIR / f"parameters_nights_k{K}.csv"
EPOCH_OUT.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # __main__ guard: nothing runs
    return mod

epochs_mod  = _load_module("epochs_mod",  "a1_epochs.py")
backfit_mod = _load_module("backfit_mod", "a2_cluster_backfit.py")


def epoch_file(pid, night):
    """Path of the epoch file for one participant/night (re-using main files)."""
    main = EPOCHS_DIR / FILTER_TYPE
    if night == "BL":
        return main / f"{pid}_sham-epo.fif"
    main_stim_night = "E2" if pid in USE_NIGHT_E2 else "E1"
    if night == main_stim_night:
        return main / f"{pid}_stim-epo.fif"
    return EPOCH_OUT / f"{pid}_{night}-epo.fif"


# ── Step 1: epochs for the additional nights ──────────────────────────────────
logger.info("step 1: epochs for E1/E2/E3")
for i, pid in enumerate(PARTICIPANT_IDS, 1):
    for night in NIGHTS:
        path = epoch_file(pid, night)
        if path.exists() or (EPOCH_OUT / f"{pid}_{night}.missing").exists():
            continue
        logger.info(f"[{i}/{len(PARTICIPANT_IDS)}] {pid} {night}")
        epochs_mod.STIM_NIGHT = night            # 01 picks night 03/04/05 from this
        try:
            epochs_mod.process_one_night(pid, "stim", path, FILTER_TYPE)
        except FileNotFoundError:
            logger.info(f"  {pid} {night}: no recording, skipped")
            (EPOCH_OUT / f"{pid}_{night}.missing").touch()
        except Exception as exc:
            logger.error(f"  {pid} {night}: failed - {exc}")
        gc.collect()


# ── Step 2: backfitting with the main K=4 group model ─────────────────────────
if PARAMS_CSV.exists():
    logger.info(f"step 2: loading existing {PARAMS_CSV.name} (delete it to recompute)")
    df_all = pd.read_csv(PARAMS_CSV)
else:
    logger.info("step 2: backfitting with the main group model")
    with open(BASE / "2. group_models" / f"group_k{K}_model.pkl", "rb") as f:
        model = pickle.load(f)                   # original numbering
    rows = []
    for i, pid in enumerate(PARTICIPANT_IDS, 1):
        for night in ["BL"] + list(NIGHTS):
            path = epoch_file(pid, night)
            if not path.exists():
                continue
            ep = backfit_mod._load_fif_epochs(path)          # avg reference, drop Cz
            if len(ep) < MIN_EPOCHS_NIGHT:
                logger.warning(f"  {pid} {night}: {len(ep)} epochs (< {MIN_EPOCHS_NIGHT}), skipped")
                continue
            seg = model.predict(
                ep,
                factor             = BACKFIT_SMOOTHING_FACTOR,
                half_window_size   = BACKFIT_SMOOTHING_WINDOW,
                min_segment_length = BACKFIT_MIN_SEGMENT_SAMPLES,
                reject_edges       = BACKFIT_REJECT_EDGES,
            )
            params = seg.compute_parameters()
            row = dict(participant_id=pid,
                       group="healthy" if pid in HEALTHY_IDS else "impaired",
                       night=night, k=K, n_epochs=len(ep))
            for ms in model.cluster_names:
                for metric in ("timecov", "meandurs", "occurrences", "gev"):
                    row[f"{ms}_{metric}"] = params.get(f"{ms}_{metric}", np.nan)
            bins, n_bins = backfit_mod.compute_time_binned_coverage(
                backfit_mod._extract_labels(seg), ep.info["sfreq"],
                EPOCH_TMIN, COVERAGE_BIN_DURATION_S, K)
            row.update(bins)
            row["n_time_bins"] = n_bins
            rows.append(row)
            logger.info(f"[{i}/{len(PARTICIPANT_IDS)}] {pid} {night}: {len(ep)} epochs")
            del ep, seg
            gc.collect()
    df_all = pd.DataFrame(rows)
    df_all.to_csv(PARAMS_CSV, index=False)
    logger.info(f"saved {PARAMS_CSV.name} ({len(df_all)} rows)")

counts = (df_all.groupby(["night", "group"])["participant_id"].nunique()
          .unstack(fill_value=0))
logger.info("participants per night:\n" + counts.to_string())


# ── Step 3: statistics, each stimulation night vs BL ──────────────────────────
src = open("a3_stats.py", encoding="utf-8").read()
st  = {}
exec(compile(src[:src.index("\n# ── Run ──")], "a3_stats.py", "exec"), st)

rng   = np.random.default_rng(42)
scalar_cols = [f"{s}_{p}" for s in range(K) for p in ("timecov", "meandurs", "occurrences", "gev")]
results = []
for night in NIGHTS:
    pids = sorted(set(df_all.loc[df_all.night == "BL", "participant_id"])
                  & set(df_all.loc[df_all.night == night, "participant_id"]))
    df = df_all[df_all.participant_id.isin(pids) & df_all.night.isin(["BL", night])].copy()
    df["condition"] = np.where(df.night == "BL", "sham", "stim")
    df["w"] = 1.0                                          # equal weighting
    # fresh model each time: the relabelling reorders the model in place
    df, _ = st["match_labels_to_reference"](df, st["load_group_model"](FILTER_TYPE, K), K)
    n_bins   = int(df["n_time_bins"].iloc[0])
    bin_cols = [f"bin{b}_state{s}_cov" for b in range(n_bins) for s in range(K)]
    n_g = df.drop_duplicates("participant_id")["group"].value_counts().to_dict()
    logger.info(f"{night} vs BL: {len(pids)} participants {n_g}")
    res = pd.DataFrame(st["run_tests"](df, scalar_cols, "scalar", rng)
                       + st["run_tests"](df, bin_cols, "timecov_bin", rng))
    res.insert(0, "night", night)
    res["n_participants"] = len(pids)
    results.append(res)

results = pd.concat(results, ignore_index=True)
results["p_fdr"], results["sig_fdr"] = np.nan, False
for _, idx in results.groupby(["night", "family", "test"]).groups.items():
    is_sig, p_adj = st["fdr_correct"](results.loc[idx, "p_value"].values)
    results.loc[idx, "p_fdr"], results.loc[idx, "sig_fdr"] = p_adj, is_sig
results.to_csv(OUT_DIR / f"permutation_results_nights_k{K}.csv", index=False)


# ── Summary: key effects of the main analysis, per night ──────────────────────
def _label(col):
    if col.startswith("bin"):
        b, s = col.replace("bin", "").replace("_cov", "").split("_state")
        t0 = EPOCH_TMIN + int(b) * COVERAGE_BIN_DURATION_S
        return f"MS {int(s) + 1} cov {t0 * 1000:+.0f}..{(t0 + COVERAGE_BIN_DURATION_S) * 1000:+.0f} ms"
    s, p = col.split("_", 1)
    return f"MS {int(s) + 1} {p}"

key_cols = ["2_meandurs", "2_timecov", "3_timecov", "3_occurrences",
            "bin7_state2_cov", "bin7_state3_cov", "bin9_state2_cov"]
summ = results[results.col.isin(key_cols) & (results.test == "all_condition_effect")].copy()
summ["effect"] = summ["col"].map(_label)
summ["dz (p_fdr)"] = summ.apply(lambda r: f"{r.effect_size:+.2f} ({r.p_fdr:.3f})", axis=1)
table = summ.pivot(index="effect", columns="night", values="dz (p_fdr)")
logger.info("key effects, all participants, stim night vs BL: dz (p_fdr)\n" + table.to_string())

sig = results[results.sig_fdr & (results.test == "all_condition_effect")]
logger.info(f"FDR-significant condition effects (all participants): "
            f"{sig.groupby('night').size().to_dict()}")
logger.info("done")
