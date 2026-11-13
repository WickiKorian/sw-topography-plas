"""
Average-level (RAGU) control: stimulation effect separately for each experimental night.

Counterpart of d1_all_nights.py (single-trial level). For each night (E1, E2, E3),
per participant: average ERP of the baseline night (BL) and of that stimulation
night, exported as RAGU text files exactly like c4_ragu_overlap_subsets.py
(baseline and stimulation night both from the Python epochs;
Cz appended as zero channel, average reference, first 900 samples = -1500 to
2995 ms, 900 x 129, tab-delimited).

Epochs: the files of d1_all_nights.py (same preprocessing as the main analysis);
participants with < 50 epochs in a night are skipped for that night, as in d1
(n = 37 / 39 / 38).

Output: matlab_ragu/derived/RAGU_TXT_nights/<night>/{ID}_BL.txt, {ID}_STIM.txt
        matlab_ragu/derived/RAGU_TXT_nights/epoch_counts_by_night.csv
Then in MATLAB: c02_ragu_nights.m (copies the settings of the main TANOVA file,
keeps only the participants of that night) and run the TANOVA in RAGU.

Resumable: existing TXT files are skipped (the device shell has a time limit).
"""
import time
import mne
import numpy as np
import pandas as pd
from config import *

mne.set_log_level("ERROR")
logger = setup_logger("ragu_nights", "d6_ragu_nights.log")

NIGHTS           = ("E1", "E2", "E3")
MIN_EPOCHS_NIGHT = 50                                   # as in d1_all_nights.py
N_SAMPLES        = 900
EPOCHS_NIGHTS    = EPOCHS_DIR / f"{FILTER_TYPE}_nights"
REF_TXT          = RAGU_DIR / "derived" / "RAGU_TXT_RowsTime_Final"
OUT              = RAGU_DIR / "derived" / "RAGU_TXT_nights"
TIME_BUDGET_S    = 150
for n in NIGHTS + ("BL",): (OUT / n).mkdir(parents=True, exist_ok=True)


def epoch_file(pid, night):
    """Same file logic as d1_all_nights.py."""
    main = EPOCHS_DIR / FILTER_TYPE
    if night == "BL":
        return main / f"{pid}_sham-epo.fif"
    if night == ("E2" if pid in USE_NIGHT_E2 else "E1"):
        return main / f"{pid}_stim-epo.fif"
    return EPOCHS_NIGHTS / f"{pid}_{night}-epo.fif"


def erp_129(path):
    ep = mne.read_epochs(str(path), preload=False)
    assert len(ep.ch_names) == 128 and ep.ch_names[0] == "E1" and ep.ch_names[-1] == "E128"
    s = np.zeros((128, len(ep.times)))
    for start in range(0, len(ep), 300):                        # chunks: limits memory
        s += (ep[start:start + 300].get_data() * 1e6).sum(axis=0)
    erp = s / len(ep)
    erp = np.vstack([erp, np.zeros((1, erp.shape[1]))])        # + Cz (reference) = 129
    return (erp - erp.mean(axis=0))[:, :N_SAMPLES], len(ep)   # average reference


counts_file = OUT / "epoch_counts_by_night.csv"
counts = pd.read_csv(counts_file).to_dict("records") if counts_file.exists() else []
done = {(r["participant_id"], r["night"]) for r in counts}
t0 = time.time()
for pid in PARTICIPANT_IDS:
    for night in NIGHTS:
        if (pid, night) in done: continue
        if time.time() - t0 > TIME_BUDGET_S:
            pd.DataFrame(counts).to_csv(counts_file, index=False)
            logger.info("time budget reached - run again to continue"); raise SystemExit(0)
        f = epoch_file(pid, night)
        if not f.exists():
            counts.append(dict(participant_id=pid, night=night, n_epochs=0, included=False)); continue
        erp, n = erp_129(f)
        inc = n >= MIN_EPOCHS_NIGHT
        if inc:
            np.savetxt(OUT / night / f"{pid}_STIM.txt", erp.T, delimiter="\t", fmt="%.10g")
            bl = OUT / "BL" / f"{pid}_BL.txt"                    # baseline: same source (Python epochs)
            if not bl.exists():
                np.savetxt(bl, erp_129(epoch_file(pid, "BL"))[0].T, delimiter="\t", fmt="%.10g")
            np.savetxt(OUT / night / f"{pid}_BL.txt", np.loadtxt(bl), delimiter="\t", fmt="%.10g")
        counts.append(dict(participant_id=pid, night=night, n_epochs=n, included=inc,
                           group="HC" if pid in HEALTHY_IDS else "CI"))
        logger.info(f"{pid} {night}: {n} epochs" + ("" if inc else " (< 50, skipped)"))
pd.DataFrame(counts).to_csv(counts_file, index=False)

# ── Validation: the main stimulation night must reproduce the MATLAB export ───
r = []
for pid in PARTICIPANT_IDS:
    night = "E2" if pid in USE_NIGHT_E2 else "E1"
    a = OUT / night / f"{pid}_STIM.txt"
    if a.exists():
        r.append(np.corrcoef(np.loadtxt(a).ravel(), np.loadtxt(REF_TXT / f"{pid}_STIM.txt").ravel())[0, 1])
logger.info(f"validation main stim night vs MATLAB export: r median {np.median(r):.4f}, min {np.min(r):.4f} (n = {len(r)})")
c = pd.DataFrame(counts)
logger.info("participants per night: " + c[c.included].groupby("night").size().to_string().replace("\n", ", "))

# ── Step 2: results of the per-night TANOVAs (after RAGU, c02_ragu_nights.m) ──
#    PHitCount / PAUCCount: rows = between effect (1 none, 2 group),
#    columns = within effect (1 none, 2 condition); CritDuration in samples (5 ms)
from scipy.io import loadmat
res = []
for night in NIGHTS:
    f = RAGU_DIR / "Ragu_results" / f"TANOVA_-1500_3000ms_ConditionXGroup_{night}.mat"
    if not f.exists():
        logger.info(f"{night}: RAGU file not found - run c02_ragu_nights.m and the TANOVA"); continue
    rd = loadmat(str(f), squeeze_me=True, struct_as_record=False)["rd"]
    hit, auc, crit = (np.atleast_2d(np.asarray(getattr(rd, k, []), float)) for k in ("PHitCount", "PAUCCount", "CritDuration"))
    if hit.size < 8 or crit[0, 1] > 1e10:
        logger.info(f"{night}: global statistics not computed yet in RAGU"); continue
    res.append(dict(night=night, n=int(np.size(rd.IndFeature)),
                    p_count_condition=hit[0, 1], p_auc_condition=auc[0, 1], crit_ms_condition=crit[0, 1] * 5,
                    p_count_group=hit[1, 0], p_count_interaction=hit[1, 1], p_auc_interaction=auc[1, 1]))
if res:
    r = pd.DataFrame(res); r.to_csv(OUT / "tanova_nights_global.csv", index=False)
    logger.info("per-night TANOVA (global statistics):\n" + r.round(4).to_string(index=False))
logger.info("done")
