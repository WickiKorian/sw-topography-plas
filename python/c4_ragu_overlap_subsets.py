"""
Average-level (RAGU) check of the overlap with neighbouring stimuli.

Question: is the second induced peak (and the topographic condition effect)
present when the ERP is computed only from epochs WITHOUT neighbouring stimuli,
or is it partly produced by overlapping responses to preceding / following
stimuli?

Subsets (same criteria as c2_overlap_variants.py, fixed in advance):
  all              all epochs (Python reference; validated against the MATLAB export)
  prev3.7          no preceding stimulus within 3.7 s (1.5 s pre-window + 2.2 s response)
  next3.0          no following stimulus within the epoch (3.0 s)
                   (see c1_stimulus_intervals.py: not depleted of "successful" trials)
  prev3.7_next3.0  both (fully isolated)

Step 1  Per participant x night x subset: average ERP, Cz (reference) appended as
        zero channel, average reference, first 900 samples (-1500 to 2995 ms),
        written as tab-delimited TXT (900 time points x 129 channels) exactly
        like a02_export_ragu.m -> matlab_ragu/derived/RAGU_TXT_overlap/<subset>/
        File names as in RAGU_TXT_RowsTime_Final ({ID}_BL.txt / {ID}_STIM.txt),
        so the same group file (RAGU_groups.xlsx) can be used in RAGU.
Step 2  Validation: 'all' vs. the MATLAB export (spatio-temporal correlation per file).
Step 3  Quick topographic check in Python: grand-average STIM map in the second-peak
        window (1080-1380 ms) and in the first-peak window (0-300 ms) of each subset,
        spatial correlation with the 'all' map; fronto-central ROI amplitude
        stim - sham in the second-peak window (Wilcoxon across participants).

Then in RAGU (per subset folder): TANOVA condition x group, -1500 to 3000 ms,
5000 randomizations, as for the main analysis.
"""
import mne
import numpy as np
import pandas as pd
from scipy import stats
from config import *

mne.set_log_level("ERROR")
logger = setup_logger("ragu_overlap", "c4_ragu_overlap_subsets.log")

OWN_MARKER_TOL = 0.02
SUBSETS = {"all": (None, None), "prev3.7": (3.7, None), "next3.0": (None, 3.0), "prev3.7_next3.0": (3.7, 3.0)}
N_SAMPLES = 900
RAGU_DERIVED = RAGU_DIR / "derived"
REF_TXT   = RAGU_DERIVED / "RAGU_TXT_RowsTime_Final"
OUT       = RAGU_DERIVED / "RAGU_TXT_overlap"
ROI       = ["E5", "E6", "E7", "E11", "E12", "E106"]
WIN_P2    = (1.080, 1.380)            # second-peak window (collapsed localizer, c5_second_peak_single_trial.py)
WIN_P1    = (0.000, 0.300)            # first (targeted) peak
TAG       = {"sham": "BL", "stim": "STIM"}
for s in SUBSETS: (OUT / s).mkdir(parents=True, exist_ok=True)


def _raw_file(pid, condition):
    folder = DATA_ROOT / pid[:5] / EEG_SUBFOLDER
    night  = "02" if condition == "sham" else ("04" if pid in USE_NIGHT_E2 else "03")
    files  = list(folder.glob(f"{pid}-{night}*_scored_clean_PREP_AutoReject.set"))
    if len(files) != 1:
        raise FileNotFoundError(f"{pid} {condition}: {len(files)} files found")
    return files[0]


def neighbours(pid, condition, ep):
    marker = ("DI64" if pid in SHAM_USES_DI64 else "D128") if condition == "sham" else "DI64"
    ann = mne.read_annotations(str(_raw_file(pid, condition)))
    on  = np.sort(ann.onset[ann.description == marker])
    t   = ep.events[:, 0] / ep.info["sfreq"]
    i   = np.searchsorted(on, t - OWN_MARKER_TOL)
    j   = np.searchsorted(on, t + OWN_MARKER_TOL)
    prev = t - np.where(i > 0, on[np.maximum(i - 1, 0)], -np.inf)
    nxt  = np.where(j < len(on), on[np.minimum(j, len(on) - 1)], np.inf) - t
    return prev, nxt


rows, maps, roi = [], {}, {}
for k, pid in enumerate(PARTICIPANT_IDS, 1):
    for cond in ("sham", "stim"):
        ep = mne.read_epochs(str(EPOCHS_DIR / FILTER_TYPE / f"{pid}_{cond}-epo.fif"), preload=False)
        assert len(ep.ch_names) == 128 and ep.ch_names[0] == "E1" and ep.ch_names[-1] == "E128"
        prev, nxt = neighbours(pid, cond, ep)
        times = ep.times[:N_SAMPLES]
        masks = {}
        for s, (p, n) in SUBSETS.items():
            m = np.ones(len(ep), bool)
            if p is not None: m &= prev > p
            if n is not None: m &= nxt > n
            masks[s] = m
        sums = {s: np.zeros((128, len(ep.times))) for s in SUBSETS}
        for start in range(0, len(ep), 300):                      # chunks: limits memory
            xc = ep[start:start + 300].get_data() * 1e6           # µV
            for s, m in masks.items():
                sums[s] += xc[m[start:start + 300]].sum(axis=0)
            del xc
        msg = []
        for s, m in masks.items():
            msg.append(f"{s} {m.sum()}")
            if m.sum() == 0:
                logger.warning(f"  {pid} {cond} {s}: no epochs - file not written"); continue
            erp = sums[s] / m.sum()                               # (128, 901)
            erp = np.vstack([erp, np.zeros((1, erp.shape[1]))])  # + Cz (reference) = 129
            erp = (erp - erp.mean(axis=0))[:, :N_SAMPLES]         # average reference
            np.savetxt(OUT / s / f"{pid}_{TAG[cond]}.txt", erp.T, delimiter="\t", fmt="%.10g")
            maps[(s, pid, cond)] = erp
            rows.append(dict(participant_id=pid, group="HC" if pid in HEALTHY_IDS else "CI",
                             condition=cond, subset=s, n_epochs=int(m.sum()), n_all=len(ep)))
        logger.info(f"[{k}/{len(PARTICIPANT_IDS)}] {pid} {cond}: " + ", ".join(msg))
        del ep

df = pd.DataFrame(rows); df.to_csv(OUT / "epoch_counts_by_subset.csv", index=False)
logger.info("epochs per participant (median [min-max]):\n" + df.groupby(["subset", "condition"]).n_epochs
            .agg(lambda v: f"{int(np.median(v))} [{v.min()}-{v.max()}]").unstack().to_string())

# ── Step 2: validation against the MATLAB export ──────────────────────────────
r = []
for pid in PARTICIPANT_IDS:
    for cond in ("sham", "stim"):
        ref = np.loadtxt(REF_TXT / f"{pid}_{TAG[cond]}.txt").T
        r.append(np.corrcoef(ref.ravel(), maps[("all", pid, cond)].ravel())[0, 1])
logger.info(f"validation 'all' vs MATLAB export: r median {np.median(r):.4f}, min {np.min(r):.4f}")

# ── Step 3: topographic check ─────────────────────────────────────────────────
roi_idx = [int(c[1:]) - 1 for c in ROI]
def win(a, b): return (times >= a) & (times <= b)
norm = lambda v: (v - v.mean()) / np.linalg.norm(v - v.mean())
out = []
for s in SUBSETS:
    pids = [p for p in PARTICIPANT_IDS if (s, p, "sham") in maps and (s, p, "stim") in maps]
    res = dict(subset=s, n=len(pids))
    for name, w in (("P1", win(*WIN_P1)), ("P2", win(*WIN_P2))):
        ga = np.mean([norm(maps[(s, p, "stim")][:, w].mean(axis=1)) for p in pids], axis=0)
        ga_all = np.mean([norm(maps[("all", p, "stim")][:, w].mean(axis=1)) for p in PARTICIPANT_IDS], axis=0)
        res[f"r_stim_map_vs_all_{name}"] = np.corrcoef(ga, ga_all)[0, 1]
    w = win(*WIN_P2)
    d = np.array([maps[(s, p, "stim")][roi_idx][:, w].mean() - maps[(s, p, "sham")][roi_idx][:, w].mean() for p in pids])
    res["roi_P2_stim_minus_sham_uV"] = d.mean()
    res["roi_P2_dz"] = d.mean() / d.std(ddof=1)
    res["roi_P2_p_wilcoxon"] = stats.wilcoxon(d).pvalue
    out.append(res)
out = pd.DataFrame(out); out.to_csv(OUT / "topographic_check_by_subset.csv", index=False)
logger.info("topographic check (second-peak window 1080-1380 ms; P1 = 0-300 ms):\n" + out.round(3).to_string(index=False))
logger.info(f"TXT files for RAGU: {OUT}\\<subset>  (use RAGU_groups.xlsx for the group factor)")
logger.info("done")
