"""
Sleep architecture and stimulus counts by night and group (reviewer concerns:
comparability of the nights, night-order effects, PLAS changing sleep).

Sleep variables from all_vars_in_one_Korian.xlsx (scoring of all nights):
  TIB, TST, SE, SL, WASO (min / %), N1, N2, N3, REM (% of TST)
Stimulus counts: number of markers per night, read from the annotations of the
recordings (BL: sham marker D128 / DI64, E1-E3: stimulation marker DI64).

Analyses
  A. BL vs the analysed stimulation night (E1; E2 for the participants in
     USE_NIGHT_E2), i.e. exactly the nights of the main analysis
       - paired Wilcoxon signed-rank: all, HC, CI
       - group difference per night: Mann-Whitney U
       - group x night: Mann-Whitney U on the (stim - BL) differences
  B. All four nights (BL, E1, E2, E3): mean +/- SD per group, Friedman test
     across nights (all participants) as a check for night-order effects
  FDR (Benjamini-Hochberg) across the sleep variables, separately per test.

Output: microstates/filter_03_20/9. sleep_architecture/
  sleep_main_nights.csv, sleep_all_nights_descriptives.csv, stimulus_counts.csv
"""

import mne
import numpy as np
import pandas as pd
from scipy import stats
from config import *

mne.set_log_level("ERROR")
logger = setup_logger("sleep_architecture", "b1_sleep_architecture.log")

SLEEP_FILE = PROJECT_ROOT / "all_vars_in_one_Korian.xlsx"
OUT_DIR    = MICROSTATES_DIR / FILTER_TYPE / "9. sleep_architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NIGHTS    = ["BL", "E1", "E2", "E3"]
NIGHT_NR  = {"BL": "02", "E1": "03", "E2": "04", "E3": "05"}
VARIABLES = {                       # column stem: label
    "TIB": "TIB (min)", "TST": "TST (min)", "SE": "SE (%)", "SL": "SL (min)",
    "WASO": "WASO (min)", "N1_perc": "N1 (%)", "N2_perc": "N2 (%)",
    "N3_perc": "N3 (%)", "REM_perc": "REM (%)",
}


def fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def msd(x):
    return f"{np.nanmean(x):.1f} ± {np.nanstd(x, ddof=1):.1f}"


# ── Data ──────────────────────────────────────────────────────────────────────
raw = pd.read_excel(SLEEP_FILE)
df  = raw[raw["ID"].isin(PARTICIPANT_IDS)].set_index("ID").loc[PARTICIPANT_IDS]
assert len(df) == 40, len(df)
group   = pd.Series(np.where(df.index.isin(HEALTHY_IDS), "HC", "CI"), index=df.index)
stim_nt = pd.Series(np.where(df.index.isin(USE_NIGHT_E2), "E2", "E1"), index=df.index)
logger.info(f"40 participants; analysed stimulation night: {stim_nt.value_counts().to_dict()}")


# ── Stimulus counts per night ─────────────────────────────────────────────────
def n_markers(pid, night):
    folder = DATA_ROOT / pid[:5] / EEG_SUBFOLDER
    files  = list(folder.glob(f"{pid}-{NIGHT_NR[night]}*_scored_clean_PREP_AutoReject.set"))
    if len(files) != 1:
        return np.nan
    marker = ("DI64" if pid in SHAM_USES_DI64 else "D128") if night == "BL" else "DI64"
    ann = mne.read_annotations(str(files[0]))
    return int(np.sum(ann.description == marker))

counts = pd.DataFrame({n: [n_markers(p, n) for p in PARTICIPANT_IDS] for n in NIGHTS},
                      index=PARTICIPANT_IDS)
counts["stim_analysed"] = [counts.loc[p, stim_nt[p]] for p in PARTICIPANT_IDS]
counts.insert(0, "group", group)
counts.to_csv(OUT_DIR / "stimulus_counts.csv")
logger.info("markers per night (mean ± SD):\n" + counts.groupby("group")[NIGHTS + ["stim_analysed"]]
            .agg(lambda x: msd(x)).T.to_string())
u = stats.mannwhitneyu(counts.loc[group == "HC", "stim_analysed"],
                       counts.loc[group == "CI", "stim_analysed"])
logger.info(f"stimuli in the analysed night, HC vs CI: Mann-Whitney p = {u.pvalue:.3f}")


# ── A. BL vs analysed stimulation night ───────────────────────────────────────
rows = []
for stem, label in VARIABLES.items():
    bl   = df[f"{stem}_BL"].astype(float)
    st_  = pd.Series([df.loc[p, f"{stem}_{stim_nt[p]}"] for p in df.index], index=df.index, dtype=float)
    diff = st_ - bl
    row  = dict(variable=label)
    for g, mask in (("HC", group == "HC"), ("CI", group == "CI")):
        row[f"{g} BL"]   = msd(bl[mask])
        row[f"{g} stim"] = msd(st_[mask])
    row["p_night_all"] = stats.wilcoxon(st_, bl).pvalue
    row["p_night_HC"]  = stats.wilcoxon(st_[group == "HC"], bl[group == "HC"]).pvalue
    row["p_night_CI"]  = stats.wilcoxon(st_[group == "CI"], bl[group == "CI"]).pvalue
    row["p_group_BL"]   = stats.mannwhitneyu(bl[group == "HC"], bl[group == "CI"]).pvalue
    row["p_group_stim"] = stats.mannwhitneyu(st_[group == "HC"], st_[group == "CI"]).pvalue
    row["p_group_x_night"] = stats.mannwhitneyu(diff[group == "HC"], diff[group == "CI"]).pvalue
    rows.append(row)
main = pd.DataFrame(rows)
for c in [c for c in main.columns if c.startswith("p_")]:
    main["pfdr_" + c[2:]] = fdr(main[c])
main.to_csv(OUT_DIR / "sleep_main_nights.csv", index=False)

show = main[["variable", "HC BL", "HC stim", "CI BL", "CI stim"]
            + [c for c in main.columns if c.startswith("pfdr_")]].copy()
for c in show.columns:
    if c.startswith("pfdr_"):
        show[c] = show[c].map(lambda v: f"{v:.3f}")
logger.info("A. BL vs analysed stimulation night (p FDR-corrected across variables):\n"
            + show.to_string(index=False))


# ── B. All four nights ────────────────────────────────────────────────────────
desc, fried = [], []
for stem, label in VARIABLES.items():
    vals = df[[f"{stem}_{n}" for n in NIGHTS]].astype(float)
    for g in ("HC", "CI"):
        desc.append(dict(variable=label, group=g,
                         **{n: msd(vals.loc[group == g, f"{stem}_{n}"]) for n in NIGHTS}))
    ok = vals.dropna()
    fried.append(dict(variable=label, n=len(ok),
                      p_friedman=stats.friedmanchisquare(*[ok[c] for c in ok.columns]).pvalue))
desc  = pd.DataFrame(desc)
fried = pd.DataFrame(fried); fried["pfdr_friedman"] = fdr(fried["p_friedman"])
desc.merge(fried, on="variable").to_csv(OUT_DIR / "sleep_all_nights_descriptives.csv", index=False)
logger.info("B. all nights, mean ± SD:\n" + desc.to_string(index=False))
logger.info("B. Friedman test across BL/E1/E2/E3 (all participants):\n"
            + fried.round(3).to_string(index=False))
logger.info("done")
