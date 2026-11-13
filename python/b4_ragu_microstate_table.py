"""
Complete table of the average-level (RAGU) microstate statistics with an
explicit multiplicity correction (Reviewer 2, point 3).

Input: the two RAGU microstate analyses with the revised groups
  -200 to 800 ms   Microstates_-200_800ms_ConditionXGroup_HCCI.mat
   800 to 1900 ms  Microstates_800_1800ms_ConditionXGroup_new.mat (file name says 1800)
Both use the same 8 maps (checked below).

RAGU stores, per effect, microstate class and parameter, the p-value of the
randomisation test (rd.pMSStats[group_effect, within_effect, class, 0, parameter]):
  effects     condition          [0, 1]
              group              [1, 0]
              group x condition  [1, 1]
  parameters  onset, offset, duration, AUC, centre of gravity, mean GFP
              (onset/offset/centre of gravity are undefined (NaN) for classes
               that are absent in some cells; these tests are not counted)

Multiplicity: Benjamini-Hochberg FDR within each analysis window x effect,
across all classes and parameters (one family = all tests of one effect in one
window). p = 0 from the randomisation test is set to 1 / (iterations + 1).

Class numbers are reported in the numbering of Figure 3 (FIGURE_ORDER, the
same mapping as in e3_figure5_correspondence.py).

Output: Ragu_results/ragu_microstate_stats_table.csv
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from config import *

logger = setup_logger("ragu_microstate_table", "b4_ragu_microstate_table.log")

RAGU_RESULTS = RAGU_DIR / "Ragu_results"
FILES = {
    "-200 to 800 ms":  "Microstates_-200_800ms_ConditionXGroup_HCCI.mat",   # re-run 02.11.2026 with HC/CI grouping (the _100k-mean_new file had been saved with the sex grouping)
    "800 to 1900 ms":  "Microstates_800_1800ms_ConditionXGroup_new.mat",   # file name says 1800; window is frames 461-681 = 800-1900 ms
}
EFFECTS    = {"condition": (0, 1), "group": (1, 0), "group x condition": (1, 1)}
PARAMETERS = ["onset", "offset", "duration", "AUC", "centre of gravity", "mean GFP"]
# Figure 3 number -> map index in the RAGU file (1-based); see e3_figure5_correspondence.py
FIGURE_ORDER = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 3}
FILE_TO_FIG  = {v: k for k, v in FIGURE_ORDER.items()}


def fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


z = lambda a: (a - a.mean(-1, keepdims=True)) / a.std(-1, keepdims=True)
rows, maps_ref = [], None
for window, fname in FILES.items():
    rd = loadmat(str(RAGU_RESULTS / fname), squeeze_me=True, struct_as_record=False)["rd"]
    maps = np.asarray(rd.MSMaps, float)
    if maps_ref is None:
        maps_ref = maps
    else:
        same = np.diag(z(maps_ref) @ z(maps).T / maps.shape[1])
        assert (same > 0.999).all(), f"maps differ between windows: {same}"
    P = np.asarray(rd.pMSStats, float)
    if P.size == 0:
        # p-values not stored: compute them from the randomisation distribution
        # (rd.MSEffectSize[group_effect, within_effect, class, iteration, parameter];
        #  iteration 0 = observed data); p = share of randomisations >= observed
        ES  = np.asarray(rd.MSEffectSize, float)
        obs = ES[:, :, :, :1, :]
        with np.errstate(invalid="ignore"):
            P = np.mean(ES >= obs, axis=3, keepdims=True)
        P[np.isnan(obs)] = np.nan
    n_iter = int(rd.Iterations)
    logger.info(f"{window}: {fname}, {n_iter} randomisations, {rd.nStates} classes")
    for effect, (g, c) in EFFECTS.items():
        for file_cls in range(1, P.shape[2] + 1):
            for j, par in enumerate(PARAMETERS):
                p = P[g, c, file_cls - 1, 0, j]
                rows.append(dict(window=window, effect=effect,
                                 figure_class=FILE_TO_FIG[file_cls], ragu_file_class=file_cls,
                                 parameter=par, p=p))

df = pd.DataFrame(rows)
df["tested"] = df["p"].notna()
df["p_used"] = df["p"].clip(lower=1 / (n_iter + 1))
df["p_fdr"]  = np.nan
for _, idx in df[df.tested].groupby(["window", "effect"]).groups.items():
    df.loc[idx, "p_fdr"] = fdr(df.loc[idx, "p_used"].values)
df["sig_uncorrected"] = df["p"] < 0.05
df["sig_fdr"] = df["p_fdr"] < 0.05
df = df.sort_values(["window", "effect", "figure_class", "parameter"])
df.to_csv(RAGU_RESULTS / "ragu_microstate_stats_table.csv", index=False)

for (window, effect), d in df[df.tested].groupby(["window", "effect"], sort=False):
    sig = d[d.sig_fdr]; unc = d[d.sig_uncorrected & ~d.sig_fdr]
    logger.info(f"{window} | {effect}: {len(d)} tests, FDR-significant {len(sig)}, "
                f"uncorrected only {len(unc)}"
                + "".join(f"\n    class {r.figure_class} {r.parameter}: p = {r.p:.3f}, "
                          f"p_fdr = {r.p_fdr:.3f}" for r in sig.itertuples())
                + "".join(f"\n    (uncorrected) class {r.figure_class} {r.parameter}: p = {r.p:.3f}, "
                          f"p_fdr = {r.p_fdr:.3f}" for r in unc.itertuples()))
logger.info("done")
