"""
Do the single-trial stimulation effects change across the three stimulation
nights, and differently in HC and CI? (Reviewer 1: in Zeller et al., 2024 the
response of the CI group was weak in the first night and reached the level of
HC only in the last night.)

Data: parameters_nights_k4.csv from d1_all_nights.py (same K=4 maps, labels as
in the manuscript). Participants with all three stimulation nights (E1, E2, E3).
For each parameter, the stimulation effect per night is the value of that night
minus the participant's baseline (BL).

Tests per parameter (permutation, 10,000 iterations, equal weighting):
  night            repeated-measures F across E1/E2/E3 (night labels permuted
                   within participants)
  night x group    interaction F of a mixed ANOVA (group labels permuted
                   across participants)
FDR within each family (scalar parameters, 250-ms bins) x test.
Descriptives: stimulation effect (night - BL) per night and group as dz.

Output: microstates/filter_03_20/7. all_nights/nights_comparison_k4.csv
"""

import numpy as np
import pandas as pd
from config import *

logger = setup_logger("nights_comparison", "d2_nights_comparison.log")

K       = 4
N_PERM  = 10000
NIGHTS  = ["E1", "E2", "E3"]
BASE    = MICROSTATES_DIR / FILTER_TYPE
OUT_DIR = BASE / "7. all_nights"

src = open("a3_stats.py", encoding="utf-8").read()
st  = {}
exec(compile(src[:src.index("\n# ── Run ──")], "a3_stats.py", "exec"), st)


def rm_f(Y):
    """One-way repeated-measures F (Y: subjects x conditions)."""
    n, k = Y.shape
    gm   = Y.mean()
    ss_c = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_s = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_e = ((Y - gm) ** 2).sum() - ss_c - ss_s
    return (ss_c / (k - 1)) / (ss_e / ((n - 1) * (k - 1)))


def interaction_f(Y, g):
    """Group x condition interaction F of a mixed ANOVA (g: boolean group)."""
    D   = Y - Y.mean(1, keepdims=True)                 # within-subject profiles
    k   = Y.shape[1]
    m   = D.mean(0)
    ss_i = sum(gi.sum() * ((D[gi].mean(0) - m) ** 2).sum() for gi in (g, ~g))
    ss_e = sum(((D[gi] - D[gi].mean(0)) ** 2).sum() for gi in (g, ~g))
    df_i, df_e = k - 1, (len(Y) - 2) * (k - 1)
    return (ss_i / df_i) / (ss_e / df_e)


def p_perm(obs, null):
    return (np.sum(null >= obs) + 1) / (len(null) + 1)


# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_DIR / f"parameters_nights_k{K}.csv")
df, _ = st["match_labels_to_reference"](df, st["load_group_model"](FILTER_TYPE, K), K)
complete = df.groupby("participant_id")["night"].apply(lambda s: set(["BL"] + NIGHTS) <= set(s))
pids = sorted(complete[complete].index)
df = df[df.participant_id.isin(pids)]
is_hc = np.array([p in HEALTHY_IDS for p in pids])
logger.info(f"participants with BL, E1, E2 and E3: {len(pids)} "
            f"(HC {is_hc.sum()}, CI {(~is_hc).sum()})")

n_bins   = int(df["n_time_bins"].iloc[0])
families = {
    "scalar":      [f"{s}_{p}" for s in range(K) for p in ("timecov", "meandurs", "occurrences", "gev")],
    "timecov_bin": [f"bin{b}_state{s}_cov" for b in range(n_bins) for s in range(K)],
}
wide = df.pivot(index="participant_id", columns="night").loc[pids]

rng  = np.random.default_rng(42)
perm_within = np.array([rng.permuted(np.tile(np.arange(3), (len(pids), 1)), axis=1)
                        for _ in range(N_PERM)])            # (N_PERM, n, 3)
perm_groups = np.array([rng.permutation(is_hc) for _ in range(N_PERM)])

rows = []
for family, cols in families.items():
    for col in cols:
        bl = wide[(col, "BL")].values
        Y  = np.column_stack([wide[(col, n)].values - bl for n in NIGHTS])   # stim effect per night
        f_n = rm_f(Y)
        null_n = np.array([rm_f(np.take_along_axis(Y, idx, 1)) for idx in perm_within])
        f_i = interaction_f(Y, is_hc)
        null_i = np.array([interaction_f(Y, gp) for gp in perm_groups])
        row = dict(family=family, col=col,
                   F_night=f_n, p_night=p_perm(f_n, null_n),
                   F_night_x_group=f_i, p_night_x_group=p_perm(f_i, null_i))
        for gname, gm in (("HC", is_hc), ("CI", ~is_hc), ("all", np.ones(len(pids), bool))):
            for j, n in enumerate(NIGHTS):
                d = Y[gm, j]
                row[f"dz_{gname}_{n}"] = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
        rows.append(row)

res = pd.DataFrame(rows)
for test in ("night", "night_x_group"):
    res[f"pfdr_{test}"] = np.nan
    for fam, idx in res.groupby("family").groups.items():
        _, padj = st["fdr_correct"](res.loc[idx, f"p_{test}"].values)
        res.loc[idx, f"pfdr_{test}"] = padj
res.to_csv(OUT_DIR / f"nights_comparison_k{K}.csv", index=False)


# ── Summary ───────────────────────────────────────────────────────────────────
def _label(col):
    if col.startswith("bin"):
        b, s = col.replace("bin", "").replace("_cov", "").split("_state")
        t0 = EPOCH_TMIN + int(b) * COVERAGE_BIN_DURATION_S
        return f"MS {int(s) + 1} cov {t0 * 1000:+.0f}..{(t0 + COVERAGE_BIN_DURATION_S) * 1000:+.0f} ms"
    s, p = col.split("_", 1)
    return f"MS {int(s) + 1} {p}"

for test in ("night", "night_x_group"):
    sig = res[res[f"pfdr_{test}"] < 0.05]
    logger.info(f"{test}: FDR-significant {len(sig)} of {len(res)}"
                + "".join(f"\n    {_label(r.col)}: p = {r[f'p_{test}']:.4f}, "
                          f"p_fdr = {r[f'pfdr_{test}']:.3f}" for _, r in sig.iterrows()))
    unc = res[res[f"p_{test}"] < 0.05]
    logger.info(f"{test}: uncorrected p < .05 in {len(unc)} of {len(res)} tests "
                f"(about {0.05 * len(res):.0f} expected by chance)")

key = ["2_meandurs", "2_timecov", "3_timecov", "3_occurrences",
       "bin7_state2_cov", "bin7_state3_cov", "bin9_state2_cov"]
k = res[res.col.isin(key)].set_index("col")
k.index = k.index.map(_label)
cols = [f"dz_{g}_{n}" for g in ("HC", "CI") for n in NIGHTS] + ["pfdr_night", "pfdr_night_x_group"]
logger.info("key effects: stimulation effect (night - BL) as dz per group and night, "
            "and FDR-corrected p for night and night x group:\n" + k[cols].round(2).to_string())
logger.info("done")
