"""
Sensitivity analysis: sex as covariate (reviewer concern on the sex distribution;
HC 19 F / 6 M, CI 7 F / 8 M, Fisher p = .09).

Linear models per parameter (single-trial results of the main analysis,
labels as in the manuscript), each fitted with and without sex so that the
effect of the adjustment can be read directly:

  condition effect   d = stim - sham        d ~ 1 (+ sex)           -> intercept
  interaction        d                      d ~ group (+ sex)       -> group
  group effect       sham / stim value      y ~ group (+ sex)       -> group
  sex moderation     d                      d ~ group + sex         -> sex
                     (does the stimulation effect differ between women and men?)

Sex is coded -0.5 / +0.5 and centred, so the intercept is the condition effect
at the average sex composition of the sample. Ordinary least squares,
two-sided t-tests; FDR (Benjamini-Hochberg) within each family x test x model,
as in a3_stats.py. Equal weighting of participants.

Output: microstates/filter_03_20/8. sex_sensitivity/sex_sensitivity_k4.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from config import *

logger = setup_logger("sex_sensitivity", "d3_sex_sensitivity.log")

K       = 4
BASE    = MICROSTATES_DIR / FILTER_TYPE
OUT_DIR = BASE / "8. sex_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# functions of a3_stats.py (relabelling, FDR) without running its main part
src = open("a3_stats.py", encoding="utf-8").read()
st  = {}
exec(compile(src[:src.index("\n# ── Run ──")], "a3_stats.py", "exec"), st)


def ols(y, X):
    """OLS coefficients, t-values and two-sided p-values."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid    = y - X @ beta
    df       = len(y) - X.shape[1]
    se       = np.sqrt(np.diag(resid @ resid / df * np.linalg.inv(X.T @ X)))
    t        = beta / se
    return beta, t, 2 * stats.t.sf(np.abs(t), df)


# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(BASE / "3. backfitting" / f"parameters_k{K}.csv")
df, _ = st["match_labels_to_reference"](df, st["load_group_model"](FILTER_TYPE, K), K)

master = pd.read_csv(MASTER_PATH)
sex    = master.set_index("ID")["gender"].str.lower().str[0]           # 'f' / 'm'
df["sex"] = df["participant_id"].map(sex)
assert df["sex"].isin(["f", "m"]).all(), df.loc[~df["sex"].isin(["f", "m"]), "participant_id"]

wide = df.pivot(index="participant_id", columns="condition")
pids = wide.index
group = np.where(np.isin(pids, HEALTHY_IDS), 0.5, -0.5)               # HC +0.5, CI -0.5
sexc  = np.where(sex.reindex(pids).values == "m", 0.5, -0.5)
sexc  = sexc - sexc.mean()
logger.info(f"{len(pids)} participants; sex x group: "
            + str(pd.crosstab(np.where(group > 0, "HC", "CI"), sex.reindex(pids).values).to_dict()))

n_bins   = int(df["n_time_bins"].iloc[0])
families = {
    "scalar":      [f"{s}_{p}" for s in range(K) for p in ("timecov", "meandurs", "occurrences", "gev")],
    "timecov_bin": [f"bin{b}_state{s}_cov" for b in range(n_bins) for s in range(K)],
}
one = np.ones(len(pids))

# ── Models ────────────────────────────────────────────────────────────────────
rows = []
for family, cols in families.items():
    for col in cols:
        sham, stim = wide[(col, "sham")].values, wide[(col, "stim")].values
        d = stim - sham
        specs = {
            # test name: (y, X without sex, coefficient index)
            "condition_effect":  (d,    np.c_[one],        0),
            "interaction":       (d,    np.c_[one, group], 1),
            "group_effect_sham": (sham, np.c_[one, group], 1),
            "group_effect_stim": (stim, np.c_[one, group], 1),
        }
        for test, (y, X, j) in specs.items():
            for model, Xm in (("unadjusted", X), ("sex_adjusted", np.c_[X, sexc])):
                b, t, p = ols(y, Xm)
                rows.append(dict(family=family, col=col, test=test, model=model,
                                 estimate=b[j], t=t[j], p_value=p[j]))
        b, t, p = ols(d, np.c_[one, group, sexc])
        rows.append(dict(family=family, col=col, test="sex_moderation", model="sex_adjusted",
                         estimate=b[2], t=t[2], p_value=p[2]))

res = pd.DataFrame(rows)
res["p_fdr"], res["sig_fdr"] = np.nan, False
for _, idx in res.groupby(["family", "test", "model"]).groups.items():
    sig, padj = st["fdr_correct"](res.loc[idx, "p_value"].values)
    res.loc[idx, "p_fdr"], res.loc[idx, "sig_fdr"] = padj, sig
res.to_csv(OUT_DIR / f"sex_sensitivity_k{K}.csv", index=False)


# ── Summary ───────────────────────────────────────────────────────────────────
def _label(col):
    if col.startswith("bin"):
        b, s = col.replace("bin", "").replace("_cov", "").split("_state")
        t0 = EPOCH_TMIN + int(b) * COVERAGE_BIN_DURATION_S
        return f"MS {int(s) + 1} cov {t0 * 1000:+.0f}..{(t0 + COVERAGE_BIN_DURATION_S) * 1000:+.0f} ms"
    s, p = col.split("_", 1)
    return f"MS {int(s) + 1} {p}"

piv = res[res.test != "sex_moderation"].pivot_table(
    index=["test", "col"], columns="model", values=["p_fdr", "sig_fdr"], aggfunc="first")
for test in ("condition_effect", "interaction", "group_effect_sham", "group_effect_stim"):
    sub = piv.loc[test]
    n_u = int(sub[("sig_fdr", "unadjusted")].sum()); n_a = int(sub[("sig_fdr", "sex_adjusted")].sum())
    changed = sub[sub[("sig_fdr", "unadjusted")] != sub[("sig_fdr", "sex_adjusted")]]
    logger.info(f"{test}: FDR-significant unadjusted {n_u}, sex-adjusted {n_a}; "
                f"changed significance: {len(changed)}")
    for col, r in changed.iterrows():
        logger.info(f"    {_label(col)}: p_fdr {r[('p_fdr', 'unadjusted')]:.3f} -> "
                    f"{r[('p_fdr', 'sex_adjusted')]:.3f}")

key = ["2_meandurs", "2_timecov", "3_timecov", "3_occurrences", "0_occurrences",
       "bin7_state2_cov", "bin7_state3_cov"]
k = res[(res.test == "condition_effect") & res.col.isin(key)].pivot(
    index="col", columns="model", values="p_fdr")
k.index = k.index.map(_label)
logger.info("key condition effects, p_fdr unadjusted vs sex-adjusted:\n" + k.round(4).to_string())

mod = res[(res.test == "sex_moderation") & res.sig_fdr]
logger.info(f"sex moderation of the stimulation effect: {len(mod)} FDR-significant"
            + ("".join(f"\n    {_label(r.col)}: estimate (m - f) {r.estimate:+.4f}, p_fdr {r.p_fdr:.3f}"
                       for r in mod.itertuples())))
logger.info("done")
