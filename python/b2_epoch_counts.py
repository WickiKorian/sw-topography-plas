"""
Table 2: number of epochs per participant after artifact rejection.

Source: n_epochs in parameters_k4.csv (written by a2_cluster_backfit.py), i.e.
the epochs of a1_epochs.py that enter both the single-trial and (after the
harmonisation) the average-level analysis.

Tests (as in the manuscript)
  between groups, per condition: Mann-Whitney U
  between conditions, per group: Wilcoxon signed-rank

Output: microstates/filter_03_20/3. backfitting/epoch_counts_table2.csv
"""

import pandas as pd
from scipy import stats
from config import *

logger = setup_logger("epoch_counts", "b2_epoch_counts.log")

BF  = MICROSTATES_DIR / FILTER_TYPE / "3. backfitting"
df  = pd.read_csv(BF / "parameters_k4.csv")[["participant_id", "condition", "n_epochs"]]
df  = df[df.participant_id.isin(PARTICIPANT_IDS)]
w   = df.pivot(index="participant_id", columns="condition", values="n_epochs")
w["group"] = ["HC" if p in HEALTHY_IDS else "CI" for p in w.index]
assert (w.group == "HC").sum() == 25 and (w.group == "CI").sum() == 15

rows = []
for cond, label in (("sham", "Baseline"), ("stim", "Experimental")):
    hc, ci = w.loc[w.group == "HC", cond], w.loc[w.group == "CI", cond]
    rows.append(dict(condition=label, HC_M=hc.mean(), HC_SD=hc.std(), CI_M=ci.mean(), CI_SD=ci.std(),
                     p_MannWhitney=stats.mannwhitneyu(hc, ci).pvalue))
tab = pd.DataFrame(rows)
tab.to_csv(BF / "epoch_counts_table2.csv", index=False)
logger.info("Table 2 (epochs after artifact rejection):\n" + tab.round({"HC_M": 0, "HC_SD": 0, "CI_M": 0, "CI_SD": 0, "p_MannWhitney": 2}).to_string(index=False))

for g in ("HC", "CI"):
    x = w[w.group == g]
    logger.info(f"{g}: baseline vs experimental, Wilcoxon p = {stats.wilcoxon(x.sham, x.stim).pvalue:.2f}")
logger.info(f"total epochs: baseline {w.sham.sum()}, experimental {w.stim.sum()}")
logger.info("done")
