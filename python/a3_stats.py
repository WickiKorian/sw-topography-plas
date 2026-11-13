"""
Statistical analysis of microstate parameters.

Runs three permutation contrasts per variable (paired condition effect,
independent group effect, group x condition interaction), applies FDR
correction (Benjamini-Hochberg) within each test family, and saves
figures for significant results.

Revision changes (2026-10)
--------------------------
- microstate labels matched to the templates of the submitted analysis
  (spatial correlation), so MS 1-4 keep the numbering used in the manuscript
- participants weighted equally (WEIGHT_BY_EPOCHS = False); epoch-count
  weighting as in the submission is kept as a sensitivity option
- condition main effect across all participants added (factorial approach);
  within-group condition effects, group effects and interaction as before
- FDR correction within each test family AND contrast (as stated in Methods);
  previously all contrasts of a family were pooled
- effect sizes (dz / d) and 95% CI of the raw difference for every test
- permutation p-values computed as (b + 1) / (n_perm + 1)
- time-binned coverage tested with the same number of permutations as all
  other parameters (N_PERMUTATIONS); previously 2000
- figure labels MS 1-4 (previously 0-3)
- FDR-adjusted p-values made monotone (step-up), consistent with significance

Figure colour convention (consistent with Figure 3 of the paper)
-----------------------------------------------------------------
- Colour encodes CONDITION:  orange-red = stim, Klein blue = sham
- Line style encodes GROUP:  long dash = healthy, short dash = impaired
- Topomaps use the pastel diverging colormap PASTEL_RDBU
"""

import pickle
import warnings
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy as scipy_entropy

from config import *

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = setup_logger("statistics", "a3_stats.log")

# ── Revision settings ──────────────────────────────────────────────────────────
WEIGHT_BY_EPOCHS   = False             # False = main analysis; True = sensitivity analysis (Supplementary Table S4)
MIN_TEMPLATE_CORR  = 0.80                # minimal |r| to accept a label match
STATS_FOLDER       = "4. statistics_weighted" if WEIGHT_BY_EPOCHS else "4. statistics"



# ── Permutation tests ──────────────────────────────────────────────────────────

def perm_paired(sham_values, stim_values, weights=None,
                n_perm=N_PERMUTATIONS, rng=None):
    """
    Paired permutation test (stim vs sham) with optional sample weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    condition_differences = (np.asarray(stim_values, float)
                             - np.asarray(sham_values, float))
    if weights is None:
        weights = np.ones(len(condition_differences))
    normalised_weights = np.asarray(weights, float) / np.asarray(weights, float).sum()

    observed_stat    = np.sum(normalised_weights * condition_differences)
    sign_flips       = rng.choice([-1., 1.], size=(n_perm, len(condition_differences)))
    null_distribution = (sign_flips * condition_differences * normalised_weights).sum(axis=1)

    p_value = _perm_p(null_distribution, observed_stat)
    return float(observed_stat), p_value


def perm_independent(healthy_values, impaired_values,
                     healthy_weights=None, impaired_weights=None,
                     n_perm=N_PERMUTATIONS, rng=None):
    """
    Two-sample permutation test (healthy vs impaired) with optional weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    healthy_values  = np.asarray(healthy_values,  float)
    impaired_values = np.asarray(impaired_values, float)

    if healthy_weights is None:
        healthy_weights  = np.ones(len(healthy_values))
    if impaired_weights is None:
        impaired_weights = np.ones(len(impaired_values))

    normalised_healthy_weights  = (np.asarray(healthy_weights,  float)
                                   / np.asarray(healthy_weights,  float).sum())
    normalised_impaired_weights = (np.asarray(impaired_weights, float)
                                   / np.asarray(impaired_weights, float).sum())

    observed_stat = (np.sum(normalised_healthy_weights  * healthy_values)
                   - np.sum(normalised_impaired_weights * impaired_values))

    pooled_values  = np.concatenate([healthy_values,  impaired_values])
    pooled_weights = np.concatenate([normalised_healthy_weights,
                                     normalised_impaired_weights])
    n_healthy      = len(healthy_values)

    perm_idx         = rng.permuted(
        np.tile(np.arange(len(pooled_values)), (n_perm, 1)), axis=1
    )
    perm_weights_h   = pooled_weights[perm_idx[:, :n_healthy]]
    perm_weights_h  /= perm_weights_h.sum(axis=1, keepdims=True)
    perm_weights_i   = pooled_weights[perm_idx[:, n_healthy:]]
    perm_weights_i  /= perm_weights_i.sum(axis=1, keepdims=True)

    null_distribution = (
        (pooled_values[perm_idx[:, :n_healthy]] * perm_weights_h).sum(axis=1)
      - (pooled_values[perm_idx[:, n_healthy:]] * perm_weights_i).sum(axis=1)
    )

    p_value = _perm_p(null_distribution, observed_stat)
    return float(observed_stat), p_value


def perm_interaction(participant_df, column, weight_col="w",
                     n_perm=N_PERMUTATIONS, rng=None):
    """
    Group x condition interaction test via permutation of group labels.
    """
    if rng is None:
        rng = np.random.default_rng()

    participant_ids = participant_df["participant_id"].unique()
    group_per_pid   = (participant_df.drop_duplicates("participant_id")
                                     .set_index("participant_id")["group"])
    n_healthy       = int((group_per_pid == "healthy").sum())
    n_participants  = len(participant_ids)

    condition_values  = np.full((n_participants, 2), np.nan)
    condition_weights = np.full((n_participants, 2), np.nan)

    for pid_idx, pid in enumerate(participant_ids):
        pid_rows = participant_df[participant_df["participant_id"] == pid]
        for cond_idx, cond in enumerate(("sham", "stim")):
            cond_row = pid_rows[pid_rows["condition"] == cond]
            if len(cond_row):
                condition_values[pid_idx, cond_idx]  = cond_row[column].values[0]
                condition_weights[pid_idx, cond_idx] = (
                    cond_row[weight_col].values[0]
                    if weight_col in cond_row.columns else 1.
                )

    participant_weights   = np.nanmean(condition_weights, axis=1)
    stim_minus_sham       = condition_values[:, 1] - condition_values[:, 0]
    is_healthy            = np.array([group_per_pid[pid] == "healthy"
                                      for pid in participant_ids])

    def _weighted_mean(values, mask, weights):
        v, w = values[mask], weights[mask]
        valid = ~np.isnan(v)
        if not valid.any():
            return np.nan
        w = w[valid] / w[valid].sum()
        return np.sum(w * v[valid])

    healthy_stim_effect  = _weighted_mean(stim_minus_sham,  is_healthy,  participant_weights)
    impaired_stim_effect = _weighted_mean(stim_minus_sham, ~is_healthy,  participant_weights)
    observed_interaction = healthy_stim_effect - impaired_stim_effect

    permuted_healthy_masks = (rng.permuted(
        np.tile(np.arange(n_participants), (n_perm, 1)), axis=1
    ) < n_healthy)
    weight_matrix        = np.tile(participant_weights, (n_perm, 1))
    effect_matrix        = np.tile(stim_minus_sham,    (n_perm, 1))

    perm_weights_healthy  = np.where( permuted_healthy_masks, weight_matrix, 0.)
    perm_weights_impaired = np.where(~permuted_healthy_masks, weight_matrix, 0.)

    null_healthy  = np.nansum(
        effect_matrix * permuted_healthy_masks
        * (perm_weights_healthy
           / perm_weights_healthy.sum(axis=1, keepdims=True).clip(1e-12)),
        axis=1,
    )
    null_impaired = np.nansum(
        effect_matrix * ~permuted_healthy_masks
        * (perm_weights_impaired
           / perm_weights_impaired.sum(axis=1, keepdims=True).clip(1e-12)),
        axis=1,
    )
    null_distribution = null_healthy - null_impaired

    p_value = _perm_p(null_distribution, observed_interaction)
    return (float(observed_interaction), float(healthy_stim_effect),
            float(impaired_stim_effect), p_value)


def _perm_p(null_distribution, observed):
    """Two-sided permutation p-value, (b + 1) / (n_perm + 1) (Phipson & Smyth, 2010)."""
    n_extreme = np.sum(np.abs(null_distribution) >= np.abs(observed))
    return float((n_extreme + 1) / (len(null_distribution) + 1))


def _effect_paired(diffs):
    """Cohen's dz and t-based 95% CI of the mean difference (unweighted)."""
    diffs = np.asarray(diffs, float)
    n, m, sd = len(diffs), diffs.mean(), diffs.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return dict(effect_size=m / sd if sd > 0 else np.nan, es_type="dz",
                diff_unweighted=m, ci_low=m - half, ci_high=m + half)


def _effect_independent(a, b):
    """Cohen's d (pooled SD) and Welch 95% CI of the mean difference a - b (unweighted)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    m = a.mean() - b.mean()
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    half = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    return dict(effect_size=m / sp if sp > 0 else np.nan, es_type="d",
                diff_unweighted=m, ci_low=m - half, ci_high=m + half)


def fdr_correct(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    Adjusted p-values are made monotone (step-up), so that p_fdr < alpha
    exactly when the test is significant. Previously the adjusted p-values
    were not monotone and could exceed alpha for significant tests.
    """
    p_array = np.asarray(p_values, float)
    n_tests = len(p_array)
    if n_tests == 0:
        return np.array([], bool), np.array([])

    order       = np.argsort(p_array)
    scaled      = p_array[order] * n_tests / np.arange(1, n_tests + 1)
    monotone    = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_p  = np.empty(n_tests)
    adjusted_p[order] = np.minimum(1.0, monotone)
    is_significant = adjusted_p <= alpha
    return is_significant, adjusted_p



def run_tests(participant_df, columns: list, family_label: str,
              rng, n_perm: int = N_PERMUTATIONS) -> list:
    """
    Run paired, independent, and interaction permutation tests for every
    column in *columns*, returning a list of result dicts.
    """
    result_rows = []

    for column in columns:
        if column not in participant_df.columns:
            continue

        col_parts  = column.split("_", 1)
        microstate = col_parts[0]
        parameter  = col_parts[1] if len(col_parts) > 1 else column

        # Condition main effect across all participants (factorial approach)
        sham_all, stim_all, w_all = [], [], []
        for group in ("healthy", "impaired"):
            s_, t_, w_ = _get_paired_values(participant_df, group, column)
            sham_all.append(s_); stim_all.append(t_); w_all.append(w_)
        sham_all, stim_all, w_all = (np.concatenate(x) for x in (sham_all, stim_all, w_all))
        if len(sham_all) > 5:
            observed_diff, p_value = perm_paired(
                sham_all, stim_all, weights=w_all, n_perm=n_perm, rng=rng,
            )
            result_rows.append(dict(
                family     = family_label,
                col        = column,
                microstate = microstate,
                parameter  = parameter,
                test       = "all_condition_effect",
                n          = len(sham_all),
                mean_sham  = float(np.average(sham_all, weights=w_all)),
                mean_stim  = float(np.average(stim_all, weights=w_all)),
                difference = observed_diff,
                p_value    = p_value,
                **_effect_paired(stim_all - sham_all),
            ))

        # Paired condition effect within each group
        for group in ("healthy", "impaired"):
            sham_vals, stim_vals, weights = _get_paired_values(
                participant_df, group, column
            )
            if len(sham_vals) > 5:
                observed_diff, p_value = perm_paired(
                    sham_vals, stim_vals, weights=weights,
                    n_perm=n_perm, rng=rng,
                )
                result_rows.append(dict(
                    family     = family_label,
                    col        = column,
                    microstate = microstate,
                    parameter  = parameter,
                    test       = f"{group}_condition_effect",
                    n          = len(sham_vals),
                    mean_sham  = float(np.average(sham_vals, weights=weights)),
                    mean_stim  = float(np.average(stim_vals, weights=weights)),
                    difference = observed_diff,
                    p_value    = p_value,
                    **_effect_paired(stim_vals - sham_vals),
                ))

        # Independent group effect within each condition
        for condition in ("sham", "stim"):
            healthy_vals,  healthy_weights  = _get_independent_values(
                participant_df, "healthy",  condition, column
            )
            impaired_vals, impaired_weights = _get_independent_values(
                participant_df, "impaired", condition, column
            )
            if len(healthy_vals) > 5 and len(impaired_vals) > 5:
                observed_diff, p_value = perm_independent(
                    healthy_vals, impaired_vals,
                    healthy_weights, impaired_weights,
                    n_perm=n_perm, rng=rng,
                )
                result_rows.append(dict(
                    family         = family_label,
                    col            = column,
                    microstate     = microstate,
                    parameter      = parameter,
                    test           = f"group_effect_{condition}",
                    mean_healthy   = float(np.average(healthy_vals,  weights=healthy_weights)),
                    mean_impaired  = float(np.average(impaired_vals, weights=impaired_weights)),
                    difference     = observed_diff,
                    p_value        = p_value,
                    **_effect_independent(healthy_vals, impaired_vals),
                ))

        # Group x condition interaction
        observed_interaction, healthy_effect, impaired_effect, p_value = (
            perm_interaction(participant_df, column, n_perm=n_perm, rng=rng)
        )
        h_s, h_t, _ = _get_paired_values(participant_df, "healthy",  column)
        i_s, i_t, _ = _get_paired_values(participant_df, "impaired", column)
        result_rows.append(dict(
            family               = family_label,
            col                  = column,
            microstate           = microstate,
            parameter            = parameter,
            test                 = "interaction_group_x_condition",
            healthy_stim_effect  = healthy_effect,
            impaired_stim_effect = impaired_effect,
            interaction          = observed_interaction,
            p_value              = p_value,
            **_effect_independent(h_t - h_s, i_t - i_s),
        ))

    return result_rows


# ── Switch-only transition columns ─────────────────────────────────────────────

def add_switch_only_cols(participant_df, k: int) -> pd.DataFrame:
    """
    Derive renormalised off-diagonal transitions and per-state switch entropy.
    Columns are added in-place and the DataFrame is returned.
    """
    if "Tsw_0_to_1" in participant_df.columns:
        return participant_df   # already computed in a previous run

    has_raw_transitions = all(
        f"T_{i}_to_{j}" in participant_df.columns
        for i in range(k) for j in range(k)
    )
    if not has_raw_transitions:
        logger.warning("raw transition columns not found, skipping switch-only derivation")
        return participant_df

    for from_state in range(k):
        off_diagonal_sum = sum(
            participant_df[f"T_{from_state}_to_{to_state}"]
            for to_state in range(k) if to_state != from_state
        )
        for to_state in range(k):
            if to_state != from_state:
                participant_df[f"Tsw_{from_state}_to_{to_state}"] = (
                    participant_df[f"T_{from_state}_to_{to_state}"].divide(
                        off_diagonal_sum.replace(0, np.nan)
                    )
                )

    for from_state in range(k):
        switch_probs = np.array([
            participant_df[f"Tsw_{from_state}_to_{to_state}"].fillna(0).values
            for to_state in range(k) if to_state != from_state
        ]) + 1e-10
        switch_probs /= switch_probs.sum(axis=0, keepdims=True)
        participant_df[f"sw_entropy_state{from_state}"] = scipy_entropy(
            switch_probs, axis=0
        )

    return participant_df


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_interaction(participant_df, param: str, microstate_names: list,
                     output_dir, tag: str = "",
                     sig_results=None, model=None, eeg_info=None) -> None:
    """
    Sham vs stimulation mean per microstate.
    Only microstates with at least one FDR-significant effect are shown.

    Colour = condition (markers); line style = group (connecting line).
    """
    significant_microstates = [
        ms for ms in microstate_names
        if f"{ms}_{param}" in participant_df.columns and (
            sig_results is None or
            ((sig_results["col"] == f"{ms}_{param}") & sig_results["sig_fdr"]).any()
        )
    ]
    if not significant_microstates:
        return

    cluster_maps = model.cluster_centers_ if model is not None else None
    fig, axes    = plt.subplots(
        1, len(significant_microstates),
        figsize=(4.2 * len(significant_microstates), 4.2),
        squeeze=False,
    )

    for ax, ms in zip(axes[0], significant_microstates):
        column             = f"{ms}_{param}"
        significant_groups = _get_significant_groups(column, sig_results)
        group_means        = (participant_df
                              .groupby(["group", "condition"])[column]
                              .mean()
                              .reset_index())

        for group in ("healthy", "impaired"):
            if group not in significant_groups:
                continue
            group_means_by_condition = (
                group_means[group_means["group"] == group]
                .set_index("condition")
            )
            if {"sham", "stim"} <= set(group_means_by_condition.index):
                ax.plot(
                    ["sham", "stim"],
                    [group_means_by_condition.loc["sham", column],
                     group_means_by_condition.loc["stim", column]],
                    ls=GROUP_LINE_STYLES[group], color="gray", lw=2,
                    label=group, zorder=1,
                )
                ax.scatter([0], [group_means_by_condition.loc["sham", column]],
                           color=CONDITION_COLORS["sham"], s=70, zorder=2)
                ax.scatter([1], [group_means_by_condition.loc["stim", column]],
                           color=CONDITION_COLORS["stim"], s=70, zorder=2)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["sham", "stim"])
        ax.set_title(f"MS {int(ms) + 1}")
        ax.set_xlabel("condition")
        ax.set_ylabel(param)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if cluster_maps is not None and eeg_info is not None:
            try:
                state_idx = int(ms)
            except (ValueError, TypeError):
                try:
                    state_idx = list(microstate_names).index(ms)
                except ValueError:
                    state_idx = None
            if state_idx is not None and state_idx < len(cluster_maps):
                add_topomap_inset(ax, cluster_maps, state_idx, eeg_info)

    fig.tight_layout()
    _save_figure(fig, output_dir, f"interaction_{tag}_{param}")
    plt.close(fig)


def plot_transition_matrices(participant_df, k: int, output_dir) -> None:
    """Heatmaps of the mean switch-only transition matrix per group/condition."""
    group_condition_pairs = [
        ("healthy",  "sham"), ("healthy",  "stim"),
        ("impaired", "sham"), ("impaired", "stim"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for ax, (group, condition) in zip(axes.flatten(), group_condition_pairs):
        subset            = participant_df[
            (participant_df["group"] == group)
            & (participant_df["condition"] == condition)
        ]
        transition_matrix = np.zeros((k, k))

        for from_state in range(k):
            for to_state in range(k):
                col = f"Tsw_{from_state}_to_{to_state}"
                if col in subset.columns and from_state != to_state:
                    transition_matrix[from_state, to_state] = subset[col].mean()

        off_diagonal_matrix = np.where(
            np.eye(k, dtype=bool), np.nan, transition_matrix
        )
        vmax    = (np.nanmax(off_diagonal_matrix)
                   if not np.all(np.isnan(off_diagonal_matrix)) else 0.5)
        heatmap = ax.imshow(off_diagonal_matrix, vmin=0, vmax=vmax,
                            cmap="plasma", aspect="auto")

        ax.set_title(f"{group} — {condition}")
        ax.set_xlabel("to state")
        ax.set_ylabel("from state")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))

        for from_state in range(k):
            for to_state in range(k):
                if from_state != to_state:
                    ax.text(to_state, from_state,
                            f"{transition_matrix[from_state, to_state]:.3f}",
                            ha="center", va="center", fontsize=7)
        plt.colorbar(heatmap, ax=ax, fraction=0.046)

    fig.suptitle(f"Switch-only transition matrices (K={k})", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"transition_matrices_k{k}")
    plt.close(fig)


def plot_transition_entropy(participant_df, k: int, output_dir,
                             sig_results=None, model=None, eeg_info=None) -> None:
    """
    Strip-plot of switch-only transition entropy per state (significant only).

    Colour = condition (stim/sham); line style of the mean bar = group.
    """
    all_entropy_cols = [f"sw_entropy_state{s}" for s in range(k)
                        if f"sw_entropy_state{s}" in participant_df.columns]
    sig_entropy_cols = (
        [col for col in all_entropy_cols
         if ((sig_results["col"] == col) & sig_results["sig_fdr"]).any()]
        if sig_results is not None else all_entropy_cols
    )
    if not sig_entropy_cols:
        return

    cluster_maps = model.cluster_centers_ if model is not None else None
    fig, axes    = plt.subplots(
        1, len(sig_entropy_cols),
        figsize=(4.2 * len(sig_entropy_cols), 4.2),
        squeeze=False, sharey=True,
    )

    for ax, column in zip(axes[0], sig_entropy_cols):
        state_label        = column.replace("sw_entropy_state", "")
        significant_groups = _get_significant_groups(column, sig_results)

        for group in ("healthy", "impaired"):
            if group not in significant_groups:
                continue
            group_df = participant_df[participant_df["group"] == group]
            for condition in ("sham", "stim"):
                condition_values = (group_df[group_df["condition"] == condition]
                                    [column].dropna().values)
                x_pos = 1 if condition == "stim" else 0
                ax.scatter(
                    x_pos + np.random.uniform(-0.1, 0.1, len(condition_values)),
                    condition_values,
                    color=CONDITION_COLORS[condition], alpha=0.55, s=30,
                )
                ax.plot(
                    [x_pos - 0.15, x_pos + 0.15],
                    [condition_values.mean()] * 2,
                    color=CONDITION_COLORS[condition],
                    ls=GROUP_LINE_STYLES[group], lw=2.5,
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["sham", "stim"])
        ax.set_title(f"MS {int(state_label) + 1}")
        ax.grid(True, alpha=0.2)

        if cluster_maps is not None and eeg_info is not None:
            try:
                state_idx = int(state_label)
                if state_idx < len(cluster_maps):
                    add_topomap_inset(ax, cluster_maps, state_idx, eeg_info)
            except ValueError:
                pass

    axes[0][0].set_ylabel("switch-transition entropy (bits)")
    axes[0][-1].legend(handles=_build_condition_legend(), fontsize=7, loc="lower right")
    fig.suptitle(f"Transition entropy — switch only (K={k})", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"transition_entropy_k{k}")
    plt.close(fig)


def plot_timecourse(participant_df, k: int, output_dir, erp_data=None,
                    model=None, eeg_info=None, sig_results=None) -> None:
    """
    Time-resolved microstate coverage with a grand-average ERP/GFP panel.

    Layout
    ------
    - One row per microstate (plus an optional ERP row at the top).
    - Colour = condition: orange-red = stim, Klein blue = sham.
    - Line style = group: long dash = healthy, short dash = impaired.
    - Time bins with FDR-significant condition effects are shaded using
      SIG_BIN_COLORS from config.
    - A compact ms map sits in the upper-right corner of each microstate panel
    """
    bin_time_cols = sorted(
        [c for c in participant_df.columns if c.startswith("bin") and "tcenter" in c],
        key=lambda x: int(x.split("bin")[1].split("_")[0]),
    )
    if not bin_time_cols:
        logger.warning("no bin*_tcenter columns found — skipping timecourse plot")
        return

    bin_centers    = participant_df[bin_time_cols].iloc[0].values.astype(float)
    n_time_bins    = len(bin_centers)
    bin_half_width = (
        COVERAGE_BIN_DURATION_S / 2
        if n_time_bins < 2
        else (bin_centers[1] - bin_centers[0]) / 2
    )

    group_condition_combos = [
        ("healthy",  "stim", CONDITION_COLORS["stim"], GROUP_LINE_STYLES["healthy"]),
        ("healthy",  "sham", CONDITION_COLORS["sham"], GROUP_LINE_STYLES["healthy"]),
        ("impaired", "stim", CONDITION_COLORS["stim"], GROUP_LINE_STYLES["impaired"]),
        ("impaired", "sham", CONDITION_COLORS["sham"], GROUP_LINE_STYLES["impaired"]),
    ]

    show_erp_panel      = erp_data is not None and "times" in erp_data
    n_panel_rows        = k + 1 if show_erp_panel else k
    cluster_maps        = model.cluster_centers_ if model is not None else None
    panel_height_ratios = ([1.8] + [1.0] * k) if show_erp_panel else [1.0] * k

    fig, axes = plt.subplots(
        n_panel_rows, 1,
        figsize=(14, 3.2 * n_panel_rows),
        sharex=True,
        gridspec_kw={"height_ratios": panel_height_ratios, "hspace": 0.45},
    )
    if n_panel_rows == 1:
        axes = [axes]

    #  ERP/GFP panel
    if show_erp_panel:
        ax_erp = axes[0]
        for group, condition, line_color, line_style in group_condition_combos:
            bucket_key = f"{group}_{condition}"
            if bucket_key in erp_data:
                ax_erp.plot(
                    erp_data["times"], erp_data[bucket_key],
                    color=line_color, ls=line_style, lw=1.8, alpha=0.9,
                    label=f"{group.capitalize()} — {condition}",
                )
        ax_erp.axvline(0, color="black", lw=1, ls=":", alpha=0.6)
        ax_erp.axhline(0, color="gray",  lw=0.5, alpha=0.4)
        erp_ylabel = f"Channel {ERP_CHANNEL} (µV)" if ERP_CHANNEL else "GFP (µV)"
        ax_erp.set_ylabel(erp_ylabel, fontsize=9)
        ax_erp.set_title(
            f"Grand-average ERP — {ERP_CHANNEL}" if ERP_CHANNEL else "Grand-average GFP",
            fontsize=10, fontweight="bold", pad=6,
        )
        ax_erp.grid(True, alpha=0.15)
        ax_erp.legend(handles=_build_condition_legend(), fontsize=8,
                      ncol=2, loc="upper right", framealpha=0.85)
        ax_erp.spines[["top", "right"]].set_visible(False)
        state_axes = axes[1:]
    else:
        state_axes = axes


    significant_bins = {s: {"healthy": set(), "impaired": set()} for s in range(k)}
    if sig_results is not None:
        fdr_mask = sig_results["sig_fdr"]
        for bin_idx in range(n_time_bins):
            for state in range(k):
                bin_col      = f"bin{bin_idx}_state{state}_cov"
                sig_for_bin  = sig_results[fdr_mask & (sig_results["col"] == bin_col)]
                for _, sig_row in sig_for_bin.iterrows():
                    test_name = sig_row["test"]
                    if "healthy_condition_effect"  in test_name:
                        significant_bins[state]["healthy"].add(bin_idx)
                    if "impaired_condition_effect" in test_name:
                        significant_bins[state]["impaired"].add(bin_idx)
                    if "group_effect" in test_name or "interaction" in test_name:
                        significant_bins[state]["healthy"].add(bin_idx)
                        significant_bins[state]["impaired"].add(bin_idx)

    # ms panels
    for state, ax in enumerate(state_axes):

        # Significance shading
        for group in ("healthy", "impaired"):
            for bin_idx in significant_bins[state][group]:
                ax.axvspan(
                    bin_centers[bin_idx] - bin_half_width,
                    bin_centers[bin_idx] + bin_half_width,
                    color=SIG_BIN_COLORS[group],
                    alpha=0.55, zorder=0, lw=0,
                )

        # Coverage lines
        for group, condition, line_color, line_style in group_condition_combos:
            subset       = participant_df[
                (participant_df["group"] == group)
                & (participant_df["condition"] == condition)
            ]
            coverage_cols = [
                f"bin{b}_state{state}_cov" for b in range(n_time_bins)
                if f"bin{b}_state{state}_cov" in subset.columns
            ]
            if coverage_cols and not subset.empty:
                ax.plot(
                    bin_centers[:len(coverage_cols)],
                    subset[coverage_cols].mean().values,
                    color=line_color, ls=line_style, lw=2, zorder=2,
                )

        ax.axvline(0, color="black", lw=1, ls=":", alpha=0.6, zorder=1)
        ax.set_ylabel("Coverage", fontsize=9)
        ax.set_title(f"MS {state + 1}", fontsize=10, fontweight="bold", pad=5)
        ax.grid(True, alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

        if cluster_maps is not None and eeg_info is not None and state < len(cluster_maps):
            add_topomap_inset(ax, cluster_maps, state, eeg_info)

    # Legend
    state_axes[-1].set_xlabel("Time relative to stimulus onset (s)", fontsize=10)

    filter_description = FILTER_TYPE.replace("_", " ").replace("filter", "").strip()
    fig.suptitle(
        f"Time-resolved microstate coverage  ·  K={k}"
        + (f"  ·  {filter_description} Hz band-pass" if filter_description else ""),
        fontsize=13, fontweight="bold", y=1.01,
    )

    legend_handles = _build_condition_legend() + [
        Patch(facecolor=SIG_BIN_COLORS["healthy"],  alpha=0.7,
              label="Significant bins, Healthy (FDR)"),
        Patch(facecolor=SIG_BIN_COLORS["impaired"], alpha=0.7,
              label="Significant bins, Impaired (FDR)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save_figure(fig, output_dir, f"timecourse_coverage_k{k}")
    plt.close(fig)


# ── Diagnostic figure ──────────────────────────────────────────────────────────

def plot_epoch_counts(participant_df, output_dir) -> None:
    """
    Scatter plot of per-participant epoch counts, with a Mann-Whitney test.

    This is a pure group diagnostic — the x-axis already separates
    healthy / impaired, so colour here is a neutral group grey and does NOT
    carry the stim/sham meaning used elsewhere.
    """
    from scipy.stats import mannwhitneyu

    group_grey = {"healthy": "#444444", "impaired": "#999999"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, condition in zip(axes, ("sham", "stim")):
        condition_df = participant_df[participant_df["condition"] == condition]
        for x_pos, group in enumerate(("healthy", "impaired")):
            epoch_counts = (condition_df[condition_df["group"] == group]
                            ["n_epochs"].values)
            ax.scatter(
                np.random.uniform(-0.1, 0.1, len(epoch_counts)) + x_pos,
                epoch_counts,
                color=group_grey[group], alpha=0.7, s=35, zorder=3,
            )
            ax.plot(
                [x_pos - 0.15, x_pos + 0.15], [epoch_counts.mean()] * 2,
                color=group_grey[group], lw=3, zorder=4,
            )
        ax.set_title(condition)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["healthy", "impaired"])
        ax.set_ylabel("n_epochs")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Epoch counts (used as permutation weights)", fontsize=11)
    fig.tight_layout()
    _save_figure(fig, output_dir, "epoch_counts")
    plt.close(fig)

    for condition in ("sham", "stim"):
        condition_df    = participant_df[participant_df["condition"] == condition]
        healthy_counts  = (condition_df[condition_df["group"] == "healthy"]
                           ["n_epochs"].values)
        impaired_counts = (condition_df[condition_df["group"] == "impaired"]
                           ["n_epochs"].values)
        _, p_value = mannwhitneyu(
            healthy_counts, impaired_counts, alternative="two-sided"
        )
        logger.info(
            f"  {condition}: healthy {healthy_counts.mean():.0f}"
            f"±{healthy_counts.std():.0f} "
            f"vs impaired {impaired_counts.mean():.0f}"
            f"±{impaired_counts.std():.0f} "
            f"(Mann-Whitney p={p_value:.3f})"
        )



# ── Helper ────────────────────────────────────────────────────────────────

def _get_output_dirs(filter_type: str) -> dict:
    base = MICROSTATES_DIR / filter_type
    return {
        "group":       base / "2. group_models",
        "backfitting": base / "3. backfitting",
        "statistics":  base / STATS_FOLDER,
    }


def _get_paired_values(participant_df, group: str, column: str,
                       weight_col: str = "w") -> tuple:
    """
    Collect matched sham stim values and weights for one group.
    """
    group_df = participant_df[participant_df["group"] == group]
    sham_vals, stim_vals, weights = [], [], []

    for pid in group_df["participant_id"].unique():
        pid_df   = group_df[group_df["participant_id"] == pid]
        sham_row = pid_df[pid_df["condition"] == "sham"]
        stim_row = pid_df[pid_df["condition"] == "stim"]

        if len(sham_row) and len(stim_row):
            sham_val = sham_row[column].values[0]
            stim_val = stim_row[column].values[0]
            if not (np.isnan(sham_val) or np.isnan(stim_val)):
                sham_vals.append(sham_val)
                stim_vals.append(stim_val)
                weight_sham = (sham_row[weight_col].values[0]
                               if weight_col in sham_row.columns else 1.)
                weight_stim = (stim_row[weight_col].values[0]
                               if weight_col in stim_row.columns else 1.)
                weights.append((weight_sham + weight_stim) / 2.)

    return np.array(sham_vals), np.array(stim_vals), np.array(weights)


def _get_independent_values(participant_df, group: str, condition: str,
                             column: str, weight_col: str = "w") -> tuple:
    """
    Collect values and weights for one group/condition cell.
    """
    subset  = participant_df[
        (participant_df["group"] == group)
        & (participant_df["condition"] == condition)
    ]
    values  = subset[column].dropna().values
    weights = (subset.loc[subset[column].notna(), weight_col].values
               if weight_col in subset.columns else np.ones(len(values)))
    return values, weights

def _save_figure(fig, output_dir, stem: str, dpi: int = 150) -> None:
    """Save a matplotlib figure as both PNG and SVG."""
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg",            bbox_inches="tight")

def load_group_model(filter_type: str, k: int):
    """Load and return the group-level K-means model for the given K"""
    model_path = _get_output_dirs(filter_type)["group"] / f"group_k{k}_model.pkl"
    if not model_path.exists():
        logger.warning(f"group model not found: {model_path}")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_eeg_info(filter_type: str):
    """Return MNE Info from the first available saved epoch file"""
    epochs_folder = EPOCHS_DIR / filter_type
    for participant_id in PARTICIPANT_IDS:
        for fname in (f"{participant_id}_sham-epo.fif",
                      f"{participant_id}_stim-epo.fif"):
            epoch_file = epochs_folder / fname
            if epoch_file.exists():
                return mne.read_epochs(
                    str(epoch_file), preload=False, verbose=False
                ).info
    return None


def add_topomap_inset(ax, cluster_maps, state_idx, eeg_info, loc="upper right"):
    """Place a compact topomap in a corner of *ax* (pastel colormap, 8 contours)."""
    try:
        inset_ax = inset_axes(ax, width="18%", height="75%",
                              loc=loc, borderpad=0.5)
        mne.viz.plot_topomap(
            cluster_maps[state_idx], eeg_info,
            axes=inset_ax, show=False,
            cmap=PASTEL_RDBU, contours=8, outlines="head",
            vlim=(-np.abs(cluster_maps[state_idx]).max(),
                   np.abs(cluster_maps[state_idx]).max()),
        )
        inset_ax.set_title(f"MS {state_idx + 1}", fontsize=6, pad=1)
        return inset_ax
    except Exception as exc:
        logger.debug(f"topomap inset failed for state {state_idx}: {exc}")
        return None


def _get_significant_groups(column, sig_results):
    """
    Return the set of groups that have an FDR-significant effect on *column*.
    Falls back to both groups when no sig_results are provided.
    """
    if sig_results is None:
        return {"healthy", "impaired"}

    sig_rows_for_col   = sig_results[(sig_results["col"] == column) & sig_results["sig_fdr"]]
    significant_groups = set()

    for _, row in sig_rows_for_col.iterrows():
        test_name = row["test"]
        if "healthy_condition_effect"  in test_name:
            significant_groups.add("healthy")
        if "impaired_condition_effect" in test_name:
            significant_groups.add("impaired")
        if "group_effect" in test_name or "interaction" in test_name:
            significant_groups.update({"healthy", "impaired"})

    return significant_groups if significant_groups else {"healthy", "impaired"}


def _build_condition_legend():
    """
    Legend handles for the new convention:
      colour  = condition (stim / sham)
      line style = group  (healthy / impaired)
    """
    return [
        Line2D([0], [0], color=CONDITION_COLORS["stim"], lw=2, label="Stim"),
        Line2D([0], [0], color=CONDITION_COLORS["sham"], lw=2, label="Sham"),
        Line2D([0], [0], color="gray", lw=2, ls=GROUP_LINE_STYLES["healthy"],
               label="Healthy"),
        Line2D([0], [0], color="gray", lw=2, ls=GROUP_LINE_STYLES["impaired"],
               label="Impaired"),
    ]


def _stim_night_number(participant_id: str) -> str:
    """Mirror of the same helper in a1_epochs """
    if STIM_NIGHT == "E1":
        return "03"
    if STIM_NIGHT == "E2":
        return "04"
    if STIM_NIGHT == "E3":
        return "05"
    return "04" if participant_id in USE_NIGHT_E2 else "03"


def _load_epochs_for_erp(participant_id: str, condition: str,
                          filter_type: str):
    """
    Load epochs for one participant / condition for ERP averaging.

    Loading priority
    ----------------
    1. Pre-saved .fif file from a1_epochs.py  (if saved)
    2. Raw EEGLAB .set/.fdt file epoched on-the-fly

    Returns an MNE Epochs object, or None if no source file was found.
    """
    # Fast path: pre-saved .fif
    saved_epochs_path = EPOCHS_DIR / filter_type / f"{participant_id}_{condition}-epo.fif"
    if saved_epochs_path.exists():
        epochs = mne.read_epochs(str(saved_epochs_path), preload=True, verbose=False)
        epochs.set_eeg_reference("average", projection=False)
        if "Cz" in epochs.ch_names:
            epochs.drop_channels(["Cz"])
        return epochs

    # Slow path: raw EEGLAB file
    subject_folder = DATA_ROOT / participant_id[:5] / EEG_SUBFOLDER

    if condition == "sham":
        file_pattern = f"{participant_id}-02*_scored_clean_PREP_AutoReject.set"
        marker_id    = "DI64" if participant_id in SHAM_USES_DI64 else "D128"
    else:
        night_number = _stim_night_number(participant_id)
        file_pattern = f"{participant_id}-{night_number}*_scored_clean_PREP_AutoReject.set"
        marker_id    = "DI64"

    matching_files = list(subject_folder.glob(file_pattern))
    if not matching_files:
        return None

    raw = mne.io.read_raw_eeglab(matching_files[0], preload=True, verbose=False)
    raw.set_montage(
        mne.channels.make_standard_montage("GSN-HydroCel-129"),
        on_missing="ignore",
    )
    raw.pick_types(eeg=True)

    low_freq, high_freq = FILTER_PRESETS[filter_type]
    if low_freq is not None or high_freq is not None:
        raw.filter(low_freq, high_freq, method="iir",
                   iir_params=dict(order=4, ftype="butter"), verbose=False)

    all_events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    if marker_id not in event_id_map:
        return None
    trigger_events = all_events[all_events[:, 2] == event_id_map[marker_id]]

    epochs = mne.Epochs(
        raw, trigger_events,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, reject=None, verbose=False,
    )
    del raw

    epochs.set_eeg_reference("average", projection=False)
    if "Cz" in epochs.ch_names:
        epochs.drop_channels(["Cz"])

    channels_without_position = [
        ch["ch_name"] for ch in epochs.info["chs"]
        if not np.isfinite(ch["loc"][:3]).all() or not np.any(ch["loc"][:3] != 0)
    ]
    if channels_without_position:
        epochs.drop_channels(channels_without_position)

    return epochs


def build_grand_avg_erp(filter_type: str, channel=None) -> dict:
    """
    Build grand-average evoked responses per group x condition.

    Returns a dict with keys 'times', 'healthy_stim', 'healthy_sham', etc.
    If *channel* is None, the GFP is returned instead of a single channel.
    """
    fif_dir   = EPOCHS_DIR / filter_type
    using_fif = fif_dir.exists() and bool(list(fif_dir.glob("*-epo.fif")))
    logger.info(
        f"  ERP source: "
        f"{'pre-saved .fif files' if using_fif else 'raw .set/.fdt files (on-the-fly)'}"
    )

    evoked_buckets = {
        f"{group}_{condition}": []
        for group     in ("healthy", "impaired")
        for condition in ("sham",    "stim")
    }
    n_loaded = n_failed = 0

    for participant_id in PARTICIPANT_IDS:
        group = "healthy" if participant_id in HEALTHY_IDS else "impaired"
        for condition in ("sham", "stim"):
            try:
                epochs = _load_epochs_for_erp(participant_id, condition, filter_type)
                if epochs is None:
                    n_failed += 1
                    continue
                evoked_buckets[f"{group}_{condition}"].append(epochs.average())
                n_loaded += 1
            except Exception as exc:
                logger.warning(
                    f"  ERP load failed — {participant_id} {condition}: {exc}"
                )
                n_failed += 1

    logger.info(
        f"  loaded {n_loaded} participant-conditions, {n_failed} failed / not found"
    )

    erp_data = {}
    for bucket_key, evoked_list in evoked_buckets.items():
        if not evoked_list:
            continue
        grand_average = mne.grand_average(evoked_list)
        if channel and channel in grand_average.ch_names:
            erp_data[bucket_key] = (
                grand_average.data[grand_average.ch_names.index(channel)] * 1e6
            )
        else:
            erp_data[bucket_key] = (
                np.sqrt((grand_average.data ** 2).mean(axis=0)) * 1e6
            )
        erp_data["times"] = grand_average.times

    logger.info(
        f"  grand-average ERP built for: "
        f"{[k for k in erp_data if k != 'times']}"
    )
    return erp_data



def match_labels_to_reference(participant_df, group_model, k: int):
    """
    Give the new microstates the numbering of the submitted analysis.

    K-means numbers clusters arbitrarily, so after re-clustering the same maps
    can come out in a different order. Each new map is assigned to the
    reference map with the highest absolute spatial correlation (polarity is
    ignored in microstate analysis), using a one-to-one assignment. Columns of
    the parameter table are renamed accordingly; the group model is reordered
    and its polarity aligned to the reference (display only).
    """
    assert k == 4, "reference maps exist for K = 4 only"
    ref_path = REFERENCE_MAPS
    ref_maps = np.load(ref_path)
    new_maps = group_model.cluster_centers_

    z = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    corr = z(new_maps) @ z(ref_maps).T / new_maps.shape[1]       # (new, ref)
    new_idx, ref_idx = linear_sum_assignment(-np.abs(corr))
    new_to_ref = dict(zip(new_idx.tolist(), ref_idx.tolist()))

    for n, r in new_to_ref.items():
        logger.info(f"  new map {n} -> MS {r + 1} (reference), r = {corr[n, r]:+.3f}")
    worst = min(abs(corr[n, r]) for n, r in new_to_ref.items())
    assert worst >= MIN_TEMPLATE_CORR, f"label match too weak (|r| = {worst:.2f})"

    # rename columns: state index n -> reference index r
    import re
    patterns = [r"^(\d+)(_.+)$", r"^(T_)(\d+)(_to_)(\d+)$",
                r"^(state)(\d+)(_transition_entropy)$", r"^(bin\d+_state)(\d+)(_cov)$"]
    def _rename(col):
        m = re.match(patterns[0], col)
        if m:
            return f"{new_to_ref[int(m.group(1))]}{m.group(2)}"
        m = re.match(patterns[1], col)
        if m:
            return f"T_{new_to_ref[int(m.group(2))]}_to_{new_to_ref[int(m.group(4))]}"
        for pat in patterns[2:]:
            m = re.match(pat, col)
            if m:
                return f"{m.group(1)}{new_to_ref[int(m.group(2))]}{m.group(3)}"
        return col
    participant_df = participant_df.rename(columns=_rename)

    # reorder model so that position r holds the map matched to reference r
    order = [n for n, _ in sorted(new_to_ref.items(), key=lambda x: x[1])]
    group_model.reorder_clusters(order=order)
    signs = [bool(corr[n, new_to_ref[n]] < 0) for n in order]
    group_model.invert_polarity(signs)
    return participant_df, group_model


# ── Sensitivity power analysis ────────────────────────────────────────────────

def sensitivity_power(n_hc: int, n_ci: int, alpha: float = 0.05, power: float = 0.80) -> pd.DataFrame:
    """Smallest effect detectable with the given power (two-sided t-test approximation of the
    permutation tests): paired tests across all participants and within each group (dz),
    independent HC vs. CI comparison, also used for the interaction on difference scores (d)."""
    from scipy import stats as _st, optimize as _opt
    def _pow(nc, df):
        tc = _st.t.ppf(1 - alpha / 2, df)
        return _st.nct.sf(tc, df, nc) + _st.nct.cdf(-tc, df, nc)
    rows = []
    for label, n in (("all participants", n_hc + n_ci), ("HC", n_hc), ("CI", n_ci)):
        rows.append(dict(test=f"within {label}", es="dz", n=n,
                         min_es=_opt.brentq(lambda d: _pow(d * np.sqrt(n), n - 1) - power, 0.05, 1.5)))
    rows.append(dict(test="HC vs. CI (between groups, interaction)", es="d", n=n_hc + n_ci,
                     min_es=_opt.brentq(lambda d: _pow(d * np.sqrt(n_hc * n_ci / (n_hc + n_ci)), n_hc + n_ci - 2) - power, 0.05, 1.5)))
    return pd.DataFrame(rows)


# ── Run ────────────────────────────────────────────────────────────────

logger.info("building grand-average ERP...")
erp_data = build_grand_avg_erp(FILTER_TYPE, channel=ERP_CHANNEL)

for K in ANALYSIS_K_VALUES:
    output_dirs = _get_output_dirs(FILTER_TYPE)
    params_file = output_dirs["backfitting"] / f"parameters_k{K}.csv"

    if not params_file.exists():
        logger.warning(f"K={K}: parameters file not found, skipping")
        continue

    output_dir = output_dirs["statistics"] / f"k{K}"
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_df = pd.read_csv(params_file)
    logger.info(f"K={K}: {len(participant_df)} rows loaded")

    n_per_group = participant_df.drop_duplicates("participant_id")["group"].value_counts().to_dict()
    logger.info(f"  group sizes: {n_per_group}")
    assert n_per_group == {"healthy": 25, "impaired": 15}, n_per_group
    sp = sensitivity_power(n_per_group["healthy"], n_per_group["impaired"])
    sp.to_csv(output_dir / "sensitivity_power.csv", index=False)
    logger.info("  sensitivity power (80%, alpha .05, two-sided):\n" + sp.round(2).to_string(index=False))
    
    microstate_names = [c.split("_")[0] for c in participant_df.columns if "_timecov" in c]
    logger.info(f"  microstates: {microstate_names}")

    # Participant weighting
    # main analysis: every participant counts equally (WEIGHT_BY_EPOCHS = False)
    # sensitivity analysis (as submitted): weighted by epoch count (True)
    if WEIGHT_BY_EPOCHS:
        participant_df["w"] = (participant_df["n_epochs"]
                               / participant_df["n_epochs"].mean())
    else:
        participant_df["w"] = 1.0
    logger.info(f"  weighting by epoch count: {WEIGHT_BY_EPOCHS}")

    # Microstate numbering as in the submitted analysis (before deriving columns)
    group_model = load_group_model(FILTER_TYPE, K)
    participant_df, group_model = match_labels_to_reference(participant_df, group_model, K)

    plot_epoch_counts(participant_df, output_dir)
    participant_df = add_switch_only_cols(participant_df, K)

    eeg_info    = getattr(group_model, "info", None) if group_model else None
    if eeg_info is None:
        eeg_info = get_eeg_info(FILTER_TYPE)

    rng         = np.random.default_rng(42)
    result_rows = []

    logger.info("  testing scalar parameters...")
    scalar_cols = [f"{ms}_{p}" for ms in microstate_names
                   for p in ("timecov", "meandurs", "occurrences", "gev")]
    result_rows += run_tests(participant_df, scalar_cols, "scalar", rng)

    logger.info("  testing switch-only transitions...")
    switch_cols = [
        f"Tsw_{i}_to_{j}" for i in range(K) for j in range(K)
        if i != j and f"Tsw_{i}_to_{j}" in participant_df.columns
    ]
    result_rows += run_tests(participant_df, switch_cols, "transition", rng)

    logger.info("  testing switch-only entropy...")
    entropy_cols = [f"sw_entropy_state{s}" for s in range(K)
                    if f"sw_entropy_state{s}" in participant_df.columns]
    result_rows += run_tests(participant_df, entropy_cols, "entropy", rng)

    logger.info("  testing time-binned coverage...")
    n_time_bins = (int(participant_df["n_time_bins"].iloc[0])
                   if "n_time_bins" in participant_df.columns else 0)
    bin_cols    = [
        f"bin{b}_state{s}_cov"
        for b in range(n_time_bins) for s in range(K)
        if f"bin{b}_state{s}_cov" in participant_df.columns
    ]
    result_rows += run_tests(participant_df, bin_cols, "timecov_bin", rng)

    # FDR correction within each test family
    stats_df               = pd.DataFrame(result_rows)
    stats_df["p_fdr"]      = np.nan
    stats_df["sig_fdr"]    = False
    stats_df["sig_uncorr"] = stats_df["p_value"] < 0.05

    # FDR correction within each test family AND contrast (as stated in Methods)
    for (family, test), idx in stats_df.groupby(["family", "test"]).groups.items():
        is_sig, adjusted_p = fdr_correct(stats_df.loc[idx, "p_value"].values)
        stats_df.loc[idx, "p_fdr"]   = adjusted_p
        stats_df.loc[idx, "sig_fdr"] = is_sig

    stats_df.to_csv(output_dir / f"permutation_results_k{K}.csv", index=False)

    sig_rows = stats_df[stats_df["sig_fdr"]]
    logger.info(f"  FDR-significant results: {len(sig_rows)}")
    for _, sig_row in sig_rows.iterrows():
        logger.info(
            f"    [{sig_row['family']}] {sig_row['col']}  {sig_row['test']}"
            f"  p={sig_row['p_value']:.4f}  p_fdr={sig_row['p_fdr']:.4f}"
            f"  {sig_row['es_type']}={sig_row['effect_size']:+.2f}"
        )

    for param in ("timecov", "meandurs", "occurrences", "gev"):
        plot_interaction(participant_df, param, microstate_names, output_dir,
                         tag=f"k{K}", sig_results=stats_df,
                         model=group_model, eeg_info=eeg_info)

    if switch_cols:
        plot_transition_matrices(participant_df, K, output_dir)

    plot_transition_entropy(participant_df, K, output_dir, sig_results=stats_df,
                            model=group_model, eeg_info=eeg_info)

    plot_timecourse(participant_df, K, output_dir, erp_data=erp_data,
                    model=group_model, eeg_info=eeg_info, sig_results=stats_df)

    logger.info(f"K={K} done")

logger.info("all done")