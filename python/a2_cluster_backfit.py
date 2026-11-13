"""
Microstate clustering and backfitting pipeline.

Step 1  Individual clustering   fit K-means per participant on GFP peaks
Step 2  Group clustering        re-cluster individual maps into group templates
Step 3  Backfitting             assign group templates to every epoch and
                                extract microstate parameters
"""

import gc
import pickle
import traceback
import warnings
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks
from scipy.stats import entropy as scipy_entropy
from config import *

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = setup_logger("cluster_backfit", "a2_cluster_backfit.log")


# ── Create folder layout ──────────────────────────────────────────────────────────────
#   MICROSTATES_DIR / filter_type / 1. individual_models / {pid}_k{k}_model.pkl
#                                 / 2. group_models      / group_k{k}_model.pkl
#                                 / 3. backfitting       / parameters_k{k}.csv

def _get_output_dirs(filter_type: str) -> dict:
    base = MICROSTATES_DIR / filter_type
    return {
        "individual": base / "1. individual_models",
        "group":      base / "2. group_models",
        "backfitting": base / "3. backfitting",
    }



# ── Step 1: individual clustering ─────────────────────────────────────────────

def individual_clustering(filter_type: str, k_values: list) -> None:
    """Fit a separate K-means model for each participant and each K in k_values."""
    output_dirs = _get_output_dirs(filter_type)
    output_dirs["individual"].mkdir(parents=True, exist_ok=True)
    epochs_folder = EPOCHS_DIR / filter_type

    logger.info(f"step 1: individual clustering ({filter_type})")

    n_participants = len(PARTICIPANT_IDS)
    for idx, participant_id in enumerate(PARTICIPANT_IDS, 1):
        k_values_to_fit = [
            k for k in k_values
            if not (output_dirs["individual"] / f"{participant_id}_k{k}_model.pkl").exists()
        ]

        if not k_values_to_fit:
            logger.info(f"[{idx}/{n_participants}] {participant_id}: all models exist, skipping")
            continue

        logger.info(f"[{idx}/{n_participants}] {participant_id}: fitting K={k_values_to_fit}")

        try:
            sham = _load_fif_epochs(epochs_folder / f"{participant_id}_sham-epo.fif")
            peaks_sham = extract_gfp_peaks(sham)
            del sham; gc.collect()

            stim = _load_fif_epochs(epochs_folder / f"{participant_id}_stim-epo.fif")
            peaks_stim = extract_gfp_peaks(stim)
            del stim; gc.collect()

            gfp_peaks = ChData(np.hstack([peaks_sham.get_data(), peaks_stim.get_data()]),
                               peaks_sham.info)
            logger.info(f"  {gfp_peaks.get_data().shape[1]} GFP peaks extracted")

            for k in k_values_to_fit:
                kmeans_model = ModKMeans(
                    n_clusters=k, random_state=RANDOM_SEED, n_init=KMEANS_RESTARTS
                )
                kmeans_model.fit(gfp_peaks, n_jobs= 2, verbose=False)

                model_path = output_dirs["individual"] / f"{participant_id}_k{k}_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(kmeans_model, f)

                logger.info(f"  K={k}: GEV={kmeans_model.GEV_:.3f}")
                del kmeans_model

            del gfp_peaks
            gc.collect()

        except Exception as exc:
            logger.error(f"  failed: {exc}")
            gc.collect()


# ── Step 2: group clustering ───────────────────────────────────────────────────

def group_clustering(filter_type: str, k_values: list) -> None:
    """
    Stack individual cluster maps and re-cluster them into group-level templates,
    one model per K value.
    """
    output_dirs = _get_output_dirs(filter_type)
    output_dirs["group"].mkdir(parents=True, exist_ok=True)
    epochs_folder = EPOCHS_DIR / filter_type

    gev_scores_file = output_dirs["group"] / "gev_scores.csv"
    if gev_scores_file.exists():
        logger.info("step 2: group clustering already complete, skipping")
        return

    logger.info(f"step 2: group clustering ({filter_type})")

    gev_results = []

    for k in k_values:
        group_model_path = output_dirs["group"] / f"group_k{k}_model.pkl"

        if group_model_path.exists():
            with open(group_model_path, "rb") as f:
                kmeans_model = pickle.load(f)
            gev_results.append({"k": k, "gev": kmeans_model.GEV_})
            logger.info(f"  K={k}: model exists (GEV={kmeans_model.GEV_:.3f}), skipping")
            continue

        # Collect and mean-centre the individual cluster maps
        individual_maps = []
        for participant_id in PARTICIPANT_IDS:
            individual_model_path = (
                output_dirs["individual"] / f"{participant_id}_k{k}_model.pkl"
            )
            if individual_model_path.exists():
                with open(individual_model_path, "rb") as f:
                    kmeans_model = pickle.load(f)
                cluster_maps  = kmeans_model.cluster_centers_.copy()
                cluster_maps -= cluster_maps.mean(axis=1, keepdims=True)
                individual_maps.append(cluster_maps)

        if not individual_maps:
            logger.warning(f"  K={k}: no individual models found, skipping")
            continue

        logger.info(f"  K={k}: re-clustering {len(individual_maps)} participant maps")

        # Use channel info from the first participant's sham epochs
        reference_epochs = _load_fif_epochs(
            epochs_folder / f"{PARTICIPANT_IDS[0]}_sham-epo.fif"
        )
        group_ch_data = ChData(np.vstack(individual_maps).T, reference_epochs.info)
        del reference_epochs

        kmeans_model = ModKMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=50)
        kmeans_model.fit(group_ch_data, n_jobs=-1, verbose=False)
        del group_ch_data
        gc.collect()

        gev_results.append({"k": k, "gev": kmeans_model.GEV_})

        with open(group_model_path, "wb") as f:
            pickle.dump(kmeans_model, f)
        np.save(output_dirs["group"] / f"group_k{k}_maps.npy",
                kmeans_model.cluster_centers_)
        
        # plot the final ms maps
        fig = kmeans_model.plot(cmap="RdBu_r", contours=True)
        fig.savefig(output_dirs["group"] / f"group_k{k}_maps.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        min_map_correlation = np.corrcoef(kmeans_model.cluster_centers_).min()
        logger.info(f"  K={k}: GEV={kmeans_model.GEV_:.3f}, "
                    f"min_map_corr={min_map_correlation:.3f}")
        del kmeans_model

    if not gev_results:
        logger.warning("no GEV scores collected")
        return

    gev_table = pd.DataFrame(gev_results)
    gev_table.to_csv(gev_scores_file, index=False)

    # GEV plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gev_table["k"], gev_table["gev"], "o-", lw=2, ms=7)
    ax.set_xlabel("K (number of microstates)")
    ax.set_ylabel("Global Explained Variance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dirs["group"] / "gev_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Step 3: backfitting ────────────────────────────────────────────────────────

def backfit(filter_type: str, k: int) -> pd.DataFrame | None:
    """
    Assign the group-level microstate templates to every epoch of every
    participant and extract per-participant/condition parameters.
    Saves results to a CSV and returns the DataFrame.
    """
    output_dirs = _get_output_dirs(filter_type)
    output_dirs["backfitting"].mkdir(parents=True, exist_ok=True)
    epochs_folder = EPOCHS_DIR / filter_type

    output_csv = output_dirs["backfitting"] / f"parameters_k{k}.csv"
    if output_csv.exists():
        logger.info(f"step 3: backfitting K={k} already complete, skipping")
        return pd.read_csv(output_csv)

    group_model_path = output_dirs["group"] / f"group_k{k}_model.pkl"
    if not group_model_path.exists():
        logger.error(f"step 3: group model not found at {group_model_path}")
        return None

    with open(group_model_path, "rb") as f:
        group_model = pickle.load(f)

    microstate_names = group_model.cluster_names
    logger.info(f"step 3: backfitting K={k}, microstates: {microstate_names}")
    logger.info(
        f"  smoothing factor={BACKFIT_SMOOTHING_FACTOR}, "
        f"window={BACKFIT_SMOOTHING_WINDOW} samples, "
        f"min segment={BACKFIT_MIN_SEGMENT_SAMPLES} samples"
    )

    # Resume from checkpoint if a previous run was interrupted
    checkpoint_csv = output_dirs["backfitting"] / f"parameters_k{k}_temp.csv"
    if checkpoint_csv.exists():
        results = pd.read_csv(checkpoint_csv).to_dict("records")
        logger.info(f"  resuming from checkpoint ({len(results)} rows already saved)")
    else:
        results = []

    n_participants = len(PARTICIPANT_IDS)
    for idx, participant_id in enumerate(PARTICIPANT_IDS, 1):
        for condition, epoch_filename in [
            ("sham", f"{participant_id}_sham-epo.fif"),
            ("stim", f"{participant_id}_stim-epo.fif"),
        ]:
            # Skip if already processed in a previous (interrupted) run
            already_done = any(
                r["participant_id"] == participant_id and r["condition"] == condition
                for r in results
            )
            if already_done:
                continue

            epoch_file = epochs_folder / epoch_filename
            if not epoch_file.exists():
                logger.warning(
                    f"[{idx}/{n_participants}] {participant_id} {condition}: "
                    f"file not found, skipping"
                )
                continue

            try:
                epochs      = _load_fif_epochs(epoch_file)
                sampling_rate = epochs.info["sfreq"]

                segmentation = group_model.predict(
                    epochs,
                    factor            = BACKFIT_SMOOTHING_FACTOR,
                    half_window_size  = BACKFIT_SMOOTHING_WINDOW,
                    min_segment_length = BACKFIT_MIN_SEGMENT_SAMPLES,
                    reject_edges      = BACKFIT_REJECT_EDGES,
                )
                ms_params = segmentation.compute_parameters()

                result_row = {
                    "participant_id": participant_id,
                    "group":          ("healthy" if participant_id in HEALTHY_IDS
                                       else "impaired" if participant_id in IMPAIRED_IDS
                                       else "UNKNOWN"),
                    "condition":      condition,
                    "k":              k,
                    "n_epochs":       len(epochs),
                }

                for ms_name in microstate_names:
                    for metric in ("timecov", "meandurs", "occurrences", "gev", "mean_corr"):
                        result_row[f"{ms_name}_{metric}"] = ms_params.get(
                            f"{ms_name}_{metric}", np.nan
                        )

                state_labels = _extract_labels(segmentation)
                labels_flat  = state_labels.flatten()
                transition_matrix = compute_transition_matrix(labels_flat, k)
                for from_state in range(k):
                    for to_state in range(k):
                        result_row[f"T_{from_state}_to_{to_state}"] = (
                            transition_matrix[from_state, to_state]
                        )
                for from_state in range(k):
                    result_row[f"state{from_state}_transition_entropy"] = float(
                        scipy_entropy(transition_matrix[from_state] + 1e-10)
                    )

                bin_coverage, n_time_bins = compute_time_binned_coverage(
                    state_labels, sampling_rate,
                    EPOCH_TMIN, COVERAGE_BIN_DURATION_S, k,
                )
                result_row.update(bin_coverage)
                result_row["n_time_bins"] = n_time_bins

                results.append(result_row)
                logger.info(f"[{idx}/{n_participants}] {participant_id} {condition}: done")

                pd.DataFrame(results).to_csv(checkpoint_csv, index=False)

                del epochs, segmentation, state_labels
                gc.collect()

            except Exception as exc:
                logger.error(
                    f"[{idx}/{n_participants}] {participant_id} {condition}: failed — {exc}"
                )
                logger.debug(traceback.format_exc())
                gc.collect()

    if not results:
        logger.warning("no backfitting results to save")
        return None

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    if checkpoint_csv.exists():
        checkpoint_csv.unlink()

    logger.info(f"step 3: done. {len(results_df)} rows saved to {output_csv.name}")
    return results_df


# ── Parameters ────────────────────────────────────────────────────────────────────

def compute_transition_matrix(labels_flat: np.ndarray,
                               n_states: int) -> np.ndarray:
    """
    Compute a row-normalised transition probability matrix.
    Self-transitions are included. Unlabelled samples are skipped.
    """
    transition_counts = np.zeros((n_states, n_states))
    for t in range(len(labels_flat) - 1):
        from_state = int(labels_flat[t])
        to_state   = int(labels_flat[t + 1])
        if from_state >= 0 and to_state >= 0:
            transition_counts[from_state, to_state] += 1

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        transition_counts, row_sums,
        out=np.zeros_like(transition_counts), where=row_sums > 0
    )
    return transition_matrix


def compute_time_binned_coverage(state_labels: np.ndarray, sampling_rate: float,
                                  epoch_tmin: float, bin_duration_s: float,
                                  n_states: int) -> tuple:
    """
    Compute the fraction of each time bin occupied by each microstate.
    """
    samples_per_bin = int(bin_duration_s * sampling_rate)
    n_time_bins     = state_labels.shape[1] // samples_per_bin
    bin_coverage    = {}

    for bin_idx in range(n_time_bins):
        bin_center_time = epoch_tmin + (bin_idx + 0.5) * bin_duration_s
        bin_coverage[f"bin{bin_idx}_tcenter"] = round(bin_center_time, 4)

        bin_labels  = state_labels[:, bin_idx * samples_per_bin:
                                      (bin_idx + 1) * samples_per_bin].flatten()
        valid_labels = bin_labels[bin_labels >= 0]

        for state in range(n_states):
            bin_coverage[f"bin{bin_idx}_state{state}_cov"] = (
                float(np.mean(valid_labels == state))
                if len(valid_labels) > 0 else np.nan
            )

    return bin_coverage, n_time_bins


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_labels(segmentation) -> np.ndarray:
    """
    Pull the state-label array from a pycrostates segmentation object,
    handling attribute-name differences across library versions.
    """
    for attr_name in ("labels_", "_labels", "labels"):
        if hasattr(segmentation, attr_name):
            value = getattr(segmentation, attr_name)
            return np.array(value() if callable(value) else value)
    raise AttributeError(
        f"cannot find labels attribute on {type(segmentation).__name__}"
    )


def _load_fif_epochs(path) -> mne.Epochs:
    """Load a .fif epoch file, apply average reference, and drop Cz."""
    epochs = mne.read_epochs(str(path), preload=True, verbose=False)
    epochs.set_eeg_reference("average", projection=False)
    if "Cz" in epochs.ch_names:
        epochs.drop_channels(["Cz"])
    return epochs


# ── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    individual_clustering(FILTER_TYPE, K_VALUES)
    group_clustering(FILTER_TYPE, K_VALUES)

    gev_scores_file = _get_output_dirs(FILTER_TYPE)["group"] / "gev_scores.csv"
    if not gev_scores_file.exists():
        logger.error("GEV scores file missing, group clustering may not have finished")
    else:
        gev_table = pd.read_csv(gev_scores_file)
        for k in ANALYSIS_K_VALUES:          # K = 4, fixed a priori
            backfit(FILTER_TYPE, k=k)

    logger.info("pipeline complete")