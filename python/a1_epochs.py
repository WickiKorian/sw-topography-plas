"""
Epoch extraction pipeline.

Loads raw EEGLAB .set/.fdt files, applies a band-pass filter,
epochs around the trigger of interest, and saves each participant's
sham and stimulation nights as MNE .fif files.

Revision changes (2026-09):
- epoch window taken from config (EPOCH_TMIN / EPOCH_TMAX = -1.5 / 3.0 s);
  previous epoch files had been created with -1.5 to 3.5 s
- filtering is applied separately to each continuous segment between
  boundary markers (cuts left by artefact removal), as in EEGLAB;
  segments shorter than one epoch are skipped (they cannot contain a valid epoch)
- epochs whose window contains a boundary are dropped, as in EEGLAB pop_epoch
- targeted rejection (7 flagged participants) harmonised with the MATLAB/RAGU
  pipeline: absolute amplitude > +/-300 uV only (as pop_eegthresh);
  previously 300 uV peak-to-peak plus a drift criterion. The drift criterion
  is not used, because it was not applied in the MATLAB pipeline
  (pop_rejtrend was called with reject=0, i.e. marking only)
- filter changed from IIR Butterworth to FIR (zero-phase, Hamming window,
  0.3-20 Hz), matching the MATLAB/RAGU pipeline (pop_eegfiltnew); verified
  on one sham recording: identical number of epochs rejected (285) in both pipelines
- run() accepts a separate output folder name (for test runs)
"""

import gc
import warnings
import mne
import numpy as np
from config import *

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = setup_logger("epochs", "a1_epochs.log")


# ── Core functions  ────────────────────────────────────────────────────────────

def process_one_night(participant_id: str, night_type: str,
                      output_path, filter_type: str) -> None:
    """
    Load, filter, epoch, and save one participant/night combination.

    Parameters
    ----------
    participant_id : str
    night_type     : "sham" or "stim"
    output_path    : Path  where to write the .fif file
    filter_type    : key of FILTER_PRESETS
    """
    subject_folder = DATA_ROOT / participant_id[:5] / EEG_SUBFOLDER

    if night_type == "sham":
        file_pattern = f"{participant_id}-02*_scored_clean_PREP_AutoReject.set"
        marker_id    = "DI64" if participant_id in SHAM_USES_DI64 else "D128"
    else:
        night_number = _stim_night_number(participant_id)
        file_pattern = f"{participant_id}-{night_number}*_scored_clean_PREP_AutoReject.set"
        marker_id    = "DI64"

    matching_files = list(subject_folder.glob(file_pattern))
    if not matching_files:
        raise FileNotFoundError(f"no file matching {subject_folder / file_pattern}")
    if len(matching_files) > 1:
        raise RuntimeError(f"more than one file matching {file_pattern}: {matching_files}")

    raw = mne.io.read_raw_eeglab(matching_files[0], preload=True)
    raw.set_montage(
        mne.channels.make_standard_montage("GSN-HydroCel-129"),
        on_missing="ignore",
    )
    raw.pick_types(eeg=True)
    _replace_non_finite(raw._data, "after load")

    # ── Band-pass filter, separately for each segment between boundaries ─────
    low_freq, high_freq = FILTER_PRESETS[filter_type]
    if low_freq is not None or high_freq is not None:
        _filter_by_segment(raw, low_freq, high_freq,
                           min_len_s=EPOCH_TMAX - EPOCH_TMIN,
                           phase="minimum" if filter_type.endswith("_causal") else "zero")
        _replace_non_finite(raw._data, "after filter")

    # ── Events ───────────────────────────────────────────────────────────────
    all_events, event_id_map = mne.events_from_annotations(raw)
    if marker_id not in event_id_map:
        raise KeyError(f"marker '{marker_id}' not found. available: {list(event_id_map)}")
    trigger_events = all_events[all_events[:, 2] == event_id_map[marker_id]]

    # ── Drop events whose epoch window contains a boundary ───────────────────
    sfreq = raw.info["sfreq"]
    b_on  = raw.annotations.onset[raw.annotations.description == "boundary"]
    t_ev  = (trigger_events[:, 0] - raw.first_samp) / sfreq
    has_b = np.array([np.any((b_on >= t + EPOCH_TMIN) & (b_on <= t + EPOCH_TMAX))
                      for t in t_ev], dtype=bool)
    logger.info(f"  {len(t_ev)} markers, {has_b.sum()} epochs contain a boundary -> dropped")
    trigger_events = trigger_events[~has_b]

    # ── Epoching ─────────────────────────────────────────────────────────────
    epochs = mne.Epochs(
        raw, trigger_events,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, reject=None, verbose=False,
    )
    del raw
    gc.collect()
    logger.info(f"  {len(epochs)} epochs after epoching "
                f"(window {EPOCH_TMIN} to {EPOCH_TMAX} s)")

    # ── Targeted epoch rejection (only for flagged subjects) ─────────────────
    # Same as MATLAB pop_eegthresh: absolute amplitude > ±300 µV (no drift criterion)
    if participant_id in PREPROCESS_SUBJECT_IDS:
        n_before = len(epochs)
        max_abs  = np.abs(epochs.get_data()).max(axis=(1, 2))
        bad      = np.where(max_abs > AMP_THRESH_UV * 1e-6)[0]
        if len(bad):
            epochs.drop(bad, reason="amplitude")

        n_after    = len(epochs)
        n_rejected = n_before - n_after
        rej_frac   = n_rejected / n_before if n_before > 0 else 0
        logger.info(f"  QC: {n_before} -> {n_after} epochs "
                    f"({n_rejected} rejected, {rej_frac:.1%})")

        if n_after < MIN_EPOCHS:
            raise RuntimeError(f"Too few epochs after rejection: {n_after} < {MIN_EPOCHS}")
        if rej_frac > MAX_REJ_FRAC:
            raise RuntimeError(f"Too many epochs rejected: {rej_frac:.1%} > {MAX_REJ_FRAC:.0%}")

    _replace_non_finite(epochs._data, "epochs")
    epochs = _drop_unlocated_channels(epochs)
    epochs.save(output_path, overwrite=True)
    logger.info(f"  saved {len(epochs)} epochs  ->  {output_path.name}")

    del epochs
    gc.collect()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _filter_by_segment(raw, l_freq, h_freq, min_len_s, phase="zero"):
    """
    Band-pass filter each continuous segment between boundary markers separately
    (as EEGLAB does). Segments shorter than one epoch cannot contain a valid
    epoch and are left unfiltered.
    phase: "zero" (default, as in the main analysis) or "minimum" (causal
    control filter: no backward projection of post-stimulus activity).
    """
    sfreq   = raw.info["sfreq"]
    n       = raw.n_times
    b_on    = raw.annotations.onset[raw.annotations.description == "boundary"]
    b_idx   = raw.time_as_index(b_on, use_rounding=True)
    edges   = np.unique(np.clip(np.r_[0, b_idx, n], 0, n))
    min_len = int(np.ceil(min_len_s * sfreq))

    n_filtered = n_skipped = 0
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop - start < min_len:
            n_skipped += 1
            continue
        raw._data[:, start:stop] = mne.filter.filter_data(
            raw._data[:, start:stop], sfreq, l_freq, h_freq,
            method="fir", phase=phase, fir_window="hamming",
            fir_design="firwin", verbose=False)
        n_filtered += 1
    logger.info(f"  filtered {n_filtered} segments, skipped {n_skipped} "
                f"shorter than {min_len_s:.1f} s")


def _replace_non_finite(data: np.ndarray, stage: str = "") -> None:
    """Replace NaN and Inf values in-place with zero (logged)."""
    non_finite_mask = ~np.isfinite(data)
    if non_finite_mask.any():
        logger.warning(f"  {non_finite_mask.sum()} NaN/Inf values found ({stage}), replacing with 0")
        data[non_finite_mask] = 0


def _drop_unlocated_channels(epochs: mne.Epochs) -> mne.Epochs:
    """Remove channels that have no valid 3D position in the montage."""
    channels_without_position = [
        ch["ch_name"] for ch in epochs.info["chs"]
        if not np.isfinite(ch["loc"][:3]).all() or not np.any(ch["loc"][:3] != 0)
    ]
    if channels_without_position:
        logger.info(f"  dropping {len(channels_without_position)} "
                    f"channel(s) without montage position: {channels_without_position}")
        epochs.drop_channels(channels_without_position)
    return epochs


def _stim_night_number(participant_id: str) -> str:
    """Night number ("03", "04", "05") of the stimulation night (see STIM_NIGHT in config)."""
    if STIM_NIGHT == "E1":
        return "03"
    if STIM_NIGHT == "E2":
        return "04"
    if STIM_NIGHT == "E3":
        return "05"
    return "04" if participant_id in USE_NIGHT_E2 else "03"   # per_participant


# ── Run ────────────────────────────────────────────────────────────────────────

def run(filter_type: str = "filter_03_20", participant_ids=None,
        output_name: str | None = None) -> None:
    """
    filter_type     key of FILTER_PRESETS (defines the filter)
    participant_ids list of IDs to process (default: all included)
    output_name     name of the output folder in EPOCHS_DIR (default: filter_type)
    """
    ids           = list(participant_ids) if participant_ids else list(PARTICIPANT_IDS)
    output_folder = EPOCHS_DIR / (output_name or filter_type)
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"output folder: {output_folder} | window {EPOCH_TMIN} to {EPOCH_TMAX} s")

    for idx, participant_id in enumerate(ids, 1):
        sham_path = output_folder / f"{participant_id}_sham-epo.fif"
        stim_path = output_folder / f"{participant_id}_stim-epo.fif"

        if sham_path.exists() and stim_path.exists():
            logger.info(f"[{idx}/{len(ids)}] {participant_id}: already done, skipping")
            continue

        logger.info(f"[{idx}/{len(ids)}] {participant_id}")
        try:
            if not sham_path.exists():
                logger.info("  sham night...")
                process_one_night(participant_id, "sham", sham_path, filter_type)
            if not stim_path.exists():
                logger.info("  stimulation night...")
                process_one_night(participant_id, "stim", stim_path, filter_type)
        except Exception as exc:
            logger.error(f"  failed: {exc}")
            gc.collect()


if __name__ == "__main__":
    run("filter_03_20")