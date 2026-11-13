"""Central paths, analysis parameters, logger and font setup used by all scripts."""
import logging
import pandas as pd
from pathlib import Path


# ── Paths: adapt the two lines below; everything else is derived from them ─────

PROJECT_ROOT  = Path(r"G:\SW Topography\Analysis")   # analysis folder (participant lists, RAGU exports, results)
DATA_ROOT     = Path(r"G:\Data")                       # preprocessed EEG (<study>/<EEG_SUBFOLDER>/*.set)

EEG_SUBFOLDER   = "EEG-03-automated_artifact_rejection"
OUTPUT_BASE     = PROJECT_ROOT / "python" / "data_pycrostates"
EPOCHS_DIR      = OUTPUT_BASE / "epochs"
MICROSTATES_DIR = OUTPUT_BASE / "microstates"
LOG_DIR         = OUTPUT_BASE / "logs"
RAGU_DIR        = PROJECT_ROOT / "matlab_ragu"            # MATLAB/RAGU exports and results
PARTICIPANTS_FILE = PROJECT_ROOT / "python" / "Analysis_Manon" / "Analysis_Manon" / "participants_marina_selection.csv"
MASTER_PATH       = PROJECT_ROOT / "participants_master.csv"
SUPPLEMENT_DOCX   = PROJECT_ROOT.parent / "Review" / "Manuskript" / "Supplementary_Material_v2.docx"
# K = 4 template maps of the submitted analysis: reference for numbering the new maps MS 1-4
# (4 maps x 128 channels, no participant data; copied from microstates/filter_03_20_old/2. group_models)
REFERENCE_MAPS    = Path(__file__).resolve().parent / "reference" / "group_k4_maps_submitted.npy"

for _d in [EPOCHS_DIR, MICROSTATES_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Participant list ──────────────────────────────────────────────────────────

_df = pd.read_csv(PARTICIPANTS_FILE)

PARTICIPANT_IDS = _df[_df["inclusion"] == 1]["ID"].tolist()
USE_NIGHT_E2    = _df[_df["use_E2_instead"] == 1]["ID"].tolist()

# Group membership: MoCA-based (<=25 = CI), read from the master file
_master = pd.read_csv(MASTER_PATH)
_inc    = _master[_master["included"] == 1]
HEALTHY_IDS  = _inc.loc[_inc["group"] == "HC", "ID"].tolist()
IMPAIRED_IDS = _inc.loc[_inc["group"] == "CI", "ID"].tolist()

# Safety checks
assert set(PARTICIPANT_IDS) == set(_inc["ID"]), set(PARTICIPANT_IDS) ^ set(_inc["ID"])
assert (len(HEALTHY_IDS), len(IMPAIRED_IDS)) == (25, 15), (len(HEALTHY_IDS), len(IMPAIRED_IDS))

# Stim night selection
#   "E1"   use night 03 (1st stim night) for every participant 
#   "E2"   use night 04 (2nd stim night) for every participant
#.  "E3"   use night 05 (3rd stim night) for every participant
#   "per_participant"  use night 04 only for participants listed in USE_NIGHT_E2,
#                          night 03 for everyone else  (default)
STIM_NIGHT = "per_participant"


# ── Epoch parameters ───────────────────────────────────────────────────────────

EPOCH_TMIN = -1.5
EPOCH_TMAX =  3

# Band-pass filters
FILTER_PRESETS = {
    "as_is":        (None,  None),
    "filter_03_20": (0.3,  20.0),
    "filter_03_20_causal": (0.3,  20.0),   # same band, minimum-phase (causal) FIR: control for d4_filter_check.py
}

# Participant-specific lists are kept outside the repository (participant codes):
# PROJECT_ROOT/participant_lists.csv with columns ID, sham_marker_DI64, targeted_preprocessing (0/1)
_pl = pd.read_csv(PROJECT_ROOT / "participant_lists.csv", dtype={"ID": str})
# Participants whose sham night used DI64 instead of the D128 marker
SHAM_USES_DI64 = set(_pl.loc[_pl["sham_marker_DI64"] == 1, "ID"])
# Targeted preprocessing: only these subjects get epoch rejection.
PREPROCESS_SUBJECT_IDS = _pl.loc[_pl["targeted_preprocessing"] == 1, "ID"].tolist()

# Epoch rejection thresholds (from a00_config.m)
AMP_THRESH_UV       = 300      # ±µV amplitude threshold
TREND_MAX_UV_PER_SEC = 80      # max drift slope
MIN_EPOCHS           = 50      # minimum epochs to keep subject
MAX_REJ_FRAC         = 0.40    # max fraction of rejected epochs


# ── Clustering & backfitting parameters (microstate analysis) ─────────────

FILTER_TYPE           = "filter_03_20"   # which filter preset to use for the ms analysis (name should match epoch parameters)
K_VALUES              = [4, 5, 6, 7, 8]  # cluster counts to test in clustering
KMEANS_RESTARTS       = 20               # n_init for individual-level K-means
RANDOM_SEED           = 42

# Backfitting smoothing
# single-trial backfitting only (the average-level RAGU microstates use no smoothing and no minimum duration)
BACKFIT_SMOOTHING_FACTOR = 10
BACKFIT_SMOOTHING_WINDOW = 10  
BACKFIT_MIN_SEGMENT_SAMPLES = 5    
BACKFIT_REJECT_EDGES     = True

COVERAGE_BIN_DURATION_S  = 0.25  # temporal resolution for coverage time-courses (seconds)


# ── Statistics parameters ──────────────────────────────────────────────────────

ANALYSIS_K_VALUES = [4]        # K values to analyse in the stats script 
N_PERMUTATIONS    = 10000      # permutation iterations
ERP_CHANNEL       = "E11"      # channel for grand-average ERP panel. None -> GFP

# Color encodes CONDITION (matches Figure 3: orange = stim, blue = sham)
CONDITION_COLORS = {
    "stim": "#E63300",   # orange-red
    "sham": "#002FA7",   # Klein blue
}

# Line style encodes GROUP
GROUP_LINE_STYLES = {
    "healthy":  (0, (5, 2)),       # long dash
    "impaired": (0, (2, 2)),       # short dash
}

SIG_BIN_COLORS = {
    "healthy":  "#add8e6",   # light blue
    "impaired": "#ffd5a8",   # light orange
}

# Pastel diverging colormap for topomaps (matches Figure 3 maps)
from matplotlib.colors import LinearSegmentedColormap
PASTEL_RDBU = LinearSegmentedColormap.from_list('soft_RdBu', [
    '#002FA7', '#4d6fc4', '#a8b8e0', '#ffffff',
    '#f0b0a0', '#e85a3c', '#E63300'])


# ── Figure font: Arial for all figures ─────────────────────────────────────────
# Arial is not redistributable, so it is not part of the repository. If the font
# files (arial.ttf, arialbd.ttf) are placed in python/fonts/, they are registered
# here; otherwise matplotlib falls back to its default font.
import matplotlib
from matplotlib import font_manager as _fm
_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
if _FONT_DIR.is_dir():
    for _f in _FONT_DIR.glob("*.ttf"):
        _fm.fontManager.addfont(str(_f))
matplotlib.rcParams["font.family"] = "Arial" if any("Arial" == f.name for f in _fm.fontManager.ttflist) else "sans-serif"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


# ── Logger ─────────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_DIR / log_file, mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger