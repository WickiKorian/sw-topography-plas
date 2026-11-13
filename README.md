# Topography of natural versus PLAS-enhanced slow waves – analysis code

Code for: Wicki et al., *Topography of Natural versus PLAS-Enhanced Slow Waves in
Healthy and Cognitively Impaired Older Adults* (submitted to *Sleep*).

## Structure
| Folder | Content |
|---|---|
| `matlab/` | Average-level path: subject ERPs and export to RAGU (EEGLAB). RAGU (Koenig et al.) was used for TANOVA and microstate analysis of the averages; settings and result files in `matlab/README.md`. |
| `python/` | Single-trial microstate analysis (pycrostates), control analyses, Python parts of the figures, Supplementary Material. See `python/README.md`. |
| `r/` | Sample characteristics (Table 1). |

## Requirements
- Python 3.11: see `requirements.txt` (`pip install -r requirements.txt`).
- MATLAB with EEGLAB; RAGU for the average-level analysis.

## Input data
The scripts start from preprocessed, sleep-scored EEG recordings
(`*_scored_clean_PREP_AutoReject.set`; 128-channel EGI, 200 Hz) of the baseline
and stimulation nights, and from participant lists with inclusion and group
(MoCA). A further list, `participant_lists.csv` in `PROJECT_ROOT`, gives the participants with a
different sham marker and those with targeted epoch rejection (columns `ID`, `sham_marker_DI64`,
`targeted_preprocessing`). The data and these lists are not included: they contain sensitive clinical data and are available in
controlled access upon reasonable request to the corresponding author (see the Data availability
statement of the manuscript).

## Running
1. Set the paths at the top of `python/config.py` (`PROJECT_ROOT`, `DATA_ROOT`)
   and in `matlab/a00_config.m`.
2. MATLAB: `matlab/run_all.m` (a00 → a01 → a02); then the RAGU analyses listed in `matlab/README.md`.
3. Python: run the scripts in `python/` in alphabetical order (`run_all.ipynb`).
4. Average-level control analyses: the RAGU input for the overlap subsets and the per-night
   analysis is exported by `python/c4_ragu_overlap_subsets.py` and `python/d6_ragu_nights.py`;
   `matlab/c01_ragu_overlap_subset.m` and `matlab/c02_ragu_nights.m` create the RAGU files
   with the settings of the main TANOVA (see `matlab/README.md`).

Figures use Arial. Because Arial cannot be redistributed, it is not included: to reproduce the
figures exactly, place `arial.ttf` and `arialbd.ttf` in a folder `fonts/` at the top of the
repository; otherwise the default font is used.

The single-trial microstate pipeline (a1–a3) is based on scripts by Manon Haulotte.

## License
MIT (see `LICENSE`).
