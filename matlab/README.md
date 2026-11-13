# Average-level analysis (MATLAB / RAGU)

## Scripts
| Script | What it does |
|---|---|
| a00_config.m | Paths (adapt `cfg.projectRoot`, `cfg.dataRoot`), participants, epoch window, filter, quality-control settings |
| a01_compute_erps.m | Per participant and night: band-pass filter 0.3–20 Hz (zero-phase FIR, `pop_eegfiltnew`, per continuous segment), epochs −1.5 to 3 s around the (pseudo-)stimulus markers, ±300 µV rejection for the 7 flagged participants, average ERP. No baseline correction. Channel detection/interpolation is implemented but switched off (`cfg.qc.doBadChannelDetection = false`, `cfg.qc.doInterpolation = false`). |
| a02_export_ragu.m | Export of the participant ERPs for RAGU: `derived/RAGU_TXT_RowsTime_Final/<ID>_BL.txt`, `<ID>_STIM.txt` (900 time points × 129 channels) and `derived/RAGU_export_BL_vs_STIM.mat` |
| c01_ragu_overlap_subset.m | RAGU file for an overlap subset (data from `python/c4_ragu_overlap_subsets.py`), same settings as the main TANOVA |
| c02_ragu_nights.m | RAGU file for one experimental night (data from `python/d6_ragu_nights.py`), same settings as the main TANOVA, participants without that night removed |
| run_all.m | a00 → a01 → a02 |
| EGI129.xyz | Electrode positions (EGI HydroCel 129) |

## RAGU analyses (files in `Ragu_results/`)
Design: condition (baseline vs. stimulation night, within) × group (HC vs. CI, between;
MoCA > 25 / ≤ 25), N = 40 (25 / 15). L2-normalised maps (topography independent of
strength), 5000 randomisations, p < .05. RAGU version: see the Methods of the paper.
Groups are read by RAGU from an Excel file (`RAGU_groups.xlsx`, not shared because it
contains participant IDs); it assigns each participant to HC (MoCA > 25) or CI (MoCA ≤ 25).
The global duration criterion of the TANOVA is not a parameter: RAGU derives it from the
randomisation distribution (Koenig et al., 2011).

| File | Analysis | Paper |
|---|---|---|
| TANOVA_-1500_3000ms_ConditionXGroup_new.mat | TANOVA, full epoch | Results (TANOVA) |
| TANOVA_-200_800ms_ConditionXGroup_new.mat | TANOVA, first slow-wave cycle | Results |
| TANOVA_800_1900ms_ConditionXGroup_new.mat | TANOVA, second induced peak | Results |
| TANOVA_-1500_3000ms_ConditionXSex.mat | TANOVA with sex instead of group | Methods / Results (sex) |
| TANOVA_-1500_3000ms_ConditionXGroup_prev3.7.mat | TANOVA, epochs without a preceding stimulus within 3.7 s (c01) | overlap check |
| TANOVA_-1500_3000ms_ConditionXGroup_E1/E2/E3.mat | TANOVA per experimental night, baseline vs. E1/E2/E3 (c02) | control analyses (nights) |
| Microstates_-200_800ms_ConditionXGroup_HCCI.mat | Microstates, 8 classes (cross-validation 3–10), −200 to 800 ms | Figure 3, Table S2, Figure 5 |
| Microstates_800_1800ms_ConditionXGroup_new.mat | Same 8 maps, 800 to 1900 ms (frames 461–681; file name historical) | Table S2 |

Microstate settings: k-means, 50 restarts, 100 re-iterations, cross-validation 3–10
classes, no temporal smoothing of labels, polarity taken into account.

Python scripts that read these files: b4_ragu_microstate_table, c4_ragu_overlap_subsets,
d4_filter_check, d6_ragu_nights (results), e1_figure2, e2_figure4, e3_figure5_correspondence.

## archive/
Earlier scripts (epoch export to Python, ERP figures, 4-night version, old runners).
