# Single-trial microstate analysis (Python) – SW topography manuscript

Scripts for the single-trial (pycrostates) analysis, the Python parts of the
figures and the Supplementary Material. The average-level analysis (RAGU/TANOVA,
Figure 3) was run in MATLAB/RAGU; several scripts read its exported results.

## How to run
- All paths and shared settings are in `config.py` (data and output folders,
  participant list and groups, filter, epoch window, backfitting and statistics
  parameters). Adapt the paths there; nothing else needs to be changed.
- `reference/group_k4_maps_submitted.npy`: the K = 4 maps of the submitted analysis (4 maps × 128
  channels, no participant data). The new maps are numbered MS 1–4 by their spatial
  correlation with these maps, so that the numbering matches the manuscript.
- Run the scripts from this folder, in alphabetical order (letter = group,
  number = order within the group), or with `run_all.ipynb`.

## Scripts

| Script | What it does | Paper |
|---|---|---|
| **a — Main pipeline** | | |
| a1_epochs | Band-pass filter (0.3–20 Hz, zero-phase FIR, per continuous segment) and epochs (−1.5 to 3 s) of the baseline and stimulation night | Methods |
| a2_cluster_backfit | Individual and group modified k-means (K = 4), backfitting, microstate parameters | Methods, Results |
| a3_stats | Permutation tests with FDR correction on all parameters (`WEIGHT_BY_EPOCHS = True`: weighted sensitivity analysis) | Results, Tables S3, S4, Figure S2 |
| a4_coverage_clusters | Time-resolved coverage, cluster-based permutation tests | Results, Figure 4 |
| **b — Sample and data description** | | |
| b1_sleep_architecture | Sleep architecture and stimulus counts | Table S1 |
| b2_epoch_counts | Epochs per participant | Table 2 |
| b3_interpolated_channels | Interpolated channels per recording | Methods |
| b4_ragu_microstate_table | Average-level (RAGU) microstate parameters with FDR | Table S2, Results |
| **c — Overlap with neighbouring stimuli** | | |
| c1_stimulus_intervals | Time to the preceding stimulus (share of epochs with overlap, sleep stage of stimuli) and to the following stimulus (stimulation vs. sham) | Figure S3A, Supplement (Figure S3), Methods, Discussion |
| c2_overlap_variants | Subsets without neighbouring stimuli (parameters, 250-ms bins) | Table S5 |
| c3_coverage_clusters_isolated | Cluster tests (as a4) in these subsets; exploratory test of MS 1 in the second-peak window | Figure S3C, Results |
| c4_ragu_overlap_subsets | Second peak and topography (ERP level) by subset; RAGU input per subset | Table S8 |
| c5_second_peak_single_trial | Single-trial expression of the second induced peak (amplitude, not topography) | Response letter only |
| **d — Further control analyses** | | |
| d1_all_nights | Analysis per stimulation night (E1–E3) | Table S6 |
| d2_nights_comparison | Night and night × group effects | Table S6 |
| d3_sex_sensitivity | Sex as covariate | Table S7 |
| d4_filter_check | Zero-phase vs. causal filter (pre-stimulus effect) | Table S9 |
| d6_ragu_nights | RAGU text files per experimental night (baseline vs. E1/E2/E3) for the per-night TANOVA (then matlab_ragu/c02_ragu_nights.m) | Results, Control analyses; Table S6 note |
| **e — Figures** (Python parts; layout in Illustrator) | | |
| e1_figure2 | ERP at Fz with 95% CI, GFP-normalised topographies, colour bar | Figure 2 |
| e2_figure4 | Coverage time courses, clusters, templates | Figure 4, Figure S2 |
| e3_figure5_correspondence | Spatial correlation of single-trial and RAGU maps | Figure 5, Results |
| **f — Supplementary Material** | | |
| f1_supplement | Builds the Supplementary Material (.docx) from all results | Supplement |

## Dependencies
- a1 → a2 → a3 → a4 first. c2, d1, d2, d3 reuse functions of a3; c1 (part B) and
  c3 reuse the epoch selection of c2; e2 uses a4.
- b4, c4, d4, e1, e2 and e3 need the exported RAGU results (MATLAB).
- c3 can be run in several calls (`TIME_BUDGET_S`); it resumes from its cache.
  c5 likewise (`C5_BUDGET_S`).

## Output folders
Results are written to `data_pycrostates/microstates/filter_03_20/`. The folder
numbers there are historical and do not correspond to the script names:

| Folder | Script |
|---|---|
| 1. individual_models, 2. group_models, 3. backfitting | a2 |
| 4. statistics, 4. statistics_weighted | a3 |
| 5. coverage_clusters | a4 |
| 5. overlap_check | c1 (part A) |
| 7. all_nights | d1, d2 |
| 8. sex_sensitivity | d3 |
| 9. sleep_architecture | b1 |
| 10. second_peak | c5 |
| 13. overlap_variants | c2 |
| 15. filter_check | d4 |
| 16. interpolated_channels | b3 |
| 17. figure4 | e2 |
| 18. next_stimulus_timing | c1 (part B) |
| 19. coverage_clusters_isolated | c3 |

Average-level outputs (b4, c4, d4, e1, e3) are in `matlab_ragu/derived/` and
`matlab_ragu/Ragu_results/`.

## archive/
Earlier versions, superseded analyses and helper scripts not used in the paper.
