%% ========================================================================
%  00_config.m
%
%  Project: Slow-wave topography analysis (EEG)
%
%  Description:
%  ------------
%  Central configuration file for the analysis pipeline.
%  Defines all paths, preprocessing parameters, and analysis settings
%  used across scripts.
%
%  This file must be executed before running any other pipeline script.
%
%  Contents:
%  ---------
%  - Project and data paths
%  - ERP epoching and baseline parameters
%  - Sampling and time-axis definitions
%  - Marker definitions (baseline / stimulation)
%  - Output file locations
%
%  All downstream scripts assume that the variable `cfg` exists.
%
%  ========================================================================


clc;

%% ------------------------------------------------------------------------
%  PROJECT PATHS
%  ------------------------------------------------------------------------

% ---- adapt these two lines; everything else is derived from them ----
cfg.projectRoot = "G:\SW Topography\Analysis\matlab_ragu";   % this folder (derived data, logs, RAGU files)
cfg.dataRoot    = "G:\Data";                                    % preprocessed EEG (<study>/<eegSubfolder>/*.set)
cfg.metadataFile = fullfile(cfg.dataRoot, "Subjects_different_projects.xlsx");
cfg.sheetName    = "Microst_comp_algo_peaks_BL_E1E2";

cfg.eegSubfolder = "EEG-03-automated_artifact_rejection";

%% ------------------------------------------------------------------------
%  DEBUG / TEST MODE
% -------------------------------------------------------------------------

% Set to true to process ONLY ONE SUBJECT (fast testing)
cfg.RUN_ONE_SUBJECT = false;

% Which subject index to test (1 = first included subject)
cfg.iSubTest = 1;


%% ------------------------------------------------------------------------
%  OUTPUT DIRECTORIES
%  ------------------------------------------------------------------------

cfg.derivedRoot = fullfile(cfg.projectRoot, "derived");
cfg.raguTxtDir  = fullfile(cfg.derivedRoot, "RAGU_TXT_RowsTime_Final");
cfg.logsDir     = fullfile(cfg.projectRoot, "logs");

if ~isfolder(cfg.derivedRoot); mkdir(cfg.derivedRoot); end
if ~isfolder(cfg.raguTxtDir);  mkdir(cfg.raguTxtDir);  end
if ~isfolder(cfg.logsDir);     mkdir(cfg.logsDir);     end

%% ------------------------------------------------------------------------
%  EPOCH AND BASELINE PARAMETERS
%  ------------------------------------------------------------------------

cfg.epochWindowSec = [-1.5 3];   % seconds
cfg.baselineMs     = [-500 0];     % milliseconds; NOT USED (no baseline correction is applied, see a01)

%% ------------------------------------------------------------------------
%  FILTERING (continuous, before epoching)
%  ------------------------------------------------------------------------
cfg.doFiltering = true;
cfg.bpLow  = 0.3;   % Hz
cfg.bpHigh = 20;    % Hz

%% ------------------------------------------------------------------------
%  QC / REJECTION / INTERPOLATION PROTOCOL
%  ------------------------------------------------------------------------
cfg.qc.minEpochs          = 50;
cfg.qc.minEpochKeepRatio  = 0.60;
cfg.qc.maxRejFrac         = 0.40;
cfg.qc.ampThreshUv        = 300;
cfg.qc.trendMaxUvPerSec   = 80;
cfg.qc.rangeMult          = 15;
cfg.qc.driftMadK          = 6;
cfg.qc.maxBadChansKeep    = 12;
cfg.qc.doBadChannelDetection = false;
cfg.qc.doInterpolation       = false;

% --- OPTIONAL: location-aware + imbalance knobs ---
cfg.qc.maxBadChansHard          = 20;
cfg.qc.periphFracRelaxThreshold = 0.70;
cfg.qc.periphBonusBadChans      = 6;
cfg.qc.clusterTightThreshold    = 0.15;
cfg.qc.clusterPenaltyBadChans   = 4;
cfg.qc.periphQuantile           = 0.75;

cfg.qc.imbalanceDiffBadChans    = 6;
cfg.qc.imbalanceRatioBadChans   = 2.0;


%% ------------------------------------------------------------------------
%  TARGETED PREPROCESSING
%  ------------------------------------------------------------------------
% Only subjects listed here will undergo additional preprocessing
% (epoch rejection + bad channel detection + interpolation).
% All other subjects will only be filtered and epoched.

cfg.doTargetedPreprocessing = true;

% Participant-specific lists are kept outside the repository (participant codes):
% participant_lists.csv in the parent folder of cfg.projectRoot, with columns
% ID, sham_marker_DI64, targeted_preprocessing (0/1)
plOpts = detectImportOptions(fullfile(fileparts(cfg.projectRoot), "participant_lists.csv"));
plOpts = setvartype(plOpts, "ID", "string");
pl = readtable(fullfile(fileparts(cfg.projectRoot), "participant_lists.csv"), plOpts);
cfg.preprocessSubjectIDs = pl.ID(pl.targeted_preprocessing == 1);

%% ------------------------------------------------------------------------
%  SAMPLING AND TIME AXIS (RAGU)
%  ------------------------------------------------------------------------

cfg.fs        = 200;                % Hz
cfg.deltaXms  = 1000 / cfg.fs;      % 4 ms
cfg.xOnsetMs  = -1500;              % ms
cfg.xLabel    = "ms";

cfg.analysisWindowMs = [-500 1500];

%% ------------------------------------------------------------------------
%  EVENT MARKERS
%  ------------------------------------------------------------------------

cfg.markerBL_default = {'D128'};
cfg.markerST_default = {'DI64'};

cfg.baselineExceptionIDs = pl.ID(pl.sham_marker_DI64 == 1);   % sham night used DI64 instead of D128

%% ------------------------------------------------------------------------
%  OUTPUT FILES
%  ------------------------------------------------------------------------

cfg.erpAllMatPath = fullfile(cfg.derivedRoot, "ERP_all_subjects.mat");
cfg.raguMatPath   = fullfile(cfg.derivedRoot, "RAGU_export_BL_vs_STIM.mat");

%% ------------------------------------------------------------------------
%  STATUS OUTPUT
%  ------------------------------------------------------------------------

fprintf("\n================================================\n");
fprintf("Configuration loaded successfully\n");
fprintf("================================================\n");
fprintf("Project root   : %s\n", cfg.projectRoot);
fprintf("Data root      : %s\n", cfg.dataRoot);
fprintf("Derived folder : %s\n", cfg.derivedRoot);
fprintf("RAGU TXT dir   : %s\n", cfg.raguTxtDir);
fprintf("Metadata file  : %s\n", cfg.metadataFile);
fprintf("Sheet name     : %s\n", cfg.sheetName);
fprintf("\nEpoch window   : [%.1f  %.1f] s\n", cfg.epochWindowSec);
fprintf("Baseline       : [%d  %d] ms\n", cfg.baselineMs);
fprintf("Sampling rate  : %d Hz\n", cfg.fs);
fprintf("Delta X        : %.2f ms\n", cfg.deltaXms);
fprintf("X onset        : %d ms\n", cfg.xOnsetMs);
fprintf("Analysis win   : [%d  %d] ms\n", cfg.analysisWindowMs);
fprintf("Targeted preprocessing: %s\n", string(cfg.doTargetedPreprocessing));
if cfg.doTargetedPreprocessing
    fprintf("Flagged subjects     : %s\n", strjoin(cellstr(cfg.preprocessSubjectIDs), ", "));
end
fprintf("================================================\n\n");
