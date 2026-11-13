%% ========================================================================
%  a01_compute_erps.m
%
%  Project: Slow-Wave Topography Analysis (Baseline vs Stimulation)
%
%  Purpose:
%  --------
%  This script computes subject-level ERPs for Baseline (sham) and
%  Stimulation nights using preprocessed and automatically artifact-
%  rejected EEG data.
%
%  The pipeline implements a structured and reproducible quality-control
%  protocol prior to ERP averaging, including:
%
%  1) Epoch-level rejection:
%       - Amplitude threshold criterion
%       - Linear trend (drift) criterion
%
%  2) Channel-level quality assessment:
%       - Robust range criterion (epoched data)
%       - Drift criterion (continuous data)
%       - Location-aware evaluation (peripheral vs central electrodes)
%       - Clustering assessment of bad channels
%
%  3) Channel interpolation:
%       - Bad channels are removed and interpolated using spherical
%         interpolation when within acceptable limits.
%       - Datasets exceeding predefined repair thresholds are excluded.
%
%  4) Condition imbalance monitoring:
%       - Flags cases where stimulation nights require substantially
%         more interpolation than baseline nights.
%
%  No baseline correction is applied.
%  Continuous band-pass filtering (0.3–20 Hz) is applied prior to epoching.
%
%  Output:
%  -------
%  - Subject-level ERPs (Baseline and Stimulation)
%  - Detailed QC metrics per condition
%  - List of interpolated channels per subject and condition
%  - Log file documenting all QC decisions
%
%  Dependencies:
%  -------------
%  - a00_config.m (creates cfg structure)
%  - EEGLAB functions (pop_loadset, pop_epoch, pop_eegfiltnew,
%    pop_eegthresh, pop_rejtrend, pop_interp, pop_select, eeg_checkset)
%
% ========================================================================

clear; clc;

%% ------------------------------------------------------------------------
%  Load configuration
%  ------------------------------------------------------------------------
run(fullfile(fileparts(mfilename("fullpath")), "a00_config.m"));   % a00_config.m in the same folder

%% ------------------------------------------------------------------------
%  Pre-flight checks
%  ------------------------------------------------------------------------
reqFns = {'pop_loadset','pop_epoch','pop_eegfiltnew','eeg_checkset', ...
          'pop_eegthresh','pop_rejtrend','pop_interp','pop_select'};
isOK   = cellfun(@(f) exist(f,'file')==2, reqFns);
missing = reqFns(~isOK);
if ~isempty(missing)
    error('EEGLAB functions not found on path: %s', strjoin(missing, ', '));
end
if ~isfile(cfg.metadataFile)
    error("Metadata Excel file not found: %s", cfg.metadataFile);
end
if ~isfield(cfg,'qc')
    error("cfg.qc not found. Add QC protocol settings to a00_config.m.");
end

%% ------------------------------------------------------------------------
%  Fill defaults for NEW location-aware / imbalance settings (safe)
%  ------------------------------------------------------------------------
cfg.qc = fill_qc_defaults(cfg.qc);

%% ------------------------------------------------------------------------
%  Logging
%  ------------------------------------------------------------------------
ts = datestr(now, "yyyymmdd_HHMMSS");
logPath = fullfile(cfg.logsDir, "01_compute_erps_" + string(ts) + ".log");
fidLog = fopen(logPath, "w");
cleanupObj = onCleanup(@() fclose(fidLog)); %#ok<NASGU>

logmsg(fidLog, "01_compute_erps started");
logmsg(fidLog, "EpochWindowSec: [" + cfg.epochWindowSec(1) + " " + cfg.epochWindowSec(2) + "]");
logmsg(fidLog, "NO baseline correction");
logmsg(fidLog, "Filter: doFiltering=" + string(cfg.doFiltering) + " | " + cfg.bpLow + "-" + cfg.bpHigh + " Hz");
logmsg(fidLog, "QC: minEpochs=" + cfg.qc.minEpochs + ...
    " maxRejFrac=" + cfg.qc.maxRejFrac + ...
    " ampThresh=" + cfg.qc.ampThreshUv + ...
    " trendMax=" + cfg.qc.trendMaxUvPerSec + ...
    " rangeMult=" + cfg.qc.rangeMult + ...
    " driftMadK=" + cfg.qc.driftMadK + ...
    " maxBadBase=" + cfg.qc.maxBadChansKeep + ...
    " maxBadHard=" + cfg.qc.maxBadChansHard);

%% ------------------------------------------------------------------------
%  Read metadata table
%  ------------------------------------------------------------------------
T = readtable(cfg.metadataFile, ...
    "Sheet", cfg.sheetName, ...
    "VariableNamesRange", "A1", ...
    "DataRange", "A2", ...
    "VariableNamingRule", "preserve");

reqCols = ["inclusion","ID","condition","use_E2_instead"];
for c = reqCols
    if ~any(strcmp(T.Properties.VariableNames, c))
        error("Required column '%s' not found in sheet: %s", c, cfg.sheetName);
    end
end

Tincl = T(T.("inclusion")==1, :);
if height(Tincl) == 0
    error("No included subjects found (inclusion==1).");
end
logmsg(fidLog, "Included subjects: " + height(Tincl));

%% ------------------------------------------------------------------------
%  Prepare Results struct
%  ------------------------------------------------------------------------
Results = repmat(struct( ...
    "ID","", ...
    "condition","", ...
    "error","", ...
    "ERP_BL",[], ...
    "ERP_STIM",[], ...
    "nEpoch_BL",[], ...
    "nEpoch_STIM",[], ...
    "nEvent_BL",[], ...
    "nEvent_STIM",[], ...
    "nDropped_BL",[], ...
    "nDropped_STIM",[], ...
    "badChans_BL_qc","", ...      % raw drift screen (unfiltered continuous)
    "badChans_STIM_qc","", ...    % raw drift screen (unfiltered continuous)
    "interpChans_BL","", ...      % FINAL interpolated channels (from protocol)
    "interpChans_STIM","", ...    % FINAL interpolated channels (from protocol)
    "qc_BL",struct(), ...
    "qc_STIM",struct(), ...
    "qc_imbalance",struct(), ...  % condition imbalance summary
    "times",[], ...
    "chanlocs",[]), height(Tincl), 1);

% Decide which subjects to run
if cfg.RUN_ONE_SUBJECT
    subList = cfg.iSubTest;
    fprintf("\n*** TEST MODE: running only subject %d ***\n", cfg.iSubTest);
else
    subList = 1:height(Tincl);
end

%% ------------------------------------------------------------------------
%  Main loop
%  ------------------------------------------------------------------------
for iSub = subList
    fprintf("\nProcessing subject %d / %d\n", iSub, height(Tincl));

    idStr = string(Tincl.("ID")(iSub));

    condVal = Tincl.("condition")(iSub);
    if ismissing(condVal)
        cond = "";
    else
        cond = string(condVal);
    end

    doExtraPreproc = should_preprocess_subject(idStr, cfg);
    logmsg(fidLog, "Subject " + idStr + ...
    " → targeted preprocessing = " + string(doExtraPreproc));
    fprintf("Subject: %s | Condition: %s\n", idStr, cond);

    Results(iSub).ID        = idStr;
    Results(iSub).condition = cond;
    Results(iSub).error     = "";

    try
        logmsg(fidLog, "----");
        logmsg(fidLog, "Processing: " + idStr + " | condition=" + cond);

        % Parse wave/subj from ID (e.g., "19-01-123")
        parts = split(idStr, "-");
        if numel(parts) ~= 3
            error("Unexpected ID format (expected 'xx-xx-xxx'): %s", idStr);
        end
        wave = parts(1) + "-" + parts(2);
        subj = parts(3);

        % Nights
        baselineNight = "02";
        useE2 = Tincl.("use_E2_instead")(iSub);
        if isnan(useE2); useE2 = 0; end
        stimNight = "03";
        if useE2 == 1
            stimNight = "04";
        end

        % Locate .set files
        eegFolder = fullfile(cfg.dataRoot, wave, cfg.eegSubfolder);
        patBL = wave + "-" + subj + "-" + baselineNight + "_*.set";
        patST = wave + "-" + subj + "-" + stimNight     + "_*.set";

        blFiles = dir(fullfile(eegFolder, patBL));
        stFiles = dir(fullfile(eegFolder, patST));

        if isempty(blFiles); error("No baseline .set found in %s (pattern %s)", eegFolder, patBL); end
        if isempty(stFiles); error("No stim .set found in %s (pattern %s)", eegFolder, patST); end

        blPath = fullfile(blFiles(1).folder, blFiles(1).name);
        stPath = fullfile(stFiles(1).folder, stFiles(1).name);

        fprintf("Baseline file: %s\n", string(blFiles(1).name));
        fprintf("Stim file:     %s\n", string(stFiles(1).name));

        logmsg(fidLog, "Baseline file: " + string(blFiles(1).name));
        logmsg(fidLog, "Stim file: " + string(stFiles(1).name));

        % Marker selection
        markerBL = cfg.markerBL_default;
        if any(idStr == cfg.baselineExceptionIDs)
            markerBL = cfg.markerST_default;
        end
        markerST = cfg.markerST_default;

        % ---- QC inventory (stim count + raw drift screen; descriptive only) ----
        EEG = pop_loadset(blPath); EEG = eeg_checkset(EEG);
        nEvent_BL = count_events_by_type(EEG, markerBL);
        [~, badBL_raw] = drift_flag_channels(EEG);

        EEG = pop_loadset(stPath); EEG = eeg_checkset(EEG);
        nEvent_ST = count_events_by_type(EEG, markerST);
        [~, badST_raw] = drift_flag_channels(EEG);

        Results(iSub).nEvent_BL   = nEvent_BL;
        Results(iSub).nEvent_STIM = nEvent_ST;
        Results(iSub).badChans_BL_qc   = string(strjoin(badBL_raw, ","));
        Results(iSub).badChans_STIM_qc = string(strjoin(badST_raw, ","));

        % ---- Baseline ERP (with protocol) ----
        [ERP_BL, nEpoch_BL, timesBL, chanlocsBL, qcBL] = compute_erp( ...
        blPath, markerBL, cfg.epochWindowSec, cfg, nEvent_BL, "BL", doExtraPreproc);

        % ---- Stim ERP (with protocol) ----
        [ERP_STIM, nEpoch_STIM, timesST, chanlocsST, qcST] = compute_erp( ...
        stPath, markerST, cfg.epochWindowSec, cfg, nEvent_ST, "STIM", doExtraPreproc);

        % Store qc + interpolated channel lists
        Results(iSub).qc_BL   = qcBL;
        Results(iSub).qc_STIM = qcST;

        Results(iSub).interpChans_BL   = join_labels_safe(qcBL.interpChanLabels);
        Results(iSub).interpChans_STIM = join_labels_safe(qcST.interpChanLabels);

        % Dropped events (safe)
        Results(iSub).nDropped_BL   = max(0, nEvent_BL - nEpoch_BL);
        Results(iSub).nDropped_STIM = max(0, nEvent_ST - nEpoch_STIM);

        % Consistency checks (times / channels)
        if numel(timesBL) ~= numel(timesST) || any(timesBL ~= timesST)
            error("Time axis mismatch between BL and STIM for %s.", idStr);
        end
        if numel(chanlocsBL) ~= numel(chanlocsST)
            error("Channel count mismatch between BL and STIM for %s.", idStr);
        end

        % Store ERPs
        Results(iSub).ERP_BL      = single(ERP_BL);
        Results(iSub).ERP_STIM    = single(ERP_STIM);
        Results(iSub).nEpoch_BL   = nEpoch_BL;
        Results(iSub).nEpoch_STIM = nEpoch_STIM;
        Results(iSub).times       = timesBL;
        Results(iSub).chanlocs    = chanlocsBL;

        % ---- Condition imbalance (flag only; do not auto-exclude) ----
        imb = struct();
        imb.nBad_BL   = qcBL.nBadChans;
        imb.nBad_STIM = qcST.nBadChans;
        imb.diffBad   = qcST.nBadChans - qcBL.nBadChans;

        denom = max(1, qcBL.nBadChans);
        imb.ratioBad = qcST.nBadChans / denom;

        imb.isImbalanced = (imb.diffBad >= cfg.qc.imbalanceDiffBadChans) || ...
                           (imb.ratioBad >= cfg.qc.imbalanceRatioBadChans);

        Results(iSub).qc_imbalance = imb;

        % Log summary + channels interpolated
        logmsg(fidLog, "QC BL: raw=" + qcBL.nEpoch_raw + " clean=" + nEpoch_BL + ...
            " rej=" + round(100*qcBL.rejFrac,1) + "% badCh=" + qcBL.nBadChans + ...
            " allowedMax=" + qcBL.allowedMaxBadChans + ...
            " periphFrac=" + round(100*qcBL.periphFracBad,1) + "% cluster=" + round(qcBL.clusterScore,3) + ...
            " interp=" + string(qcBL.didInterp));

        if qcBL.didInterp
            logmsg(fidLog, "Interpolated BL: " + strjoin(qcBL.interpChanLabels, ","));
        end

        logmsg(fidLog, "QC ST: raw=" + qcST.nEpoch_raw + " clean=" + nEpoch_STIM + ...
            " rej=" + round(100*qcST.rejFrac,1) + "% badCh=" + qcST.nBadChans + ...
            " allowedMax=" + qcST.allowedMaxBadChans + ...
            " periphFrac=" + round(100*qcST.periphFracBad,1) + "% cluster=" + round(qcST.clusterScore,3) + ...
            " interp=" + string(qcST.didInterp));

        if qcST.didInterp
            logmsg(fidLog, "Interpolated STIM: " + strjoin(qcST.interpChanLabels, ","));
        end

        if imb.isImbalanced
            logmsg(fidLog, "IMBALANCE FLAG: STIM vs BL badCh diff=" + imb.diffBad + " ratio=" + round(imb.ratioBad,2));
        end

        fprintf("BL epochs kept: %d | STIM epochs kept: %d\n", nEpoch_BL, nEpoch_STIM);

    catch ME
        Results(iSub).error = string(ME.message);
        logmsg(fidLog, "ERROR: " + string(ME.message));
        fprintf("ERROR for %s: %s\n", idStr, ME.message);
        continue
    end
end

%% ------------------------------------------------------------------------
%  Save output
%  ------------------------------------------------------------------------
cfg_used = cfg; %#ok<NASGU>
save(cfg.erpAllMatPath, "Results", "Tincl", "cfg_used", "-v7.3");

nOK = sum(arrayfun(@(s) strlength(s.error)==0, Results));
logmsg(fidLog, "----");
logmsg(fidLog, "Completed. Successful subjects: " + nOK + " / " + numel(Results));
logmsg(fidLog, "Saved: " + cfg.erpAllMatPath);

fprintf("Done. Successful subjects: %d / %d\n", nOK, numel(Results));
fprintf("Saved: %s\n", cfg.erpAllMatPath);
fprintf("Log:   %s\n", logPath);

%% ========================================================================
%  Local functions
% ========================================================================

function qc = fill_qc_defaults(qc)
% Adds defaults for location-aware + imbalance checks if missing.
    def = struct();
    def.maxBadChansHard          = 20;   % absolute hard stop (exclude if > this)
    def.periphFracRelaxThreshold = 0.70; % if >=70% bad chans are peripheral -> relax
    def.periphBonusBadChans      = 6;    % how many extra channels allowed when mostly peripheral
    def.clusterTightThreshold    = 0.15; % smaller => more clustered (unitless normalized score)
    def.clusterPenaltyBadChans   = 4;    % subtract allowed channels if tightly clustered
    def.periphQuantile           = 0.75; % define "peripheral" as top 25% of radius
    def.imbalanceDiffBadChans    = 6;    % flag if STIM has >=6 more bad chans than BL
    def.imbalanceRatioBadChans   = 2.0;  % or if STIM/BL >= 2.0

    fns = fieldnames(def);
    for i = 1:numel(fns)
        if ~isfield(qc, fns{i}) || isempty(qc.(fns{i}))
            qc.(fns{i}) = def.(fns{i});
        end
    end
end

function logmsg(fid, msg)
    fprintf(fid, "[%s] %s\n", datestr(now, "yyyy-mm-dd HH:MM:SS"), msg);
end

function [ERP, nEpoch_clean, times, chanlocs, qc] = compute_erp(setPath, marker, epochWindowSec, cfg, nEvent_total, tag, doExtraPreproc)
% Implements targeted preprocessing:
% - if doExtraPreproc == true: full QC pipeline
% - if doExtraPreproc == false: only filter + epoch + average

    qc = struct();
    qc.tag = string(tag);
    qc.setPath = string(setPath);
    qc.marker  = string(marker{1});
    qc.nEvent_total = double(nEvent_total);
    qc.targeted_preprocessing = doExtraPreproc;

    EEG = pop_loadset(setPath);
    EEG = eeg_checkset(EEG);

    % Save original montage for interpolation back to full montage
    templateChanlocs = EEG.chanlocs;

    % ---- Continuous filter BEFORE epoching ----
    if isfield(cfg,"doFiltering") && cfg.doFiltering
        EEG = pop_eegfiltnew(EEG, 'locutoff', cfg.bpLow, 'hicutoff', cfg.bpHigh);
        EEG = eeg_checkset(EEG);
    end

    % ---- Epoch ----
    EEG_ep = pop_epoch(EEG, marker, epochWindowSec);
    if EEG_ep.trials == 0
        error("No epochs found (marker %s) in %s", marker{1}, setPath);
    end

    qc.nEpoch_raw = double(EEG_ep.trials);
    qc.epochKeepRatio = NaN;
    qc.nEpoch_after_rej = qc.nEpoch_raw;
    qc.rejFrac = 0;
    qc.badChanLabels = string.empty(1,0);
    qc.nBadChans = 0;
    qc.badByRange = string.empty(1,0);
    qc.badByDrift = string.empty(1,0);
    qc.periphFracBad = 0;
    qc.clusterScore = 1;
    qc.allowedMaxBadChans = NaN;
    qc.didInterp = false;
    qc.interpChanLabels = string.empty(1,0);

    % =====================================================================
    % If subject is NOT flagged: only filter + epoch + average
    % =====================================================================
    if ~doExtraPreproc
        qc.epochKeepRatio      = NaN;
        qc.nEpoch_after_rej    = qc.nEpoch_raw;
        qc.rejFrac             = 0;
        qc.badChanLabels       = "";
        qc.nBadChans           = 0;
        qc.badByRange          = "";
        qc.badByDrift          = "";
        qc.periphFracBad       = 0;
        qc.clusterScore        = 1;
        qc.allowedMaxBadChans  = 0;
        qc.didInterp           = false;
        qc.interpChanLabels    = "";
    
        ERP = mean(EEG_ep.data, 3);
        nEpoch_clean = double(EEG_ep.trials);
        times = EEG_ep.times;
        chanlocs = EEG_ep.chanlocs;
    
        if any(~isfinite(ERP(:)))
            error("Non-finite values detected in ERP for %s", setPath);
        end
        return
    end

    % =====================================================================
    % Full preprocessing only for flagged subjects
    % =====================================================================

    % ---- Dataset-level minimums (before epoch rejection) ----
    if qc.nEpoch_raw < cfg.qc.minEpochs
        error("Exclude dataset: nEpoch=%d < %d (minEpochs).", qc.nEpoch_raw, cfg.qc.minEpochs);
    end

    if ~isnan(qc.nEvent_total) && qc.nEvent_total > 0
        keepRatio = qc.nEpoch_raw / qc.nEvent_total;
        qc.epochKeepRatio = keepRatio;
        if keepRatio < cfg.qc.minEpochKeepRatio
            error("Exclude dataset: nEpoch/nEvent=%.2f < %.2f (minEpochKeepRatio).", keepRatio, cfg.qc.minEpochKeepRatio);
        end
    end

    % =====================================================================
    % 1) Epoch rejection
    % =====================================================================

    EEG_ep = pop_eegthresh(EEG_ep, 1, 1:EEG_ep.nbchan, ...
        -cfg.qc.ampThreshUv, cfg.qc.ampThreshUv, ...
        epochWindowSec(1), epochWindowSec(2), 0, 1);

    epLenSec = epochWindowSec(2) - epochWindowSec(1);
    maxSlopeUvPerEpoch = cfg.qc.trendMaxUvPerSec * epLenSec;

    EEG_ep = pop_rejtrend(EEG_ep, 1, 1:EEG_ep.nbchan, ...
        EEG_ep.pnts, maxSlopeUvPerEpoch, 0.3, 0.1, 0, 1);

    qc.nEpoch_after_rej = double(EEG_ep.trials);
    qc.rejFrac = (qc.nEpoch_raw - qc.nEpoch_after_rej) / qc.nEpoch_raw;

    if qc.nEpoch_after_rej < cfg.qc.minEpochs
        error("Exclude dataset: after epoch rejection nEpoch=%d < %d.", qc.nEpoch_after_rej, cfg.qc.minEpochs);
    end
    if qc.rejFrac > cfg.qc.maxRejFrac
        error("Exclude dataset: rejected %.1f%% epochs (>%.1f%% max).", 100*qc.rejFrac, 100*cfg.qc.maxRejFrac);
    end

    % =====================================================================
    % 2) Bad channel detection
    % =====================================================================

    if cfg.qc.doBadChannelDetection
    
        % ===== EXISTING CODE (unchanged) =====
        labels = {templateChanlocs.labels};
    
        Xep = double(EEG_ep.data);
        Xflat = reshape(Xep, size(Xep,1), []);
    
        p1  = pctile_simple(Xflat, 1);
        p99 = pctile_simple(Xflat, 99);
        chanRange = p99 - p1;
    
        medRange = median(chanRange);
        badByRange = chanRange > (cfg.qc.rangeMult * medRange);
    
        Xc = double(EEG.data);
        [~, badByDrift] = drift_flag_logic(Xc, cfg.qc.driftMadK);
    
        badIdx = find(badByRange | badByDrift);
    
        qc.badChanLabels = string(labels(badIdx));
        qc.nBadChans = numel(badIdx);
    
    else
        % ===== SKIPPED =====
        badIdx = [];
        qc.badChanLabels = "";
        qc.nBadChans = 0;
    end

    % =====================================================================
    % 2C) Location-aware metrics
    % =====================================================================
    [periphMask, periphThreshR, ~] = classify_peripheral(templateChanlocs, cfg.qc.periphQuantile);
    qc.periphThreshR = periphThreshR;

    if ~isempty(badIdx)
        qc.periphFracBad = mean(periphMask(badIdx));
        qc.clusterScore  = cluster_score_xy(templateChanlocs, badIdx);
    else
        qc.periphFracBad = 0;
        qc.clusterScore  = 1;
    end

    % =====================================================================
    % 2D) Allowed max bad channels
    % =====================================================================
    allowed = cfg.qc.maxBadChansKeep;
    if qc.periphFracBad >= cfg.qc.periphFracRelaxThreshold
        allowed = allowed + cfg.qc.periphBonusBadChans;
    end
    if qc.clusterScore <= cfg.qc.clusterTightThreshold && qc.nBadChans > cfg.qc.maxBadChansKeep
        allowed = allowed - cfg.qc.clusterPenaltyBadChans;
    end

    % IMPORTANT: real lower clamp, so cluster penalty can actually work
    allowed = max(1, min(allowed, cfg.qc.maxBadChansHard));
    qc.allowedMaxBadChans = allowed;
    
    if cfg.qc.doBadChannelDetection
    
        if qc.nBadChans > cfg.qc.maxBadChansHard
            error("Exclude dataset: %d bad channels (> %d hard stop).", ...
                qc.nBadChans, cfg.qc.maxBadChansHard);
        end
    
        if qc.nBadChans > qc.allowedMaxBadChans
            error("Exclude dataset: %d bad channels (> %d allowed).", ...
                qc.nBadChans, qc.allowedMaxBadChans);
        end
    
    end

    % =====================================================================
    % 3) Interpolate bad channels
    % =====================================================================
    if cfg.qc.doInterpolation && qc.nBadChans > 0
        qc.didInterp = true;
        qc.interpChanLabels = qc.badChanLabels;

        badLabels = cellstr(qc.badChanLabels);
        present = intersect({EEG_ep.chanlocs.labels}, badLabels);

        if ~isempty(present)
            EEG_ep = pop_select(EEG_ep, 'nochannel', present);
            EEG_ep = eeg_checkset(EEG_ep);

            EEG_ep = pop_interp(EEG_ep, templateChanlocs, 'spherical');
            EEG_ep = eeg_checkset(EEG_ep);
        end
    end

    % =====================================================================
    % 4) Average
    % =====================================================================
    ERP = mean(EEG_ep.data, 3);
    nEpoch_clean = double(EEG_ep.trials);
    times = EEG_ep.times;
    chanlocs = EEG_ep.chanlocs;

    if any(~isfinite(ERP(:)))
        error("Non-finite values detected in ERP for %s", setPath);
    end
end

%% -------- helpers --------

function n = count_events_by_type(EEG, types)
    if isstring(types); types = cellstr(types); end
    ev = EEG.event;
    if isempty(ev); n = 0; return; end

    t = strings(1, numel(ev));
    for i = 1:numel(ev)
        t(i) = string(ev(i).type);
    end

    isBoundary = strcmpi(t, "boundary");
    isMatch = false(1, numel(ev));
    for k = 1:numel(types)
        isMatch = isMatch | strcmp(t, string(types{k}));
    end
    n = sum(isMatch & ~isBoundary);
end

function [driftPerChan, badLabels] = drift_flag_channels(EEG)
% Raw drift screen (continuous, unfiltered here): median(last10%) - median(first10%)
    X = double(EEG.data); % ch x samples
    nS = size(X,2);
    a = 1:max(1, round(0.1*nS));
    b = (nS-max(1, round(0.1*nS))+1):nS;

    driftPerChan = median(X(:,b),2) - median(X(:,a),2);
    driftAbs = abs(driftPerChan);

    med = median(driftAbs);
    madv = median(abs(driftAbs - med)) + eps;

    isBad = driftAbs > (med + 6*madv);

    labels = {EEG.chanlocs.labels};
    badLabels = labels(isBad);
end

function p = pctile_simple(X, q)
% X: ch x N
% q: scalar percentile (0..100)
    Xs = sort(X, 2, 'ascend');
    n  = size(Xs,2);
    idx = max(1, min(n, round((q/100) * n)));
    p = Xs(:, idx);
end

function [driftAbs, isBad] = drift_flag_logic(Xc, kMad)
% Xc: ch x samples
% driftAbs: abs(median(last10%) - median(first10%))
% isBad: abs(drift) > med + kMad*MAD
    nS = size(Xc,2);
    a = 1:max(1, round(0.1*nS));
    b = (nS-max(1, round(0.1*nS))+1):nS;

    drift = median(Xc(:,b),2) - median(Xc(:,a),2);
    driftAbs = abs(drift);

    med = median(driftAbs);
    madv = median(abs(driftAbs - med)) + eps;

    isBad = driftAbs > (med + kMad*madv);
end

function [periphMask, thrR, r] = classify_peripheral(chanlocs, q)
% Peripheral if radius r = sqrt(X^2+Y^2) is above the q-quantile.
% If X/Y missing, falls back to "no peripheral classification".
    n = numel(chanlocs);
    X = nan(n,1); Y = nan(n,1);
    for i = 1:n
        if isfield(chanlocs(i),'X') && isfield(chanlocs(i),'Y') && ~isempty(chanlocs(i).X) && ~isempty(chanlocs(i).Y)
            X(i) = chanlocs(i).X;
            Y(i) = chanlocs(i).Y;
        end
    end
    ok = isfinite(X) & isfinite(Y);
    r = nan(n,1);
    r(ok) = sqrt(X(ok).^2 + Y(ok).^2);

    if sum(ok) < max(10, round(0.5*n))
        % Too few coordinates -> disable
        periphMask = false(n,1);
        thrR = NaN;
        return;
    end

    thrR = quantile(r(ok), q);
    periphMask = false(n,1);
    periphMask(ok) = r(ok) >= thrR;
end

function score = cluster_score_xy(chanlocs, idx)
% Returns a normalized clustering score in [0..1] approx.
% Smaller => more clustered. Larger => more scattered.
% Computed as mean pairwise distance among bad channels divided by
% mean pairwise distance among all channels (XY plane).
    n = numel(chanlocs);
    X = nan(n,1); Y = nan(n,1);
    for i = 1:n
        if isfield(chanlocs(i),'X') && isfield(chanlocs(i),'Y') && ~isempty(chanlocs(i).X) && ~isempty(chanlocs(i).Y)
            X(i) = chanlocs(i).X;
            Y(i) = chanlocs(i).Y;
        end
    end
    ok = isfinite(X) & isfinite(Y);
    if sum(ok) < max(10, round(0.5*n)) || numel(idx) < 2
        score = 1; % can't estimate -> treat as not clustered
        return;
    end

    allXY = [X(ok) Y(ok)];
    badOk = ok(idx);
    if sum(badOk) < 2
        score = 1;
        return;
    end
    badXY = [X(idx(badOk)) Y(idx(badOk))];

    dAll = mean(pdist(allXY));
    dBad = mean(pdist(badXY));

    score = dBad / (dAll + eps);
end


function tf = should_preprocess_subject(idStr, cfg)
    if ~isfield(cfg, 'doTargetedPreprocessing') || ~cfg.doTargetedPreprocessing
        tf = true;   % fallback: preprocess everyone
        return;
    end

    if ~isfield(cfg, 'preprocessSubjectIDs') || isempty(cfg.preprocessSubjectIDs)
        tf = false;  % nobody selected
        return;
    end

    tf = any(string(idStr) == string(cfg.preprocessSubjectIDs));
end

function s = join_labels_safe(x)
    if isempty(x)
        s = "";
        return;
    end
    x = string(x);
    x = x(~ismissing(x) & strlength(x) > 0);
    if isempty(x)
        s = "";
    else
        s = strjoin(cellstr(x), ",");
    end
end