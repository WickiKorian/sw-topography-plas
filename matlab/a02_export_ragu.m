%% ========================================================================
%  a02_export_ragu.m
%
%  Description:
%  ------------
%  Exports subject-level ERPs (Baseline vs Stim) for RAGU from the
%  Results struct saved by a01_compute_erps.m.
%
%  Outputs:
%  --------
%  1) cfg.raguMatPath:
%       BL(subject, channel, time)
%       STIM(subject, channel, time)
%       times, chanlocs, subjID
%
%  2) One TXT per subject per condition in cfg.raguTxtDir:
%       <ID>_BL.txt
%       <ID>_STIM.txt
%     Format: rows = time, columns = channels, tab-delimited, numeric only
%
%  Dependencies:
%  -------------
%  - a00_config.m (creates cfg)
%  - cfg.erpAllMatPath created by a01_compute_erps.m
%  ========================================================================

%% ------------------------------------------------------------------------
%  Load configuration (only if needed)
%  ------------------------------------------------------------------------
if ~exist("cfg","var")
    run(fullfile(fileparts(mfilename("fullpath")), "a00_config.m"));   % a00_config.m in the same folder
end
%% ------------------------------------------------------------------------
%  Logging
%  ------------------------------------------------------------------------
ts = datestr(now, "yyyymmdd_HHMMSS");
logPath = fullfile(cfg.logsDir, "02_export_ragu_" + string(ts) + ".log");
fidLog = fopen(logPath, "w");
cleanupObj = onCleanup(@() fclose(fidLog));

logmsg(fidLog, "a02_export_ragu started");
logmsg(fidLog, "Input:  " + cfg.erpAllMatPath);
logmsg(fidLog, "Output MAT: " + cfg.raguMatPath);
logmsg(fidLog, "Output TXT dir: " + cfg.raguTxtDir);

%% ------------------------------------------------------------------------
%  Load ERP results
%  ------------------------------------------------------------------------
if ~isfile(cfg.erpAllMatPath)
    error("Missing ERP results file: %s", cfg.erpAllMatPath);
end

S = load(cfg.erpAllMatPath, "Results");
Results = S.Results;

if isempty(Results)
    error("Results is empty in %s", cfg.erpAllMatPath);
end

%% ------------------------------------------------------------------------
%  Keep only subjects with valid ERP data
%  ------------------------------------------------------------------------
ok = arrayfun(@(s) ...
    isfield(s,"error") && strlength(string(s.error))==0 && ...
    isfield(s,"ERP_BL") && ~isempty(s.ERP_BL) && ...
    isfield(s,"ERP_STIM") && ~isempty(s.ERP_STIM) && ...
    isfield(s,"times") && ~isempty(s.times) && ...
    isfield(s,"chanlocs") && ~isempty(s.chanlocs), ...
    Results);

R = Results(ok);

logmsg(fidLog, "Total entries in Results: " + numel(Results));
logmsg(fidLog, "Subjects with valid ERPs: " + numel(R));

if isempty(R)
    error("No subjects with valid ERP data found. Run a01_compute_erps first.");
end

%% ------------------------------------------------------------------------
%  Validate consistency across subjects
%  ------------------------------------------------------------------------
nCh0 = size(R(1).ERP_BL, 1);
nT0  = size(R(1).ERP_BL, 2);
times0 = R(1).times;
chanlocs0 = R(1).chanlocs;

for i = 1:numel(R)

    if any(size(R(i).ERP_BL) ~= [nCh0 nT0])
        error("Size mismatch in ERP_BL for ID=%s. Expected [%d %d].", string(R(i).ID), nCh0, nT0);
    end
    if any(size(R(i).ERP_STIM) ~= [nCh0 nT0])
        error("Size mismatch in ERP_STIM for ID=%s. Expected [%d %d].", string(R(i).ID), nCh0, nT0);
    end

    if numel(R(i).times) ~= numel(times0) || any(R(i).times ~= times0)
        error("Time axis mismatch for ID=%s.", string(R(i).ID));
    end

    if numel(R(i).chanlocs) ~= numel(chanlocs0)
        error("Channel count mismatch for ID=%s.", string(R(i).ID));
    end

    if any(~isfinite(R(i).ERP_BL(:))) || any(~isfinite(R(i).ERP_STIM(:)))
        error("Non-finite values found for ID=%s.", string(R(i).ID));
    end
end

logmsg(fidLog, "Validation passed: consistent channels/time across subjects.");

%% ------------------------------------------------------------------------
%  Export compact MAT for downstream processing
%  ------------------------------------------------------------------------
nSub = numel(R);
nCh  = nCh0;
nT   = nT0;

BL   = zeros(nSub, nCh, nT, "single");
STIM = zeros(nSub, nCh, nT, "single");

for i = 1:nSub
    BL(i,:,:)   = R(i).ERP_BL;
    STIM(i,:,:) = R(i).ERP_STIM;
end

times    = times0;     %#ok<NASGU>
chanlocs = chanlocs0;  %#ok<NASGU>
subjID   = string({R.ID})'; %#ok<NASGU>

save(cfg.raguMatPath, "BL","STIM","times","chanlocs","subjID", "-v7.3");
logmsg(fidLog, "Saved MAT: " + cfg.raguMatPath);

%% ------------------------------------------------------------------------
%  Export TXT per subject per condition (rows=time, cols=channels)
%  ------------------------------------------------------------------------
if ~isfolder(cfg.raguTxtDir); mkdir(cfg.raguTxtDir); end

for i = 1:nSub
    idStr = string(R(i).ID);

    blFile   = fullfile(cfg.raguTxtDir, idStr + "_BL.txt");
    stimFile = fullfile(cfg.raguTxtDir, idStr + "_STIM.txt");

    % Data are stored as channels x time -> transpose to time x channels
    writematrix(double(R(i).ERP_BL')  , blFile,   "Delimiter","\t");
    writematrix(double(R(i).ERP_STIM'), stimFile, "Delimiter","\t");

    logmsg(fidLog, "Wrote TXT: " + idStr + " (BL, STIM)");
end

logmsg(fidLog, "Completed TXT export: " + cfg.raguTxtDir);

fprintf("Done.\n");
fprintf("Saved MAT: %s\n", cfg.raguMatPath);
fprintf("Saved TXT: %s\n", cfg.raguTxtDir);
fprintf("Log:       %s\n", logPath);

%% ========================================================================
%  Local function
%  ========================================================================
function logmsg(fid, msg)
    fprintf(fid, "[%s] %s\n", datestr(now, "yyyy-mm-dd HH:MM:SS"), msg);
end
