%% c01_ragu_overlap_subset.m
%  Creates a RAGU file for one overlap subset (see python/scripts/20_ragu_overlap_subsets.py)
%  by taking the group TANOVA file of the main analysis (all settings: design, group
%  labels, 5000 randomizations, normalisation, window 1-900) and replacing only the
%  data (rd.V) with the subset ERPs. All previous results are cleared, so RAGU
%  computes the TANOVA from scratch on the new data.
%
%  Usage (MATLAB command window):
%     subset = "prev3.7";            % or "prev3.7_next3.0", "next3.0", "all"
%     run("<path to matlab_ragu>\c01_ragu_overlap_subset.m")
%  Then in RAGU: File -> Open the new file, start the TANOVA directly (do not open
%  the analysis-window dialog), save when finished.

if ~exist("subset", "var"); subset = "prev3.7"; end
root   = fileparts(mfilename("fullpath"));                   % this folder (matlab_ragu)
src    = fullfile(root, "Ragu_results", "TANOVA_-1500_3000ms_ConditionXGroup_new.mat");
txtDir = fullfile(root, "derived", "RAGU_TXT_overlap", subset);
dst    = fullfile(root, "Ragu_results", "TANOVA_-1500_3000ms_ConditionXGroup_" + subset + ".mat");

S  = load(src);  rd = S.rd;
tags = ["STIM", "BL"];                         % rd.V(:, 1, :, :) = STIM, rd.V(:, 2, :, :) = BL
assert(isequal(upper(reshape(string(rd.conds), 1, [])), tags), "unexpected condition order");
for s = 1:size(rd.V, 1)
    id = extractBefore(string(rd.Names{s, 1}), 10);           % e.g. "19-01-123"
    for c = 1:2
        f = fullfile(txtDir, id + "_" + tags(c) + ".txt");
        x = readmatrix(f, "FileType", "text", "Delimiter", "\t");   % 900 x 129
        assert(isequal(size(x), [900 129]), "wrong size: " + f);
        rd.V(s, c, :, :) = permute(x', [3 4 1 2]);
    end
end

% clear all results of the source file (as in a file that was never analysed)
resultFields = ["PTanova","TanovaEffectSize","TanovaHits","PHitCount","PHitDuration","PAUCCount", ...
                "TanovaMeanEffectSize","PTanovaOverall","TanovaAUC","MDDelta", ...
                "GFPPTanova","GFPTanovaEffectSize","MSEffectSize","pTopCons"];
for f = resultFields
    if isfield(rd, f); rd.(f) = []; end
end
rd.CritDuration = ones(2, 4) * 1e15;
rd.StartFrame   = 1;
rd.EndFrame     = 900;
rd.Directory    = char(txtDir);
S.rd = rd;
save(dst, "-struct", "S", "-v7.3");
fprintf("Saved %s\n(%d participants, data from %s)\n", dst, size(rd.V, 1), txtDir);
