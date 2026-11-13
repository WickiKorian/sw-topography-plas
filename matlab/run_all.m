%% run_all.m
%  Average-level pipeline: configuration -> subject ERPs -> export for RAGU.
%  Adapt the paths at the top of a00_config.m first. Requires EEGLAB on the path.
%  The TANOVA and microstate analyses are then run in RAGU (see README.md).

here = fileparts(mfilename("fullpath"));
run(fullfile(here, "a00_config.m"));
run(fullfile(here, "a01_compute_erps.m"));
run(fullfile(here, "a02_export_ragu.m"));
