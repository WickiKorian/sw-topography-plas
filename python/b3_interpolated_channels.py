"""
Number of channels interpolated by the PREP pipeline per participant and
analysed night (Reviewer 2: "interpolated-channel counts should describe the
retained sample and analyzed nights").

PREP stores its results in the EEGLAB structure, usually in
EEG.etc.noiseDetection.reference.interpolatedChannels (field "all": indices of
all interpolated channels). The script searches EEG.etc recursively for fields
named "interpolatedChannels" (or "badChannels" as a fallback), so that small
differences between PREP versions do not matter. If nothing is found, the
available fields are logged for inspection.

Files: the same cleaned files the epochs come from (baseline night and the
analysed stimulation night, see a1_epochs.py).

Output: microstates/filter_03_20/16. interpolated_channels/interpolated_channels.csv
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat
from config import *

logger = setup_logger("interpolated_channels", "b3_interpolated_channels.log")
OUT = MICROSTATES_DIR / FILTER_TYPE / "16. interpolated_channels"; OUT.mkdir(parents=True, exist_ok=True)


def _file(pid, night):
    folder = DATA_ROOT / pid[:5] / EEG_SUBFOLDER
    f = list(folder.glob(f"{pid}-{night}*_scored_clean_PREP_AutoReject.set"))
    return f[0] if len(f) == 1 else None


def _load_etc(path):
    try:
        m = loadmat(str(path), squeeze_me=True, struct_as_record=False, variable_names=["EEG", "etc"])
        if "EEG" in m:
            return getattr(m["EEG"], "etc", None)
        return m.get("etc")                      # pop_saveset can store fields at top level
    except NotImplementedError:                  # MATLAB v7.3 (HDF5)
        import h5py
        f = h5py.File(str(path), "r")
        return f["EEG"]["etc"] if "EEG" in f else f.get("etc")


def _find(node, names, path="etc", depth=0):
    """Recursively collect (path, value) for fields whose name is in `names`."""
    hits = []
    if depth > 6 or node is None:
        return hits
    fields = getattr(node, "_fieldnames", None) or (list(node.keys()) if hasattr(node, "keys") else [])
    for k in fields:
        v = getattr(node, k) if hasattr(node, "_fieldnames") else node[k]
        if k in names:
            hits.append((f"{path}.{k}", v))
        hits += _find(v, names, f"{path}.{k}", depth + 1)
    return hits


def _count(v):
    """Number of channels in a PREP field (struct with .all, or index array)."""
    if hasattr(v, "_fieldnames") and "all" in v._fieldnames:
        v = v.all
    elif hasattr(v, "keys") and "all" in v:
        v = v["all"][()]
    a = np.atleast_1d(np.asarray(v)).ravel()
    return int(np.unique(a[np.isfinite(a.astype(float))]).size) if a.size else 0


rows = []
for i, pid in enumerate(PARTICIPANT_IDS, 1):
    for cond, night in (("baseline", "02"), ("stimulation", "04" if pid in USE_NIGHT_E2 else "03")):
        f = _file(pid, night)
        if f is None:
            logger.warning(f"[{i}] {pid} {cond}: file not found"); continue
        etc = _load_etc(f)
        hits = _find(etc, {"interpolatedChannels"})
        source = "interpolatedChannels"
        if not hits:
            hits = _find(etc, {"badChannels", "noisyChannels"}); source = "badChannels (fallback)"
        if not hits:
            top = getattr(etc, "_fieldnames", None) or (list(etc.keys()) if hasattr(etc, "keys") else None)
            logger.warning(f"[{i}] {pid} {cond}: no PREP channel information found; EEG.etc fields: {top}")
            rows.append(dict(participant_id=pid, group="HC" if pid in HEALTHY_IDS else "CI", night=cond, n_interpolated=np.nan, source="not found"))
            continue
        # prefer the final reference step if several entries exist
        hits.sort(key=lambda h: ("reference" not in h[0], len(h[0])))
        p, v = hits[0]
        n = _count(v)
        rows.append(dict(participant_id=pid, group="HC" if pid in HEALTHY_IDS else "CI", night=cond,
                         n_interpolated=n, source=f"{source}: {p}"))
        logger.info(f"[{i}/{len(PARTICIPANT_IDS)}] {pid} {cond}: {n} channels ({p})")

df = pd.DataFrame(rows); df.to_csv(OUT / "interpolated_channels.csv", index=False)
ok = df.dropna(subset=["n_interpolated"])
if len(ok):
    s = ok.groupby(["night", "group"]).n_interpolated.agg(["mean", "std", "min", "max", "count"]).round(1)
    logger.info("interpolated channels (of 128) per recording:\n" + s.to_string())
    logger.info(f"all recordings: M = {ok.n_interpolated.mean():.1f}, SD = {ok.n_interpolated.std():.1f}, "
                f"range {int(ok.n_interpolated.min())}-{int(ok.n_interpolated.max())}")
logger.info("done")
