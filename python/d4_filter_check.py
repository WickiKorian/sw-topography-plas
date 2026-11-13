"""
Is the pre-stimulus condition effect a filter artefact?

Background
  The band-pass filter of the main analysis is a zero-phase FIR filter
  (0.3-20 Hz, order 2200 = 11 s). Zero-phase filters are symmetric in time: they
  spread part of a response that starts at stimulus onset also BACKWARDS, into
  the pre-stimulus interval. With topographic normalisation (TANOVA), such a small
  but consistent projection can become significant when the pre-stimulus signal
  is weak (e.g. in epochs without preceding stimuli, c4_ragu_overlap_subsets.py).

Steps (criteria fixed in advance)
  1  Projection test (existing data): the stimulation response (STIM - BL) after
     stimulus onset is filtered again with the same filter; whatever appears
     before 0 is backward projection. Compared with the observed pre-stimulus
     difference: strength (GFP) and topography (spatial r), per window.
  2  Causal control: epochs are recomputed with the same band (0.3-20 Hz) but a
     causal (minimum-phase) FIR filter, which cannot project anything backwards
     (a1_epochs.run("filter_03_20_causal")). Only epochs also present in the main
     analysis are used (same trials).
  3  TANOVA screen (Python; condition effect across all participants, paired
     sign-flip permutation of normalised maps, 1000 permutations): zero-phase vs.
     causal filter, all epochs and epochs without a preceding stimulus within 3.7 s.
     Prediction if the pre-stimulus effect is a filter artefact: present with the
     zero-phase filter, absent (or strongly reduced) with the causal filter.
     Note: the causal filter delays post-stimulus responses slightly; only the
     pre-stimulus interval is interpreted.
  4  RAGU export of the causal ERPs (same format as c4_ragu_overlap_subsets.py)
     for confirmation in RAGU.

Output: microstates/filter_03_20/15. filter_check/   (+ RAGU TXT in matlab_ragu/derived/RAGU_TXT_causal/)
"""
import importlib.util
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

mne.set_log_level("ERROR")
logger = setup_logger("filter_check", "d4_filter_check.log")

OUT      = MICROSTATES_DIR / FILTER_TYPE / "15. filter_check"; OUT.mkdir(parents=True, exist_ok=True)
RAGU_DERIVED = RAGU_DIR / "derived"
CAUSAL   = "filter_03_20_causal"
WINDOWS  = {"-1500 to -1000": (-1.5, -1.0), "-1000 to -500": (-1.0, -0.5), "-500 to -150": (-0.5, -0.15)}
N_PERM, OWN_TOL, PREV_CRIT, N_SAMPLES = 1000, 0.02, 3.7, 900
rng = np.random.default_rng(RANDOM_SEED)
gfp = lambda x: x.std(-1)


# ── helpers ────────────────────────────────────────────────────────────────────
def _raw_file(pid, condition):
    folder = DATA_ROOT / pid[:5] / EEG_SUBFOLDER
    night  = "02" if condition == "sham" else ("04" if pid in USE_NIGHT_E2 else "03")
    f = list(folder.glob(f"{pid}-{night}*_scored_clean_PREP_AutoReject.set")); assert len(f) == 1
    return f[0]

def prev_interval(pid, condition, events, sfreq):
    marker = ("DI64" if pid in SHAM_USES_DI64 else "D128") if condition == "sham" else "DI64"
    ann = mne.read_annotations(str(_raw_file(pid, condition)))
    on  = np.sort(ann.onset[ann.description == marker])
    t   = events[:, 0] / sfreq
    i   = np.searchsorted(on, t - OWN_TOL)
    return t - np.where(i > 0, on[np.maximum(i - 1, 0)], -np.inf)

def erps(folder, pid, cond, keep_samples=None):
    """Average reference ERPs (129 ch incl. Cz = reference, 900 samples): all and prev3.7."""
    ep = mne.read_epochs(str(EPOCHS_DIR / folder / f"{pid}_{cond}-epo.fif"), preload=False)
    ev = ep.events[:, 0]
    sel = np.ones(len(ev), bool) if keep_samples is None else np.isin(ev, keep_samples)
    prev = prev_interval(pid, cond, ep.events, ep.info["sfreq"])
    masks = {"all": sel, "prev3.7": sel & (prev > PREV_CRIT)}
    sums = {k: np.zeros((128, len(ep.times))) for k in masks}
    for s in range(0, len(ep), 300):
        x = ep[s:s + 300].get_data() * 1e6
        for k, m in masks.items(): sums[k] += x[m[s:s + 300]].sum(0)
    out = {}
    for k, m in masks.items():
        e = np.vstack([sums[k] / max(m.sum(), 1), np.zeros((1, len(ep.times)))])
        out[k] = (e - e.mean(0))[:, :N_SAMPLES].T            # time x 129
    return out, ev, ep.times[:N_SAMPLES], {k: int(m.sum()) for k, m in masks.items()}

def tanova_condition(S, B):
    """Condition effect: GFP of the mean difference of normalised maps; sign-flip permutation."""
    n = lambda X: X / np.maximum(gfp(X)[..., None], 1e-12)
    D = n(S) - n(B)
    obs = gfp(D.mean(0))
    null = np.array([gfp((D * rng.choice([-1, 1], (len(D), 1, 1))).mean(0)) for _ in range(N_PERM)])
    return obs, (np.sum(null >= obs, 0) + 1) / (N_PERM + 1)


# ── 2. causal epochs ───────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("epochs01", "a1_epochs.py")
ep01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep01)
ep01.run(CAUSAL)                                           # skips participants already done

# ── ERPs: zero-phase (main) and causal, same trials ───────────────────────────
data = {f: {k: {"sham": [], "stim": []} for k in ("all", "prev3.7")} for f in ("zero", "causal")}
counts = []
for i, pid in enumerate(PARTICIPANT_IDS, 1):
    for cond in ("sham", "stim"):
        z, ev_z, times, nz = erps(FILTER_TYPE, pid, cond)
        c, _, _, nc = erps(CAUSAL, pid, cond, keep_samples=ev_z)
        for k in ("all", "prev3.7"):
            data["zero"][k][cond].append(z[k]); data["causal"][k][cond].append(c[k])
        counts.append(dict(participant_id=pid, condition=cond, **{f"zero_{k}": v for k, v in nz.items()},
                           **{f"causal_{k}": v for k, v in nc.items()}))
        for k in ("all", "prev3.7"):                         # RAGU export (causal)
            d = RAGU_DERIVED / "RAGU_TXT_causal" / k; d.mkdir(parents=True, exist_ok=True)
            np.savetxt(d / f"{pid}_{'BL' if cond == 'sham' else 'STIM'}.txt", c[k], delimiter="\t", fmt="%.10g")
    logger.info(f"[{i}/{len(PARTICIPANT_IDS)}] {pid}: " + ", ".join(f"{k}={v}" for k, v in counts[-1].items() if k != "participant_id"))
pd.DataFrame(counts).to_csv(OUT / "epoch_counts.csv", index=False)
t = times

# ── 1. projection test (zero-phase data) ──────────────────────────────────────
h = mne.filter.create_filter(np.zeros(4000), 200.0, 0.3, 20.0, method="fir", phase="zero",
                             fir_window="hamming", fir_design="firwin", verbose=False)
ramp = np.clip(t / 0.05, 0, 1)                             # 50-ms onset ramp (no artificial step)
rows = []
for k in ("all", "prev3.7"):
    D = np.array(data["zero"][k]["stim"]) - np.array(data["zero"][k]["sham"])
    r = D * ramp[None, :, None]
    pad = np.zeros((len(D), len(h), D.shape[2]))
    proj = np.apply_along_axis(lambda v: np.convolve(v, h, mode="same"), 1, np.concatenate([pad, r, pad], 1))
    proj = proj[:, len(h):len(h) + len(t)]
    obs, pj = D.mean(0), proj.mean(0)
    for name, (a, b) in WINDOWS.items():
        w = (t >= a) & (t < b)
        rows.append(dict(subset=k, window=name, observed_gfp=gfp(obs)[w].mean(), projected_gfp=gfp(pj)[w].mean(),
                         share=gfp(pj)[w].mean() / gfp(obs)[w].mean(), spatial_r=np.corrcoef(obs[w].mean(0), pj[w].mean(0))[0, 1]))
proj_df = pd.DataFrame(rows); proj_df.to_csv(OUT / "projection_test.csv", index=False)
logger.info("1. backward projection of the response by the zero-phase filter:\n" + proj_df.round(3).to_string(index=False))

# ── 3. TANOVA screen ───────────────────────────────────────────────────────────
rows, curves = [], {}
for f in ("zero", "causal"):
    for k in ("all", "prev3.7"):
        S, B = np.array(data[f][k]["stim"]), np.array(data[f][k]["sham"])
        obs, p = tanova_condition(S, B); curves[(f, k)] = p
        for name, (a, b) in WINDOWS.items():
            w = (t >= a) & (t < b)
            rows.append(dict(filter=f, subset=k, window=name, effect_size=obs[w].mean(), pct_p_below_05=100 * np.mean(p[w] < .05)))
tan = pd.DataFrame(rows); tan.to_csv(OUT / "tanova_screen.csv", index=False)
logger.info("3. TANOVA screen, condition effect before stimulus onset (effect size; % of time points p < .05):\n"
            + tan.pivot_table(index=["subset", "window"], columns="filter", values=["effect_size", "pct_p_below_05"]).round(2).to_string())

fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
for ax, k in zip(axes, ("all", "prev3.7")):
    for f, col in (("zero", "#E63300"), ("causal", "#002FA7")):
        ax.plot(t, curves[(f, k)], color=col, lw=1, label=f"{'zero-phase' if f == 'zero' else 'causal'} filter")
    ax.axhline(.05, color="grey", ls=":"); ax.axvline(0, color="k", lw=0.8); ax.set_ylabel("p (TANOVA)")
    ax.set_title(f"{'all epochs' if k == 'all' else 'no preceding stimulus within 3.7 s'}", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
axes[-1].set_xlabel("Time relative to stimulus onset (s)")
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"tanova_zero_vs_causal.{ext}", dpi=300)
logger.info("done")
