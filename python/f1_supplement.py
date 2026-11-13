"""
Builds the Supplementary Material (Word) from the saved results of scripts
02-17. Nothing is recomputed except effect-size CIs and figure layouts.

Output: <SW Topography>/Review/Manuskript/Supplementary_Material_v2.docx
        (+ figure PNGs in Review/Manuskript/supp_figures)

Items that need results not yet available are written as highlighted
placeholders ([...]).
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_COLOR_INDEX
from docx.enum.section import WD_ORIENT
from config import PROJECT_ROOT, MICROSTATES_DIR, FILTER_TYPE, SUPPLEMENT_DOCX
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

matplotlib.rcParams.update({"font.size": 8, "pdf.fonttype": 42})   # font family (Arial) set in config.py

ANA    = PROJECT_ROOT
RES    = MICROSTATES_DIR / FILTER_TYPE
OUT    = SUPPLEMENT_DOCX.parent
FIGDIR = OUT / "supp_figures"; FIGDIR.mkdir(parents=True, exist_ok=True)
DOCX   = SUPPLEMENT_DOCX

HC_C, CI_C, SHAM_C, STIM_C = "#3b6fb6", "#d1495b", "#002FA7", "#E63300"
PAR = {"timecov": "Coverage", "meandurs": "Mean duration", "occurrences": "Occurrence", "gev": "GEV"}
MSNAME = {1: "MS 1 (central positive)", 2: "MS 2 (right-lateralized)", 3: "MS 3 (frontal positive)", 4: "MS 4 (central-posterior positive)"}

# ── helpers ───────────────────────────────────────────────────────────────────
def fp(p):
    if pd.isna(p): return "–"
    return "< .001" if p < .001 else (f"{p:.3f}".lstrip("0") if p < .9995 else "1.000")

def ci_es(es, typ, n, n1=25, n2=15):
    if typ == "dz": se = np.sqrt(1 / n + es ** 2 / (2 * n))
    else:           se = np.sqrt((n1 + n2) / (n1 * n2) + es ** 2 / (2 * (n1 + n2)))
    return es - 1.96 * se, es + 1.96 * se

def es_txt(es, lo=None, hi=None):
    s = f"{es:.2f}"
    return s if lo is None else f"{s} [{lo:.2f}, {hi:.2f}]"

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(10)
st.paragraph_format.space_after = Pt(4)

def heading(t, lvl=2):
    h = doc.add_heading(t, lvl)
    for r in h.runs: r.font.name = "Arial"; r.font.color.rgb = RGBColor(0, 0, 0)

def para(txt, size=None, italic=False, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix); r.bold = True
        if size: r.font.size = Pt(size)
    for part in re.split(r"(\[\[.*?\]\])", txt):
        if not part: continue
        hl = part.startswith("[[")
        r = p.add_run("[" + part[2:-2] + "]" if hl else part); r.italic = italic
        if size: r.font.size = Pt(size)
        if hl: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p

def _border(cell, side):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None: b = OxmlElement("w:tcBorders"); tcPr.append(b)
    e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); b.append(e)

def table(header, rows, size=8, widths=None):
    t = doc.add_table(rows=1, cols=len(header)); t.autofit = widths is None
    for c, h in zip(t.rows[0].cells, header):
        c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size)
        _border(c, "bottom"); _border(c, "top")
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = ""; pp = c.paragraphs[0]; pp.paragraph_format.space_after = Pt(0)
            pp.add_run(str(v)).font.size = Pt(size)
    for c in t.rows[-1].cells: _border(c, "bottom")
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths): c.width = Cm(w)
        for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(Cm(w).twips)))
        tblW = t._tbl.tblPr.find(qn("w:tblW"))
        if tblW is None: tblW = OxmlElement("w:tblW"); t._tbl.tblPr.append(tblW)
        tblW.set(qn("w:type"), "dxa"); tblW.set(qn("w:w"), str(int(Cm(sum(widths)).twips)))
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(lay)
    return t

def note(txt): para(txt, size=8)

def figure(path, width_cm=16, name=None):
    """Insert a figure; a copy under its final name goes to supp_figures (overview of all supplementary figures)."""
    doc.add_picture(str(path), width=Cm(width_cm))
    if name and Path(path).resolve() != (FIGDIR / name).resolve():
        import shutil; shutil.copyfile(path, FIGDIR / name)

def newpage(): doc.add_page_break()

from docx.enum.section import WD_SECTION
def orient(landscape):
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    w, h = sec.page_width, sec.page_height
    if landscape != (w > h):
        sec.page_width, sec.page_height = h, w
    sec.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT

# ── title and contents ────────────────────────────────────────────────────────
heading("Supplementary Material", 1)
para("Topography of Natural versus PLAS-Enhanced Slow Waves in Healthy and Cognitively Impaired Older Adults", italic=True)
para("Korian Wicki, Manon Haulotte, Marina Wunderlin, Leila Tarokh, Thomas Koenig, Marc Alain Züst")
heading("Contents", 2)
TOC = [
 None,   # removed: participant overview table
 "Figure S1. Individual MoCA, age and PSQI values by group",
 "Table S1. Sleep architecture and stimulus counts",
 None,   # removed: k = 4-8 correspondence figure
 "Table S2. Average-level (RAGU) microstate parameters",
 "Table S3. Single-trial microstate parameters",
 "Table S4. Single-trial results weighted by epoch count",
 "Figure S2. Time-resolved coverage in 250-ms bins",
 None,   # removed: GFP-matched coverage (table + figure)
 "Figure S3. Overlap with neighbouring stimuli: intervals between stimuli, ERP and coverage clusters by subset",
 "Table S5. Single-trial results by overlap subset",
 "Table S6. Single-trial results per experimental night",
 "Table S7. Single-trial results with sex as a covariate",
 "Table S8. Second induced peak: robustness to overlap",
 "Table S9. Pre-stimulus TANOVA effect: zero-phase vs. causal filter",
 "Table S10. Single-trial transition probabilities and transition entropy",
]
for x in TOC:
    if x: para(x)

# Participant overview table removed: not requested by the reviewers; the sample is described in the Methods.
master = pd.read_csv(ANA / "participants_master.csv")

# ── Figure S1 ─────────────────────────────────────────────────────────────────
inc = master[master.included == 1]
fig, axes = plt.subplots(1, 3, figsize=(7, 2.6))
rng = np.random.default_rng(1)
for ax, (col, lab) in zip(axes, [("MOCA_score", "MoCA"), ("age", "Age (years)"), ("PSQI_Score", "PSQI")]):
    for k, (g, c) in enumerate((("HC", HC_C), ("CI", CI_C))):
        y = inc.loc[inc.group == g, col].astype(float).dropna().values
        ax.scatter(k + rng.uniform(-0.12, 0.12, y.size), y, s=14, color=c, alpha=0.8, lw=0)
        m, h = y.mean(), 1.96 * y.std(ddof=1) / np.sqrt(y.size)
        ax.errorbar(k + 0.3, m, yerr=h, fmt="o", color="black", ms=4, capsize=3)
    if col == "MOCA_score": ax.axhline(25.5, color="grey", ls=":", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HC (n = 25)", "CI (n = 15)"]); ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel(lab); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIGDIR / "FigureS1.png", dpi=300); plt.close(fig)
newpage(); heading(TOC[1]); figure(FIGDIR / "FigureS1.png")
note("Each dot is one participant; black: mean with 95% confidence interval. Dotted line: group cutoff (MoCA ≤ 25 = CI). "
     "HC: MoCA M = 27.68, 95% CI [27.09, 28.27], range 26–30; CI: M = 23.13, 95% CI [22.27, 23.99], range 20–25.")

# ── Table S1 ──────────────────────────────────────────────────────────────────
sl = pd.read_csv(RES / "9. sleep_architecture" / "sleep_main_nights.csv", encoding="utf-8")
rows = [[r.variable, r["HC BL"], r["HC stim"], r["CI BL"], r["CI stim"], fp(r.pfdr_night_all), fp(r.pfdr_group_BL), fp(r.pfdr_group_x_night)] for _, r in sl.iterrows()]
sc = pd.read_csv(RES / "9. sleep_architecture" / "stimulus_counts.csv", index_col=0)
f = lambda x: f"{x.mean():.0f} ± {x.std():.0f}"
rows.append(["(Pseudo-)stimuli (n)", f(sc[sc.group == "HC"].BL), f(sc[sc.group == "HC"].stim_analysed),
             f(sc[sc.group == "CI"].BL), f(sc[sc.group == "CI"].stim_analysed), "–", "–", "–"])
newpage(); heading(TOC[2])
table(["Variable", "HC baseline", "HC stimulation", "CI baseline", "CI stimulation", "p_FDR night", "p_FDR group (baseline)", "p_FDR group × night"], rows, size=7.5)
note("Note. M ± SD for the baseline night and the analysed stimulation night. TIB = time in bed; TST = total sleep time; SE = sleep efficiency; SL = sleep latency; WASO = wake after sleep onset; stages as % of TST. "
     "Night: baseline vs. stimulation night across participants; group × night: interaction. p-values FDR-corrected across sleep variables. "
     "(Pseudo-)stimuli: sham markers in the baseline night, delivered stimuli in the analysed stimulation night; the number of stimuli in the analysed night did not differ between groups (Mann-Whitney U, p = .81). "
     "Arousal scores were not available for the present re-analysis.")

# k = 4-8 correspondence figure (archive/23_correspondence_k4_k8.py) removed: not requested by the reviewers.

# ── Table S2: RAGU microstate parameters ─────────────────────────────────────
rg = pd.read_csv(ANA / "matlab_ragu" / "Ragu_results" / "ragu_microstate_stats_table.csv")
WIN = ["-200 to 800 ms", "800 to 1900 ms"]; EFF = ["condition", "group", "group x condition"]
def rcell(d):
    if d.empty or pd.isna(d.p.iloc[0]): return "–"
    r = d.iloc[0]; return f"{fp(max(r.p, 1/5001))} ({fp(r.p_fdr)})" + ("*" if r.sig_fdr else "")
rows = []
for cls in range(1, 9):
    for par in ["onset", "offset", "duration", "AUC", "centre of gravity", "mean GFP"]:
        rows.append([f"Class {cls}" if par == "onset" else "", par] +
                    [rcell(rg[(rg.window == w) & (rg.effect == e) & (rg.figure_class == cls) & (rg.parameter == par)]) for w in WIN for e in EFF])
orient(True); heading(TOC[4])
table(["Class", "Parameter", "−200–800 ms: condition", "group", "group × condition", "800–1900 ms: condition", "group", "group × condition"], rows, size=6.5,
      widths=[2.0, 3.0, 3.2, 2.6, 3.2, 3.2, 2.6, 3.2])
note("Note. p (p_FDR) of RAGU's randomization tests (5000 randomizations) for each microstate class (numbering of Figure 3) and parameter. "
     "p_FDR: Benjamini–Hochberg within each analysis window and effect, across classes and parameters. *p_FDR < .05. –: not tested (class absent in at least one cell, parameter undefined). "
     "p-values of 0 are reported as < .001 (1/(randomizations + 1)). In the second window (second induced peak), which was characterized descriptively, condition effects are shown for completeness.")

# ── Table S3 / S4: single-trial scalar parameters ─────────────────────────────
def scalar_table(path):
    p = pd.read_csv(path); p = p[p.family == "scalar"]
    rows = []
    for ms in range(1, 5):
        for par in PAR:
            g = p[p.col == f"{ms-1}_{par}"].set_index("test")
            cells = [MSNAME[ms] if par == "timecov" else "", PAR[par]]
            for test in ("all_condition_effect", "healthy_condition_effect", "impaired_condition_effect", "interaction_group_x_condition"):
                r = g.loc[test]; lo, hi = ci_es(r.effect_size, r.es_type, r.n)
                cells += [es_txt(r.effect_size, lo, hi), fp(r.p_fdr)]
            rows.append(cells)
    grp = p[p.test.str.startswith("group_effect")]
    return rows, grp.p_fdr.min()
H4 = ["State", "Parameter", "All: dz [95% CI]", "p_FDR", "HC: dz [95% CI]", "p_FDR", "CI: dz [95% CI]", "p_FDR", "Group × Condition: d [95% CI]", "p_FDR"]
rows, gmin = scalar_table(RES / "4. statistics" / "k4" / "permutation_results_k4.csv")
sec = doc.sections[-1]
orient(True); heading(TOC[5]); table(H4, rows, size=7, widths=[3.6, 2.2, 3.0, 1.3, 3.0, 1.3, 3.0, 1.3, 3.4, 1.3])
note("Note. Stimulation vs. baseline (sham) night; positive values = higher under stimulation. Paired sign-flip permutation tests (10,000 permutations); interaction: permutation of group labels on within-participant differences. "
     "dz = Cohen's dz (within participants), d = Cohen's d (interaction: HC vs. CI difference in the stimulation effect); approximate 95% CIs, not adjusted for multiple comparisons. "
     f"p_FDR: Benjamini–Hochberg within the family of temporal parameters (4 states × 4 parameters), separately per contrast. Group main effects (baseline and stimulation night) did not survive correction (smallest p_FDR = {fp(gmin)}). GEV = global explained variance.")
rows_w, _ = scalar_table(RES / "4. statistics_weighted" / "k4" / "permutation_results_k4.csv")
newpage(); heading(TOC[6]); table(H4, rows_w, size=7, widths=[3.6, 2.2, 3.0, 1.3, 3.0, 1.3, 3.0, 1.3, 3.4, 1.3])
note("Note. As Table S3, but with participants weighted by their number of epochs in the permutation tests instead of equally. Effect sizes are computed unweighted and are therefore identical to Table S3; only the p-values differ.")

# ── Figure S2: 250-ms bins (e2_figure4.py: same layout as Figure 4) ─────────
# final version finished in Illustrator (as Figure 4) if present, otherwise the script output
_s2 = PROJECT_ROOT.parent / "figures" / "Figure 4" / "FigureS2.png"
orient(False); heading(TOC[7]); figure(_s2 if _s2.exists() else RES / "17. figure4" / "FigureS2.png", 16, "FigureS2.png")
note("Same layout as Figure 4, with microstate coverage averaged in non-overlapping 250-ms bins from −1.5 to 3.0 s instead of full temporal resolution "
     "(sensitivity analysis; analysis of the original submission). Top: grand-average ERP at Fz. Below: coverage for healthy controls (solid) and participants "
     "with cognitive impairment (dashed) in the stimulation (red) and sham (grey) conditions, plotted at the bin centres. Blue shading: bins with a significant "
     "stimulation effect across all participants (paired permutation tests, Benjamini–Hochberg correction across all bins and states, 18 bins × 4 states). "
     "The pattern is largely consistent with the cluster-based analysis at full temporal resolution reported in the main text (Figure 4). "
     "Right: group-level microstate templates (normalised amplitude, a.u.).")

# GFP-matched coverage table and figure (archive/16_gfp_matched_coverage.py) removed: not requested by the reviewers; kept in case it is asked for.

# ── Figure S3 / Table S5: overlap with neighbouring stimuli ──────────────────
ov = pd.read_csv(RES / "5. overlap_check" / "overlap_per_participant.csv")
pp = ov.groupby("condition").pct_prev_in_window.mean()
nx = pd.read_csv(RES / "18. next_stimulus_timing" / "next_stimulus_stim_vs_sham.csv").set_index("measure")   # c1_stimulus_intervals.py
ns, nm = nx.loc["pct_next3"], nx.loc["median_next_s"]
VAR = [("all_epochs", "All epochs"), ("prev1.5", "No prec. < 1.5 s"), ("prev3.7", "No prec. < 3.7 s"),
       ("next3.0", "No foll. < 3 s"), ("prev3.7_next3.0", "Fully isolated")]
newpage(); heading(TOC[9])
para("Because stimuli were triggered by detected slow-wave peaks, they often occurred in short sequences, so that responses to neighbouring stimuli could overlap within an epoch "
     f"(minimum interval between stimuli {ov.min_isi_s.min():.2f} s). On average, {pp['sham']:.0f}% (baseline) and {pp['stim']:.0f}% (stimulation) of epochs contained a preceding (pseudo-)stimulus "
     "within the 1.5-s pre-stimulus interval, and a similar proportion a following one within 1.5 s. The stimulation response lasts about 2.2 s (end of the TANOVA condition effect). "
     "The analyses were therefore repeated on subsets defined in advance from this duration: no preceding stimulus within 1.5 s (onset of a preceding response excluded); within 3.7 s "
     "(1.5 s pre-stimulus interval + 2.2 s response: pre-stimulus interval free of any earlier response); no following stimulus within the epoch (3 s); and both (fully isolated). "
     f"Under stimulation, epochs were not more often followed by a stimulus within 3 s than in the baseline night ({ns.stim_mean:.0f}% vs. {ns.sham_mean:.0f}%), "
     f"but the following stimulus occurred slightly earlier (median {nm.stim_mean:.2f} vs. {nm.sham_mean:.2f} s), so that its response falls into the late part of the epoch.", size=9)
figure(RES / "5. overlap_check" / "overlap_isi_hist.png", 13, "FigureS3A.png")
note("Figure S3A. Time since the preceding (pseudo-)stimulus for each analysed epoch, pooled across participants, in the baseline (grey) and stimulation night (red). Bars give the proportion of epochs per 0.25-s interval; intervals longer than 10 s are combined in the last bar. Epochs to the left of the line at 1.5 s (start of the epoch) contain a preceding stimulus within the pre-stimulus interval, on average 32% and 36% of the epochs per participant; epochs to the right of the line at 3.7 s, the pre-stimulus interval plus the 2.2-s response, were classified as having no preceding stimulus. No interval was shorter than 0.85 s, and the peak at about 1 s reflects the stimulation of consecutive slow waves; the distribution was similar in both nights.")
figure(RES / "10. second_peak" / "second_peak_roi_by_overlap_subset.png", 15, "FigureS3B.png")
note("Figure S3B. Fronto-central ERP (electrodes around Fz/FCz; average reference), stimulation vs. baseline, for all epochs and each overlap subset. Grey: second-peak window (1080–1380 ms). "
     "The induced trough and second peak are present in every subset, including fully isolated epochs, whereas the positive deflection at about −1 s in all epochs (the preceding slow wave of a sequence) disappears without preceding stimuli.")
iso = RES / "19. coverage_clusters_isolated"                       # c3_coverage_clusters_isolated.py
ms1 = pd.read_csv(iso / "ms1_second_peak_exploratory.csv").set_index("subset")
figure(iso / "coverage_clusters_isolated_k4.png", 16, "FigureS3C.png")
_m = lambda k: f"{ms1.loc[k, 'mean_diff_pp']:+.1f} pp (dz = {ms1.loc[k, 'dz']:.2f}, p = {fp(ms1.loc[k, 'p_two_sided'])})"
note("Figure S3C. Stimulation effect on single-trial microstate coverage (stimulation − baseline, percentage points; display smoothing 150 ms) for all epochs, "
     "epochs without a preceding stimulus within 3.7 s, and fully isolated epochs. Bars: significant clusters (cluster-based permutation test as in Figure 4, Bonferroni-corrected across the four states). "
     f"Exploratory post-hoc test: MS 1 coverage in the second-peak window (1080–1380 ms), stimulation vs. baseline: all epochs {_m('all_epochs')}, "
     f"no preceding stimulus {_m('prev3.7')}, fully isolated {_m('prev3.7_next3.0')} (two-sided paired permutation tests).")

summ = pd.read_csv(RES / "13. overlap_variants" / "summary_overlap_variants_k4.csv").set_index("col")
def cell(col, v):
    r = summ.loc[col]; return f"{r['dz_' + v]:.2f} ({fp(r['pfdr_' + v])})"
rows = []
for ms in range(1, 5):
    for par in PAR:
        rows.append([MSNAME[ms] if par == "timecov" else "", PAR[par]] + [cell(f"{ms-1}_{par}", v) for v, _ in VAR])
for label, bins in (("Pre-stimulus", (1, 2, 3)), ("Early (250–1000 ms)", (7, 8, 9)), ("Late (1000–2250 ms)", (10, 11, 12, 13, 14))):
    for b in bins:
        t0 = -1500 + 250 * b
        for ms in range(1, 5):
            rows.append([f"{label}: {t0} to {t0 + 250} ms" if ms == 1 else "", f"MS {ms} coverage"] + [cell(f"bin{b}_state{ms-1}_cov", v) for v, _ in VAR])
orient(True); heading(TOC[10])
table(["State / bin", "Parameter"] + [h for _, h in VAR], rows, size=6.5, widths=[4.6, 2.6, 3.0, 3.0, 3.0, 3.0, 3.0])
note("Note. Stimulation effect (stimulation vs. baseline, all participants): dz (p_FDR). FDR within family (temporal parameters; 250-ms bins) and contrast, separately for each subset. "
     "Subsets as described for Figure S3. Participants: 40 in all subsets except fully isolated (39; one participant with < 20 epochs). "
     "Early post-stimulus effects (higher MS 3, lower MS 2 and MS 4 coverage at 250–500 ms) persisted in every subset; pre-stimulus effects were absent without preceding stimuli within 3.7 s; "
     "later effects (MS 1 and MS 3 after about 1000 ms) were not significant in fully isolated epochs (cluster-based results: Figure S3C).")
orient(False)

def _tanova_nights_note():
    """Average-level TANOVA per night (d6_ragu_nights.py + RAGU, c02_ragu_nights.m)."""
    f = ANA / "matlab_ragu" / "derived" / "RAGU_TXT_nights" / "tanova_nights_global.csv"
    t = pd.read_csv(f).set_index("night")
    assert list(t.index) == ["E1", "E2", "E3"]
    pmax = max(t.p_count_condition.max(), t.p_auc_condition.max())   # smallest attainable p = 1/5001
    pmin = min(t.p_count_group.min(), t.p_count_interaction.min())
    ns = ", ".join(str(int(x)) for x in t.n)
    return (f"Average level: the TANOVA repeated for each night (baseline vs. E1, E2, E3; n = {ns}) showed a significant "
            f"condition effect in every night (global count and p-AUC: p = {f'{pmax:.4f}'[1:]}) and no group effect or "
            f"Group × Condition interaction (global count: all p ≥ {f'{pmin:.2f}'[1:]}).")


# ── Table S6: nights ──────────────────────────────────────────────────────────
nc = pd.read_csv(RES / "7. all_nights" / "nights_comparison_k4.csv"); nc = nc[nc.family == "scalar"]
pn = pd.read_csv(RES / "7. all_nights" / "permutation_results_nights_k4.csv")
nn = pn.groupby("night").n_participants.first().to_dict()
pa = pn[(pn.test == "all_condition_effect") & (pn.family == "scalar")].set_index(["col", "night"])   # d1: per night
def _dz(col, night):
    r = pa.loc[(col, night)]; return f"{r.effect_size:.2f}" + ("*" if r.sig_fdr else "")
n_all = int(pd.read_csv(RES / "7. all_nights" / "parameters_nights_k4.csv").groupby("participant_id").night.nunique().eq(4).sum())
rows = []
for ms in range(1, 5):
    for par in PAR:
        r = nc[nc.col == f"{ms-1}_{par}"].iloc[0]
        c = f"{ms-1}_{par}"
        rows.append([MSNAME[ms] if par == "timecov" else "", PAR[par], _dz(c, "E1"), _dz(c, "E2"), _dz(c, "E3"), fp(r.pfdr_night), fp(r.pfdr_night_x_group)])
heading(TOC[11])
table(["State", "Parameter", f"E1: dz (n = {nn.get('E1')})", f"E2: dz (n = {nn.get('E2')})", f"E3: dz (n = {nn.get('E3')})", "p_FDR night", "p_FDR night × group"], rows, size=7.5)
note("Note. Stimulation effect (stimulation night vs. baseline night, all participants) separately for each experimental night. "
     "Night: does the stimulation effect differ between E1, E2 and E3; night × group: does this differ between HC and CI (permutation tests, FDR across the 16 parameters). "
     "dz per night: stimulation night vs. baseline, all participants with ≥ 50 artefact-free epochs in that night (sample sizes therefore differ between nights); "
     "*p_FDR < .05 (FDR within the temporal parameters, per night). "
     f"Night and night × group: participants with all three experimental nights (n = {n_all}). "
     + _tanova_nights_note())

# ── Table S7: sex ─────────────────────────────────────────────────────────────
sx = pd.read_csv(RES / "8. sex_sensitivity" / "sex_sensitivity_k4.csv"); sx = sx[sx.family == "scalar"]
rows = []
for ms in range(1, 5):
    for par in PAR:
        g = sx[sx.col == f"{ms-1}_{par}"]
        ce = g[g.test == "condition_effect"].set_index("model")
        mo = g[g.test == "sex_moderation"].iloc[0]
        rows.append([MSNAME[ms] if par == "timecov" else "", PAR[par],
                     f"{ce.loc['unadjusted','t']:.2f}", fp(ce.loc['unadjusted','p_fdr']),
                     f"{ce.loc['sex_adjusted','t']:.2f}", fp(ce.loc['sex_adjusted','p_fdr']), fp(mo.p_fdr)])
newpage(); heading(TOC[12])
table(["State", "Parameter", "Condition effect: t (unadjusted)", "p_FDR", "t (sex-adjusted)", "p_FDR", "Sex × Condition: p_FDR"], rows, size=7.5)
note("Note. Stimulation effect (stimulation − baseline) without and with sex as a covariate, and test of whether the stimulation effect differs between women and men (Sex × Condition). "
     "FDR across the 16 parameters per test. Cell sizes are small (HC: 19 F / 6 M; CI: 7 F / 8 M); the analysis cannot establish that sex is irrelevant. "
     "At the average level, a TANOVA with sex as between-subject factor (5000 randomizations) showed no effect of sex on scalp topography and no Sex × Condition interaction (global count: p = .61 and p = .74; global p-AUC: p = .35 and p = .58; no period exceeded the duration criterion of 240 and 220 ms), whereas the condition effect was unchanged (p = .0002).")

# Figure S4 (single-trial amplitude of the second peak, c5_second_peak_single_trial.py) removed:
# reported in the response letter only (amplitude, not topography).

topo  = pd.read_csv(ANA / "matlab_ragu" / "derived" / "RAGU_TXT_overlap" / "topographic_check_by_subset.csv").set_index("subset")
rows = []
for key, lab in (("all", "All epochs"), ("prev3.7", "No preceding < 3.7 s"),
                 ("next3.0", "No following < 3 s"), ("prev3.7_next3.0", "Fully isolated")):
    r = topo.loc[key]
    rows.append([lab, int(r.n), f"{r.roi_P2_dz:.2f}", f"{r.r_stim_map_vs_all_P2:.3f}"])
newpage(); heading(TOC[13])
table(["Subset", "n", "ERP stim. vs. baseline (dz)", "Second-peak map: r with all epochs"], rows, size=7.5)
note("Note. ERP: fronto-central amplitude in the second-peak window (1080–1380 ms), stimulation minus baseline across participants (all p < .001). "
     "Map: spatial correlation of the grand-average stimulation topography in the second-peak window with that of all epochs. "
     "The second induced peak and its topography were preserved in every subset, including fully isolated epochs.")

# ── Table S9: filter check (d4_filter_check.py) ─────────────────────────────
fc   = RES / "15. filter_check"
proj = pd.read_csv(fc / "projection_test.csv").set_index(["subset", "window"])
scr  = pd.read_csv(fc / "tanova_screen.csv").set_index(["filter", "subset", "window"])
rows = []
for sub, lab in (("all", "All epochs"), ("prev3.7", "No preceding stimulus < 3.7 s")):
    for w in ("-1500 to -1000", "-1000 to -500", "-500 to -150"):
        pr = proj.loc[(sub, w)]
        rows.append([lab if w.startswith("-1500") else "", w.replace("-", "−") + " ms",
                     f"{100 * pr.share:.0f}", f"{pr.spatial_r:.2f}",
                     f"{scr.loc[('zero', sub, w), 'pct_p_below_05']:.0f}", f"{scr.loc[('causal', sub, w), 'pct_p_below_05']:.0f}"])
newpage(); heading(TOC[14])
table(["Epochs", "Window", "Projection: share of observed GFP (%)", "Projection: spatial r with observed difference",
       "Zero-phase filter: time points p < .05 (%)", "Causal filter: time points p < .05 (%)"], rows, size=7.5)
note("Note. Projection: the post-stimulus stimulation response (stimulation minus baseline, grand average) was filtered again with the zero-phase filter of the main analysis "
     "(FIR, 0.3–20 Hz, order 2200); the part projected into the pre-stimulus window is compared with the observed pre-stimulus difference in strength (global field power, GFP) and topography (spatial correlation). "
     "TANOVA screen: condition effect (stimulation vs. baseline, all participants) on L2-normalised maps, paired sign-flip randomisation (1000 permutations), computed in Python for each time point; "
     "reported is the proportion of time points with p < .05 per window (point-wise, without the global duration criterion of RAGU). "
     "Causal filter: same pass band, minimum-phase FIR filter, which cannot project activity backwards in time; the same epochs as in the main analysis were used (except 0.1% that exceeded the ±300 µV criterion only after causal filtering). "
     "Without preceding stimuli, the pre-stimulus effect was abolished by the causal filter; the residual effect across all epochs is attributable to responses to preceding stimuli (Figure S3, Table S5).")

# ── Table S10: transitions and entropy (a3_stats.py) ─────────────────────────
pt = pd.read_csv(RES / "4. statistics" / "k4" / "permutation_results_k4.csv")
pt = pt[pt.family.isin(["transition", "entropy"])]
TESTS = ("all_condition_effect", "healthy_condition_effect", "impaired_condition_effect", "interaction_group_x_condition")
def trow(col, lab1, lab2):
    g = pt[pt.col == col].set_index("test"); cells = [lab1, lab2]
    for test in TESTS:
        r = g.loc[test]; lo, hi = ci_es(r.effect_size, r.es_type, r.n)
        cells += [es_txt(r.effect_size, lo, hi), fp(r.p_fdr) + ("*" if r.sig_fdr else "")]
    return cells
rows = []
for i in range(4):
    for j in range(4):
        if i != j:
            rows.append(trow(f"Tsw_{i}_to_{j}", "Transition probability" if (i, j) == (0, 1) else "", f"MS {i+1} → MS {j+1}"))
for i in range(4):
    rows.append(trow(f"sw_entropy_state{i}", "Transition entropy" if i == 0 else "", f"from MS {i+1}"))
gmin_t = pt[pt.test.str.startswith("group_effect")].p_fdr.min()
orient(True); heading(TOC[15]); table(["Measure", "Transition", "All: dz [95% CI]", "p_FDR", "HC: dz [95% CI]", "p_FDR", "CI: dz [95% CI]", "p_FDR", "Group × Condition: d [95% CI]", "p_FDR"],
                                     rows, size=7, widths=[3.6, 2.2, 3.0, 1.3, 3.0, 1.3, 3.0, 1.3, 3.4, 1.3])
note("Note. Stimulation vs. baseline (sham) night; positive values = higher under stimulation. Transition probability: probability of switching from one state to another, "
     "excluding self-transitions; transition entropy: entropy of the outgoing transitions of a state (lower = more predictable next state). "
     "Tests and effect sizes as in Table S3. p_FDR: Benjamini–Hochberg within each family (12 transition probabilities; 4 entropies), separately per contrast; *p_FDR < .05. "
     f"Group main effects (baseline and stimulation night) did not survive correction (smallest p_FDR = {fp(gmin_t)}).")


# ── Formatting: statistical symbols in italics, p_FDR as p with subscript FDR ──

import copy
SYM = r'(SD|dz|M|N|n|p|r|d|t|F|k|U)'
RX = [re.compile(r'(?<![A-Za-z])'+SYM+r'(?=\s?[=<>≥≤]|-AUC)'),
      re.compile(r'\|(r)\|'),
      re.compile(r'(?<![A-Za-z])(M)(?= ±)'), re.compile(r'(?<=± )(SD)(?![A-Za-z])'),
      re.compile(r'(?<=Cohen[’\']s )(dz|d)(?![A-Za-z])'),
      re.compile(r'(?<![A-Za-z])(dz|d|M)(?= \[95)'),
      re.compile(r'(?<![A-Za-z])(dz|p|t|d)(?= \((?:n =|p_FDR|unadjusted|sex-adjusted))'),
      re.compile(r'(?<=\()(dz|d)(?=\))'),
      re.compile(r'(?<![A-Za-z])(dz)(?= per night)'),
      re.compile(r'(?<=: )(r)(?= with)'),
      re.compile(r'(?<![A-Za-z])(p)(?=_FDR)')]
NOT_IT = re.compile(r'\[95% CI\]')
def _is_it(r):
    rp = r.find(qn('w:rPr')); i = None if rp is None else rp.find(qn('w:i'))
    return i is not None and i.get(qn('w:val')) not in ('0', 'false')
def _set(rp, tag, on, val=None):
    for e in rp.findall(qn(tag)): rp.remove(e)
    if not on: return
    e = OxmlElement(tag)
    if val: e.set(qn('w:val'), val)
    order = ['rStyle','rFonts','b','bCs','i','iCs','caps','smallCaps','strike','dstrike','outline','shadow','emboss','imprint','noProof','snapToGrid','vanish','webHidden','color','spacing','w','kern','position','sz','szCs','highlight','u','effect','bdr','shd','fitText','vertAlign','rtl','cs','em','lang','eastAsianLayout','specVanish','oMath']
    k = order.index(tag.split(':')[1]); idx = len(rp)
    for j, ch in enumerate(rp):
        n = ch.tag.split('}')[1]
        if n in order and order.index(n) > k or n in ('ins','del','rPrChange'): idx = j; break
    rp.insert(idx, e)
def format_paragraph(p, fix_ci=True):
    runs = []
    for r in p.iter(qn('w:r')):
        if any(a.tag == qn('w:del') and a.getparent().tag != qn('w:rPr') for a in r.iterancestors()): continue
        if r.find(qn('w:instrText')) is not None: continue
        ts = r.findall(qn('w:t'))
        if len(ts) != 1: continue
        runs.append(r)
    chars = []
    for r in runs:
        t = r.find(qn('w:t')).text or ''; chars += [(r, i) for i in range(len(t))]
    s = ''.join((r.find(qn('w:t')).text or '')[i] for r, i in chars)
    want = {}                       # char index -> 'i' | 'sub' | 'del' | 'noi'
    for rx in RX:
        for m in rx.finditer(s):
            for k in range(m.start(1), m.end(1)): want[k] = 'i'
    for m in re.finditer(r'(?<=p)_(FDR)', s):
        want[m.start()] = 'del'
        for k in range(m.start(1), m.end(1)): want[k] = 'sub'
    if fix_ci:
        for m in NOT_IT.finditer(s):
            for k in range(m.start(), m.end()): want[k] = 'noi'
    changed = 0
    for r in runs:
        idxs = [k for k, (rr, _) in enumerate(chars) if rr is r]
        if not any(k in want for k in idxs): continue
        t = r.find(qn('w:t')).text; base_it = _is_it(r)
        segs = []
        for j, k in enumerate(idxs):
            w = want.get(k)
            if w == 'del': continue
            key = ('i' if w == 'i' else 'sub' if w == 'sub' else 'noi' if w == 'noi' else None)
            if segs and segs[-1][0] == key: segs[-1][1] += t[j]
            else: segs.append([key, t[j]])
        need = any(k is not None for k, _ in segs) or len(segs) != 1 or ''.join(x for _, x in segs) != t
        if not need: continue
        for key, txt in segs:
            n = copy.deepcopy(r); nt = n.find(qn('w:t')); nt.text = txt
            nt.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            rp = n.find(qn('w:rPr'))
            if rp is None: rp = OxmlElement('w:rPr'); n.insert(0, rp)
            if key == 'i' and not base_it: _set(rp, 'w:i', True)
            if key == 'noi': _set(rp, 'w:i', False)
            if key == 'sub':
                _set(rp, 'w:i', False); _set(rp, 'w:vertAlign', True, 'subscript')
            r.addprevious(n)
        r.getparent().remove(r); changed += 1
    return changed
def format_document(doc):
    n = 0
    for p in doc.paragraphs: n += format_paragraph(p._p)
    seen = set()
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                if c._tc in seen: continue
                seen.add(c._tc)
                for p in c.paragraphs: n += format_paragraph(p._p)
    return n

format_document(doc)

doc.save(DOCX); print("saved", DOCX)
