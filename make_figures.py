#!/usr/bin/env python
"""
FIGURES  -  7 main figures (PDF vector + PNG 300 dpi, 180 mm wide) + supplementary tables (one Excel workbook)
=============================================================================================================
Reads ONLY outputs written by stages 1-5 on this machine. Any missing input -> that panel says
"input missing: <file>" instead of being drawn (nothing is invented, nothing is hard-coded).

  Fig 1  single-cell atlas (UMAP lineage / site, composition, malignant cells, SE- vs EC-like markers)
  Fig 2  deconvolution (LODO benchmark, method comparison, TCGA composition, tumour-state read-out vs pathology)
  Fig 3  histology-independent TME programmes (Cramer's V before/after, loadings, scores, k selection)
  Fig 4  spatial (programme maps, Moran's I, co-localisation with TLS / tumour)
  Fig 5  chromosomal copy gain + tumour -> TME communication (MIF axis)
  Fig 6  pre-specified clinical analysis (TCGA stage I) + GSE99420 validation
  Fig 7  drug repurposing (positive controls, druggable targets, LINCS reversal)

Layout engine (v3, overlap-free):
  * matplotlib "constrained" layout -> room is reserved automatically for tick labels, axis labels,
    titles, colour bars and legends (all legends sit OUTSIDE the data area, to the right).
  * panel letter is part of the left-aligned panel title -> letter and title can never collide.
  * crowded tick labels are shrunk automatically; point labels are repelled from each other.
  * an automatic overlap checker measures every text box after rendering and writes any remaining
    collision to FIGURES_LOG.txt (target: "0 overlaps" for every figure).

Run:  conda activate scvi-env ; python make_figures.py        (pip install openpyxl for the Excel tables)
"""
import os, re, json, textwrap, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("TGCT_BASE", "/home/arshia/testis project"))
PROJ = BASE / "TGCT_ecotype"
S1, S2, S3, S4, S5 = (PROJ / x for x in ("stage1_atlas", "stage2_tcga", "stage3_clinical", "stage4_mechanism", "stage5_drugs"))
OUT = PROJ / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
W = 180 * MM
FS = 8          # Frontiers: smallest visible text >= 8 pt at final size (180 mm wide)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.titlesize": FS + 0.5, "axes.labelsize": FS,
                     "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "legend.title_fontsize": FS,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "axes.titlelocation": "left", "mathtext.default": "regular", "legend.frameon": False,
                     "legend.handlelength": 1.0, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
                     "legend.labelspacing": 0.25, "legend.columnspacing": 0.8})
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000", "#999999"]
# 24 mutually distinguishable colours (Okabe-Ito first, then hand-picked; no near-duplicates)
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999",
       "#8C564B", "#9467BD", "#17BECF", "#BCBD22", "#E377C2", "#393B79", "#AD494A", "#637939",
       "#C5B0D5", "#FF9896", "#98DF8A", "#C49C94", "#7F7F7F", "#DBDB8D", "#9EDAE5", "#843C39"]
HIST_C = {"SE": "#0072B2", "NSGCT": "#D55E00", "other": "#999999"}
SITE_C = {"primary": "#0072B2", "LN": "#D55E00", "blood": "#CC79A7", "normal": "#009E73"}
# drug-target pairs judged mechanistically implausible on inspection (kept in Suppl. tables, not plotted)
ARTEFACT_DRUGS = {"DOCOSANOL", "HALOTHANE", "TRIPARANOL", "AZATHIOPRINE", "AZATHIOPRINE SODIUM", "AZARIBINE", "CEDAZURIDINE"}
LOG = []
OVERLAP_SUMMARY = {}

NAME = {"P1_TME_immune": "P1 immune", "P2_TME_stromal": "P2 stromal", "P3_EC_likeness": "P3 EC-likeness",
        "P4_IFNG_programme": "P4 IFNG", "P5_CD163_macro": "P5 CD163 macro", "TLS12_chemokine": "TLS12 chemokine",
        "Tumour_fraction": "tumour fraction", "P1_TME_immune_noTLSgenes": "P1 immune (no TLS genes)",
        "Tumour_GCT": "Tumour (GCT)", "T_NK": "T/NK", "Germ_normal": "normal germ", "Myoid_SMC": "myoid / SMC",
        "chrX_gain": "chrX gain", "chr12p_gain": "12p gain", "KIT_mut": "KIT mut", "KRAS_mut": "KRAS mut"}


def pretty(s):
    s = str(s)
    if s in NAME:
        return NAME[s]
    s = s.replace("_TME_", "_").replace("_programme", "")
    return s.replace("_", " ")


def wrap(s, w=18):
    return "\n".join(textwrap.wrap(str(s), w, break_long_words=False)) or str(s)


_CREG = {}


def ccol(name):
    """one colour per cell type across ALL figures"""
    name = str(name)
    if name not in _CREG:
        _CREG[name] = PAL[len(_CREG) % len(PAL)]
    return _CREG[name]


def log(m):
    print(m, flush=True); LOG.append(m)


def read(f, **kw):
    f = Path(f)
    if not f.exists():
        log(f"  missing: {f.relative_to(PROJ) if PROJ in f.parents else f}")
        return None
    return pd.read_csv(f, **kw)


def new_fig(h_mm, scale=1.25, max_mm=230):
    """8-pt text needs ~25% more height than the original 6-pt design; one figure must fit one page"""
    fig = plt.figure(figsize=(W, min(h_mm * scale, max_mm) * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=3 / 72, h_pad=3 / 72, wspace=0.04, hspace=0.04)
    fig._repel = []
    fig._post_layout = []
    return fig


def title(ax, letter, txt=""):
    """letter + title as ONE left-aligned text -> cannot overlap each other; layout reserves its space"""
    lab = r"$\bf{" + letter.upper() + "}$" if letter else ""   # Frontiers: panels (A), (B), ...
    ax.set_title(f"{lab}   {txt}" if txt else lab, loc="left", fontsize=FS + 0.5, pad=4)


def missing(ax, what, letter=""):
    ax.text(0.5, 0.5, "input missing:\n" + wrap(what, 30), ha="center", va="center", fontsize=FS, color="grey",
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    title(ax, letter)


def legend_right(ax, handles=None, labels=None, ncol=1, title_=None, **kw):
    """legend outside the data area (right side) - constrained layout makes room for it"""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if labels is None:
        labels = [h.get_label() for h in handles]
    if not handles:
        return None
    return ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=ncol, title=title_,
                     alignment="left", **kw)


def patches(names, colors):
    return [Patch(facecolor=c, edgecolor="none", label=n) for n, c in zip(names, colors)]


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p)
    q = np.empty_like(p); q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def label_points(fig, ax, xs, ys, labels, fontsize=FS, sizes=None):
    anns = []
    rad0 = np.sqrt(np.asarray(sizes, float)) / 2 if sizes is not None else np.full(len(labels), 2.0)
    for x, y, l, r0 in zip(xs, ys, labels, rad0):
        a = ax.annotate(l, (x, y), xytext=(0.75 * r0 + 1.5, 0.75 * r0 + 1.0), textcoords="offset points", fontsize=fontsize, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="0.45", shrinkA=0, shrinkB=1.5),
                        annotation_clip=False, zorder=5)
        a.set_in_layout(False)
        anns.append(a)
    rad = np.sqrt(np.asarray(sizes, float)) / 2 if sizes is not None else np.full(len(anns), 2.0)   # marker radius (pt)
    fig._repel.append((ax, anns, rad))
    return anns


# ------------------------------------------------------------------ geometry helpers
def tbb(t, r):
    """window extent of the TEXT only (annotations: without their leader line)"""
    if isinstance(t, Annotation):
        t.update_positions(r)
    return Text.get_window_extent(t, r)


def _ovl(a, b, tol=0.6):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol) and (min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def _ticklabels(axis):
    lo, hi = sorted(axis.get_view_interval()); eps = 1e-9 * (abs(hi - lo) + 1)
    out = []
    for tk in axis.get_major_ticks():
        loc = tk.get_loc()
        if loc is None or not tk.get_visible() or not (lo - eps <= loc <= hi + eps):
            continue
        for lab in (tk.label1, tk.label2):
            if lab.get_visible() and lab.get_text().strip():
                out.append(lab)
    return out


def _slanted(t):
    return round(t.get_rotation()) % 90 != 0


def _axes_texts(ax):
    T = [(t, "title") for t in (ax.title, ax._left_title, ax._right_title)]
    T += [(ax.xaxis.label, "xlabel"), (ax.yaxis.label, "ylabel")]
    T += [(t, "xtick") for t in _ticklabels(ax.xaxis)] + [(t, "ytick") for t in _ticklabels(ax.yaxis)]
    T += [(t, "text") for t in ax.texts]
    leg = ax.get_legend()
    if leg is not None:
        T += [(t, "legend") for t in leg.get_texts()] + [(leg.get_title(), "legend")]
    return [(t, k) for t, k in T if t.get_visible() and t.get_text().strip()]


def _region(ax, r):
    """screen area an axes really occupies (axis-off axes: only their legend / image / texts)"""
    if ax.axison and any(s.get_visible() for s in ax.spines.values()):
        return ax.get_window_extent(r)
    if ax.images:
        return ax.get_window_extent(r)
    return None


def shrink_crowded(fig, r):
    """reduce the tick-label font of any axis whose own tick labels touch each other"""
    changed = False
    for ax in fig.axes:
        for which, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            labs = [t for t in _ticklabels(axis) if not _slanted(t)]   # slanted labels: boxes overlap by design
            bbs = [tbb(t, r) for t in labs]
            crowd = any(_ovl(bbs[i], bbs[j], 0.3) for i in range(len(bbs)) for j in range(i + 1, len(bbs)))
            if crowd and labs:
                fs = labs[0].get_fontsize()
                if fs > FS + 0.05:                         # never shrink below the 8-pt journal minimum
                    ax.tick_params(axis=which, labelsize=max(FS, fs - 0.5)); changed = True
    return changed


def _hits_marker(b, px, py, rp):
    return (b.x0 - rp < px < b.x1 + rp) and (b.y0 - rp < py < b.y1 + rp)


def repel(fig, r, ax, anns, rad, iters=600):
    k = 72 / fig.dpi
    radpx = np.asarray(rad) / k + 1.0
    axbb = ax.get_window_extent(r)
    pts = ax.transData.transform(np.array([a.xy for a in anns]))
    for _ in range(iters):
        bbs = [tbb(a, r) for a in anns]
        moved = False
        for i, a in enumerate(anns):
            b = bbs[i]; dx = dy = 0.0
            for j, c in enumerate(bbs):
                if j == i:
                    continue
                ox = min(b.x1, c.x1) - max(b.x0, c.x0); oy = min(b.y1, c.y1) - max(b.y0, c.y0)
                if ox > 0 and oy > 0:
                    sy = np.sign((b.y0 + b.y1) - (c.y0 + c.y1)) or (1 if i > j else -1)
                    sx = np.sign((b.x0 + b.x1) - (c.x0 + c.x1)) or (1 if i > j else -1)
                    if oy <= ox:
                        dy += sy * (oy / 2 + 0.6)
                    else:
                        dx += sx * (ox / 2 + 0.6)
            for j, (px, py) in enumerate(pts):           # keep labels off every marker (incl. its own)
                if _hits_marker(b, px, py, radpx[j]):
                    cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
                    dy += (1.0 if cy >= py else -1.0) * 1.2
                    dx += (1.0 if cx >= px else -1.0) * 0.6
            if b.x0 < axbb.x0: dx += axbb.x0 - b.x0
            if b.x1 > axbb.x1: dx -= b.x1 - axbb.x1
            if b.y0 < axbb.y0: dy += axbb.y0 - b.y0
            if b.y1 > axbb.y1: dy -= b.y1 - axbb.y1
            if abs(dx) > 0.05 or abs(dy) > 0.05:
                a.xyann = (a.xyann[0] + dx * k, a.xyann[1] + dy * k); moved = True
        if not moved:
            break
    # greedy fallback for any label still touching a marker / another label / the panel edge
    def bad(i, bbs):
        b = bbs[i]
        if b.x0 < axbb.x0 or b.x1 > axbb.x1 or b.y0 < axbb.y0 or b.y1 > axbb.y1:
            return True
        if any(_hits_marker(b, px, py, radpx[j]) for j, (px, py) in enumerate(pts)):
            return True
        return any(_ovl(b, c, 0.3) for j, c in enumerate(bbs) if j != i)
    bbs = [tbb(a, r) for a in anns]
    for i, a in enumerate(anns):
        if not bad(i, bbs):
            continue
        best = None
        for dist in np.arange(4, 40, 2.5):                 # points
            for ang in np.deg2rad([45, 135, -45, -135, 90, -90, 0, 180, 22, 158, -22, -158, 68, 112, -68, -112]):
                ha = "left" if np.cos(ang) > 0.2 else ("right" if np.cos(ang) < -0.2 else "center")
                va = "bottom" if np.sin(ang) > 0.2 else ("top" if np.sin(ang) < -0.2 else "center")
                a.set_ha(ha); a.set_va(va)
                a.xyann = (dist * np.cos(ang) + (rad[i] * np.cos(ang) if ha != "center" else 0),
                           dist * np.sin(ang) + (rad[i] * np.sin(ang) if va != "center" else 0))
                bbs[i] = tbb(a, r)
                if not bad(i, bbs):
                    best = True; break
            if best:
                break
    for a in anns:                                         # leader line only when the label moved away
        if a.arrow_patch is not None and np.hypot(*a.xyann) < 7:
            a.arrow_patch.set_visible(False)


def check_overlaps(fig, r):
    issues = []
    items = []
    for ai, ax in enumerate(fig.axes):
        for t, kind in _axes_texts(ax):
            items.append((t, kind, ai, tbb(t, r)))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            items.append((t, "figtext", -1, tbb(t, r)))
    lab = lambda t: repr(t.get_text().replace("\n", " ")[:40])
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ti, ki, ai, bi = items[i]; tj, kj, aj, bj = items[j]
            if ki == kj and ki in ("xtick", "ytick") and ai == aj and (_slanted(ti) or _slanted(tj)):
                continue                                     # neighbouring slanted labels: box overlap only
            if _ovl(bi, bj):
                issues.append(f"text/text: {lab(ti)} [{ki}] x {lab(tj)} [{kj}]")
    regions = [(ai, _region(ax, r)) for ai, ax in enumerate(fig.axes)]
    for ai, ax in enumerate(fig.axes):                       # legends as regions too
        leg = ax.get_legend()
        if leg is not None:
            regions.append((ai, leg.get_window_extent(r)))
    for t, kind, ai, b in items:
        for aj, reg in regions:
            if reg is not None and aj != ai and _ovl(b, reg, 1.0):
                issues.append(f"text over other panel: {lab(t)} [{kind}] -> axes #{aj}")
    for ai, ax in enumerate(fig.axes):                       # legend covering data marks of its own axes
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(r)
        if not _ovl(lb, ax.get_window_extent(r), 1.0):
            continue
        n = sum(_ovl(lb, p.get_window_extent(r), 0.5) for p in ax.patches)
        for c in ax.collections:
            try:
                off = c.get_offset_transform().transform(c.get_offsets())
                n += int(((off[:, 0] > lb.x0) & (off[:, 0] < lb.x1) & (off[:, 1] > lb.y0) & (off[:, 1] < lb.y1)).sum())
            except Exception:
                pass
        if n:
            issues.append(f"legend covers {n} data marks in axes #{ai}")
    for ax, anns, rad in getattr(fig, "_repel", []):          # point labels must not sit on a marker
        pts = ax.transData.transform(np.array([a.xy for a in anns]))
        rp = np.asarray(rad) * fig.dpi / 72
        for a in anns:
            b = tbb(a, r)
            if any(_hits_marker(b, px, py, q * 0.8) for (px, py), q in zip(pts, rp)):
                issues.append(f"label on a marker: {lab(a)}")
    for ai, ax in enumerate(fig.axes):                       # point labels must stay inside their panel
        axbb = ax.get_window_extent(r)
        for t in ax.texts:
            if isinstance(t, Annotation) and t.get_visible():
                b = tbb(t, r)
                if b.x0 < axbb.x0 - 1 or b.x1 > axbb.x1 + 1 or b.y0 < axbb.y0 - 1 or b.y1 > axbb.y1 + 1:
                    issues.append(f"label outside its panel: {lab(t)}")
    return sorted(set(issues))


def save(fig, name):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        fig.canvas.draw()
    for w_ in wl:
        if "layout" in str(w_.message).lower():
            log(f"  LAYOUT WARNING {name}: {w_.message}")
    for _ in range(6):                                       # layout <-> tick-label size fixed point
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        if not shrink_crowded(fig, r):
            break
    for _ in range(5):                                       # outside legends must not widen the PNG beyond 180 mm
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        tb = fig.get_tightbbox(r)
        excess = tb.width + 0.08 - W                         # 0.08 in = pad_inches on both sides
        if excess <= 0.01:
            break
        fig.set_figwidth(fig.get_figwidth() - excess - 0.01)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for fn_ in getattr(fig, "_post_layout", []):
        fn_(fig, r)
    if fig._post_layout:
        fig.canvas.draw(); r = fig.canvas.get_renderer()
    if fig._repel:
        fig.set_layout_engine("none")                        # freeze geometry, then place point labels
        for ax, anns, rad in fig._repel:
            repel(fig, r, ax, anns, rad)
        fig.canvas.draw(); r = fig.canvas.get_renderer()
    issues = check_overlaps(fig, r)
    OVERLAP_SUMMARY[name] = len(issues)
    log(f"  overlap check {name}: {len(issues)} overlap(s)")
    for s in issues:
        log(f"    - {s}")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.04, dpi=300)
    try:                                                     # Frontiers accepts TIFF/JPEG/EPS: RGB TIFF, 300 dpi, LZW
        from PIL import Image
        im = Image.open(OUT / f"{name}.png")
        if im.mode != "RGB":
            bg = Image.new("RGB", im.size, "white"); bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1]); im = bg
        im.save(OUT / f"{name}.tif", dpi=(300, 300), compression="tiff_lzw")
    except Exception as e:
        log(f"  TIFF export failed for {name}: {e}")
    plt.close(fig)
    log(f"  saved {name}")


def jitter_box(ax, data, colors, labels, s=3):
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                    boxprops=dict(lw=0.5))
    for p_, c in zip(bp["boxes"], colors):
        p_.set_facecolor(c); p_.set_alpha(0.45)
    for i, v in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i + 1, 0.07, len(v)), v, s=s, c="k", lw=0, zorder=3)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)


# ============================================================ Fig 1
def fig1():
    fig = new_fig(165)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1.25, 0.8, 0.8])
    top = gs[0, :].subgridspec(1, 2, width_ratios=[1, 1])
    axA, axB = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1]); axE = fig.add_subplot(gs[1, 2])
    fa = S1 / "TGCT_atlas_stage1.h5ad"
    if fa.exists():
        import anndata as ad
        a = ad.read_h5ad(fa, backed="r")
        obs = a.obs[["major", "site"]].astype(str).copy()
        um = np.asarray(a.obsm["X_umap"]) if "X_umap" in a.obsm else None
        if um is not None:
            idx = np.random.default_rng(0).permutation(len(obs))
            cats = obs["major"].value_counts().index.tolist()
            for c in cats:
                ccol(c)
            axA.scatter(um[idx, 0], um[idx, 1], c=[ccol(c) for c in obs["major"].values[idx]], s=0.15, lw=0, rasterized=True)
            n_c = obs["major"].value_counts()
            legend_right(axA, patches([f"{pretty(c)} ({n_c[c]:,})" for c in cats], [ccol(c) for c in cats]),
                         ncol=1 if len(cats) <= 14 else 2)
            title(axA, "a", f"{len(obs):,} cells - major lineages")
            axB.scatter(um[idx, 0], um[idx, 1], c=[SITE_C.get(c, "#999999") for c in obs["site"].values[idx]], s=0.15, lw=0, rasterized=True)
            sites = [k for k in SITE_C if k in set(obs["site"])] + sorted(set(obs["site"]) - set(SITE_C))
            n_s = obs["site"].value_counts()
            legend_right(axB, patches([f"{k} ({n_s[k]:,})" for k in sites], [SITE_C.get(k, "#999999") for k in sites]))
            title(axB, "b", "sample site")
            for ax in (axA, axB):
                ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
                ax.set_aspect("equal", adjustable="datalim")
        else:
            missing(axA, "X_umap in atlas", "a"); missing(axB, "X_umap in atlas", "b")
    else:
        missing(axA, "TGCT_atlas_stage1.h5ad", "a"); missing(axB, "TGCT_atlas_stage1.h5ad", "b")

    d = read(S1 / "tables" / "S7_major_fractions_per_sample.csv", index_col=0)
    if d is not None:
        meta = d[["site", "histology"]]
        f = d.drop(columns=["dataset", "patient", "site", "histology"], errors="ignore")
        f = f.loc[meta.sort_values(["site", "histology"]).index]
        bottom = np.zeros(len(f))
        for c in f.columns:
            axC.bar(range(len(f)), f[c], bottom=bottom, color=ccol(c), width=0.85, lw=0)
            bottom += f[c].values
        axC.set_xticks(range(len(f)))
        axC.set_xticklabels([s.split("_", 1)[-1] for s in f.index], rotation=90, fontsize=FS)
        axC.set_xlim(-0.6, len(f) - 0.4); axC.set_ylim(0, 1.0)
        axC.set_ylabel("fraction of cells")
        # site band under the bars (colour = site, same as panel b)
        site_sorted = meta.loc[f.index, "site"].values
        for i, s_ in enumerate(site_sorted):
            axC.add_patch(plt.Rectangle((i - 0.5, -0.045), 1, 0.03, color=SITE_C.get(s_, "#999999"), clip_on=False, lw=0))
        legend_right(axC, patches([pretty(c) for c in f.columns], [ccol(c) for c in f.columns]))
        title(axC, "c", "composition per sample (bar under axis = site)")
    else:
        missing(axC, "S7_major_fractions_per_sample.csv", "c")

    m = read(S1 / "tables" / "S5_malignant_per_sample.csv", index_col=0)
    if m is not None:
        m = m[m.sum(1) > 0]
        mm = m.get("malignant", pd.Series(0, index=m.index))
        mm = mm[~mm.index.str.contains("GSE112013|GSE182786")].sort_values()
        axD.barh(range(len(mm)), mm.values, color="#D55E00", height=0.7)
        axD.set_yticks(range(len(mm))); axD.set_yticklabels([s.split("_", 1)[-1] for s in mm.index], fontsize=FS)
        axD.set_ylim(-0.6, len(mm) - 0.4)
        axD.set_xlabel("malignant cells\n(lineage + CNV / 12p)")
        title(axD, "d", "malignant cells")
    else:
        missing(axD, "S5_malignant_per_sample.csv", "d")

    lm = read(S1 / "tables" / "S7_malignant_lineage_markers.csv", index_col=0)
    if lm is not None:
        lm = lm[lm["n"] >= 10].drop(columns="n")
        im = axE.imshow(lm.values.astype(float), aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axE.set_xticks(range(lm.shape[1])); axE.set_xticklabels(lm.columns, rotation=90, fontsize=FS)
        axE.set_yticks(range(len(lm))); axE.set_yticklabels([s.split("_", 1)[-1] for s in lm.index], fontsize=FS)
        cb = fig.colorbar(im, ax=axE, shrink=0.6, aspect=15, pad=0.02)
        cb.set_label("fraction\nexpressing", fontsize=FS); cb.ax.tick_params(labelsize=FS)
        title(axE, "e", "SE vs EC markers\n(malignant cells)")
    else:
        missing(axE, "S7_malignant_lineage_markers.csv", "e")
    save(fig, "Fig1_atlas")


# ============================================================ Fig 2
def fig2():
    fig = new_fig(120)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
    axA, axB, axC, axD = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
    pv = S2 / "S4_benchmark_pred_vs_true.csv"
    if pv.exists():
        d = pd.read_csv(pv, header=[0, 1], index_col=0)
        types = list(d["pred"].columns)
        for t in types:
            axA.scatter(d["true"][t], d["pred"][t], s=6, color=ccol(t), lw=0, label=pretty(t))
        mx = max(d["true"].values.max(), d["pred"].values.max()) * 1.05
        axA.plot([0, mx], [0, mx], "--", c="grey", lw=0.5); axA.set_xlim(0, mx); axA.set_ylim(0, mx)
        axA.set_aspect("equal", adjustable="box")
        axA.set_xlabel("true fraction (scRNA)"); axA.set_ylabel("estimated fraction")
        legend_right(axA, markerscale=1.3)
        title(axA, "a", "leave-one-dataset-out benchmark")
    else:
        missing(axA, "S4_benchmark_pred_vs_true.csv", "a")
    b = read(S2 / "S4_benchmark_per_type.csv")
    if b is not None:
        b = b[~b["cell_type"].isin(["Germ_normal", "Leydig", "Sertoli"])]
        piv = b.pivot(index="cell_type", columns="method", values="ccc")
        order = [c for c in ["ALL (pooled)"] if c in piv.index] + [c for c in piv.index if c != "ALL (pooled)"]
        piv = piv.reindex(order)
        x = np.arange(len(piv)); meths = [m_ for m_ in ["final", "pass1", "nnls"] if m_ in piv]
        w = 0.8 / max(len(meths), 1)
        lab = {"final": "final (2-pass EM)", "pass1": "pass 1 only", "nnls": "NNLS"}
        for j, mth in enumerate(meths):
            axB.bar(x + (j - (len(meths) - 1) / 2) * w, piv[mth], w, color=["#0072B2", "#56B4E9", "#999999"][j], label=lab[mth], lw=0)
        axB.set_xticks(x); axB.set_xticklabels([pretty(c) for c in piv.index], rotation=45, ha="right", rotation_mode="anchor")
        axB.set_ylabel("Lin's CCC"); axB.axhline(0, c="k", lw=0.4)
        legend_right(axB, title_="method")
        title(axB, "b", "accuracy per cell type")
    else:
        missing(axB, "S4_benchmark_per_type.csv", "b")
    ty = read(S2 / "tcga_type_fractions.csv", index_col=0)
    sel = read(S2 / "S5c_tumour_SE_likeness.csv", index_col=0)
    if ty is not None and sel is not None:
        h = sel["histology_used"].reindex(ty.index)
        g = ty.groupby(h).mean().reindex([k for k in ("SE", "NSGCT") if k in set(h)])
        bottom = np.zeros(len(g))
        for c in g.columns:
            axC.bar(range(len(g)), g[c], bottom=bottom, color=ccol(c), width=0.6, lw=0, label=pretty(c))
            bottom += g[c].values
        axC.set_xticks(range(len(g))); axC.set_xticklabels([f"{k}\n(n={int((h == k).sum())})" for k in g.index])
        axC.set_xlim(-0.6, len(g) - 0.4)
        axC.set_ylabel("mean cell fraction")
        hs, ls = axC.get_legend_handles_labels()
        legend_right(axC, hs[::-1], ls[::-1])
        title(axC, "c", "TCGA-TGCT composition")
        known = sel["histology_pathology"].isin(["SE", "NSGCT"])
        dd = sel[known]
        data = [dd.loc[dd["histology_pathology"] == k, "SE_likeness"].values for k in ("SE", "NSGCT")]
        jitter_box(axD, data, [HIST_C["SE"], HIST_C["NSGCT"]],
                   [f"SE\n(n={len(data[0])})", f"NSGCT\n(n={len(data[1])})"])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score((dd["histology_pathology"] == "SE").astype(int), dd["SE_likeness"])
        axD.set_ylabel("tumour SE-likeness\n(deconvolved)"); axD.set_xlabel("pathology")
        title(axD, "d", f"read-out vs pathology (AUC = {auc:.2f})")
    else:
        missing(axC, "tcga_type_fractions / S5c_tumour_SE_likeness", "c"); missing(axD, "S5c_tumour_SE_likeness.csv", "d")
    save(fig, "Fig2_deconvolution")


# ============================================================ Fig 3
def cramers_v(t):
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(t)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))


def fig3():
    fig = new_fig(118)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35])       # 8-pt text: three panels per row
    axA = fig.add_subplot(gs[0, 0]); axC = fig.add_subplot(gs[0, 1]); axD = fig.add_subplot(gs[0, 2])
    axE = fig.add_subplot(gs[1, 0:2]); axB = fig.add_subplot(gs[1, 2])
    t0 = read(S2 / "S5_ecotype_x_histology.csv", index_col=0)
    t1 = read(S2 / "S5c_ecotype_x_histology.csv", index_col=0)
    if t0 is not None and t1 is not None:
        v = [cramers_v(t0[[c for c in t0.columns if c in ("SE", "NSGCT")]]),
             cramers_v(t1[[c for c in t1.columns if c in ("SE", "NSGCT")]])]
        axA.bar([0, 1], v, color=["#999999", "#0072B2"], width=0.6)
        for i, x in enumerate(v):
            axA.text(i, x + 0.03, f"{x:.2f}", ha="center", va="bottom", fontsize=FS)
        axA.set_xticks([0, 1]); axA.set_xticklabels(["raw", "adjusted"]); axA.set_ylim(0, 1.12)
        axA.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axA.set_ylabel("Cramer's V\n(ecotype vs histology)")
        title(axA, "a", "confounding")
    else:
        missing(axA, "S5(c)_ecotype_x_histology.csv", "a")
    L = read(S2 / "S5c_programme_state_loadings.csv", index_col=0)
    if L is not None:
        top = list(dict.fromkeys(L.apply(lambda c: c.sort_values(ascending=False).head(8).index).values.T.ravel()))
        H = L.loc[top]; H = H / H.max()
        im = axB.imshow(H.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axB.set_xticks(range(H.shape[1])); axB.set_xticklabels([pretty(c) for c in H.columns])
        axB.set_yticks(range(len(top))); axB.set_yticklabels([pretty(s) for s in top], fontsize=FS)
        axB.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        cb = fig.colorbar(im, ax=axB, location="bottom", shrink=0.7, aspect=20, pad=0.01)
        cb.set_label("loading (scaled to programme max)", fontsize=FS); cb.ax.tick_params(labelsize=FS)
        title(axB, "b", "programme composition\n(top 8 states each)")
    else:
        missing(axB, "S5c_programme_state_loadings.csv", "b")
    e = read(S2 / "S5c_adjusted_ecotypes.csv", index_col=0)
    if e is not None:
        sc_cols = [c for c in e.columns if c.startswith("score_")][:2]
        for ax, c, l in zip((axC, axD), sc_cols, "cd"):
            hs = [h for h in ("SE", "NSGCT") if (e["histology_used"] == h).any()]
            data = [e.loc[e["histology_used"] == h, c].values for h in hs]
            jitter_box(ax, data, [HIST_C[h] for h in hs], [f"{h}\n(n={len(x)})" for h, x in zip(hs, data)], s=2)
            ax.set_ylabel(f"{c.replace('score_', '')} score")
            title(ax, l, f"{c.replace('score_', '')}\nby histology")
        for ax, l in zip((axC, axD)[len(sc_cols):], "cd"[len(sc_cols):]):
            missing(ax, "score_* column", l)
    else:
        missing(axC, "S5c_adjusted_ecotypes.csv", "c"); missing(axD, "S5c_adjusted_ecotypes.csv", "d")
    k = read(S2 / "S5c_k_selection.csv")
    if k is not None:
        axE.plot(k["k"], k["cophenetic"], "o-", c="#0072B2", ms=3, lw=0.8, label="cophenetic")
        axE.plot(k["k"], k["silhouette"], "s-", c="#E69F00", ms=3, lw=0.8, label="silhouette")
        axE.set_xlabel("number of programmes (k)"); axE.set_ylabel("stability"); axE.set_ylim(0, 1.05)
        axE.set_xticks(k["k"])
        legend_right(axE)
        title(axE, "e", "consensus NMF model selection")
    else:
        missing(axE, "S5c_k_selection.csv", "e")
    save(fig, "Fig3_TME_programmes")


# ============================================================ Fig 4
def fig4():
    S = read(S4 / "A2_spot_programme_scores.csv", index_col=0)
    A3 = read(S4 / "A3_spatial_autocorrelation.csv")
    cols = ["Tumour_fraction", "P1_TME_immune", "P2_TME_stromal", "TLS12_chemokine"]
    have_xy = S is not None and {"x", "y"} <= set(S.columns) and all(c in S for c in cols)
    libs = list(S["library_id"].unique())[:3] if have_xy else []
    nl = max(len(libs), 1) if have_xy else 3
    fig = new_fig(40 * nl + 78)
    # ONE flat grid (no nesting): map rows | colour-bar row | bottom row (b = cols 0-1, c = cols 2-3)
    # two independent sub-figures -> legends / tick labels of b,c can never distort the map grid
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[40 * nl + 14, 64])
    gs = sf_top.add_gridspec(nl, 4)
    map_axes = []
    if have_xy:
        from scipy.spatial import cKDTree
        from matplotlib.colors import LinearSegmentedColormap
        # one vivid colour ramp per map: grey tissue (low) -> saturated hue -> dark (high)
        ramp = {"Tumour_fraction": ["#D9D9D9", "#F4A582", "#D6604D", "#B2182B", "#67001F"],
                "P1_TME_immune": ["#D9D9D9", "#92C5DE", "#4393C3", "#2166AC", "#053061"],
                "P2_TME_stromal": ["#D9D9D9", "#A6DBA0", "#5AAE61", "#1B7837", "#00441B"],
                "TLS12_chemokine": ["#D9D9D9", "#C2A5CF", "#9970AB", "#762A83", "#40004B"]}
        cmaps = {c: LinearSegmentedColormap.from_list(c, ramp[c]) for c in cols}
        col_axes = {c: [] for c in cols}
        spot_sets = []
        for i, lib in enumerate(libs):
            s = S[S["library_id"] == lib].dropna(subset=["x", "y"])
            xy = s[["x", "y"]].to_numpy(float)
            nn = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]) if len(xy) > 2 else 1.0   # spot spacing
            for j, c in enumerate(cols):
                ax = sf_top.add_subplot(gs[i, j]); map_axes.append(ax); col_axes[c].append(ax)
                v = s[c].astype(float).to_numpy()
                lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
                vv = np.clip((v - lo) / (hi - lo if hi > lo else 1), 0, 1)
                o = np.argsort(np.nan_to_num(vv, nan=-1))                   # high spots drawn on top
                sc = ax.scatter(xy[o, 0], -xy[o, 1], c=vv[o], s=4, cmap=cmaps[c], vmin=0, vmax=1, lw=0,
                                marker="o", rasterized=True)
                spot_sets.append((ax, sc, nn))
                pad = nn * 1.5
                ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
                ax.set_ylim(-xy[:, 1].max() - pad, -xy[:, 1].min() + pad)
                ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if i == 0:
                    title(ax, "a" if j == 0 else "", pretty(c))
                if j == 0:
                    ax.set_ylabel(str(lib), fontsize=FS)
        for j, c in enumerate(cols):                                     # one colour bar per programme
            sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmaps[c])
            cb = sf_top.colorbar(sm, ax=col_axes[c], location="bottom", shrink=0.75, aspect=14, pad=0.02, ticks=[0, 1])
            cb.ax.set_xticklabels(["low", "high"], fontsize=FS); cb.outline.set_linewidth(0.3)
            if j == 0:
                cb.set_label("scaled within slide (2nd-98th pct.)", fontsize=FS)

        def _spot_size(fig_, r_):
            """after layout: spot diameter = spot spacing on screen -> spots tile the tissue (no white gaps)"""
            for ax_, sc_, nn_ in spot_sets:
                d_px = abs(ax_.transData.transform((nn_, 0))[0] - ax_.transData.transform((0, 0))[0])
                d_pt = d_px * 72 / fig_.dpi * 1.12
                sc_.set_sizes([max(d_pt, 0.3) ** 2])
        fig._post_layout.append(_spot_size)
    else:
        ax = sf_top.add_subplot(gs[0:nl, :])
        img = S4 / "figures" / "A5_spatial_programme_maps.png"
        if img.exists():
            ax.imshow(plt.imread(img)); ax.set_axis_off(); title(ax, "a", "programme maps (stage-4 render)")
            log("  Fig4a: spot coordinates not in A2 -> using stage-4 PNG (re-run stage 4 block A for vector panels)")
        else:
            missing(ax, "A2 coordinates / A5 png", "a")
    gb = sf_bot.add_gridspec(1, 2, width_ratios=[1, 1.1])
    axb, axc = sf_bot.add_subplot(gb[0]), sf_bot.add_subplot(gb[1])
    if A3 is not None:
        isrho = A3["score"].astype(str).str.startswith("rho(")
        mi = A3[~isrho].pivot(index="score", columns="slide", values="morans_I")
        slides = list(mi.columns); scol = {s_: OKABE[i % len(OKABE)] for i, s_ in enumerate(slides)}
        x = np.arange(len(mi)); w = 0.8 / max(len(slides), 1)
        for j, s_ in enumerate(slides):
            axb.bar(x + (j - (len(slides) - 1) / 2) * w, mi[s_].values, w, color=scol[s_], lw=0, label=str(s_))
        axb.axhline(0, c="k", lw=0.4)
        axb.set_xticks(x); axb.set_xticklabels([pretty(i_) for i_ in mi.index], rotation=35, ha="right", rotation_mode="anchor")
        pmax = A3.loc[~isrho, "p_perm"].max() if "p_perm" in A3 else np.nan
        axb.set_ylabel("Moran's I")
        legend_right(axb, title_="slide")
        title(axb, "b", "spatial autocorrelation" + (f"\n(all permutation p <= {pmax:.3g})" if np.isfinite(pmax) else ""))
        rh = A3[isrho].pivot(index="score", columns="slide", values="morans_I")
        names = rh.index.str.replace("rho(", "", regex=False).str.replace(")", "", regex=False)
        rh.index = [wrap(" vs ".join(pretty(p_.strip()) for p_ in n.split(",")), 26) for n in names]
        y = np.arange(len(rh)); hgt = 0.8 / max(rh.shape[1], 1)
        for j, s_ in enumerate(rh.columns):
            axc.barh(y + (j - (rh.shape[1] - 1) / 2) * hgt, rh[s_].values, hgt, color=scol.get(s_, OKABE[j]), lw=0)
        axc.set_yticks(y); axc.set_yticklabels(rh.index, fontsize=FS); axc.invert_yaxis()
        axc.axvline(0, c="k", lw=0.4); axc.set_xlabel("Spearman rho across spots")
        title(axc, "c", "co-localisation (colours = slide, as in b)")
    else:
        missing(axb, "A3_spatial_autocorrelation.csv", "b"); missing(axc, "A3_spatial_autocorrelation.csv", "c")
    save(fig, "Fig4_spatial")


# ============================================================ Fig 5
def fig5():
    fig = new_fig(105)                                          # 8-pt text: A full height left, B and C stacked right
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    C = read(S4 / "C1_genomics_vs_programmes.csv")
    if C is not None:
        bcol = "beta_SD_hist_purity" if "beta_SD_hist_purity" in C else "beta_SD"
        pcol = "p_hist_purity" if "p_hist_purity" in C else "p"
        C = C[C["feature"].isin(["chrX_gain", "chr12p_gain", "KIT_mut", "KRAS_mut"])]
        C = C[C["programme"].isin(["P1_TME_immune", "P2_TME_stromal", "P4_IFNG_programme"])].copy()
        # q = stage-4 FDR over the FULL test family (all programmes x features), not re-computed on the
        # plotted subset (re-computing on a subset would understate multiplicity)
        qcol = "q_hist_purity" if "q_hist_purity" in C else ("q_BH" if "q_BH" in C else None)
        C["q"] = C[qcol] if qcol else bh(C[pcol])
        C = C.sort_values(["feature", "programme"])
        y = np.arange(len(C))[::-1]
        from scipy.stats import norm
        z = pd.Series(np.abs(norm.ppf(C[pcol].values / 2)), index=C.index).replace(0, np.nan)
        se = np.abs(C[bcol]) / z
        cols = [OKABE[3] if q < 0.05 else "#999999" for q in C["q"]]
        axes[0].errorbar(C[bcol], y, xerr=1.96 * se, fmt="none", ecolor=cols, lw=0.8)
        axes[0].scatter(C[bcol], y, c=cols, s=10, zorder=3)
        axes[0].axvline(0, c="k", lw=0.4); axes[0].set_yticks(y)
        axes[0].set_yticklabels([f"{pretty(f)} - {pretty(p)}" for f, p in zip(C["feature"], C["programme"])], fontsize=FS)
        axes[0].set_xlabel("beta per SD, adjusted for\nhistology + purity (95% CI)")
        axes[0].legend(handles=[Line2D([], [], marker="o", ls="", c=OKABE[3], ms=3, label="q < 0.05"),
                                Line2D([], [], marker="o", ls="", c="#999999", ms=3, label="q >= 0.05")],
                       loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2)
        title(axes[0], "a", "chromosomal copy gain\nvs programmes")
    else:
        missing(axes[0], "C1_genomics_vs_programmes.csv", "a")
    M = read(S4 / "B1_MIF_axis.csv")
    m = None
    if M is not None:
        m = M[(M["context"] == "primary_SE") & (M["receptor"] == "CD74")].sort_values("score").tail(12)
    if m is not None and len(m):
        axes[1].barh(range(len(m)), m["score"], color="#0072B2", height=0.7)
        axes[1].set_yticks(range(len(m))); axes[1].set_yticklabels([pretty(r_) for r_ in m["receiver"]], fontsize=FS)
        axes[1].set_ylim(-0.6, len(m) - 0.4)
        axes[1].set_xlabel("MIF (tumour) x CD74 (receiver)")
        title(axes[1], "b", "tumour MIF -> CD74\n(primary seminoma)")
    else:
        missing(axes[1], "B1_MIF_axis.csv (MIF-CD74, primary_SE)", "b")
    B4 = read(S4 / "B4_LN1_SE_vs_EC_ligands.csv")
    if B4 is not None and "log2FC_depthmatched" in B4:
        b = B4[B4["q_depthmatched"] < 0.05].sort_values("log2FC_depthmatched")
        if len(b):
            axes[2].barh(range(len(b)), b["log2FC_depthmatched"], height=0.7,
                         color=[OKABE[3] if v > 0 else OKABE[0] for v in b["log2FC_depthmatched"]])
            axes[2].set_yticks(range(len(b))); axes[2].set_yticklabels(b["ligand"], fontsize=FS)
            axes[2].set_ylim(-0.6, len(b) - 0.4)
            axes[2].axvline(0, c="k", lw=0.4)
            legend_right(axes[2], handles=patches(["up in EC-like", "up in SE-like"], [OKABE[3], OKABE[0]]))
        else:
            axes[2].text(0.5, 0.5, "no ligand with q < 0.05\nafter depth matching", ha="center", va="center",
                         transform=axes[2].transAxes, fontsize=FS)
            axes[2].set_xticks([]); axes[2].set_yticks([])
            for sp in axes[2].spines.values():
                sp.set_visible(False)
        axes[2].set_xlabel("log2FC EC-like vs SE-like\n(depth-matched)")
        title(axes[2], "c", "same lymph node (LN1)\nexploratory")
    else:
        missing(axes[2], "B4_LN1_SE_vs_EC_ligands.csv", "c")
    from matplotlib.ticker import MaxNLocator
    for ax_ in axes:                                           # thin crowded numeric ticks; keep removed ticks removed
        if len(ax_.get_xticks()):
            ax_.xaxis.set_major_locator(MaxNLocator(nbins=4))
    save(fig, "Fig5_dosage_communication")


# ============================================================ Fig 6
def fig6():
    fig = new_fig(118)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    axA, axB, axC, axD = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
    P = read(S3 / "S3_primary_stageI_PFI.csv")
    if P is not None:
        y = np.arange(len(P))[::-1]
        axA.errorbar(P["HR_adj"], y, xerr=[P["HR_adj"] - P["CI_low"], P["CI_high"] - P["HR_adj"]], fmt="o", c="k", ms=3, lw=0.8)
        axA.axvline(1, ls="--", c="grey", lw=0.5); axA.set_xscale("log")
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        lo, hi = P["CI_low"].min() * 0.9, P["CI_high"].max() * 1.1
        tk = [t for t in [0.125, 0.25, 0.5, 1, 2, 4, 8] if lo <= t <= hi] or [1]
        axA.xaxis.set_major_locator(FixedLocator(tk)); axA.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tk]))
        axA.xaxis.set_minor_locator(NullLocator())
        axA.set_yticks(y); axA.set_ylim(-0.6, len(P) - 0.4)
        axA.set_yticklabels([f"{pretty(p_)}  (q = {q:.2f})" for p_, q in zip(P["predictor"], P["q_LRT_BH"])], fontsize=FS)
        axA.set_xlabel("HR per SD (adjusted for histology + LVI)")
        title(axA, "a", f"TCGA stage I PFI (n = {int(P['n'].max())}, {int(P['events'].max())} events)")
        axB.bar(range(len(P)), P["dC_optimism_corr"], color="#0072B2", width=0.6)
        axB.axhline(0, c="k", lw=0.4); axB.set_xticks(range(len(P)))
        axB.set_xticklabels([pretty(p_) for p_ in P["predictor"]], rotation=35, ha="right", rotation_mode="anchor")
        axB.set_ylabel("delta C-index\n(optimism-corrected)")
        title(axB, "b", "added value over histology + LVI")
    else:
        missing(axA, "S3_primary_stageI_PFI.csv", "a"); missing(axB, "S3_primary_stageI_PFI.csv", "b")
    T = read(S3 / "S3_tcga_analysis_table.csv", index_col=0)
    if T is not None:
        try:
            from lifelines import KaplanMeierFitter
            s = T[(T["stage_I"] == 1) & T["PFI.time"].notna() & (T["PFI.time"] > 0)]
            hi_ = s["P1_TME_immune"] > s.groupby("histology")["P1_TME_immune"].transform("median")
            for lab, msk, c in (("P1 high", hi_, OKABE[0]), ("P1 low", ~hi_, OKABE[1])):
                KaplanMeierFitter().fit(s.loc[msk, "PFI.time"] / 365, s.loc[msk, "PFI"],
                                        label=f"{lab} (n = {int(msk.sum())})").plot_survival_function(
                    ax=axC, ci_show=False, color=c, lw=0.9, show_censors=True, censor_styles={"ms": 3, "mew": 0.5})
            axC.set_ylim(0, 1.03); axC.set_xlabel("years"); axC.set_ylabel("progression-free interval")
            axC.get_legend().remove() if axC.get_legend() else None
            legend_right(axC)
            title(axC, "c", "P1 immune programme\n(split at within-histology median)")
        except ImportError:
            missing(axC, "lifelines (pip install lifelines)", "c")
    else:
        missing(axC, "S3_tcga_analysis_table.csv", "c")
    V = read(S3 / "S3_validation_GSE99420.csv")
    if V is not None:
        y = np.arange(len(V))[::-1]
        axD.errorbar(V["AUC"], y, xerr=[V["AUC"] - V["AUC_low"], V["AUC_high"] - V["AUC"]], fmt="o", c="k", ms=3, lw=0.8)
        axD.axvline(0.5, ls="--", c="grey", lw=0.5); axD.set_xlim(0.1, 0.9); axD.set_xticks([0.2, 0.5, 0.8])
        axD.set_yticks(y); axD.set_yticklabels([pretty(p_) for p_ in V["predictor"]], fontsize=FS)
        axD.set_ylim(-0.6, len(V) - 0.4)
        axD.set_xlabel("AUC for relapse (95% bootstrap CI)")
        G = read(S3 / "S3_gse99420_scores.csv", index_col=0)
        sub = ""
        if G is not None and "relapse" in G:
            sub = f" ({int((G['relapse'] == 1).sum())} relapse / {int((G['relapse'] == 0).sum())} no relapse)"
        title(axD, "d", "external validation GSE99420" + ("\n" + sub.strip() if sub else ""))
    else:
        missing(axD, "S3_validation_GSE99420.csv", "d")
    save(fig, "Fig6_clinical")


# ============================================================ Fig 7
def fig7():
    fig = new_fig(128)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1.35])
    axA = fig.add_subplot(gs[0, 0]); axC = fig.add_subplot(gs[0, 1]); axB = fig.add_subplot(gs[1, :])
    pc = read(S5 / "T_positive_controls_expression.csv", index_col=0)
    if pc is not None:
        cols = [c for c in ("mean_tumour", "mean_normal_testis", "mean_immune") if c in pc]
        labs = {"mean_tumour": "tumour", "mean_normal_testis": "normal testis", "mean_immune": "immune"}
        colr = {"mean_tumour": "#D55E00", "mean_normal_testis": "#009E73", "mean_immune": "#999999"}
        x = np.arange(len(pc)); w = 0.8 / max(len(cols), 1)
        for j, c in enumerate(cols):
            axA.bar(x + (j - (len(cols) - 1) / 2) * w, pc[c].values, w, color=colr[c], lw=0, label=labs[c])
        axA.set_xticks(x); axA.set_xticklabels(pc.index, rotation=45, ha="right", rotation_mode="anchor")
        axA.set_ylabel("mean log-normalised\nexpression")
        legend_right(axA)
        title(axA, "a", "positive controls (scRNA)")
    else:
        missing(axA, "T_positive_controls_expression.csv", "a")
    R = read(S5 / "R_L1000CDS2_reversal_annotated.csv")
    if R is not None and len(R):
        r = R[(R["query"] == "EC_like_vs_SE_like") & (~R["pan_reverser"].astype(bool))].drop_duplicates("perturbagen").head(10)
        if len(r):
            yy = np.arange(len(r))[::-1]
            axC.barh(yy, r["score"], color="#0072B2", height=0.7)
            axC.set_yticks(yy); axC.set_yticklabels([f"{p} (rank {k})" for p, k in zip(r["perturbagen"], r["rank"])], fontsize=FS)
            axC.set_ylim(-0.6, len(r) - 0.4)
            axC.set_xlabel("L1000CDS2 reversal score")
            title(axC, "b", "reversers of the EC-like programme\n(pan-reversers removed)")
        else:
            missing(axC, "EC_like_vs_SE_like non-pan reversers", "b")
    else:
        missing(axC, "R_L1000CDS2_reversal_annotated.csv", "b")
    M = read(S5 / "M_mechanism_targets.csv")
    Ps = read(S5 / "P_drug_priority_STRICT.csv")
    if M is not None:
        bad_t = set()
        if Ps is not None:
            for _, row in Ps[Ps["drug"].isin(ARTEFACT_DRUGS)].iterrows():
                bad_t |= set(str(row["targets"]).split(";"))
        m = M[M["nominated_by"].str.contains("T1", na=False) & M["bulk_confirmed"].astype(bool)].copy()
        m = m[~(m["target"].isin(bad_t) & (m["n_approved"] <= 2))]
        m = m.dropna(subset=["pct_tumour", "pct_normal_testis"]).sort_values("logFC_tumour", ascending=False).head(25)
        if len(m):
            nag = m["n_approved"].fillna(0) + m["n_clinical_or_tool"].fillna(0)
            sz = 8 + 14 * np.sqrt(nag)
            sca = axB.scatter(m["pct_tumour"], m["pct_normal_testis"], s=sz, c=m["logFC_tumour"], cmap="viridis",
                              edgecolor="k", lw=0.3, zorder=3)
            xr = (m["pct_tumour"].min(), m["pct_tumour"].max()); yr = (m["pct_normal_testis"].min(), m["pct_normal_testis"].max())
            px = 0.08 * (xr[1] - xr[0] or 1); py = 0.12 * (yr[1] - yr[0] or 1)
            axB.set_xlim(xr[0] - px, xr[1] + 2.2 * px); axB.set_ylim(max(-0.02, yr[0] - py), yr[1] + 2 * py)
            label_points(fig, axB, m["pct_tumour"], m["pct_normal_testis"], m["target"], sizes=sz)
            axB.set_xlabel("fraction of tumour cells expressing")
            axB.set_ylabel("fraction of normal-testis\ncells expressing")
            cb = fig.colorbar(sca, ax=axB, shrink=0.8, aspect=18, pad=0.01)
            cb.set_label("log2FC tumour vs rest", fontsize=FS); cb.ax.tick_params(labelsize=FS)
            ref = [v for v in (1, 5, 20) if v <= max(nag.max(), 1)] or [1]
            axB.legend(handles=[plt.scatter([], [], s=8 + 14 * np.sqrt(v), c="w", edgecolor="k", lw=0.3) for v in ref],
                       labels=[str(v) for v in ref], title="n agents", loc="upper left", bbox_to_anchor=(1.12, 1.0))
            title(axB, "c", "druggable tumour-restricted targets (bulk-confirmed)")
        else:
            missing(axB, "T1 bulk-confirmed targets", "c")
    else:
        missing(axB, "M_mechanism_targets.csv", "c")
    save(fig, "Fig7_drugs")


# ============================================================ supplementary tables
TABLE_LEGENDS = {
    'S1_QC': ('Supplementary Table S1. Quality control of the single-cell samples.',
        'Per-sample quality-control summary of the 15 single-cell samples, including the filtering thresholds applied and the numbers of cells before and after filtering.'),
    'S2_malignant': ('Supplementary Table S2. Malignant-cell calls per sample.',
        'Number of cells per sample called malignant (tumour lineage with copy-number or chromosome 12p evidence), tumour-lineage cells without copy-number evidence (CNV-negative) and non-malignant cells.'),
    'S3_state_fractions_sc': ('Supplementary Table S3. Cell-state fractions in the single-cell samples.',
        'Fraction of each fine cell state per single-cell sample; these fractions served as the reference values in the leave-one-dataset-out benchmark.'),
    'S4_reference': ('Supplementary Table S4. Composition of the deconvolution reference.',
        'Number of reference cells per cell state and cell type (8,076 cells, 32 states, 10 cell types).'),
    'S5_benchmark': ('Supplementary Table S5. Leave-one-dataset-out deconvolution benchmark.',
        "Lin's concordance correlation coefficient between estimated and single-cell fractions per cell type for the two-pass model, pass 1 alone and non-negative least squares."),
    'S6_k_selection': ('Supplementary Table S6. Selection of the number of microenvironment programmes.',
        'Consensus non-negative matrix factorisation of the histology-adjusted microenvironment for k = 2 to 6: cophenetic coefficient, silhouette width and size of the smallest cluster (min_cluster).'),
    'S7_programme_loadings': ('Supplementary Table S7. State loadings of the microenvironment programmes.',
        'Loading of each immune and stromal cell state on the two programmes of the histology-adjusted factorisation (TME1, immune programme P1; TME2, vascular-stromal programme P2).'),
    'S8_signature_agreement': ('Supplementary Table S8. Agreement between state signatures and deconvolved fractions.',
        'Spearman correlation across TCGA tumours between rank-based state signature scores and deconvolved state fractions.'),
    'S9_clinical_primary': ('Supplementary Table S9. Pre-specified primary relapse analysis.',
        'TCGA stage I tumours, progression-free interval: univariable and adjusted hazard ratios per SD, likelihood-ratio tests against histology plus LVI, Benjamini-Hochberg q values, optimism-corrected C-indices and predictor-by-histology interaction tests.'),
    'S10_clinical_sensitivity': ('Supplementary Table S10. Secondary and sensitivity analyses.',
        'Hazard ratios per SD for disease-free interval, all stages with adjustment for stage, adjustment for adjuvant radiotherapy, pathological T2 or higher instead of LVI, and histology-specific models.'),
    'S11_validation_GSE99420': ('Supplementary Table S11. External validation in GSE99420.',
        'Area under the ROC curve with bootstrap 95% confidence intervals, histology-adjusted odds ratios per SD and within-histology Wilcoxon tests for each locked signature, with Benjamini-Hochberg correction across the five predictors.'),
    'S12_signature_sanity': ('Supplementary Table S12. Agreement of the locked signatures with their TCGA predictors.',
        'Spearman correlation in TCGA between each locked gene-signature score and the predictor it represents.'),
    'S13_spatial_moran': ('Supplementary Table S13. Spatial autocorrelation in the Visium sections.',
        "Moran's I and permutation P values (199 permutations) for each score in each of the three seminoma sections; rows labelled rho(…) give the Spearman correlation across spots between the two named scores (no permutation P value)."),
    'S14_niche_test': ('Supplementary Table S14. Programme niche test.',
        'Within-programme versus between-programme spatial correlation of state fractions per section, with P values from 999 permutations of programme labels.'),
    'S15_LR_tumour_TME': ('Supplementary Table S15. Tumour-to-microenvironment ligand-receptor scores.',
        'Scores of 25 curated ligand-receptor pairs from malignant cells to each receiver state in primary seminoma-containing tumours and lymph-node metastases, with permutation P values and Benjamini-Hochberg q values.'),
    'S16_LN1_ligands': ('Supplementary Table S16. Ligand expression of EC-like versus SE-like malignant cells in LN1.',
        'Comparison after binomial thinning to matched sequencing depth: fraction of expressing cells, mean expression, test statistics and q values for each ligand.'),
    'S17_genomics': ('Supplementary Table S17. Genomic associations with the programmes.',
        'Regression coefficients (per SD) of programme scores on 12p and chromosome X copy gain (mean positive GISTIC2 call), KIT, KRAS and NRAS mutation and number of mutated genes, adjusted for histology and tumour purity, with Benjamini-Hochberg correction across the 24 tests.'),
    'S18_targets': ('Supplementary Table S18. Tumour-intrinsic target candidates.',
        'Genes up-regulated in SE-like or EC-like malignant cells versus non-tumour cells, with fold changes, detection rates, expression in the deconvolved TCGA tumour profile and expression in normal testis cells.'),
    'S19_druggable_targets': ('Supplementary Table S19. Druggable target annotation.',
        'Tumour-intrinsic and programme-level candidates with drug-gene interactions retrieved from DGIdb 5.0.'),
    'S20_drugs_strict': ('Supplementary Table S20. Drug priority list after the strict interaction filter.',
        'Agents retained after keeping targeting interaction types with an interaction score at or above the median and excluding antibody-drug conjugate payload links; two literature-curated CLDN6-directed agents are labelled as curated. Seven entries judged mechanistically implausible (see Methods) are retained but were not interpreted.'),
    'S21_LINCS': ('Supplementary Table S21. L1000CDS2 connectivity results.',
        'Perturbagens reversing the EC-like versus SE-like malignant programme and the tumour versus normal germ cell programme, with ranks and flags for non-specific reversers.'),
    'S22_convergence': ('Supplementary Table S22. Convergence of connectivity and target-based nominations.',
        'Perturbagens that reversed the EC-like programme and also act on data-nominated targets in DGIdb.'),
    'S23_locked_signatures': ('Supplementary Table S23. Locked gene signatures.',
        'Genes of each signature locked in TCGA before scoring GSE99420 and the Visium sections (predictor, direction and gene).'),
    'S24_decision_curve_3y': ('Supplementary Table S24. Decision-curve analysis at 3 years.',
        'Net benefit by threshold probability for treat all, treat none, histology plus LVI (M0) and M0 plus the P1 immune programme score (M1), TCGA stage I tumours (n = 98); apparent estimates without correction for optimism.'),
    'S25_LIANA_tumour_sender': ('Supplementary Table S25. LIANA+ ranking of tumour-sender interactions.',
        'Resource-wide LIANA+ rank aggregation for interactions sent by malignant cells (SE-like and EC-like) in primary seminoma-containing tumours and in the lymph-node metastases (column context), with ligand and receptor complexes, mean expression, CellPhoneDB-style permutation P values, aggregate magnitude rank and specificity rank (lower ranks indicate stronger and more specific interactions).'),
}


def suppl_tables():
    sheets = {
        "S1_QC": S1 / "tables" / "S2_qc_per_sample.csv", "S2_malignant": S1 / "tables" / "S5_malignant_per_sample.csv",
        "S3_state_fractions_sc": S1 / "tables" / "S8_true_state_fractions_per_sample.csv",
        "S4_reference": S2 / "S2_reference_composition.csv", "S5_benchmark": S2 / "S4_benchmark_per_type.csv",
        "S6_k_selection": S2 / "S5c_k_selection.csv", "S7_programme_loadings": S2 / "S5c_programme_state_loadings.csv",
        "S8_signature_agreement": S2 / "S6_signature_vs_bayesprism.csv", "S9_clinical_primary": S3 / "S3_primary_stageI_PFI.csv",
        "S10_clinical_sensitivity": S3 / "S3_secondary_sensitivity.csv", "S11_validation_GSE99420": S3 / "S3_validation_GSE99420.csv",
        "S12_signature_sanity": S3 / "S3_signature_sanity.csv", "S13_spatial_moran": S4 / "A3_spatial_autocorrelation.csv",
        "S14_niche_test": S4 / "A4_programme_niche_test.csv", "S15_LR_tumour_TME": S4 / "B1_tumour_to_TME_LR_scores.csv",
        "S16_LN1_ligands": S4 / "B4_LN1_SE_vs_EC_ligands.csv", "S17_genomics": S4 / "C1_genomics_vs_programmes.csv",
        "S18_targets": S5 / "T_targets.csv", "S19_druggable_targets": S5 / "M_mechanism_targets.csv",
        "S20_drugs_strict": S5 / "P_drug_priority_STRICT.csv", "S21_LINCS": S5 / "R_L1000CDS2_reversal_annotated.csv",
        "S22_convergence": S5 / "C_mechanism_convergence.csv",
        "S24_decision_curve_3y": S3 / "S3_decision_curve_3y.csv"}
    locked = S3 / "LOCKED_signatures.json"
    try:
        import openpyxl  # noqa
        with pd.ExcelWriter(OUT / "Supplementary_Tables.xlsx") as xw:
            pd.DataFrame(sorted([(sh, t, l) for sh, (t, l) in TABLE_LEGENDS.items()], key=lambda r: int(r[0].split("_")[0][1:])),
                         columns=["sheet", "table", "legend"]).to_excel(xw, sheet_name="Contents", index=False)
            idx = []
            for name, f in sheets.items():
                if f.exists():
                    d = pd.read_csv(f)
                    d.head(1_000_000).to_excel(xw, sheet_name=name[:31], index=False)
                    idx.append(dict(sheet=name, source=str(f.relative_to(PROJ)), rows=len(d)))
                else:
                    idx.append(dict(sheet=name, source=str(f.relative_to(PROJ)), rows="MISSING"))
            lia = [(c, S4 / f"B2_liana_{c}_tumour_sender.csv") for c in ("primary_SE", "LN_metastasis")]
            lia = [pd.read_csv(f).assign(context=c) for c, f in lia if f.exists()]
            if lia:
                L_ = pd.concat(lia, ignore_index=True)
                L_ = L_[["context"] + [c for c in L_.columns if c != "context"]].sort_values(["context", "magnitude_rank"])
                L_.to_excel(xw, sheet_name="S25_LIANA_tumour_sender", index=False)
                idx.append(dict(sheet="S25_LIANA_tumour_sender", source="stage4_mechanism/B2_liana_*_tumour_sender.csv", rows=len(L_)))
            else:
                idx.append(dict(sheet="S25_LIANA_tumour_sender", source="stage4_mechanism/B2_liana_*", rows="MISSING"))
            if locked.exists():
                sig = json.load(open(locked))
                pd.DataFrame([(k, d, g) for k, v in sig.items() for d in ("up", "down") for g in (v.get(d) or [])],
                             columns=["predictor", "direction", "gene"]).to_excel(xw, sheet_name="S23_locked_signatures", index=False)
                idx.append(dict(sheet="S23_locked_signatures", source=str(locked.relative_to(PROJ)), rows="json"))
            miss = [r["sheet"] for r in idx if r["rows"] == "MISSING"]
            if miss:
                log("  !! supplementary tables missing: " + ", ".join(miss))
        log("  saved Supplementary_Tables.xlsx")
    except ImportError:
        log("  openpyxl not installed -> pip install openpyxl (supplementary workbook skipped)")


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, suppl_tables):
        try:
            log(f"{fn.__name__}"); fn()
        except Exception as e:
            import traceback
            log(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")
            (OUT / f"ERROR_{fn.__name__}.txt").write_text(traceback.format_exc())
            plt.close("all")
    log("\nOVERLAP SUMMARY: " + ", ".join(f"{k}={v}" for k, v in OVERLAP_SUMMARY.items()))
    (OUT / "FIGURES_LOG.txt").write_text("\n".join(LOG))
    print(f"done -> {OUT}")
