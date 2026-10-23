#!/usr/bin/env python
"""
STAGE 4 / 4  -  Biology of the histology-independent TME programmes
=====================================================================
A  Spatial (GSE229823 Visium): do the bulk-derived programmes form physical niches?
   A1 per-spot cell-state fractions (pass-1 multinomial EM, same reference as stage 2)
   A2 per-spot scores of the LOCKED stage-3 signatures (P1 immune, P2 stromal, P3 EC-likeness) + 12-chemokine TLS score
   A3 Moran's I (spatial autocorrelation, permutation null) of programme scores
   A4 niche test: are states of the same programme more co-localised than states of different programmes?
      (within-slide spot correlation, state-label permutation null)
   A5 maps
B  Cell-cell communication in the atlas: primary seminoma vs EC-like nodal metastasis (LN1/LN2)
   curated ligand-receptor scores (expression product, sample-aware) incl. the MIF-CD74/CXCR4/CD44 axis;
   LIANA+ consensus if installed
   + tumour SE-like vs EC-like differential expression (exploratory: EC-like cells ~ one patient)
C  TCGA genomics vs programmes (KIT / KRAS / NRAS mutation, mutation count, 12p and chrX copy gain),
   histology-adjusted, BH-FDR

Each block runs independently; a failure in one is logged and the others continue.
Run:  conda activate scvi-env ; python stage4_mechanism_spatial.py      (optional: pip install liana)
"""
import os, re, io, json, gzip, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
from scipy.stats import spearmanr, mannwhitneyu
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("TGCT_BASE", "/home/arshia/testis project"))
PROJ = BASE / "TGCT_ecotype"
S1, S2, S3 = PROJ / "stage1_atlas", PROJ / "stage2_tcga", PROJ / "stage3_clinical"
OUT = PROJ / "stage4_mechanism"
FIG = OUT / "figures"
for d in (OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)
VISIUM_DIRS = [BASE / "GSE229823_RAW"]
VISIUM_H5AD = [BASE / "step2_hypothesis" / "h5ad" / "visium_all_slides_step2.h5ad"]   # old project object, if kept
XENA = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
XENA_FILES = {"mc3": "mc3_gene_level%2FTGCT_mc3_gene_level.txt.gz",
              "gistic": "TCGA.TGCT.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"}
SEED = 0
N_PERM = 199 if not os.environ.get("TGCT_TEST") else 19
TLS12 = ["CCL2", "CCL3", "CCL4", "CCL5", "CCL8", "CCL18", "CCL19", "CCL21", "CXCL9", "CXCL10", "CXCL11", "CXCL13"]
LR = [("MIF", "CD74"), ("MIF", "CXCR4"), ("MIF", "CD44"), ("MIF", "ACKR3"), ("CD274", "PDCD1"), ("PDCD1LG2", "PDCD1"),
      ("LGALS9", "HAVCR2"), ("CD80", "CTLA4"), ("CD86", "CTLA4"), ("CXCL13", "CXCR5"), ("CCL19", "CCR7"),
      ("CCL21", "CCR7"), ("CXCL12", "CXCR4"), ("SPP1", "CD44"), ("CSF1", "CSF1R"), ("IL10", "IL10RA"),
      ("TGFB1", "TGFBR2"), ("HLA-E", "KLRC1"), ("KITLG", "KIT"), ("CD47", "SIRPA"), ("VEGFA", "KDR"),
      ("APP", "CD74"), ("COPA", "CD74"), ("IFNG", "IFNGR1"), ("TNF", "TNFRSF1B")]
LOG = []


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    LOG.append(s)


def safe(fn, name):
    (OUT / f"ERROR_{name}.txt").unlink(missing_ok=True)          # remove stale error from an earlier run
    try:
        return fn()
    except Exception as e:
        import traceback
        log(f"  !! block {name} failed: {type(e).__name__}: {e}")
        (OUT / f"ERROR_{name}.txt").write_text(traceback.format_exc())
        return None


def singscore(expr_log: pd.DataFrame, up, down=None):
    R = expr_log.rank(axis=1, method="average")
    n = expr_log.shape[1]

    def one(gs):
        g = [x for x in gs if x in expr_log.columns]
        if len(g) < 5:
            return pd.Series(np.nan, index=expr_log.index)
        lo, hi = (len(g) + 1) / 2, n - (len(g) - 1) / 2
        return (R[g].mean(1) - lo) / (hi - lo) - 0.5
    return one(up) if down is None else one(up) - one(down)


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p)
    q = np.empty_like(p); q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ============================================================ A. spatial
def _first(d, pats):
    for p in pats:
        h = sorted(x for x in d.glob(p) if not x.name.startswith("._"))
        if h:
            return h[0]
    return None


def _junk_file(f):
    return f.name.startswith(("._", ".")) or "__MACOSX" in f.parts


def extract_archives(root):
    """extract *.tar / *.tar.gz / *.tgz once (GEO ships GSE*_RAW.tar with per-sample archives inside)"""
    import tarfile
    for _ in range(2):                                    # nested archives
        for t in sorted(root.rglob("*")):
            if not t.is_file() or _junk_file(t) or not re.search(r"\.(tar|tar\.gz|tgz)$", t.name):
                continue
            dest = t.parent / (re.sub(r"\.(tar|tar\.gz|tgz)$", "", t.name) + "_extracted")
            if dest.exists():
                continue
            try:
                with tarfile.open(t) as tf:
                    tf.extractall(dest, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(dest)
                log(f"  extracted {t.name}")
            except Exception as e:
                log(f"  could not extract {t.name}: {e}")


def load_visium():
    import scanpy as sc, anndata as ad
    for f in VISIUM_H5AD:
        if f.exists():
            a = sc.read_h5ad(f)
            lib = next((c for c in ("library_id", "sample_id", "slide") if c in a.obs), None)
            X = a.layers["counts"] if "counts" in a.layers else a.X
            a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame({"library_id": a.obs[lib].astype(str).values}, index=a.obs_names),
                           var=pd.DataFrame(index=a.var_names), obsm={"spatial": np.asarray(a.obsm["spatial"])}, uns={"spatial": a.uns.get("spatial", {})})
            log(f"  Visium from existing h5ad {f.name}: {a.n_obs} spots, slides {sorted(a.obs['library_id'].unique())}")
            return a
    slides = []
    roots = [Path(x) for x in os.environ.get("TGCT_VISIUM_DIR", "").split(":") if x] + VISIUM_DIRS
    for root in roots:
        if not root.exists():
            log(f"  Visium folder not found: {root}")
            continue
        extract_archives(root)
        files = [f for f in root.rglob("*") if f.is_file() and not _junk_file(f)]
        # group every file by slide key (GSM id if present, otherwise parent folder)
        def key(f):
            m = re.search(r"GSM\d+", str(f.relative_to(root)))
            return m.group(0) if m else f.parent.name
        by = {}
        for f in files:
            by.setdefault(key(f), []).append(f)
        for k, fl in sorted(by.items()):
            names = {f: f.name.lower() for f in fl}
            h5 = [f for f, n in names.items() if n.endswith(".h5") and "filtered" in n] or \
                 [f for f, n in names.items() if n.endswith(".h5") and "raw" not in n]
            mtx = [f for f, n in names.items() if re.search(r"matrix\.mtx(\.gz)?$", n)]
            pos = [f for f, n in names.items() if "tissue_positions" in n or re.search(r"positions.*\.csv", n)]
            sf = next((f for f, n in names.items() if "scalefactors" in n), None)
            img = next((f for f, n in names.items() if "hires" in n and n.endswith((".png", ".png.gz", ".jpg"))), None)
            if not pos or not (h5 or mtx):
                continue
            if h5:
                a = sc.read_10x_h5(h5[0])
            else:
                m = mtx[0]; d = m.parent; pre = m.name.split("matrix.mtx")[0]
                bfile = _first(d, [f"{pre}barcodes.tsv*", "*barcodes.tsv*"]); ffile = _first(d, [f"{pre}features.tsv*", f"{pre}genes.tsv*", "*features.tsv*", "*genes.tsv*"])
                if not (bfile and ffile):
                    continue
                X = sp.csr_matrix(scipy.io.mmread(str(m))).T.tocsr()
                ft = pd.read_csv(ffile, sep="\t", header=None)
                a = ad.AnnData(X=X.astype(np.float32),
                               obs=pd.DataFrame(index=pd.Index(pd.read_csv(bfile, header=None)[0].astype(str).values, name=None)),
                               var=pd.DataFrame(index=pd.Index((ft[1] if ft.shape[1] > 1 else ft[0]).astype(str).values, name=None)))
            a.var_names_make_unique()
            slides.append((k, a, pos[0], sf, img))
    if not slides:
        listing = []
        for root in roots:
            if root.exists():
                listing += [str(f.relative_to(root)) for f in sorted(root.rglob("*")) if f.is_file()][:300]
        (OUT / "A0_visium_folder_listing.txt").write_text("\n".join(listing) or "folder missing or empty")
        raise FileNotFoundError(f"no Visium slides found in {[str(r) for r in roots]} - file list written to "
                                f"A0_visium_folder_listing.txt (if the folder was moved, set TGCT_VISIUM_DIR=/new/path)")
    out = []
    for lib, a, pos, sf, img in slides:
        P = pd.read_csv(pos, header=None)
        if not str(P.iloc[0, 1]).isdigit():                   # new format with header
            P = pd.read_csv(pos)
        P = P.iloc[:, :6]; P.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        P = P.set_index("barcode")
        P = P[P["in_tissue"].astype(int) == 1]
        a = a[a.obs_names.isin(P.index)].copy()
        a.obsm["spatial"] = P.loc[a.obs_names, ["pxl_col", "pxl_row"]].values.astype(float)
        a.obs["library_id"] = lib
        a.obs_names = lib + "|" + a.obs_names
        spatial = {}
        if sf is not None:
            opener = gzip.open if str(sf).endswith(".gz") else open
            spatial["scalefactors"] = json.load(opener(sf, "rt"))
        if img is not None:
            try:
                import matplotlib.image as mpimg
                spatial["images"] = {"hires": mpimg.imread(gzip.open(img) if str(img).endswith(".gz") else img)}
            except Exception:
                pass
        a.uns["spatial"] = {lib: spatial}
        out.append(a)
        log(f"  slide {lib}: {a.n_obs} in-tissue spots")
    a = ad.concat(out, join="inner", merge="first", uns_merge="unique")
    a.uns["spatial"] = {k: v for x in out for k, v in x.uns["spatial"].items()}
    return a


def spot_profiles():
    """pass-1 state profiles from the stage-2 reference (identical marker logic)"""
    W = S2 / "bayesprism_input"
    X = scipy.io.mmread(str(W / "ref_counts.mtx")).tocsr()
    genes = pd.read_csv(W / "ref_genes.txt", header=None)[0].astype(str).values
    meta = pd.read_csv(W / "ref_meta.csv")
    lib = np.asarray(X.sum(1)).ravel()
    Xn = sp.diags(1.0 / np.maximum(lib, 1)) @ X
    st = sorted(meta["cell_state"].unique())
    phi = np.vstack([np.asarray(Xn[(meta["cell_state"] == s).values].mean(0)).ravel() for s in st])
    pct = np.vstack([np.asarray((X[(meta["cell_state"] == s).values] > 0).mean(0)).ravel() for s in st])
    ty = meta.groupby("cell_state")["cell_type"].first().reindex(st).values
    sel = set()
    for i in range(len(st)):
        other = ty != ty[i]
        lfc = np.log2((phi[i] + 1e-7) / (phi[other].mean(0) + 1e-7))
        c = np.where((pct[i] >= 0.1) & (lfc >= 1))[0]
        sel.update(c[np.argsort(-lfc[c])][:150])
    g = np.array(sorted(sel))
    return phi[:, g], np.array(st), ty, genes[g], pd.Series(lib).groupby(meta["cell_state"].values).mean().reindex(st).values


def em(Xb, phi, iters=3000, tol=1e-6):
    phi = phi / phi.sum(1, keepdims=True)
    th = np.full((Xb.shape[0], phi.shape[0]), 1.0 / phi.shape[0])
    for _ in range(iters):
        D = th @ phi
        new = th * ((Xb / np.maximum(D, 1e-300)) @ phi.T)
        new /= new.sum(1, keepdims=True)
        if np.abs(new - th).max() < tol:
            return new
        th = new
    return th


def morans_i(xy, v, k=6, n_perm=N_PERM, seed=SEED):
    from scipy.spatial import cKDTree
    v = np.asarray(v, float); ok = np.isfinite(v); xy, v = xy[ok], v[ok]
    _, idx = cKDTree(xy).query(xy, k + 1)
    rows = np.repeat(np.arange(len(v)), k); cols = idx[:, 1:].ravel()
    Wm = sp.csr_matrix((np.ones(len(rows)) / k, (rows, cols)), shape=(len(v), len(v)))
    def I(z):
        z = z - z.mean()
        return (z @ (Wm @ z)) / (z @ z)
    obs = I(v)
    rng = np.random.default_rng(seed)
    null = np.array([I(rng.permutation(v)) for _ in range(n_perm)])
    return obs, (1 + (null >= obs).sum()) / (n_perm + 1)


def block_A():
    import scanpy as sc
    a = load_visium()
    a.var_names_make_unique()
    sc.pp.filter_cells(a, min_counts=200)
    phi, states, stype, genes, libsz = spot_profiles()
    g = pd.Index(genes).intersection(a.var_names)
    gi = pd.Index(genes).get_indexer(g)
    Xb = a[:, g].X
    Xb = Xb.toarray() if sp.issparse(Xb) else np.asarray(Xb)
    keep = Xb.sum(1) > 50
    a, Xb = a[keep].copy(), Xb[keep]
    th = em(Xb.astype(float), phi[:, gi])
    cells = (th / libsz) / (th / libsz).sum(1, keepdims=True)
    F = pd.DataFrame(cells, index=a.obs_names, columns=states)
    F.to_csv(OUT / "A1_spot_state_fractions.csv.gz", compression="gzip")
    log(f"  A1 spot deconvolution: {a.n_obs} spots, {len(g)} marker genes, {len(states)} states")
    # A2 signature scores (locked) + TLS
    sig = json.load(open(S3 / "LOCKED_signatures.json"))
    lx = a.copy(); sc.pp.normalize_total(lx, target_sum=1e4); sc.pp.log1p(lx)
    E = pd.DataFrame(lx.X.toarray() if sp.issparse(lx.X) else lx.X, index=a.obs_names, columns=a.var_names)
    E = E.loc[:, (E > 0).mean() > 0.05]
    S = pd.DataFrame(index=a.obs_names)
    for p in ("P1_TME_immune", "P2_TME_stromal", "P3_EC_likeness"):
        S[p] = singscore(E, sig[p]["up"], sig[p]["down"])
    S["TLS12_chemokine"] = singscore(E, TLS12)
    # honest version of the P1-vs-TLS test: P1 signature without any of the 12 TLS chemokines
    p1_noTLS = [g for g in sig["P1_TME_immune"]["up"] if g not in TLS12]
    S["P1_TME_immune_noTLSgenes"] = singscore(E, p1_noTLS)
    log(f"  P1 signature genes overlapping TLS12: {sorted(set(sig['P1_TME_immune']['up']) & set(TLS12))}")
    S["Tumour_fraction"] = F[[c for c in F.columns if c.startswith("Tumour")]].sum(1)
    S["library_id"] = a.obs["library_id"].values
    S["x"], S["y"] = a.obsm["spatial"][:, 0], a.obsm["spatial"][:, 1]          # for vector maps in make_figures.py
    S.to_csv(OUT / "A2_spot_programme_scores.csv")
    # A3 Moran's I + correlations per slide
    rows = []
    for lib in S["library_id"].unique():
        m = (S["library_id"] == lib).values
        xy = a.obsm["spatial"][m]
        for c in ("P1_TME_immune", "P2_TME_stromal", "P3_EC_likeness", "TLS12_chemokine", "Tumour_fraction"):
            I, p = morans_i(xy, S.loc[m, c].values)
            rows.append(dict(slide=lib, score=c, morans_I=I, p_perm=p))
        for c1, c2 in (("P1_TME_immune", "TLS12_chemokine"), ("P1_TME_immune_noTLSgenes", "TLS12_chemokine"),
                       ("P1_TME_immune", "Tumour_fraction"),
                       ("P2_TME_stromal", "Tumour_fraction"), ("P1_TME_immune", "P2_TME_stromal")):
            r = spearmanr(S.loc[m, c1], S.loc[m, c2], nan_policy="omit")[0]
            rows.append(dict(slide=lib, score=f"rho({c1},{c2})", morans_I=r, p_perm=np.nan))
    A3 = pd.DataFrame(rows); A3.to_csv(OUT / "A3_spatial_autocorrelation.csv", index=False)
    # A4 niche test with programme membership from stage-2 loadings
    load = pd.read_csv(S2 / "S5c_programme_state_loadings.csv", index_col=0)
    memb = load.idxmax(1)[(load.max(1) > load.max(1).median())]          # confidently assigned states
    memb = memb[memb.index.isin(F.columns)]
    res = []
    for lib in S["library_id"].unique():
        m = (S["library_id"] == lib).values
        Fm = F.loc[m, memb.index]
        Fm = Fm.loc[:, Fm.std() > 0]
        C = Fm.rank().corr().values
        mb = memb[Fm.columns].values
        iu = np.triu_indices(len(mb), 1)
        same = (mb[:, None] == mb[None, :])[iu]
        obs = C[iu][same].mean() - C[iu][~same].mean()
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(999):
            pm = rng.permutation(mb)
            sm = (pm[:, None] == pm[None, :])[iu]
            null.append(C[iu][sm].mean() - C[iu][~sm].mean())
        res.append(dict(slide=lib, n_states=len(mb), within_minus_between_rho=obs,
                        p_perm=(1 + (np.array(null) >= obs).sum()) / 1000))
    A4 = pd.DataFrame(res); A4.to_csv(OUT / "A4_programme_niche_test.csv", index=False)
    log("  A4 niche test:\n" + A4.round(3).to_string(index=False))
    # A5 maps
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        libs = list(S["library_id"].unique()); cols = ["Tumour_fraction", "P1_TME_immune", "P2_TME_stromal", "TLS12_chemokine"]
        fig, axes = plt.subplots(len(libs), len(cols), figsize=(3.2 * len(cols), 3 * len(libs)), squeeze=False)
        for i, lib in enumerate(libs):
            m = (S["library_id"] == lib).values; xy = a.obsm["spatial"][m]
            for j, c in enumerate(cols):
                ax = axes[i, j]; v = S.loc[m, c]
                ax.scatter(xy[:, 0], -xy[:, 1], c=v, s=4, cmap="magma", vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98))
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                if i == 0: ax.set_title(c, fontsize=9)
                if j == 0: ax.set_ylabel(lib, fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "A5_spatial_programme_maps.png", dpi=200); plt.close(fig)
    except Exception as e:
        log(f"  map error {e}")
    return dict(A3=A3, A4=A4)


# ============================================================ B. communication
def block_B():
    import scanpy as sc
    a = sc.read_h5ad(S1 / "TGCT_atlas_stage1.h5ad")
    a.X = a.layers["lognorm"]
    st = a.obs["state"].astype(str)
    # tumour SE-like / EC-like (same rule as stage 2)
    tum = (st == "Tumour_malignant").values
    def pos(g):
        return (np.asarray(a[:, g].layers["counts"].todense()).ravel() > 0) if g in a.var_names else np.zeros(a.n_obs, bool)
    ec = pos("SOX2").astype(int) + pos("DNMT3B")
    se = pos("SOX17").astype(int) + pos("KIT") + pos("PRDM1")
    st = st.where(~tum, np.where(ec > se, "Tumour_EC_like", "Tumour_SE_like"))
    a.obs["state_b"] = st.values
    keep = ~a.obs["state_b"].str.contains("_other|_contaminant|Unresolved|Tumour_CNVneg|Tumour_GCT")
    a = a[keep.values].copy()
    grp = np.where(a.obs["site"].astype(str) == "LN", "LN_metastasis",
                   np.where((a.obs["site"].astype(str) == "primary") & (a.obs["histology"].astype(str).isin(["SE", "MIX"])), "primary_SE", "other"))
    a.obs["context"] = grp
    a = a[a.obs["context"] != "other"].copy()
    # curated LR: mean(ligand in sender) x mean(receptor in receiver), per context;
    # null = permutation of cell labels (vectorised: all genes / groups per permutation)
    rows = []
    rng = np.random.default_rng(SEED)
    lr_genes = sorted({g for pr in LR for g in pr if g in a.var_names})
    for ctx in ("primary_SE", "LN_metastasis"):
        b = a[a.obs["context"] == ctx]
        lab = b.obs["state_b"].values
        cnt = pd.Series(lab).value_counts()
        groups = list(cnt[cnt >= 20].index)
        if not any(g.startswith("Tumour") for g in groups):
            continue
        Xg = b[:, lr_genes].X
        Xg = Xg.toarray() if sp.issparse(Xg) else np.asarray(Xg)
        def gmeans(lb):
            G = np.vstack([(lb == g) for g in groups]).astype(float)
            return (G @ Xg) / G.sum(1, keepdims=True)              # groups x genes
        M = gmeans(lab)
        nulls = np.stack([gmeans(rng.permutation(lab)) for _ in range(N_PERM)])
        gi = {g: i for i, g in enumerate(lr_genes)}
        for L, R in LR:
            if L not in gi or R not in gi:
                continue
            for si, s_ in enumerate(groups):
                if not s_.startswith("Tumour"):
                    continue
                for ri, r_ in enumerate(groups):
                    if r_.startswith("Tumour"):
                        continue
                    score = M[si, gi[L]] * M[ri, gi[R]]
                    null = nulls[:, si, gi[L]] * nulls[:, ri, gi[R]]
                    others = [i for i, g_ in enumerate(groups) if i != si]
                    spec = M[si, gi[L]] / (M[others, gi[L]].mean() + 1e-9)
                    rows.append(dict(context=ctx, sender=s_, receiver=r_, ligand=L, receptor=R, score=score,
                                     ligand_specificity=spec,
                                     ligand_mean=M[si, gi[L]], receptor_mean=M[ri, gi[R]],
                                     p_perm=(1 + (null >= score).sum()) / (N_PERM + 1)))
    B1 = pd.DataFrame(rows)
    if len(B1):
        B1["q_BH"] = bh(B1["p_perm"])
        B1.to_csv(OUT / "B1_tumour_to_TME_LR_scores.csv", index=False)
        mif = B1[B1["ligand"] == "MIF"].sort_values("score", ascending=False)
        mif.to_csv(OUT / "B1_MIF_axis.csv", index=False)
        log("  B1 top MIF interactions:\n" + mif.head(10)[["context", "sender", "receiver", "receptor", "score", "q_BH"]].round(4).to_string(index=False))
    # LIANA+ (optional)
    try:
        import liana as li
        for ctx in ("primary_SE", "LN_metastasis"):
            if (OUT / f"B2_liana_{ctx}_tumour_sender.csv").exists():
                continue
            b = a[a.obs["context"] == ctx].copy()
            li.mt.rank_aggregate(b, groupby="state_b", expr_prop=0.1, use_raw=False, verbose=False, n_perms=100)
            r = b.uns["liana_res"]
            r[r["source"].str.startswith("Tumour")].to_csv(OUT / f"B2_liana_{ctx}_tumour_sender.csv", index=False)
        log("  B2 LIANA+ consensus written")
    except ImportError:
        log("  B2 LIANA+ not installed - curated LR only (pip install liana)")
    # B4 patient-controlled comparison: SE-like vs EC-like tumour cells INSIDE the same lymph node (LN1)
    ln = a[(a.obs["sample"].astype(str).str.endswith("LN1")) & a.obs["state_b"].isin(["Tumour_SE_like", "Tumour_EC_like"])]
    if (ln.obs["state_b"] == "Tumour_EC_like").sum() >= 20 and (ln.obs["state_b"] == "Tumour_SE_like").sum() >= 20:
        # sequencing-depth control: binomial thinning of every cell to the lower group's median UMI count
        C = ln.layers["counts"].tocsr().astype(float) if sp.issparse(ln.layers["counts"]) else sp.csr_matrix(ln.layers["counts"].astype(float))
        libs = np.asarray(C.sum(1)).ravel()
        grp = ln.obs["state_b"].values
        target = min(np.median(libs[grp == "Tumour_EC_like"]), np.median(libs[grp == "Tumour_SE_like"]))
        rng = np.random.default_rng(SEED)
        frac = np.minimum(1.0, target / np.maximum(libs, 1))
        Ct = C.copy()
        Ct.data = rng.binomial(np.round(C.data).astype(int), np.repeat(frac, np.diff(C.indptr))).astype(float)
        lt = np.log1p(Ct.multiply(1e4 / np.maximum(np.asarray(Ct.sum(1)).ravel(), 1)[:, None]).tocsr())
        gidx = {g: i for i, g in enumerate(ln.var_names)}
        log(f"  B4 depth control: median UMI EC-like {np.median(libs[grp == 'Tumour_EC_like']):.0f}, "
            f"SE-like {np.median(libs[grp == 'Tumour_SE_like']):.0f} -> thinned to {target:.0f}")
        r4 = []
        for L in sorted({l for l, _ in LR}):
            if L not in ln.var_names:
                continue
            xt = np.asarray(lt[:, gidx[L]].todense()).ravel()
            et, st_ = xt[grp == "Tumour_EC_like"], xt[grp == "Tumour_SE_like"]
            x = ln[:, L].X; x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
            e, s_ = x[(ln.obs["state_b"] == "Tumour_EC_like").values], x[(ln.obs["state_b"] == "Tumour_SE_like").values]
            r4.append(dict(ligand=L, mean_EC_like=e.mean(), mean_SE_like=s_.mean(), pct_EC=(e > 0).mean(), pct_SE=(s_ > 0).mean(),
                           log2FC_EC_vs_SE=np.log2((np.expm1(e).mean() + 1e-3) / (np.expm1(s_).mean() + 1e-3)),
                           p=mannwhitneyu(e, s_).pvalue, n_EC=len(e), n_SE=len(s_),
                           pct_EC_depthmatched=(et > 0).mean(), pct_SE_depthmatched=(st_ > 0).mean(),
                           log2FC_depthmatched=np.log2((np.expm1(et).mean() + 1e-3) / (np.expm1(st_).mean() + 1e-3)),
                           p_depthmatched=mannwhitneyu(et, st_).pvalue))
        B4 = pd.DataFrame(r4); B4["q_BH"] = bh(B4["p"]); B4["q_depthmatched"] = bh(B4["p_depthmatched"])
        B4.sort_values("p_depthmatched").to_csv(OUT / "B4_LN1_SE_vs_EC_ligands.csv", index=False)
        log("  B4 within-LN1 (same patient) SE-like vs EC-like ligands:\n" +
            B4.sort_values("p_depthmatched").head(8)[["ligand", "log2FC_EC_vs_SE", "q_BH", "log2FC_depthmatched", "q_depthmatched"]].round(3).to_string(index=False))
    # B3 tumour SE-like vs EC-like differential expression (exploratory)
    t = a[a.obs["state_b"].isin(["Tumour_SE_like", "Tumour_EC_like"])].copy()
    if (t.obs["state_b"] == "Tumour_EC_like").sum() >= 20:
        sc.tl.rank_genes_groups(t, "state_b", groups=["Tumour_EC_like"], reference="Tumour_SE_like", method="wilcoxon", use_raw=False)
        de = sc.get.rank_genes_groups_df(t, "Tumour_EC_like")
        de.to_csv(OUT / "B3_EC_vs_SE_tumour_DE.csv", index=False)
        pd.crosstab(t.obs["sample"], t.obs["state_b"]).to_csv(OUT / "B3_tumour_state_by_sample.csv")
        log(f"  B3 EC-like vs SE-like DE: {(de['pvals_adj'] < 0.01).sum()} genes padj<0.01 (exploratory; patient-confounded)")
    return B1


# ============================================================ C. TCGA genomics
def xena(name):
    f = PROJ / "data" / "raw" / "TCGA_TGCT" / XENA_FILES[name].split("%2F")[-1]
    if not f.exists():
        import requests
        r = requests.get(XENA + XENA_FILES[name], timeout=300); r.raise_for_status()
        f.write_bytes(r.content)
        log(f"  downloaded {f.name}")
    return pd.read_csv(f, sep="\t", index_col=0)


def block_C():
    d = pd.read_csv(S3 / "S3_tcga_analysis_table.csv", index_col=0)
    mut = xena("mc3"); cn = xena("gistic")
    mut.columns = mut.columns.str[:15]; cn.columns = cn.columns.str[:15]
    g = pd.DataFrame(index=d.index)
    for gene in ("KIT", "KRAS", "NRAS"):
        if gene in mut.index:
            g[f"{gene}_mut"] = mut.loc[gene].reindex(d.index)
    g["mutated_genes_n"] = mut.sum(0).reindex(d.index)
    pos = pd.read_csv(PROJ / "data" / "gene_positions_hg38.tsv", sep="\t").set_index("gene") if (PROJ / "data" / "gene_positions_hg38.tsv").exists() else None
    if pos is not None:
        p12 = pos.index[(pos["chromosome"] == "chr12") & (pos["start"] < 35.5e6)].intersection(cn.index)
        chrx = pos.index[pos["chromosome"] == "chrX"].intersection(cn.index)
        g["chr12p_gain"] = cn.loc[p12].clip(lower=0).mean(0).reindex(d.index)
        g["chrX_gain"] = cn.loc[chrx].clip(lower=0).mean(0).reindex(d.index)
    import statsmodels.api as smapi
    # copy-number calls are attenuated in low-purity (immune-rich) tumours -> adjust for deconvolved tumour fraction
    ty = pd.read_csv(S2 / "tcga_type_fractions.csv", index_col=0)
    d["tumour_fraction"] = ty["Tumour"].reindex(d.index)
    rows = []
    for p in ("P1_TME_immune", "P2_TME_stromal", "P3_EC_likeness", "P4_IFNG_programme"):
        for v in g.columns:
            x = pd.concat([d[[p, "NSGCT", "tumour_fraction"]], g[v]], axis=1).dropna()
            if len(x) < 30 or x[v].std() == 0:
                continue
            yv = (x[p] - x[p].mean()) / x[p].std()
            r = dict(programme=p, feature=v, n=len(x), n_positive=int((x[v] > 0).sum()))
            for tag, cov in (("hist", [v, "NSGCT"]), ("hist_purity", [v, "NSGCT", "tumour_fraction"])):
                m = smapi.OLS(yv, smapi.add_constant(x[cov].astype(float))).fit()
                r[f"beta_SD_{tag}"], r[f"p_{tag}"] = m.params[v], m.pvalues[v]
            r["rho_feature_vs_purity"] = spearmanr(x[v], x["tumour_fraction"])[0]
            rows.append(r)
    C = pd.DataFrame(rows)
    C["q_hist"] = bh(C["p_hist"]); C["q_hist_purity"] = bh(C["p_hist_purity"])
    C["p"] = C["p_hist_purity"]; C["q_BH"] = C["q_hist_purity"]     # purity-adjusted = primary
    C.to_csv(OUT / "C1_genomics_vs_programmes.csv", index=False)
    log("  C1 genomics (histology-adjusted):\n" + C.sort_values("p").head(10).round(4).to_string(index=False))
    return C


# ============================================================ RUN
if __name__ == "__main__":
    t0 = time.time()
    log("BLOCK A spatial");        A = safe(block_A, "A_spatial")
    log("BLOCK B communication");  B = safe(block_B, "B_communication")
    log("BLOCK C TCGA genomics");  C = safe(block_C, "C_genomics")
    md = ["# STAGE 4 report - mechanism / spatial", "", f"runtime {(time.time() - t0) / 60:.1f} min", ""]
    if A:
        md += ["## A3 spatial autocorrelation (Moran's I) and per-slide correlations", A["A3"].round(4).to_markdown(index=False), "",
               "## A4 do programme states co-localise? (within - between programme spot correlation)", A["A4"].round(4).to_markdown(index=False), ""]
    if B is not None and len(B):
        md += ["## B1 MIF axis (tumour sender)", pd.read_csv(OUT / "B1_MIF_axis.csv").head(20).round(4).to_markdown(index=False), "",
               "## B1 top tumour -> TME interactions per context",
               B.sort_values("score", ascending=False).groupby("context").head(12).round(4).to_markdown(index=False), ""]
        f4 = OUT / "B4_LN1_SE_vs_EC_ligands.csv"
        if f4.exists():
            md += ["## B4 same lymph node (LN1): SE-like vs EC-like tumour cells - ligands", pd.read_csv(f4).round(4).to_markdown(index=False), ""]
        f3 = OUT / "B3_EC_vs_SE_tumour_DE.csv"
        if f3.exists():
            de = pd.read_csv(f3)
            md += ["## B3 EC-like vs SE-like tumour cells (top 25 up in EC-like)", de.head(25).round(4).to_markdown(index=False), ""]
    if C is not None:
        md += ["## C1 TCGA genomics vs programmes (OLS; adjusted for histology, and for histology + tumour purity = primary)",
               C.sort_values("p").drop(columns=["p", "q_BH"]).round(4).to_markdown(index=False), ""]
    md += ["## Log", "```", *LOG, "```"]
    (OUT / "STAGE4_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log(f"done -> {OUT / 'STAGE4_REPORT.md'}")
