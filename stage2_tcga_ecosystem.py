#!/usr/bin/env python
"""
STAGE 2 / 4  -  Tumour-immune ecosystem of TCGA-TGCT from the single-cell reference
==================================================================================
Inputs (from earlier stages; only a ~5 MB gene-ID map is downloaded if missing)
  stage1_atlas/S8_deconv_reference_counts.h5ad        single-cell reference (counts, cell_type, cell_state)
  stage1_atlas/S8_pseudobulk_sample_counts.parquet     pseudobulk of each scRNA sample (benchmark)
  stage1_atlas/tables/S8_true_state_fractions_per_sample.csv
  stage1_atlas/S8_state_signatures_top50.json          state marker genes (signature scores)
  data/raw/TCGA_TGCT/TCGA-TGCT.star_counts.tsv.gz      GDC STAR counts (log2(count+1), Ensembl IDs)
  data/processed/TCGA_TGCT_clinical_harmonised.csv     clinical table from step 0b

Steps (each writes its outputs; re-running skips finished steps)
  1  TCGA bulk: back-transform to counts, Ensembl -> symbol, one primary tumour per patient
  2  reference: nested type/state labels, down-sampling, gene clean-up
  3  deconvolution of TCGA (pure Python, BayesPrism-style two-pass EM) -> type + state fractions,
     tumour-specific expression per patient
  4  benchmark: leave-one-dataset-out on scRNA pseudobulks vs true fractions (final / pass-1 / NNLS)
  5  ecotypes: consensus NMF on TME state composition (k selected by stability)
  6  rank-based state signature scores (transferable to GSE99420) + agreement with deconvolution
  7  report

Run:  conda activate scvi-env ; python stage2_tcga_ecosystem.py
No R needed: numpy / scipy / pandas / scikit-learn / anndata only
"""
import os, re, io, json, gzip, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr, chi2_contingency
warnings.filterwarnings("ignore")

# ============================================================ config
BASE = Path(os.environ.get("TGCT_BASE", "/home/arshia/testis project"))
PROJ = BASE / "TGCT_ecotype"
S1 = PROJ / "stage1_atlas"
OUT = PROJ / "stage2_tcga"
WORK = OUT / "bayesprism_input"
for d in (OUT, WORK, OUT / "figures"):
    d.mkdir(parents=True, exist_ok=True)

TCGA_COUNTS = PROJ / "data" / "raw" / "TCGA_TGCT" / "TCGA-TGCT.star_counts.tsv.gz"
PROBEMAP = PROJ / "data" / "raw" / "TCGA_TGCT" / "gencode.v36.annotation.gtf.gene.probemap"
PROBEMAP_URL = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap"
CLIN = PROJ / "data" / "processed" / "TCGA_TGCT_clinical_harmonised.csv"

KEY = "Tumour"                   # cell type with sample-specific profile (like BayesPrism 'key')
MARKERS_PER_STATE = 150
PROFILE_SHRINK = 0.5             # weight of the scRNA profile when updating non-tumour type profiles
MAX_CELLS_PER_STATE = 300        # keeps the dense R reference matrix small (RAM)
MIN_CELLS_PER_STATE = 20
RUN_BENCHMARK = True
K_RANGE = range(2, 9)
N_RESAMPLE = 100 if not os.environ.get("TGCT_TEST") else 10
SEED = 0

# cell-state -> cell type (states must be nested in types)
TYPE_RULES = [
    (r"^Tumour", "Tumour"), (r"^Plasma$", "Plasma"), (r"^B_", "B"),
    (r"^(CD4_|CD8_|Treg|T_|NK$|MAIT)", "T_NK"),
    (r"^(Mono_|Macro_|cDC|mregDC|pDC|Mast|Neutrophil|DC$|Myeloid)", "Myeloid"),
    (r"^Endo_|^Endothelial$", "Endothelial"),
    (r"^(Fibro_|Fibroblast|Pericyte|Myoid)", "Stromal"),
    (r"^Leydig$", "Leydig"), (r"^Sertoli$", "Sertoli"), (r"^Germ_normal$", "Germ_normal"),
]
NORMAL_TESTIS_TYPES = {"Germ_normal", "Sertoli", "Leydig"}       # residual testis, not TME
# stage-1 major lineage -> deconvolution type (used to drop cells whose fine state contradicts their lineage)
MAJOR2TYPE = {"T_NK": "T_NK", "B": "B", "Plasma": "Plasma", "Myeloid": "Myeloid", "DC": "Myeloid", "pDC": "Myeloid",
              "Mast": "Myeloid", "Neutrophil": "Myeloid", "Endothelial": "Endothelial", "Fibroblast": "Stromal",
              "Myoid_SMC": "Stromal", "Leydig": "Leydig", "Sertoli": "Sertoli", "Germ_normal": "Germ_normal",
              "Tumour_malignant": "Tumour"}
SPLIT_TUMOUR = True              # Tumour_SE-like vs Tumour_EC-like states (SOX17/KIT vs SOX2/DNMT3B)
LOG = []


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    LOG.append(s)


def state_to_type(st):
    for pat, t in TYPE_RULES:
        if re.search(pat, st):
            return t
    return None


# ============================================================ STEP 1  TCGA bulk
def step1_tcga():
    f_out = WORK / "tcga_bulk_counts.csv.gz"
    if f_out.exists():
        return pd.read_csv(f_out, index_col=0)
    if not PROBEMAP.exists():
        import requests
        log("  downloading GDC gene probemap (Ensembl -> symbol)")
        r = requests.get(PROBEMAP_URL, timeout=300)
        r.raise_for_status()
        PROBEMAP.write_bytes(r.content)
    pm = pd.read_csv(PROBEMAP, sep="\t")
    pm.columns = ["id", "gene", "chrom", "start", "end", "strand"][: pm.shape[1]]
    pm = pm.set_index("id")
    log("  reading STAR counts (log2(count+1))")
    x = pd.read_csv(TCGA_COUNTS, sep="\t", index_col=0)
    x = x.loc[x.index.isin(pm.index)]
    cnt = np.round(np.power(2.0, x) - 1).clip(lower=0)
    cnt["gene"] = pm.loc[cnt.index, "gene"].values
    cnt = cnt.groupby("gene").sum()
    # one primary tumour (sample type 01) per patient
    cols = [c for c in cnt.columns if c[13:15] == "01"]
    pats = pd.Series(cols, index=cols).str[:12]
    keep = pats[~pats.duplicated()].index
    cnt = cnt[keep].T
    cnt.index = cnt.index.str[:15]
    cnt = cnt.loc[:, cnt.sum(0) > 0].astype(np.int64)
    cnt.to_csv(f_out, compression="gzip")
    # chromosome table for gene clean-up
    pm.drop_duplicates("gene").set_index("gene")["chrom"].to_csv(WORK / "gene_chrom.csv")
    log(f"  TCGA bulk: {cnt.shape[0]} primary tumours x {cnt.shape[1]} genes")
    return cnt


# ============================================================ STEP 2  reference
def step2_reference():
    import anndata as ad
    f_meta = WORK / "ref_meta.csv"
    if f_meta.exists():
        return pd.read_csv(f_meta)
    ref = ad.read_h5ad(S1 / "S8_deconv_reference_counts.h5ad")
    o = ref.obs.copy()
    o["cell_state"] = o["cell_state"].astype(str)
    o = o[~o["cell_state"].isin(["Tumour_GCT", "Tumour_CNVneg"])]
    o["major_type"] = o["cell_type"].astype(str).map(MAJOR2TYPE)
    o["cell_type"] = o["cell_state"].map(state_to_type)
    # lineage consistency: e.g. fibroblasts that a weak sub-cluster vote called 'Sertoli' are dropped
    bad = o["major_type"].notna() & o["cell_type"].notna() & (o["major_type"] != o["cell_type"])
    if bad.any():
        drop_tab = o[bad].groupby("cell_state").size().sort_values(ascending=False)
        drop_tab.rename("cells_dropped").to_csv(OUT / "S2_reference_lineage_conflicts.csv")
        log(f"  dropped {int(bad.sum())} cells with state/lineage conflict (top: {drop_tab.head(5).to_dict()})")
        o = o[~bad]
    unk = o["cell_type"].isna()
    if unk.any():                                      # fall back on the majority stage-1 type of the state
        maj = ref.obs.loc[o.index[unk]].groupby("cell_state", observed=True)["cell_type"].agg(lambda x: x.value_counts().index[0])
        o.loc[unk, "cell_type"] = o.loc[unk, "cell_state"].map(maj).astype(str)
        log(f"  states typed by majority vote: {sorted(o.loc[unk, 'cell_state'].unique())}")
    o.loc[o["cell_type"] == "Tumour", "cell_state"] = "Tumour"
    if SPLIT_TUMOUR:
        tum = o.index[o["cell_type"] == "Tumour"]
        g = {x: list(ref.var_names).index(x) for x in ("SOX2", "DNMT3B", "SOX17", "KIT", "PRDM1") if x in ref.var_names}
        if {"SOX2", "SOX17"} <= set(g) and len(tum):
            Xt = ref[tum].layers["counts"] if "counts" in ref.layers else ref[tum].X
            Xt = Xt.toarray() if sp.issparse(Xt) else np.asarray(Xt)
            ec = (Xt[:, g["SOX2"]] > 0) + (Xt[:, g["DNMT3B"]] > 0 if "DNMT3B" in g else 0)
            se = (Xt[:, g["SOX17"]] > 0) + (Xt[:, g["KIT"]] > 0 if "KIT" in g else 0) + (Xt[:, g["PRDM1"]] > 0 if "PRDM1" in g else 0)
            lab = np.where(ec > se, "Tumour_EC_like", "Tumour_SE_like")
            o.loc[tum, "cell_state"] = lab
            t = pd.crosstab(o.loc[tum, "sample"], o.loc[tum, "cell_state"])
            t.to_csv(OUT / "S2_reference_tumour_states.csv")
            log("  tumour states in reference:\n" + t.to_string())
    vc = o["cell_state"].value_counts()
    o = o[o["cell_state"].isin(vc[vc >= MIN_CELLS_PER_STATE].index)]
    rng = np.random.default_rng(SEED)
    idx = []
    for st, oo in o.groupby("cell_state"):
        idx += list(rng.choice(oo.index, min(len(oo), MAX_CELLS_PER_STATE), replace=False))
    o = o.loc[idx]
    r = ref[o.index].copy()
    X = r.layers["counts"] if "counts" in r.layers else r.X
    X = sp.csr_matrix(X.astype(np.float64)) if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
    # gene clean-up (python side): ribosomal, mitochondrial, sex chromosomes, MALAT1/NEAT1, haemoglobin
    genes = pd.Index(r.var_names)
    chrom = pd.read_csv(WORK / "gene_chrom.csv", index_col=0)["chrom"] if (WORK / "gene_chrom.csv").exists() else pd.Series(dtype=str)
    bad = genes.str.match(r"^(RP[SL]\d|MRP[SL]\d|MT-|HB[AB]\d?$|MALAT1$|NEAT1$|XIST$)")
    bad |= genes.isin(chrom[chrom.isin(["chrX", "chrY", "chrM"])].index)
    expressed = np.asarray((X > 0).sum(0)).ravel() >= 5
    keep = (~bad) & expressed
    X = X[:, keep]
    genes = genes[keep]
    scipy.io.mmwrite(str(WORK / "ref_counts.mtx"), X.astype(np.float64))
    pd.Series(genes).to_csv(WORK / "ref_genes.txt", index=False, header=False)
    o = o.assign(cell=o.index)[["cell", "cell_type", "cell_state", "dataset", "sample"]]
    o.to_csv(f_meta, index=False)
    tab = o.groupby(["cell_type", "cell_state"]).size().rename("cells").reset_index()
    tab.to_csv(OUT / "S2_reference_composition.csv", index=False)
    log(f"  reference: {len(o)} cells, {o['cell_type'].nunique()} types, {o['cell_state'].nunique()} states, "
        f"{len(genes)} genes")
    return o


# ============================================================ deconvolution engine (pure Python)
# BayesPrism-style two-pass multinomial mixture model, fitted by EM (MAP) instead of Gibbs sampling:
#   pass 1  bulk reads ~ Multinomial( sum_k theta_nk * phi_kg )  over cell STATES       -> state fractions
#   update  tumour profile re-estimated PER SAMPLE from the reads allocated to tumour;
#           non-tumour type profiles re-estimated from reads pooled over all samples
#           (absorbs scRNA-vs-bulk platform shift), shrunk toward the scRNA profile
#   pass 2  EM over cell TYPES with the updated (sample-specific tumour) profiles       -> final type fractions
# theta = fraction of transcripts (not of cells), as in BayesPrism.
_REF = {}


def load_reference():
    if not _REF:
        X = scipy.io.mmread(str(WORK / "ref_counts.mtx")).tocsr()
        genes = pd.read_csv(WORK / "ref_genes.txt", header=None)[0].astype(str).values
        meta = pd.read_csv(WORK / "ref_meta.csv")
        _REF.update(X=X, genes=genes, meta=meta)
    return _REF["X"], _REF["genes"], _REF["meta"]


def build_profiles(exclude_datasets=(), n_markers=MARKERS_PER_STATE):
    X, genes, meta = load_reference()
    keep = ~meta["dataset"].isin(exclude_datasets).values
    # states left with too few cells after excluding a dataset (e.g. EC-like tumour without GSE197778) are dropped
    vc = meta.loc[keep, "cell_state"].value_counts()
    small = vc[vc < 10].index
    if len(small):
        log(f"    states dropped (<10 cells after exclusion): {list(small)}")
    keep &= ~meta["cell_state"].isin(small).values
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    lib = np.asarray(X.sum(1)).ravel()
    Xn = sp.diags(1.0 / np.maximum(lib, 1)) @ X                    # per-cell proportions
    states = meta["cell_state"].values
    st_names = sorted(set(states))
    phi, pct = [], []
    for st in st_names:
        m = states == st
        phi.append(np.asarray(Xn[m].mean(0)).ravel())
        pct.append(np.asarray((X[m] > 0).mean(0)).ravel())
    phi, pct = np.vstack(phi), np.vstack(pct)
    st_type = meta.groupby("cell_state")["cell_type"].first().reindex(st_names).values
    # markers: per state, log2FC vs the mean of states of OTHER cell types, detected in >=10% of its cells
    eps = 1e-7
    sel = set()
    for i, st in enumerate(st_names):
        other = st_type != st_type[i]
        base = phi[other].mean(0) if other.any() else phi[np.arange(len(st_names)) != i].mean(0)
        lfc = np.log2((phi[i] + eps) / (base + eps))
        cand = np.where((pct[i] >= 0.10) & (lfc >= 1.0))[0]
        sel.update(cand[np.argsort(-lfc[cand])][:n_markers])
        # within-type resolution: vs other states of the same type
        same = (st_type == st_type[i]) & (np.arange(len(st_names)) != i)
        if same.any():
            lfc2 = np.log2((phi[i] + eps) / (phi[same].mean(0) + eps))
            cand2 = np.where((pct[i] >= 0.10) & (lfc2 >= 0.8))[0]
            sel.update(cand2[np.argsort(-lfc2[cand2])][: n_markers // 2])
    g = np.array(sorted(sel))
    # mean transcripts per cell (UMI) per state: converts transcript fractions -> cell fractions
    libsize = pd.Series(lib).groupby(states).mean().reindex(st_names).values
    phi = phi[:, g] + 1e-8
    phi = phi / phi.sum(1, keepdims=True)
    return dict(phi=phi, states=np.array(st_names), state_type=np.array(st_type), genes=genes[g], libsize=libsize)


def _em(Xb, phi, theta=None, iters=5000, tol=1e-6, phi_tum=None, tum_idx=None):
    """MAP-EM for multinomial mixture. Xb: N x G counts; phi: K x G (rows sum to 1).
    phi_tum (N x G) replaces row tum_idx per sample if given."""
    N, K = Xb.shape[0], phi.shape[0]
    th = np.full((N, K), 1.0 / K) if theta is None else theta.copy()
    for it in range(iters):
        if phi_tum is None:
            D = th @ phi
        else:
            mask = np.ones(K, bool); mask[tum_idx] = False
            D = th[:, mask] @ phi[mask] + th[:, [tum_idx]] * phi_tum
        R = Xb / np.maximum(D, 1e-300)
        num = th * (R @ phi.T)
        if phi_tum is not None:
            num[:, tum_idx] = th[:, tum_idx] * (R * phi_tum).sum(1)
        new = num / num.sum(1, keepdims=True)
        if np.abs(new - th).max() < tol:
            th = new
            break
        th = new
    return th, it + 1


def deconvolve(bulk: pd.DataFrame, prof: dict, shrink=PROFILE_SHRINK):
    genes = pd.Index(prof["genes"]).intersection(bulk.columns)
    gi = pd.Index(prof["genes"]).get_indexer(genes)
    Xb = bulk[genes].values.astype(float)
    # outlier genes (BayesPrism default): >1% of a sample's reads in >10% of samples
    frac = Xb / Xb.sum(1, keepdims=True)
    ok = (frac > 0.01).mean(0) <= 0.10
    Xb, gi, genes = Xb[:, ok], gi[ok], genes[ok]
    phi = prof["phi"][:, gi]
    phi = phi / phi.sum(1, keepdims=True)
    states, stype = prof["states"], prof["state_type"]
    # ---- pass 1: cell states
    th_s, it1 = _em(Xb, phi)
    types = np.array(sorted(set(stype)))
    T = np.vstack([(stype == t).astype(float) for t in types])            # types x states
    th_t1 = th_s @ T.T
    # expected reads per type: Z_t = X * (sum_{k in t} theta_k phi_k) / D
    D = th_s @ phi
    Z = {t: Xb * ((th_s[:, stype == t] @ phi[stype == t]) / np.maximum(D, 1e-300)) for t in types}
    # ---- reference update
    rows = []
    for t in types:
        w = th_s[:, stype == t].sum(0)
        if w.sum() < 1e-8:                                   # type absent from every sample -> plain mean of its states
            w = np.ones_like(w)
        rows.append((w / w.sum()) @ phi[stype == t])          # fraction-weighted state mix
    phi_type = np.vstack(rows) + 1e-12
    phi_type = phi_type / phi_type.sum(1, keepdims=True)
    upd = []
    for i, t in enumerate(types):
        pooled = Z[t].sum(0) + 1.0
        pooled = pooled / pooled.sum()
        upd.append(shrink * phi_type[i] + (1 - shrink) * pooled if t != KEY else phi_type[i])
    phi2 = np.vstack(upd)
    phi2 = phi2 / phi2.sum(1, keepdims=True)
    tum_idx = int(np.where(types == KEY)[0][0]) if KEY in types else None
    phi_tum = None
    if tum_idx is not None:
        zt = Z[KEY] + 1e-2 * phi_type[tum_idx] * Xb.sum(1, keepdims=True)     # tiny prior toward reference tumour
        phi_tum = zt / zt.sum(1, keepdims=True)
    # ---- pass 2: cell types with updated profiles
    th_t2, it2 = _em(Xb, phi2, theta=th_t1, phi_tum=phi_tum, tum_idx=tum_idx)
    idx = bulk.index
    # transcript -> cell fractions (divide by mean UMI per cell of each state / type)
    ls_state = prof["libsize"]
    ls_type = np.array([np.average(ls_state[stype == t], weights=th_s[:, stype == t].mean(0) + 1e-9) for t in types])
    for k, v in (("pass1", th_s), ("pass2", th_t2)):
        if not np.isfinite(v).all():
            raise ValueError(f"non-finite fractions in {k} - check reference/bulk gene overlap")
    tocell = lambda th, ls: (th / ls) / (th / ls).sum(1, keepdims=True)
    res = dict(state_cells=pd.DataFrame(tocell(th_s, ls_state), index=idx, columns=states),
               type_cells=pd.DataFrame(tocell(th_t2, ls_type), index=idx, columns=types),
               type_first_cells=pd.DataFrame(tocell(th_t1, ls_type), index=idx, columns=types),
               state=pd.DataFrame(th_s, index=idx, columns=states),
               type_first=pd.DataFrame(th_t1, index=idx, columns=types),
               type_final=pd.DataFrame(th_t2, index=idx, columns=types),
               n_genes=len(genes), iters=(it1, it2))
    if tum_idx is not None:
        res["tumour_expr"] = pd.DataFrame(phi_tum * 1e6, index=idx, columns=genes)   # tumour-specific CPM
    # simple NNLS on CPM (comparator)
    from scipy.optimize import nnls
    y = Xb / np.maximum(Xb.sum(1, keepdims=True), 1)
    A = np.vstack([phi_type[i] for i in range(len(types))]).T
    nn = np.vstack([nnls(A, y[n])[0] for n in range(len(y))])
    nn = nn / np.maximum(nn.sum(1, keepdims=True), 1e-12)
    res["type_nnls"] = pd.DataFrame(tocell(nn + 1e-12, ls_type), index=idx, columns=types)
    return res


def run_deconv(bulk: pd.DataFrame, tag: str, exclude_datasets=()):
    outp = OUT / tag
    if Path(f"{outp}_type_fractions.csv").exists():
        log(f"  deconvolution {tag}: exists, skip")
        return
    t0 = time.time()
    prof = build_profiles(exclude_datasets)
    res = deconvolve(bulk, prof)
    # main outputs = CELL fractions (comparable with scRNA truth); transcript fractions kept too
    res["type_cells"].to_csv(f"{outp}_type_fractions.csv")
    res["type_first_cells"].to_csv(f"{outp}_type_fractions_pass1.csv")
    res["type_nnls"].to_csv(f"{outp}_type_fractions_nnls.csv")
    res["state_cells"].to_csv(f"{outp}_state_fractions.csv")
    res["type_final"].to_csv(f"{outp}_type_transcript_fractions.csv")
    res["state"].to_csv(f"{outp}_state_transcript_fractions.csv")
    if "tumour_expr" in res and tag == "tcga":
        res["tumour_expr"].round(3).to_csv(f"{outp}_tumour_expression.csv.gz", compression="gzip")
    log(f"  deconvolution {tag}: {bulk.shape[0]} samples, {res['n_genes']} marker genes, "
        f"{len(prof['states'])} states, EM iters {res['iters']}, excluded={list(exclude_datasets)}, "
        f"{time.time() - t0:.0f}s")


# ============================================================ STEP 4  benchmark
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = x.var(), y.var()
    return 2 * np.cov(x, y, bias=True)[0, 1] / (sx + sy + (x.mean() - y.mean()) ** 2 + 1e-12)


def step4_benchmark(meta):
    f = OUT / "S4_benchmark_per_type.csv"
    if f.exists() or not RUN_BENCHMARK:
        return pd.read_csv(f) if f.exists() else None
    pb = pd.read_parquet(S1 / "S8_pseudobulk_sample_counts.parquet")
    truth_s = pd.read_csv(S1 / "tables" / "S8_true_state_fractions_per_sample.csv", index_col=0)
    ds_of = pd.Series(pb.index.str.split("_").str[0], index=pb.index)
    test = [s for s in pb.index if "PBMC" not in s and not s.startswith(("GSE112013", "GSE182786"))]
    for ds in sorted(ds_of[test].unique()):
        samples = [s for s in test if ds_of[s] == ds]
        run_deconv(pb.loc[samples].round(), f"bench_LODO_{ds}", exclude_datasets=[ds])
    # truth at type level (low-confidence states dropped, renormalised)
    tt = truth_s.T.assign(type=[state_to_type(s) if not re.search(r"_other$|_contaminant$|Tumour_GCT|Tumour_CNVneg", s)
                                else None for s in truth_s.columns]).dropna(subset=["type"])
    truth = tt.groupby("type").sum().T
    truth = truth.div(truth.sum(1), axis=0)
    allres = []
    for method, suf in (("final", ""), ("pass1", "_pass1"), ("nnls", "_nnls")):
        pred = pd.concat([pd.read_csv(OUT / f"bench_LODO_{ds}_type_fractions{suf}.csv", index_col=0)
                          for ds in sorted(ds_of[test].unique())])
        common_t = pred.columns.intersection(truth.columns)
        common_s = pred.index.intersection(truth.index)
        P, T = pred.loc[common_s, common_t], truth.loc[common_s, common_t]
        rows = []
        for t in common_t:
            r = pearsonr(P[t], T[t])[0] if T[t].std() > 0 and P[t].std() > 0 else np.nan
            rows.append(dict(method=method, cell_type=t, pearson_r=r, ccc=ccc(P[t], T[t]), mean_true=T[t].mean(),
                             mean_pred=P[t].mean(), abs_err=float((P[t] - T[t]).abs().mean())))
        rows.append(dict(method=method, cell_type="ALL (pooled)", pearson_r=pearsonr(P.values.ravel(), T.values.ravel())[0],
                         ccc=ccc(P.values.ravel(), T.values.ravel()), mean_true=np.nan, mean_pred=np.nan,
                         abs_err=float((P - T).abs().values.mean())))
        allres += rows
        if method == "final":
            pd.concat({"pred": P, "true": T}, axis=1).to_csv(OUT / "S4_benchmark_pred_vs_true.csv")
    res = pd.DataFrame(allres)
    res.to_csv(f, index=False)
    summ = res[res["cell_type"] == "ALL (pooled)"][["method", "pearson_r", "ccc", "abs_err"]]
    log("  pooled benchmark by method:\n" + summ.round(3).to_string(index=False))
    return res


# ============================================================ STEP 5  ecotypes
def consensus_nmf(X, k, n=N_RESAMPLE, frac=0.8, seed=SEED):
    from sklearn.decomposition import NMF
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    co = np.zeros((N, N)); cnt = np.zeros((N, N))
    for i in range(n):
        idx = rng.choice(N, int(frac * N), replace=False)
        m = NMF(n_components=k, init="nndsvda", random_state=int(rng.integers(1e6)), max_iter=2000)
        W = m.fit_transform(X[idx])
        lab = W.argmax(1)
        same = (lab[:, None] == lab[None, :]).astype(float)
        co[np.ix_(idx, idx)] += same
        cnt[np.ix_(idx, idx)] += 1
    C = np.divide(co, cnt, out=np.zeros_like(co), where=cnt > 0)
    D = 1 - C
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    coph = cophenet(Z, squareform(D, checks=False))[0]
    return C, coph, Z


def step5_ecotypes():
    f = OUT / "S5_ecotype_assignment.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    from sklearn.decomposition import NMF
    from sklearn.metrics import silhouette_score
    st = pd.read_csv(OUT / "tcga_state_fractions.csv", index_col=0)
    ty = pd.read_csv(OUT / "tcga_type_fractions.csv", index_col=0)
    st_type = pd.Series({s: state_to_type(s) for s in st.columns})
    tme_states = st_type[~st_type.isin(NORMAL_TESTIS_TYPES | {"Tumour"})].index
    tme = st[tme_states]
    tme = tme.div(tme.sum(1).replace(0, np.nan), axis=0).fillna(0)       # composition of the TME only
    tme = tme.loc[:, tme.mean() > 0.002]
    Xs = (tme / tme.max().replace(0, 1)).values                          # each state scaled to [0,1]
    rows, cons = [], {}
    for k in K_RANGE:
        C, coph, Z = consensus_nmf(Xs, k)
        lab = fcluster(Z, k, "maxclust")
        sizes = np.bincount(lab)[1:]
        sil = silhouette_score(1 - C, lab, metric="precomputed") if len(set(lab)) > 1 else np.nan
        rows.append(dict(k=k, cophenetic=coph, silhouette=sil, min_cluster=int(sizes.min())))
        cons[k] = C
        log(f"  k={k}: cophenetic={coph:.3f} silhouette={sil:.3f} min size={sizes.min()}")
    ksel = pd.DataFrame(rows)
    ksel.to_csv(OUT / "S5_k_selection.csv", index=False)
    ok = ksel[ksel["min_cluster"] >= 10]
    ok = ok if len(ok) else ksel
    # largest k whose cophenetic stays within 0.02 of the best (favours resolution without losing stability)
    best = ok["cophenetic"].max()
    k = int(ok.loc[ok["cophenetic"] >= best - 0.02, "k"].max())
    log(f"  selected k = {k}")
    m = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=5000)
    W = m.fit_transform(Xs)
    H = pd.DataFrame(m.components_, columns=tme.columns, index=[f"E{i+1}" for i in range(k)])
    Wn = pd.DataFrame(W / (W.sum(1, keepdims=True) + 1e-12), index=tme.index, columns=H.index)
    assign = Wn.idxmax(1)
    # order ecotypes by total immune share of their top states (readability only)
    out = Wn.add_prefix("w_").assign(ecotype=assign,
                                     confidence=Wn.max(1),
                                     tumour_fraction=ty.get("Tumour", pd.Series(np.nan, index=ty.index)).reindex(tme.index),
                                     residual_testis=ty[[c for c in ty.columns if c in NORMAL_TESTIS_TYPES]].sum(1).reindex(tme.index))
    out.to_csv(f)
    H.T.to_csv(OUT / "S5_ecotype_state_loadings.csv")
    tme.to_csv(OUT / "S5_tme_state_composition.csv")
    top = {e: H.loc[e].sort_values(ascending=False).head(6).round(3).to_dict() for e in H.index}
    json.dump(top, open(OUT / "S5_ecotype_top_states.json", "w"), indent=1)
    # stability: consensus within assigned ecotype
    C = cons[k]
    lab = assign.map({e: i for i, e in enumerate(H.index)}).values
    stab = {e: float(C[np.ix_(lab == i, lab == i)].mean()) for i, e in enumerate(H.index) if (lab == i).sum() > 1}
    json.dump(stab, open(OUT / "S5_ecotype_stability.json", "w"), indent=1)
    return out


def step5b_clinical_check(eco):
    if not CLIN.exists():
        return {}
    cl = pd.read_csv(CLIN, index_col=0)
    cl.index = cl.index.str[:15]
    d = eco.join(cl[["histology", "stage_group", "PFI", "PFI.time"]], how="left")
    res = {}
    for v in ("histology", "stage_group"):
        t = pd.crosstab(d["ecotype"], d[v].fillna("NA"))
        t.to_csv(OUT / f"S5_ecotype_x_{v}.csv")
        if t.shape[0] > 1 and t.shape[1] > 1:
            chi2, p, dof, _ = chi2_contingency(t)
            n = t.values.sum()
            res[v] = dict(chi2_p=float(p), cramers_v=float(np.sqrt(chi2 / (n * (min(t.shape) - 1)))))
    # crude event table only (formal modelling is Stage 3, pre-specified)
    if "PFI" in d:
        res["PFI_events_by_ecotype"] = d.groupby("ecotype")["PFI"].agg(
            lambda x: f"{int((x == 1).sum())}/{int(x.notna().sum())}").to_dict()
    json.dump(res, open(OUT / "S5_ecotype_clinical_check.json", "w"), indent=1, default=str)
    return res



# ============================================================ STEP 5c  histology-adjusted ecotypes (main analysis)
def tumour_state_histology():
    """SE-likeness of the tumour compartment from deconvolution; checks against pathology, fills missing histology"""
    st = pd.read_csv(OUT / "tcga_state_fractions.csv", index_col=0)
    cl = pd.read_csv(CLIN, index_col=0)
    cl.index = cl.index.str[:15]
    hist = cl["histology"].reindex(st.index)
    res = {}
    if {"Tumour_SE_like", "Tumour_EC_like"} <= set(st.columns):
        se = st["Tumour_SE_like"] / (st["Tumour_SE_like"] + st["Tumour_EC_like"] + 1e-12)
        known = hist.isin(["SE", "NSGCT"])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score((hist[known] == "SE").astype(int), se[known])
        res["AUC_SE_likeness_vs_pathology"] = round(float(auc), 3)
        inferred = np.where(se > 0.5, "SE", "NSGCT")
        res["agreement_at_0.5"] = round(float((inferred[known.values] == hist[known].values).mean()), 3)
        fill = ~known
        hist = hist.where(known, pd.Series(inferred, index=st.index))
        res["histology_inferred_for"] = int(fill.sum())
        pd.DataFrame({"SE_likeness": se, "histology_pathology": cl["histology"].reindex(st.index),
                      "histology_used": hist}).to_csv(OUT / "S5c_tumour_SE_likeness.csv")
        log(f"  tumour SE-likeness vs pathology: AUC={auc:.3f}; histology inferred for {int(fill.sum())} tumours")
    else:
        hist = hist.fillna("unknown")
    return hist, res


def step5c_adjusted_ecotypes():
    f = OUT / "S5c_adjusted_ecotypes.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0), json.load(open(OUT / "S5c_summary.json"))
    from sklearn.decomposition import NMF
    from sklearn.metrics import silhouette_score
    tme = pd.read_csv(OUT / "S5_tme_state_composition.csv", index_col=0)
    hist, summ = tumour_state_histology()
    hist = hist.reindex(tme.index)
    # within-histology percentile of every state: removes the SE-vs-NSGCT main effect, keeps non-negativity
    P = tme.groupby(hist).rank(pct=True)
    P = P.loc[:, tme.std() > 0]
    Xs = P.values
    rows, cons = [], {}
    for k in range(2, 7):
        C, coph, Z = consensus_nmf(Xs, k)
        lab = fcluster(Z, k, "maxclust")
        sizes = np.bincount(lab)[1:]
        sil = silhouette_score(1 - C, lab, metric="precomputed")
        rows.append(dict(k=k, cophenetic=coph, silhouette=sil, min_cluster=int(sizes.min())))
        cons[k] = C
        log(f"  [adjusted] k={k}: cophenetic={coph:.3f} silhouette={sil:.3f} min size={sizes.min()}")
    ksel = pd.DataFrame(rows)
    ksel.to_csv(OUT / "S5c_k_selection.csv", index=False)
    ok = ksel[ksel["min_cluster"] >= 15]
    ok = ok if len(ok) else ksel
    k = int(ok.loc[ok["cophenetic"] >= ok["cophenetic"].max() - 0.02, "k"].max())
    m = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=5000)
    W = m.fit_transform(Xs)
    names = [f"TME{i+1}" for i in range(k)]
    H = pd.DataFrame(m.components_, columns=P.columns, index=names)
    Wd = pd.DataFrame(W, index=P.index, columns=names)
    Wn = Wd.div(Wd.sum(1) + 1e-12, axis=0)
    out = Wd.add_prefix("score_").join(Wn.add_prefix("w_"))
    out["ecotype_adj"] = Wn.idxmax(1)
    out["histology_used"] = hist
    out.to_csv(f)
    H.T.to_csv(OUT / "S5c_programme_state_loadings.csv")
    top = {e: H.loc[e].sort_values(ascending=False).head(8).round(3).to_dict() for e in names}
    t = pd.crosstab(out["ecotype_adj"], out["histology_used"])
    t.to_csv(OUT / "S5c_ecotype_x_histology.csv")
    chi2, p, dof, _ = chi2_contingency(t)
    summ.update(selected_k=k, top_states=top, cramers_v_histology=float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))),
                chi2_p_histology=float(p))
    cl = pd.read_csv(CLIN, index_col=0); cl.index = cl.index.str[:15]
    d = out.join(cl[["PFI", "stage_group"]], how="left")
    summ["PFI_events_by_ecotype_and_histology"] = d.groupby(["histology_used", "ecotype_adj"])["PFI"].agg(
        lambda x: f"{int((x == 1).sum())}/{int(x.notna().sum())}").unstack().to_dict()
    json.dump(summ, open(OUT / "S5c_summary.json", "w"), indent=1, default=str)
    log(f"  [adjusted] selected k={k}; Cramer's V vs histology = {summ['cramers_v_histology']:.2f}")
    return out, summ

# ============================================================ STEP 6  signature scores
def singscore(expr_log: pd.DataFrame, genes):
    """rank-based single-sample score (Foroutan et al. 2018), centred: 0 = random gene set.
    expr_log: samples x genes"""
    g = [x for x in genes if x in expr_log.columns]
    if len(g) < 5:
        return pd.Series(np.nan, index=expr_log.index), len(g)
    R = expr_log.rank(axis=1, method="average")
    n = expr_log.shape[1]
    s = R[g].mean(1)
    lo, hi = (len(g) + 1) / 2, n - (len(g) - 1) / 2
    return (s - lo) / (hi - lo) - 0.5, len(g)


def step6_signatures(bulk):
    f = OUT / "S6_tcga_state_signature_scores.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    sig = json.load(open(S1 / "S8_state_signatures_top50.json"))
    cpm = bulk.div(bulk.sum(1), axis=0) * 1e6
    logx = np.log2(cpm + 1)
    logx = logx.loc[:, (cpm > 1).mean() > 0.2]
    scores, ng = {}, {}
    for st, genes in sig.items():
        scores[st], ng[st] = singscore(logx, genes)
    S = pd.DataFrame(scores)
    S.to_csv(f)
    st_frac = pd.read_csv(OUT / "tcga_state_fractions.csv", index_col=0)
    rows = []
    for s in S.columns.intersection(st_frac.columns):
        if S[s].notna().sum() > 10 and st_frac[s].std() > 0:
            rho, p = spearmanr(S[s], st_frac.loc[S.index, s])
            rows.append(dict(state=s, n_genes=ng[s], spearman_rho=rho, p=p))
    agr = pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)
    agr.to_csv(OUT / "S6_signature_vs_bayesprism.csv", index=False)
    log(f"  signature vs deconvolution agreement: median rho = {agr['spearman_rho'].median():.2f} "
        f"({(agr['spearman_rho'] > 0.3).sum()}/{len(agr)} states rho > 0.3)")
    return S


# ============================================================ RUN
if __name__ == "__main__":
    t0 = time.time()
    log("STEP 1 TCGA bulk");     bulk = step1_tcga()
    log("STEP 2 reference");     meta = step2_reference()
    log("STEP 3 deconvolution TCGA")
    run_deconv(bulk, "tcga")
    log("STEP 4 benchmark");     bench = step4_benchmark(meta)
    log("STEP 5 ecotypes");      eco = step5_ecotypes(); clin = step5b_clinical_check(eco)
    log("STEP 5c histology-adjusted TME programmes"); eco_adj, adj = step5c_adjusted_ecotypes()
    log("STEP 6 signatures");    sig = step6_signatures(bulk)

    ty = pd.read_csv(OUT / "tcga_type_fractions.csv", index_col=0)
    md = ["# STAGE 2 report - TCGA tumour-immune ecosystem", "",
          f"- TCGA primary tumours: **{bulk.shape[0]}**; reference cells: {len(meta)}; "
          f"types: {meta['cell_type'].nunique()}; states: {meta['cell_state'].nunique()}",
          f"- runtime: {(time.time() - t0) / 60:.1f} min", "",
          "## Mean cell-type fractions in TCGA", ty.mean().sort_values(ascending=False).round(3).to_frame("mean").to_markdown(), ""]
    if bench is not None:
        md += ["## Benchmark (leave-one-dataset-out pseudobulk)", bench.round(3).to_markdown(index=False), ""]
    md += ["## Ecotype k selection", pd.read_csv(OUT / "S5_k_selection.csv").round(3).to_markdown(index=False), "",
           "## Ecotype sizes", eco["ecotype"].value_counts().to_frame("n").to_markdown(), "",
           "## Ecotype top states", "```json", json.dumps(json.load(open(OUT / "S5_ecotype_top_states.json")), indent=1), "```",
           "## Ecotype stability (mean consensus within ecotype)", "```json",
           json.dumps(json.load(open(OUT / "S5_ecotype_stability.json")), indent=1), "```",
           "## Ecotype vs clinical (descriptive only)", "```json", json.dumps(clin, indent=1, default=str), "```"]
    for v in ("histology", "stage_group"):
        fv = OUT / f"S5_ecotype_x_{v}.csv"
        if fv.exists():
            md += [f"### ecotype x {v}", pd.read_csv(fv).to_markdown(index=False), ""]
    md += ["## MAIN: histology-adjusted TME programmes (within-histology percentiles)",
           pd.read_csv(OUT / "S5c_k_selection.csv").round(3).to_markdown(index=False), "",
           pd.read_csv(OUT / "S5c_ecotype_x_histology.csv").to_markdown(index=False), "",
           "```json", json.dumps(adj, indent=1, default=str), "```", ""]
    fa = OUT / "S6_signature_vs_bayesprism.csv"
    if fa.exists():
        md += ["## Signature score vs deconvolved fraction (Spearman)", pd.read_csv(fa).round(3).to_markdown(index=False), ""]
    md += ["## Log", "```", *LOG, "```"]
    (OUT / "STAGE2_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log(f"done -> {OUT / 'STAGE2_REPORT.md'}")
