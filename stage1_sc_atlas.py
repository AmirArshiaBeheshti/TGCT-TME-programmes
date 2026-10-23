#!/usr/bin/env python
"""
STAGE 1 / 4  -  Single-cell atlas of TGCT: primary tumour -> nodal metastasis -> blood, + normal testis
==========================================================================================================
Inputs (all already on disk; nothing is downloaded except a cached gene-position table)
  GSE228501  4 primary seminomas                     BASE/GSE228501_RAW (1)/GSM*/<sample>/
  GSE261811  2 SE + 1 EC + 1 mixed primary            PROJ/data/raw/GSE261811/GSE261811_RAW_extracted/
  GSE197778  1 patient: primary, LN1, LN2, PBMC       PROJ/data/raw/GSE197778/GSE197778_RAW_extracted/
  GSE112013  normal adult testis (h5ad)               BASE/GSE112013/*.h5ad
  GSE182786  normal testis, 12 donors (h5ad)          BASE/step4_aging/h5ad/GSE182786_raw_counts.h5ad

Steps (each step writes a checkpoint; re-running skips finished steps; delete a checkpoint to redo it)
  1  load raw counts, harmonise genes, attach sample metadata
  2  tumour-aware per-sample QC (MAD) + Scrublet doublets + ambient (spermatid transcript) score
  3  HVG + scVI integration (GPU, 4 GB safe)
  4  major-lineage annotation (marker scores -> cluster vote) refined by scANVI
  5  malignant-cell confirmation: inferCNV (infercnvpy) + chromosome-12p gain score
  6  fine cell states per compartment (T/NK, myeloid, B/plasma, stroma)
  7  site comparison tables (primary / LN / blood / normal), LN1 'EC-like' check
  8  exports for Stage 2: deconvolution reference (h5ad), state signatures, pseudobulk

Run:  conda activate scvi-env ; python stage1_sc_atlas.py
Needs: scanpy>=1.10, scvi-tools>=1.1, infercnvpy, anndata, scipy, matplotlib
"""
import os, io, re, json, gzip, time, zlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Variable names are not unique")
sc.settings.verbosity = 1
SEED = 0
np.random.seed(SEED)

# ============================================================ config
BASE = Path(os.environ.get("TGCT_BASE", "/home/arshia/testis project"))
PROJ = BASE / "TGCT_ecotype"
RAW = PROJ / "data" / "raw"
OUT = PROJ / "stage1_atlas"
FIG = OUT / "figures"
TAB = OUT / "tables"
CKP = OUT / "checkpoints"
for d in (OUT, FIG, TAB, CKP):
    d.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = FIG

SC_ROOTS = {
    "GSE228501": BASE / "GSE228501_RAW (1)",
    "GSE261811": RAW / "GSE261811" / "GSE261811_RAW_extracted",
    "GSE197778": RAW / "GSE197778" / "GSE197778_RAW_extracted",
}
NORMAL_H5AD = {
    "GSE112013": sorted((BASE / "GSE112013").glob("*.h5ad")),
    "GSE182786": [BASE / "step4_aging" / "h5ad" / "GSE182786_raw_counts.h5ad"],
}
# sample-name keyword -> metadata   (patient ids are project-internal)
SAMPLE_META = {
    "19355X1": dict(dataset="GSE228501", patient="L1", site="primary", histology="SE"),
    "19355X2": dict(dataset="GSE228501", patient="L2", site="primary", histology="SE"),
    "19355X3": dict(dataset="GSE228501", patient="L3", site="primary", histology="SE"),
    "19355X4": dict(dataset="GSE228501", patient="L4", site="primary", histology="SE"),
    "Tumor2EC": dict(dataset="GSE261811", patient="I2", site="primary", histology="EC"),
    "Tumor4SE": dict(dataset="GSE261811", patient="I4", site="primary", histology="SE"),
    "Tumor12MIX": dict(dataset="GSE261811", patient="I12", site="primary", histology="MIX"),
    "Tumor13SE": dict(dataset="GSE261811", patient="I13", site="primary", histology="SE"),
    "tumor": dict(dataset="GSE197778", patient="M1", site="primary", histology="SE"),
    "LN1": dict(dataset="GSE197778", patient="M1", site="LN", histology="SE"),
    "LN2": dict(dataset="GSE197778", patient="M1", site="LN", histology="SE"),
    "PBMC": dict(dataset="GSE197778", patient="M1", site="blood", histology="SE"),
}
KEEP_INTERMEDIATE = False       # False: each new checkpoint deletes the previous ones (low disk space)
MIN_CELLS_SAMPLE = 200          # samples below this after QC are kept in the object but flagged
N_HVG = 3000
SCVI_KW = dict(n_latent=30, n_layers=2, gene_likelihood="nb")
TRAIN_KW = dict(max_epochs=None, early_stopping=True, batch_size=1024, accelerator="auto")
SCANVI_EPOCHS = 30
if os.environ.get("TGCT_TEST"):                       # quick smoke test only
    TRAIN_KW["max_epochs"], SCANVI_EPOCHS = 5, 3

# marker panels -------------------------------------------------------------------
MAJOR = {
    "Tumour_GCT":   ["POU5F1", "NANOG", "TFAP2C", "PRDM1", "SOX17", "KIT", "DPPA3", "LIN28A"],
    "T_NK":         ["CD3D", "CD3E", "CD2", "TRAC", "NKG7", "GNLY", "KLRD1"],
    "B":            ["MS4A1", "CD79A", "CD19", "BANK1"],
    "Plasma":       ["JCHAIN", "MZB1", "IGHG1", "XBP1", "SDC1"],
    "Myeloid":      ["LYZ", "CD68", "CD14", "C1QA", "C1QB", "CSF1R", "FCGR3A", "AIF1"],
    "DC":           ["CLEC9A", "XCR1", "CD1C", "FCER1A", "CLEC10A", "LAMP3"],   # no CCR7: marks naive T/B
    "pDC":          ["LILRA4", "IL3RA", "CLEC4C", "IRF7"],
    "Mast":         ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "HDC"],                # no KIT: seminoma marker
    "Neutrophil":   ["CSF3R", "FCGR3B", "CXCR2", "S100A8"],
    "Endothelial":  ["PECAM1", "VWF", "CDH5", "PLVAP"],
    "Fibroblast":   ["DCN", "LUM", "COL1A1", "PDGFRA"],
    "Myoid_SMC":    ["ACTA2", "MYH11", "DES", "CNN1"],
    "Leydig":       ["CYP17A1", "STAR", "INSL3", "CYP11A1", "HSD3B2"],
    "Sertoli":      ["SOX9", "AMH", "WT1", "FATE1", "CLU"],
    "Germ_normal":  ["DDX4", "DAZL", "UTF1", "ID4", "SYCP3", "PRM1", "TNP1"],
    "Erythrocyte":  ["HBB", "HBA1", "HBA2"],
    "Platelet":     ["PPBP", "PF4"],
}
COMPARTMENT = {
    "T_NK": "T_NK", "B": "B_plasma", "Plasma": "B_plasma", "Myeloid": "Myeloid", "DC": "Myeloid",
    "pDC": "Myeloid", "Mast": "Myeloid", "Neutrophil": "Myeloid", "Endothelial": "Stroma",
    "Fibroblast": "Stroma", "Myoid_SMC": "Stroma", "Leydig": "Stroma", "Sertoli": "Stroma",
}
ALLOWED = {"T_NK": {"T_NK"}, "Myeloid": {"Myeloid", "DC", "pDC", "Mast", "Neutrophil"},
           "B_plasma": {"B", "Plasma"}, "Stroma": {"Endothelial", "Fibroblast", "Myoid_SMC", "Leydig", "Sertoli"}}
# continuous programmes stored per cell for Stage 2/3 (IFNG+ T, exhaustion, M2 ...)
PROGRAMS = {
    "prog_IFNG_effector": ["IFNG", "TNF", "CCL4", "CCL3", "XCL1", "CRTAM"],
    "prog_T_exhaustion":  ["PDCD1", "HAVCR2", "LAG3", "TOX", "ENTPD1", "LAYN", "CXCL13"],
    "prog_cytotoxic":     ["GZMB", "PRF1", "GNLY", "NKG7", "GZMA"],
    "prog_M2_CD163":      ["CD163", "MRC1", "F13A1", "STAB1", "LYVE1"],
    "prog_ISG":           ["ISG15", "IFI6", "MX1", "IFIT3", "CXCL10"],
}
FINE = {
    "T_NK": {
        "CD4_naive_CM":   ["CCR7", "SELL", "LEF1", "TCF7", "CD4"],
        "CD4_Tfh_CXCL13": ["CXCL13", "PDCD1", "ICOS", "MAF", "TOX2"],
        "Treg":           ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18"],
        "CD8_Tem_GZMK":   ["CD8A", "GZMK", "CCL5", "CMC1", "EOMES"],
        "CD8_cytotoxic":  ["CD8A", "GZMB", "PRF1", "GNLY", "FGFBP2"],
        "CD8_Tex":        ["CD8A", "PDCD1", "HAVCR2", "LAG3", "TOX", "ENTPD1", "LAYN"],
        "T_IFNG_effector": ["IFNG", "TNF", "CCL4", "CCL3", "XCL1", "CRTAM"],
        "T_ISG":          ["ISG15", "IFI6", "MX1", "IFIT3"],
        "T_prolif":       ["MKI67", "TOP2A", "STMN1", "TYMS"],
        "NK":             ["KLRD1", "NCAM1", "FCGR3A", "KLRF1", "TYROBP"],
        "MAIT_gdT":       ["SLC4A10", "KLRB1", "TRDC", "TRGC1", "ZBTB16"],
    },
    "Myeloid": {
        "Mono_classical": ["CD14", "FCN1", "S100A8", "VCAN", "LYZ"],
        "Mono_nonclassical": ["FCGR3A", "CDKN1C", "LST1", "MS4A7"],
        "Macro_C1Q":      ["C1QA", "C1QB", "C1QC", "APOE", "SELENOP"],
        "Macro_CD163_M2": ["CD163", "MRC1", "LYVE1", "F13A1", "STAB1"],
        "Macro_SPP1":     ["SPP1", "APOC1", "GPNMB", "TREM2", "FABP5"],
        "Macro_inflam":   ["IL1B", "CXCL8", "CCL3", "TNF", "NLRP3"],
        "Macro_ISG_CXCL10": ["CXCL9", "CXCL10", "GBP1", "STAT1", "IDO1"],
        "cDC1":           ["CLEC9A", "XCR1", "CADM1", "IRF8"],
        "cDC2":           ["CD1C", "FCER1A", "CLEC10A", "CD1E"],
        "mregDC_LAMP3":   ["LAMP3", "FSCN1", "CCL19", "CCL22", "IDO1", "CD274"],
        "pDC":            ["LILRA4", "IL3RA", "CLEC4C", "GZMB"],
        "Mast":           ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "HDC"],
        "Neutrophil":     ["CSF3R", "FCGR3B", "CXCR2", "G0S2"],
    },
    "B_plasma": {
        "B_naive":        ["MS4A1", "IGHD", "TCL1A", "FCER2"],
        "B_memory":       ["MS4A1", "CD27", "AIM2", "TNFRSF13B"],
        "B_GC":           ["RGS13", "LMO2", "MEF2B", "AICDA", "BCL6"],
        "Plasma":         ["JCHAIN", "MZB1", "IGHG1", "PRDM1", "XBP1"],
    },
    "Stroma": {
        "Endo_vascular":  ["PECAM1", "VWF", "ACKR1", "SELE"],
        "Endo_HEV":       ["CHST4", "GLYCAM1", "CCL21", "ACKR1"],
        "Endo_lymphatic": ["PROX1", "LYVE1", "CCL21", "TFF3"],
        "Fibro_ECM":      ["COL1A1", "COL3A1", "POSTN", "FAP", "LUM"],
        "Fibro_inflam":   ["CXCL12", "C3", "CFD", "IL6", "CCL2"],
        "Myoid_peritubular": ["ACTA2", "MYH11", "DES", "CNN1"],
        "Pericyte":       ["RGS5", "PDGFRB", "KCNJ8", "NOTCH3"],
        "Leydig":         ["CYP17A1", "STAR", "INSL3", "DLK1"],
        "Sertoli":        ["SOX9", "AMH", "FATE1", "CLU"],
    },
}

LOG = []


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    LOG.append(s)


def ck(name):
    return CKP / f"{name}.h5ad"


import hashlib
ANNOT_VERSION = "v3_knn_sitepriors"
PANEL_HASH = hashlib.md5(json.dumps([MAJOR, FINE, PROGRAMS, ANNOT_VERSION], sort_keys=True).encode()).hexdigest()[:10]
ANNOT_STEPS = ("s4_major", "s5_malig", "s6_fine")


def load_ck(name):
    """load a checkpoint; a corrupt/partial file (e.g. kernel killed while writing) is removed and the step re-run.
    annotation checkpoints made with different marker panels are ignored (re-computed)."""
    f = ck(name)
    if not f.exists():
        return None
    try:
        a = sc.read_h5ad(f)
        if name in ANNOT_STEPS and a.uns.get("panel_hash") != PANEL_HASH:
            log(f"  checkpoint {f.name} built with older marker panels -> re-computing")
            return None
        log(f"  checkpoint loaded: {f.name} ({a.n_obs} cells)")
        return a
    except Exception as e:
        log(f"  !! checkpoint {f.name} unreadable ({type(e).__name__}: {str(e)[:80]}) -> deleted, step re-runs")
        f.unlink()
        return None


def save_ck(adata, name):
    """atomic write: a crash during writing never leaves a half-written checkpoint"""
    f = ck(name)
    adata.uns["panel_hash"] = PANEL_HASH
    tmp = f.with_suffix(".tmp.h5ad")
    adata.write_h5ad(tmp, compression="gzip")
    os.replace(tmp, f)
    if not KEEP_INTERMEDIATE:
        order = ["s1_raw", "s2_qc", "s3_scvi", "s4_major", "s5_malig", "s6_fine"]
        for old in order[:order.index(name)]:
            if old == "s3_scvi" and not (OUT / "TGCT_atlas_stage1.h5ad").exists():
                continue                               # keep scVI result until an atlas exists
            if ck(old).exists():
                ck(old).unlink()
                log(f"  removed old checkpoint {old}.h5ad (KEEP_INTERMEDIATE=False)")


# ============================================================ helpers: reading
def _junk(p: Path):
    """macOS AppleDouble files ('._name', magic 0x00051607) and __MACOSX folders shipped inside some GEO tars"""
    return p.name.startswith("._") or "__MACOSX" in p.parts or p.name.startswith(".")


def _first(dirpath: Path, pats):
    for p in pats:
        hits = sorted(h for h in dirpath.glob(p) if not _junk(h) and h.is_file())
        if hits:
            return hits[0]
    return None


def read_bytes_robust(path: Path) -> io.BytesIO:
    """gzip-tolerant reader: handles trailing padding/garbage after the gzip stream
    (seen in some GEO tar extractions), concatenated members, and '.gz' files that are plain text"""
    raw = Path(path).read_bytes()
    if raw[:2] != b"\x1f\x8b":
        return io.BytesIO(raw)
    out, data = [], raw
    while data[:2] == b"\x1f\x8b":
        dec = zlib.decompressobj(wbits=31)
        out.append(dec.decompress(data))
        out.append(dec.flush())
        data = dec.unused_data.lstrip(b"\x00")        # next member, or zero padding / junk
    if data:
        log(f"    note: {len(data)} trailing bytes ignored in {Path(path).name}")
    return io.BytesIO(b"".join(out))


def read_mtx_dir(d: Path):
    """reads 10x-style triplet with any file prefix (matrix.mtx[.gz], barcodes, features|genes)"""
    m = _first(d, ["*matrix.mtx.gz", "*matrix.mtx", "*.mtx.gz", "*.mtx"])
    b = _first(d, ["*barcodes.tsv.gz", "*barcodes.tsv", "*barcodes*.gz"])
    f = _first(d, ["*features.tsv.gz", "*features.tsv", "*genes.tsv.gz", "*genes.tsv"])
    if not (m and b and f):
        return None
    X = sp.csr_matrix(scipy.io.mmread(read_bytes_robust(m)))
    bc = pd.read_csv(read_bytes_robust(b), header=None, sep="\t")[0].astype(str).values
    ft = pd.read_csv(read_bytes_robust(f), header=None, sep="\t")
    sym = ft[1].astype(str).values if ft.shape[1] > 1 else ft[0].astype(str).values
    ens = ft[0].astype(str).values
    if X.shape == (len(ft), len(bc)):
        X = X.T.tocsr()
    a = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=bc), var=pd.DataFrame({"gene_ids": ens}, index=sym))
    if ft.shape[1] > 2:                                   # keep gene expression only (drop ADT/CRISPR)
        a = a[:, ft[2].astype(str).str.contains("Gene", case=False).values].copy()
    a.var_names_make_unique()
    return a


def find_10x_dirs(root: Path):
    dirs = set()
    for m in root.rglob("*.mtx*"):
        if _junk(m):
            continue
        if "extracted" in str(m) or re.search(r"GSM\d+", str(m)):
            dirs.add(m.parent)
    return sorted(dirs)


def meta_for(path: Path):
    s = str(path)
    for key in sorted(SAMPLE_META, key=len, reverse=True):       # longest key first ('tumor' vs 'Tumor13SE')
        if re.search(rf"(^|[_/]){re.escape(key)}([_/.]|$)", s):
            return key, SAMPLE_META[key]
    return None, None


def counts_from_h5ad(a: ad.AnnData):
    """return AnnData with integer counts in X, gene symbols as var_names"""
    X = None
    for src in ("layers:counts", "raw", "X"):
        if src == "layers:counts" and "counts" in a.layers:
            X = a.layers["counts"]
        elif src == "raw" and a.raw is not None and a.raw.shape[0] == a.n_obs:
            X, var = a.raw.X, a.raw.var
            a = ad.AnnData(X=X, obs=a.obs, var=var)
        elif src == "X":
            X = a.X
        if X is not None:
            v = X[:200].toarray() if sp.issparse(X) else np.asarray(X[:200])
            if np.allclose(v, np.round(v)) and v.max() > 20:
                break
            X = None
    if X is None:
        raise ValueError("no integer count matrix found")
    b = ad.AnnData(X=sp.csr_matrix(X, dtype=np.float32), obs=a.obs.copy(), var=a.var.copy())
    if b.var_names.str.startswith("ENSG").mean() > 0.5:
        sym_col = next((c for c in b.var.columns if c.lower() in ("gene_symbols", "feature_name", "symbol", "gene_name", "gene_symbol")), None)
        if sym_col:
            b.var["gene_ids"] = b.var_names
            b.var_names = b.var[sym_col].astype(str).values
    b.var_names_make_unique()
    return b


def donor_column(obs: pd.DataFrame):
    for c in obs.columns:
        if re.search(r"donor|sample|patient|individual|orig\.ident|batch", c, re.I) and 1 < obs[c].nunique() <= 40:
            return c
    return None


# ============================================================ STEP 1  load
def step1_load():
    _a = load_ck("s1_raw")
    if _a is not None:
        return _a
    ads = []
    for ds, root in SC_ROOTS.items():
        if not root.exists():
            log(f"  !! missing {root}")
            continue
        for d in find_10x_dirs(root):
            key, meta = meta_for(d)
            if key is None or meta["dataset"] != ds:
                continue
            try:
                a = read_mtx_dir(d)
            except Exception as e:
                raise SystemExit(f"\n!! cannot read 10x files in {d}\n   {type(e).__name__}: {e}\n"
                                 f"   test with:  zcat '<file>.gz' | head   ->  if it fails, re-extract that GSM tar") from e
            if a is None:
                continue
            a.obs["sample"] = f"{ds}_{key}"
            for k, v in meta.items():
                a.obs[k] = v
            a.obs_names = a.obs["sample"] + "|" + a.obs_names
            log(f"  {ds:10s} {key:11s} {a.n_obs:6d} cells  ({d})")
            ads.append(a)
    for ds, files in NORMAL_H5AD.items():
        for f in files:
            if not f.exists():
                log(f"  !! missing {f}")
                continue
            a = counts_from_h5ad(sc.read_h5ad(f))
            dc = donor_column(a.obs)
            donors = a.obs[dc].astype(str) if dc else pd.Series("D1", index=a.obs_names)
            a.obs = pd.DataFrame(index=a.obs_names)
            a.obs["sample"] = (ds + "_" + donors).values
            a.obs["dataset"], a.obs["patient"] = ds, (ds + "_" + donors).values
            a.obs["site"], a.obs["histology"] = "normal", "normal"
            a.obs_names = a.obs["sample"] + "|" + a.obs_names
            log(f"  {ds:10s} normal      {a.n_obs:6d} cells, {donors.nunique()} donors (column '{dc}')")
            ads.append(a)
    if not ads:
        raise SystemExit("no single-cell input found - check paths in config")
    adata = ad.concat(ads, join="outer", fill_value=0, merge="same")
    adata.obs_names_make_unique()
    adata.X = sp.csr_matrix(adata.X)
    # keep genes detected in >= 2 samples (harmonises different gene annotations)
    det = pd.DataFrame({s: np.asarray((adata[adata.obs["sample"] == s].X > 0).sum(0)).ravel() > 0
                        for s in adata.obs["sample"].unique()}, index=adata.var_names)
    adata = adata[:, det.sum(1) >= 2].copy()
    for c in ("sample", "dataset", "patient", "site", "histology"):
        adata.obs[c] = adata.obs[c].astype("category")
    adata.layers["counts"] = adata.X.copy()
    log(f"  merged: {adata.n_obs} cells x {adata.n_vars} genes, {adata.obs['sample'].nunique()} samples")
    save_ck(adata, "s1_raw")
    return adata


# ============================================================ STEP 2  QC
def mad_bounds(x, nlo=4, nhi=4):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return med - nlo * mad, med + nhi * mad


def step2_qc(adata):
    _a = load_ck("s2_qc")
    if _a is not None:
        return _a
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.match(r"^RP[SL]\d")
    adata.var["hb"] = adata.var_names.str.match(r"^HB[AB]\d?$")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=True, inplace=True)
    # tumour candidates: protected from the upper-count / MT filters (seminoma cells are large, MT-rich)
    tm = [g for g in ("POU5F1", "NANOG", "TFAP2C") if g in adata.var_names]
    tumour_cand = (np.asarray((adata[:, tm].X > 0).sum(1)).ravel() >= 2) if tm else np.zeros(adata.n_obs, bool)
    adata.obs["tumour_candidate"] = tumour_cand
    # spermatid transcripts are the dominant ambient contaminant in testis
    amb = [g for g in ("PRM1", "PRM2", "TNP1") if g in adata.var_names]
    lib = np.asarray(adata.X.sum(1)).ravel() + 1
    adata.obs["ambient_spermatid_frac"] = np.asarray(adata[:, amb].X.sum(1)).ravel() / lib if amb else 0.0

    keep = np.ones(adata.n_obs, bool)
    rows = []
    for s in adata.obs["sample"].cat.categories:
        idx = np.where(adata.obs["sample"] == s)[0]
        o = adata.obs.iloc[idx]
        lo_c, hi_c = mad_bounds(o["log1p_total_counts"].values, 4, 4)
        lo_g, _ = mad_bounds(o["log1p_n_genes_by_counts"].values, 4, 4)
        _, hi_mt = mad_bounds(o["pct_counts_mt"].values, 0, 3)
        mt_cap = max(15.0, min(hi_mt, 30.0))
        tc = o["tumour_candidate"].values
        k = (o["n_genes_by_counts"].values >= 200) & (o["log1p_total_counts"].values >= lo_c) \
            & (o["log1p_n_genes_by_counts"].values >= lo_g) \
            & ((o["log1p_total_counts"].values <= hi_c) | tc) \
            & ((o["pct_counts_mt"].values <= mt_cap) | (tc & (o["pct_counts_mt"].values <= 40)))
        keep[idx] = k
        rows.append(dict(sample=s, cells_in=len(idx), cells_pass=int(k.sum()), mt_cap=round(mt_cap, 1),
                         min_counts=int(np.expm1(lo_c)), max_counts=int(np.expm1(hi_c)),
                         tumour_candidates=int(tc.sum()), tumour_cand_kept=int((tc & k).sum()),
                         median_genes=int(o["n_genes_by_counts"].median()),
                         ambient_spermatid_median=round(float(o["ambient_spermatid_frac"].median()), 4)))
    adata = adata[keep].copy()
    # doublets per sample (Scrublet on raw counts)
    adata.X = adata.layers["counts"].copy()
    try:
        sc.pp.scrublet(adata, batch_key="sample", random_state=SEED)
    except Exception as e:                                   # very small samples
        log(f"  scrublet batch run failed ({e}); per-sample fallback")
        adata.obs["predicted_doublet"] = False
        adata.obs["doublet_score"] = 0.0
        for s in adata.obs["sample"].cat.categories:
            m = (adata.obs["sample"] == s).values
            if m.sum() < 300:
                continue
            sub = adata[m].copy()
            sc.pp.scrublet(sub, random_state=SEED)
            adata.obs.loc[sub.obs_names, ["predicted_doublet", "doublet_score"]] = sub.obs[["predicted_doublet", "doublet_score"]].values
    qc = pd.DataFrame(rows).set_index("sample")
    qc["doublets_removed"] = adata.obs.groupby("sample", observed=False)["predicted_doublet"].sum().astype(int)
    adata = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()
    qc["cells_final"] = adata.obs["sample"].value_counts()
    qc["flag_small"] = qc["cells_final"] < MIN_CELLS_SAMPLE
    qc.to_csv(TAB / "S2_qc_per_sample.csv")
    log("  QC per sample:\n" + qc[["cells_in", "cells_final", "tumour_candidates", "tumour_cand_kept", "doublets_removed"]].to_string())
    save_ck(adata, "s2_qc")
    return adata


# ============================================================ STEP 3  scVI
def try_s3():
    """scVI-integrated object from the s3 checkpoint or, if that was deleted, from the final atlas"""
    _a = load_ck("s3_scvi")
    if _a is not None:
        return _a
    atlas = OUT / "TGCT_atlas_stage1.h5ad"
    if atlas.exists():
        a = sc.read_h5ad(atlas)
        if "X_scVI" in a.obsm and "counts" in a.layers and "lognorm" in a.layers:
            log("  s3 checkpoint missing -> reusing scVI latent / clustering from TGCT_atlas_stage1.h5ad")
            keep_obs = [c for c in a.obs.columns if not (c.startswith(("M_", "F_", "prog_", "sub_")) or c in (
                "major_vote", "seed", "major", "major_prob", "compartment", "cnv_score", "p12_score", "malignant", "state"))]
            a.obs = a.obs[keep_obs]
            for k in ("X_scANVI", "X_cnv"):
                a.obsm.pop(k, None)
            for k in list(a.uns):
                if k not in ("neighbors", "umap", "leiden"):
                    del a.uns[k]
            return a                                   # not re-saved (disk space)
    return None


def step3_scvi(adata):
    _a = try_s3()
    if _a is not None:
        return _a
    import scvi, torch
    scvi.settings.seed = SEED
    torch.set_float32_matmul_precision("medium")
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    excl = adata.var_names.str.match(r"^(MT-|RP[SL]\d|HB[AB]|MALAT1|NEAT1|XIST|PRM1|PRM2|TNP1)")
    tmp = adata[:, ~excl].copy()
    sc.pp.highly_variable_genes(tmp, flavor="seurat_v3", n_top_genes=N_HVG, layer="counts",
                                batch_key="sample", span=0.5)
    # always keep the marker genes used for annotation
    must = {g for panel in [MAJOR] + list(FINE.values()) for gl in panel.values() for g in gl}
    hv = set(tmp.var_names[tmp.var["highly_variable"]]) | (must & set(adata.var_names))
    adata.var["highly_variable"] = adata.var_names.isin(hv)
    log(f"  HVG: {adata.var['highly_variable'].sum()} (incl. markers)")
    bd = adata[:, adata.var["highly_variable"]].copy()
    scvi.model.SCVI.setup_anndata(bd, layer="counts", batch_key="sample",
                                  continuous_covariate_keys=["pct_counts_mt"])
    model = scvi.model.SCVI(bd, **SCVI_KW)
    model.train(**TRAIN_KW)
    model.save(str(OUT / "scvi_model"), overwrite=True)
    adata.obsm["X_scVI"] = model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2, random_state=SEED)
    sc.pl.umap(adata, color=["dataset", "site", "sample"], ncols=1, show=False, save="_S3_batches.png")
    save_ck(adata, "s3_scvi")
    return adata


# ============================================================ STEP 4  major annotation
def score_panels(adata, panels, prefix, layer="lognorm"):
    adata.X = adata.layers[layer]
    cols = []
    for name, genes in panels.items():
        g = [x for x in genes if x in adata.var_names]
        if len(g) < 2:
            continue
        sc.tl.score_genes(adata, g, score_name=f"{prefix}{name}", random_state=SEED, ctrl_size=min(50, adata.n_vars // 10))
        cols.append(f"{prefix}{name}")
    return cols


def cluster_vote(adata, cols, cluster_key, prefix, min_margin=0.05):
    z = adata.obs[cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    cl_mean = z.groupby(adata.obs[cluster_key], observed=False).mean()
    best = cl_mean.idxmax(1).str.replace(prefix, "", regex=False)
    srt = np.sort(cl_mean.values, axis=1)
    margin = pd.Series(srt[:, -1] - srt[:, -2], index=cl_mean.index)
    lab = best.where(margin > min_margin, "Unresolved")
    return lab, cl_mean, margin


def step4_annotate(adata):
    _a = load_ck("s4_major")
    if _a is not None:
        return _a
    cols = score_panels(adata, MAJOR, "M_")
    lab, cl_mean, margin = cluster_vote(adata, cols, "leiden", "M_")
    adata.obs["major_vote"] = adata.obs["leiden"].map(lab).astype("category")
    cl_mean.assign(label=lab, margin=margin).to_csv(TAB / "S4_cluster_marker_scores.csv")
    # cell-level seeds: own top score agrees with the cluster vote and is clearly positive
    z = adata.obs[cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    own = z.idxmax(1).str.replace("M_", "", regex=False)
    seed = np.where((own == adata.obs["major_vote"].astype(str)) & (z.max(1) > 0.5), own, "Unknown")
    adata.obs["seed"] = pd.Categorical(seed)
    # kNN label propagation on the scVI graph (deterministic; replaces scANVI, which mislabelled
    # blood/LN cells as Leydig/myoid when trained from scratch)
    if "connectivities" not in adata.obsp:
        sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30, random_state=SEED)
    labs = sorted(set(seed) - {"Unknown"})
    S = sp.csr_matrix((np.ones((seed != "Unknown").sum()),
                       (np.where(seed != "Unknown")[0], [labs.index(x) for x in seed[seed != "Unknown"]])),
                      shape=(adata.n_obs, len(labs)))
    C = adata.obsp["connectivities"].tocsr()
    K = np.asarray((C @ S).todense()) + S.toarray() * 2           # own seed label weighs in
    # biological site priors: testis-specific / tumour lineages cannot occur in blood; testis somatic+germ not in LN
    site = adata.obs["site"].astype(str).values
    forbid = {"blood": {"Tumour_GCT", "Germ_normal", "Leydig", "Sertoli", "Myoid_SMC", "Fibroblast", "Endothelial"},
              "LN": {"Germ_normal", "Leydig", "Sertoli", "Myoid_SMC"}}
    for st, bad in forbid.items():
        m = site == st
        for b in bad:
            if b in labs:
                K[m, labs.index(b)] = 0
    tot = K.sum(1)
    best = np.array(labs)[K.argmax(1)]
    frac = np.divide(K.max(1), tot, out=np.zeros_like(tot, dtype=float), where=tot > 0)
    vote = adata.obs["major_vote"].astype(str).values
    vote_ok = np.array([not (s_ in forbid and v in forbid[s_]) for s_, v in zip(site, vote)])
    major = np.where(tot > 0, best, np.where(vote_ok, vote, "Unresolved"))
    major = np.where((tot > 0) & (frac < 0.4) & vote_ok, vote, major)   # weak neighbourhood -> cluster vote
    adata.obs["major"] = pd.Categorical(major)
    adata.obs["major_prob"] = frac
    t = pd.crosstab(adata.obs["major_vote"], adata.obs["major"])
    t.to_csv(TAB / "S4_vote_vs_final.csv")
    log(f"  kNN refinement changed {int((major != vote).sum())} / {adata.n_obs} cell labels "
        f"(median neighbourhood agreement {np.median(frac):.2f})")
    adata.obs["major"] = adata.obs["major"].astype(str)
    adata.obs["major"] = adata.obs["major"].astype("category")
    adata.obs["compartment"] = adata.obs["major"].astype(str).map(lambda x: COMPARTMENT.get(x, x)).astype("category")
    sc.pl.umap(adata, color=["major", "major_vote", "site"], ncols=1, show=False, save="_S4_major.png")
    adata.X = adata.layers["lognorm"]
    sc.pl.dotplot(adata, {k: [g for g in v if g in adata.var_names][:5] for k, v in MAJOR.items()},
                  groupby="major", standard_scale="var", show=False, save="S4_major_markers.png")
    pd.crosstab(adata.obs["sample"], adata.obs["major"]).to_csv(TAB / "S4_major_counts_per_sample.csv")
    save_ck(adata, "s4_major")
    return adata


# ============================================================ STEP 5  malignant confirmation
def gene_positions():
    f = PROJ / "data" / "gene_positions_hg38.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    log("  fetching gene positions from Ensembl BioMart (cached once)")
    pos = sc.queries.biomart_annotations("hsapiens", ["external_gene_name", "chromosome_name",
                                                      "start_position", "end_position"], host="www.ensembl.org")
    pos.columns = ["gene", "chromosome", "start", "end"]
    pos = pos[pos["chromosome"].astype(str).str.match(r"^(\d+|X|Y)$")].drop_duplicates("gene")
    pos["chromosome"] = "chr" + pos["chromosome"].astype(str)
    pos.to_csv(f, sep="\t", index=False)
    return pos


def step5_malignant(adata):
    _a = load_ck("s5_malig")
    if _a is not None:
        return _a
    import infercnvpy as cnv
    pos = gene_positions().set_index("gene")
    g = adata.var_names.intersection(pos.index)
    adata.var.loc[g, "chromosome"] = pos.loc[g, "chromosome"].values
    adata.var.loc[g, "start"] = pos.loc[g, "start"].values
    adata.var.loc[g, "end"] = pos.loc[g, "end"].values
    # 12p (chr12: 0-35.5 Mb) relative expression; i(12p) is near-universal in TGCT
    p12 = adata.var_names[(adata.var.get("chromosome") == "chr12") & (adata.var["start"] < 35.5e6)]
    adata.obs["cnv_score"] = np.nan
    adata.obs["p12_score"] = np.nan
    ref_types = ["T_NK", "B", "Plasma", "Myeloid", "Endothelial", "Fibroblast"]
    for pt in adata.obs.loc[adata.obs["site"] != "normal", "patient"].unique():
        m = (adata.obs["patient"] == pt).values & (adata.obs["site"] != "blood").values
        sub = adata[m].copy()
        if (sub.obs["major"] == "Tumour_GCT").sum() < 10:
            log(f"  {pt}: <10 tumour-lineage cells, CNV skipped")
            continue
        sub.X = sub.layers["lognorm"]
        sub = sub[:, sub.var["chromosome"].notna()].copy()
        refcats = [r for r in ref_types if (sub.obs["major"] == r).sum() >= 20]
        cnv.tl.infercnv(sub, reference_key="major", reference_cat=refcats, window_size=150, step=10)
        cnv.tl.cnv_score(sub, groupby="leiden")
        # per-cell CNV burden (mean |signal|) is more robust than cluster score for small samples
        Xc = sub.obsm["X_cnv"]
        burden = np.asarray(np.abs(Xc).mean(1)).ravel() if sp.issparse(Xc) else np.abs(Xc).mean(1)
        adata.obs.loc[sub.obs_names, "cnv_score"] = burden
        refm = sub.obs["major"].isin(refcats).values
        e = sub[:, sub.var_names.isin(p12)].X
        e_all = sub.X
        mu = lambda M: np.asarray(M.mean(1)).ravel()
        d = mu(e) - mu(e_all)
        d = d - np.median(d[refm])
        adata.obs.loc[sub.obs_names, "p12_score"] = d
        log(f"  {pt}: CNV done on {sub.n_obs} cells (ref={refcats})")
    # call: tumour lineage AND (CNV burden or 12p gain above the 95th pct of the patient's reference cells)
    adata.obs["malignant"] = "non_malignant"
    for pt, o in adata.obs[adata.obs["cnv_score"].notna()].groupby("patient", observed=True):
        ref = o["major"].isin(ref_types)
        thr_c, thr_p = o.loc[ref, "cnv_score"].quantile(0.95), o.loc[ref, "p12_score"].quantile(0.95)
        tl = o["major"] == "Tumour_GCT"
        call = tl & ((o["cnv_score"] > thr_c) | (o["p12_score"] > thr_p))
        adata.obs.loc[o.index[call], "malignant"] = "malignant"
        adata.obs.loc[o.index[tl & ~call], "malignant"] = "tumour_lineage_CNVneg"
    adata.obs["malignant"] = adata.obs["malignant"].astype("category")
    t = pd.crosstab(adata.obs["sample"], adata.obs["malignant"])
    t.to_csv(TAB / "S5_malignant_per_sample.csv")
    log("  malignant calls:\n" + t.to_string())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    o = adata.obs[adata.obs["cnv_score"].notna()]
    for i, col in enumerate(["cnv_score", "p12_score"]):
        for lab, oo in o.groupby(o["major"].astype(str).where(o["major"] == "Tumour_GCT", "other")):
            ax[i].hist(oo[col], bins=60, alpha=.5, density=True, label=lab)
        ax[i].set_title(col); ax[i].legend()
    fig.savefig(FIG / "S5_cnv_scores.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    save_ck(adata, "s5_malig")
    return adata


# ============================================================ STEP 6  fine states
def step6_fine(adata):
    _a = load_ck("s6_fine")
    if _a is not None:
        return _a
    rep = "X_scANVI" if "X_scANVI" in adata.obsm else "X_scVI"
    adata.obs["state"] = adata.obs["major"].astype(str)
    adata.obs.loc[adata.obs["malignant"] == "malignant", "state"] = "Tumour_malignant"
    adata.obs.loc[adata.obs["malignant"] == "tumour_lineage_CNVneg", "state"] = "Tumour_CNVneg"
    for comp, panels in FINE.items():
        m = (adata.obs["compartment"] == comp).values
        if m.sum() < 100:
            continue
        sub = adata[m].copy()
        sc.pp.neighbors(sub, use_rep=rep, n_neighbors=20, random_state=SEED)
        sc.tl.leiden(sub, resolution=1.2, key_added="sub", flavor="igraph", n_iterations=2, random_state=SEED)
        cols = score_panels(sub, panels, "F_")
        lab, cl_mean, margin = cluster_vote(sub, cols, "sub", "F_", min_margin=0.03)
        lab = lab.where(lab != "Unresolved", comp + "_other")
        # purity guard: subcluster whose dominant MAJOR lineage is not of this compartment -> contaminant
        mcols = [c for c in sub.obs.columns if c.startswith("M_")]
        if mcols:
            zm = sub.obs[mcols].apply(lambda x: (x - adata.obs[x.name].mean()) / (adata.obs[x.name].std() + 1e-9))
            top_major = zm.groupby(sub.obs["sub"], observed=False).mean().idxmax(1).str.replace("M_", "", regex=False)
            bad = ~top_major.isin(ALLOWED[comp])
            lab = lab.where(~bad.reindex(lab.index, fill_value=False), comp + "_contaminant")
        # low absolute evidence (best fine score below 0.2 SD) -> other
        lab = lab.where(cl_mean.max(1).reindex(lab.index) > 0.2, comp + "_other")
        adata.obs.loc[sub.obs_names, "state"] = sub.obs["sub"].map(lab).astype(str).values
        adata.obs.loc[sub.obs_names, "sub_" + comp] = (comp + "_" + sub.obs["sub"].astype(str)).values
        cl_mean.assign(label=lab, margin=margin).to_csv(TAB / f"S6_{comp}_subcluster_scores.csv")
        sc.tl.umap(sub, random_state=SEED)
        sub.obs["state"] = adata.obs.loc[sub.obs_names, "state"].values
        sc.pl.umap(sub, color=["state", "site", "dataset"], ncols=1, show=False, save=f"_S6_{comp}.png")
        sub.X = sub.layers["lognorm"]
        sc.pl.dotplot(sub, {k: [g for g in v if g in sub.var_names][:5] for k, v in panels.items()},
                      groupby="state", standard_scale="var", show=False, save=f"S6_{comp}_markers.png")
        log(f"  {comp}: {m.sum()} cells -> {sub.obs['sub'].nunique()} subclusters, {lab.nunique()} states")
    adata.obs["state"] = adata.obs["state"].astype("category")
    adata.X = adata.layers["lognorm"]
    for name, genes in PROGRAMS.items():
        g = [x for x in genes if x in adata.var_names]
        if len(g) >= 3:
            sc.tl.score_genes(adata, g, score_name=name, random_state=SEED)
    save_ck(adata, "s6_fine")
    return adata


# ============================================================ STEP 7  site comparison
def plain_obs(adata):
    """obs copy with categoricals as plain strings (avoids pandas CategoricalIndex join/crosstab bugs)"""
    o = adata.obs.copy()
    for c in o.columns:
        if isinstance(o[c].dtype, pd.CategoricalDtype):
            o[c] = o[c].astype(str)
    return o


def step7_sites(adata):
    o = plain_obs(adata)
    tum = o[o["site"] != "normal"]
    # composition within immune compartment (denominator = all immune cells of the sample)
    imm = tum[tum["compartment"].isin(["T_NK", "Myeloid", "B_plasma"])]
    comp = pd.crosstab(imm["sample"], imm["state"], normalize="index")
    comp = comp.loc[:, comp.sum() > 0]
    meta = tum.groupby("sample", observed=True)[["dataset", "patient", "site", "histology"]].first()
    comp.join(meta).to_csv(TAB / "S7_immune_state_fractions_per_sample.csv")
    pd.crosstab(tum["sample"], tum["major"], normalize="index").join(meta).to_csv(TAB / "S7_major_fractions_per_sample.csv")
    # paired patient M1: primary vs LN1 vs LN2 vs PBMC
    m1 = imm[imm["patient"] == "M1"]
    if len(m1):
        pd.crosstab(m1["state"], m1["sample"], normalize="columns").round(4).to_csv(TAB / "S7_M1_primary_LN_blood_states.csv")
    # primary-site summary across patients (mean ± sd over samples)
    pr = comp.join(meta)
    pr = pr[pr["site"] == "primary"].drop(columns=["dataset", "patient", "site", "histology"])
    pr.agg(["mean", "std"]).T.sort_values("mean", ascending=False).to_csv(TAB / "S7_primary_state_mean_sd.csv")
    # LN1 'EC-like' check: SOX2 vs SOX17 inside malignant cells
    mal = adata[adata.obs["malignant"] == "malignant"]
    if mal.n_obs:
        rows = []
        for s in mal.obs["sample"].unique():
            x = mal[mal.obs["sample"] == s]
            f = lambda g: float((x[:, g].layers["counts"] > 0).mean()) if g in x.var_names else np.nan
            rows.append(dict(sample=s, n=x.n_obs, SOX17=f("SOX17"), KIT=f("KIT"), SOX2=f("SOX2"),
                             DNMT3B=f("DNMT3B"), AFP=f("AFP"), CGA=f("CGA")))
        pd.DataFrame(rows).to_csv(TAB / "S7_malignant_lineage_markers.csv", index=False)
    # stacked bar
    fig, ax = plt.subplots(figsize=(max(8, len(comp) * .5), 5))
    comp.plot.bar(stacked=True, ax=ax, width=.85, legend=False, colormap="tab20")
    ax.set_ylabel("fraction of immune cells")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, ncol=2)
    fig.savefig(FIG / "S7_immune_states_per_sample.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ============================================================ STEP 8  exports for Stage 2
def step8_export(adata, max_per_state=600):
    rng = np.random.default_rng(SEED)
    o = plain_obs(adata)
    keep_state = o["state"].value_counts()
    keep_state = keep_state[keep_state >= 30].index
    drop = {"Unresolved", "Erythrocyte", "Platelet", "Tumour_CNVneg"}
    lowconf = o["state"].str.endswith(("_other", "_contaminant"))
    use = o[o["state"].isin(keep_state) & ~o["state"].isin(drop) & ~lowconf & (o["site"] != "blood")]
    idx = []
    for (st, ds), oo in use.groupby(["state", "dataset"], observed=True):   # balance across datasets
        n = min(len(oo), max(50, max_per_state // use.loc[use["state"] == st, "dataset"].nunique()))
        idx += list(rng.choice(oo.index, n, replace=False))
    ref = adata[idx].copy()
    ref.X = ref.layers["counts"].copy()
    for k in list(ref.layers):
        if k != "counts":
            del ref.layers[k]
    ref.obs["cell_type"] = ref.obs["major"].astype(str).where(ref.obs["state"] != "Tumour_malignant", "Tumour_malignant")
    ref.obs["cell_state"] = ref.obs["state"].astype(str)
    ref.obs = ref.obs[["sample", "dataset", "patient", "site", "histology", "cell_type", "cell_state"]]
    ref.obsm = {}
    ref.write_h5ad(OUT / "S8_deconv_reference_counts.h5ad", compression="gzip")
    log(f"  deconvolution reference: {ref.n_obs} cells, {ref.obs['cell_state'].nunique()} states, "
        f"{ref.obs['cell_type'].nunique()} types")
    # state marker signatures (Wilcoxon vs rest, within full atlas, lognorm)
    adata.X = adata.layers["lognorm"]
    st = adata[adata.obs["state"].isin(keep_state) & ~adata.obs["state"].isin(drop)].copy()
    st.obs["state"] = st.obs["state"].astype(str).astype("category")      # drop unused categories
    sc.tl.rank_genes_groups(st, "state", method="wilcoxon", pts=True, use_raw=False)
    sig = {}
    for s in pd.Categorical(st.obs["state"]).categories:
        df = sc.get.rank_genes_groups_df(st, s)
        df = df[(df["logfoldchanges"] > 1) & (df["pvals_adj"] < 0.01) & (df["pct_nz_group"] > 0.25)]
        df = df[~df["names"].str.match(r"^(MT-|RP[SL]\d|HB[AB])")]
        sig[s] = df["names"].head(50).tolist()
    json.dump(sig, open(OUT / "S8_state_signatures_top50.json", "w"), indent=1)
    # pseudobulk counts per sample x state (for reference validation in Stage 2)
    o = plain_obs(adata)
    grp = (o["sample"].astype(str) + "::" + o["state"].astype(str)).values
    G = pd.get_dummies(grp, sparse=False).astype(np.float32)
    pb = sp.csr_matrix(G.values.T) @ adata.layers["counts"]
    pbdf = pd.DataFrame(pb.toarray() if sp.issparse(pb) else pb, index=G.columns, columns=adata.var_names)
    pbdf[pbdf.sum(1) > 0].to_parquet(OUT / "S8_pseudobulk_sample_state_counts.parquet")
    # sample-level pseudobulk (all cells) + true fractions: ground truth for deconvolution benchmark
    gs = pd.get_dummies(o["sample"].astype(str)).astype(np.float32)
    pbs = (sp.csr_matrix(gs.values.T) @ adata.layers["counts"])
    pd.DataFrame(pbs.toarray(), index=gs.columns, columns=adata.var_names).to_parquet(OUT / "S8_pseudobulk_sample_counts.parquet")
    pd.crosstab(o["sample"], o["state"], normalize="index").to_csv(TAB / "S8_true_state_fractions_per_sample.csv")


# ============================================================ RUN
if __name__ == "__main__":
    t0 = time.time()
    # resume from the latest valid point; raw data are only needed if nothing past QC exists
    adata, done = None, 0
    for i, name in ((6, "s6_fine"), (5, "s5_malig"), (4, "s4_major")):
        adata = load_ck(name)
        if adata is not None:
            done = i
            break
    if adata is None:
        adata = try_s3()
        done = 3 if adata is not None else 0
    if done < 3:
        log("STEP 1 load");        adata = step1_load()
        log("STEP 2 QC");          adata = step2_qc(adata)
        log("STEP 3 scVI");        adata = step3_scvi(adata)
    if done < 4:
        log("STEP 4 annotate");    adata = step4_annotate(adata)
    if done < 5:
        log("STEP 5 malignant");   adata = step5_malignant(adata)
    if done < 6:
        log("STEP 6 fine states"); adata = step6_fine(adata)
    log("STEP 7 sites");       step7_sites(adata)
    log("STEP 8 export");      step8_export(adata)
    tmp = OUT / "TGCT_atlas_stage1.tmp.h5ad"
    adata.write_h5ad(tmp, compression="gzip")
    os.replace(tmp, OUT / "TGCT_atlas_stage1.h5ad")
    summ = dict(cells=adata.n_obs, genes=adata.n_vars, samples=adata.obs["sample"].nunique(),
                major=adata.obs["major"].value_counts().to_dict(),
                state=adata.obs["state"].value_counts().to_dict(),
                malignant=adata.obs["malignant"].value_counts().to_dict(),
                minutes=round((time.time() - t0) / 60, 1))
    json.dump(summ, open(OUT / "STAGE1_SUMMARY.json", "w"), indent=1, default=str)
    md = ["# STAGE 1 report - single-cell atlas", "", "```json", json.dumps(summ, indent=1, default=str), "```", "",
          "## QC per sample", pd.read_csv(TAB / "S2_qc_per_sample.csv").to_markdown(index=False), "",
          "## Malignant calls", pd.read_csv(TAB / "S5_malignant_per_sample.csv").to_markdown(index=False), ""]
    f7 = TAB / "S7_M1_primary_LN_blood_states.csv"
    if f7.exists():
        md += ["## Patient M1: primary vs LN vs blood (immune state fractions)", pd.read_csv(f7).to_markdown(index=False), ""]
    f7b = TAB / "S7_malignant_lineage_markers.csv"
    if f7b.exists():
        md += ["## Malignant lineage markers (fraction of cells expressing)", pd.read_csv(f7b).round(3).to_markdown(index=False), ""]
    md += ["## Log", "```", *LOG, "```"]
    (OUT / "STAGE1_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log(f"done -> {OUT/'STAGE1_REPORT.md'}")
