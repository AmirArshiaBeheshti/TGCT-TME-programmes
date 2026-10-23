#!/usr/bin/env python
# coding: utf-8
"""
=====================================================================================
 STEP 0 — Data audit and feasibility check (TGCT cross-histology ecotype project)
=====================================================================================
 Goal: before writing any analysis code, find out what is REALLY available and usable.

 For every dataset it:
   1. lists GEO files (series matrix + supplementary) with sizes
   2. downloads what is needed (scRNA supplementary files, series matrices, TCGA from UCSC Xena)
   3. parses sample metadata and searches for clinical fields (histology, survival, IGCCCG, relapse)
   4. inspects scRNA matrices: cells, genes, raw counts?, and how many TUMOUR-LIKE cells per
      histology marker (seminoma: POU5F1/NANOG/SOX17/KIT; EC: SOX2; YST: AFP; chorio: CGA/CGB)
   5. summarises TCGA-TGCT: histology mix, PFI / OS events
   6. writes a feasibility report with explicit GO / RISK / NO-GO flags

 Outputs: <PROJ>/step0_audit/STEP0_AUDIT.md   (send this file back)
          <PROJ>/step0_audit/*.csv / *.json
 Run    : python step0_data_audit.py        (re-runs skip files already downloaded)
 Needs  : requests, pandas, numpy; scanpy optional (for 10x / h5 / h5ad files)
 Proxy  : honours HTTP(S)_PROXY environment variables.
"""

# %% =================================================================================
# 0. CONFIGURATION
# ====================================================================================
from pathlib import Path
import re, json, gzip, tarfile, zipfile, time, datetime, html
import numpy as np
import pandas as pd
import requests

BASE = Path("/home/arshia/testis project")
PROJ = BASE / "TGCT_ecotype"
RAW = PROJ / "data" / "raw"
REP = PROJ / "step0_audit"
for d in (RAW, REP):
    d.mkdir(parents=True, exist_ok=True)

DOWNLOAD_SC_SUPPL = True     # download scRNA supplementary files (can be several GB)
MAX_FILE_GB = 8.0            # skip single files larger than this (reported, not downloaded)
TIMEOUT, RETRIES = 60, 4
INSPECT_MAX_GB = 3.0         # do not parse single text matrices larger than this (uncompressed estimate)

# data already on disk from the previous project (re-used, not downloaded again)
LOCAL = {
    "GSE228501": BASE / "GSE228501_RAW (1)",
    "GSE229823": BASE / "GSE229823_RAW",
    "GSE112013": BASE / "GSE112013",
}

GEO = {  # accession: role, data kind, note
    "GSE261811": ("sc_discovery", "scRNA", "TGCT 2 seminoma + 1 EC + 1 mixed (Islam et al., Br J Cancer 2024)"),
    "GSE228501": ("sc_discovery", "scRNA", "seminoma n=4 (Lu et al., Nat Commun 2023)"),
    "GSE197778": ("sc_discovery", "scRNA", "metastatic seminoma (Mo et al., Front Oncol 2022)"),
    "GSE120508": ("sc_normal", "scRNA", "normal adult testis (Guo et al., Cell Res 2018)"),
    "GSE112013": ("sc_normal", "scRNA", "normal adult testis"),
    "GSE229823": ("spatial", "Visium", "seminoma Visium n=3"),
    "GSE3218": ("bulk_clinical", "array", "Korkola: seminoma / NSGCT / normal, cisplatin outcome?"),
    "GSE10783": ("bulk_clinical", "array", "Korkola JCO validation: NSGCT, 5-y OS?"),
    "GSE1818": ("bulk_histology", "array", "histological subtypes incl. CIS/GCNIS"),
}

# UCSC Xena (TCGA legacy hub = Pan-Cancer survival endpoints from TCGA-CDR; GDC hub = STAR counts)
XENA = {
    "TCGA_TGCT_HiSeqV2.gz": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.TGCT.sampleMap%2FHiSeqV2.gz",
    "TCGA_TGCT_clinicalMatrix.tsv": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.TGCT.sampleMap%2FTGCT_clinicalMatrix",
    "TCGA_TGCT_survival.tsv": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/survival%2FTGCT_survival.txt",
    "TCGA-TGCT.star_counts.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-TGCT.star_counts.tsv.gz",
    "TCGA-TGCT.clinical.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-TGCT.clinical.tsv.gz",
}

MARKERS = {
    "seminoma": ["POU5F1", "NANOG", "SOX17", "KIT", "TFAP2C", "PRAME"],
    "EC": ["POU5F1", "NANOG", "SOX2", "DPPA4", "LIN28A"],
    "YST": ["AFP", "APOA1", "GPC3", "TTR"],
    "chorio": ["CGA", "CGB3", "CGB5", "KRT7"],
    "teratoma": ["KRT8", "KRT18", "COL1A1", "MGP"],
    "immune": ["PTPRC"],
    "germ_normal": ["DDX4", "DAZL", "SYCP3", "PRM1"],
}
ALL_MARKERS = sorted({g for v in MARKERS.values() for g in v})
CLIN_RE = re.compile(r"surviv|vital|status|death|dead|alive|os\b|overall|follow|month|year|relapse|recur|progress|"
                     r"pfs|dfs|igcccg|risk|stage|histolog|tumou?r type|diagnos|subtype|cisplatin|chemo|outcome|"
                     r"age|metasta|response|resist", re.I)

STATUS, FLAGS = [], []


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def status(item, state, detail=""):
    STATUS.append(dict(item=item, state=state, detail=str(detail)[:300]))
    log(f"[{state}] {item} {detail}")


# %% =================================================================================
# 1. HTTP HELPERS (retries, resume, size check)
# ====================================================================================
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (TGCT-audit)"})


def get(url, stream=False, method="GET"):
    last = None
    for k in range(RETRIES):
        try:
            r = S.request(method, url, timeout=TIMEOUT, stream=stream, allow_redirects=True)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def remote_size(url):
    try:
        r = get(url, method="HEAD")
        return int(r.headers.get("Content-Length", 0)) or None
    except Exception:
        return None


def download(url, dest, max_gb=MAX_FILE_GB):
    dest = Path(dest)
    size = remote_size(url)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest, "exists"
    if size and size / 1024 ** 3 > max_gb:
        return None, f"skipped (size {size / 1024 ** 3:.1f} GB > {max_gb} GB)"
    tmp = dest.with_suffix(dest.suffix + ".part")
    pos = tmp.stat().st_size if tmp.exists() else 0
    hdr = {"Range": f"bytes={pos}-"} if pos else {}
    for k in range(RETRIES):
        try:
            with S.get(url, headers=hdr, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 404:
                    return None, "404"
                r.raise_for_status()
                mode = "ab" if pos and r.status_code == 206 else "wb"
                with open(tmp, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.rename(dest)
            return dest, "downloaded"
        except Exception as e:
            last = e
            pos = tmp.stat().st_size if tmp.exists() else 0
            hdr = {"Range": f"bytes={pos}-"} if pos else {}
            time.sleep(3 * (k + 1))
    return None, f"failed ({last})"


def geo_url(acc):
    stub = acc[:-3] + "nnn" if len(acc) > 6 else "GSEnnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{acc}/"


def list_geo_dir(url):
    r = get(url)
    if r.status_code == 404:
        return []
    names = re.findall(r'href="([^"?/][^"]*)"', r.text)
    return [html.unescape(n) for n in names if not n.startswith(("http", "/")) and n not in ("../",)]


# %% =================================================================================
# 2. GEO SERIES MATRIX PARSING (sample metadata + clinical field search)
# ====================================================================================
def parse_series_matrix(path):
    """Return (series_info dict, samples DataFrame). Characteristics 'key: value' become columns."""
    opener = gzip.open if str(path).endswith(".gz") else open
    series, rows = {}, {}
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Series_"):
                k, *v = line.rstrip("\n").split("\t")
                series.setdefault(k[8:], []).append(" ".join(x.strip('"') for x in v))
            elif line.startswith("!Sample_"):
                k, *v = line.rstrip("\n").split("\t")
                rows.setdefault(k[8:], []).append([x.strip('"') for x in v])
    if "geo_accession" not in rows:
        return series, pd.DataFrame()
    n = len(rows["geo_accession"][0])
    df = pd.DataFrame(index=rows["geo_accession"][0])
    for k, lists in rows.items():
        if k.startswith("characteristics"):
            for vals in lists:
                for i, v in enumerate(vals[:n]):
                    if ":" in v:
                        key, val = v.split(":", 1)
                        key = "char_" + re.sub(r"\W+", "_", key.strip().lower())
                    else:
                        key, val = "char_unlabelled", v
                    if key not in df:
                        df[key] = None
                    df.iloc[i, df.columns.get_loc(key)] = val.strip()
        elif k in ("title", "source_name_ch1", "platform_id", "description", "molecule_ch1",
                   "library_strategy", "supplementary_file_1", "supplementary_file_2"):
            df[k] = lists[0][:n]
    return series, df


def clinical_fields(df):
    out = {}
    for c in df.columns:
        vals = df[c].dropna().astype(str)
        if vals.empty:
            continue
        if CLIN_RE.search(c) or vals.str.contains(CLIN_RE).mean() > 0.3:
            out[c] = dict(n_nonmissing=int(vals.ne("").sum()), n_unique=int(vals.nunique()),
                          examples="; ".join(vals.unique()[:6]))
    return out


# %% =================================================================================
# 3. scRNA MATRIX INSPECTION (cells, counts, tumour-like cells per histology marker)
# ====================================================================================
BC_RE = re.compile(r"[ACGTN]{12,}")


def summarise_counts(name, X_marker, lib, genes_present, is_int):
    """X_marker: DataFrame cells x marker genes (counts); lib: library size per cell."""
    pos = lambda g: (X_marker[g] > 0) if g in X_marker else pd.Series(False, index=X_marker.index)
    tum_sem = pos("POU5F1") & pos("NANOG") & (pos("SOX17") | pos("KIT") | pos("TFAP2C"))
    tum_ec = pos("POU5F1") & pos("NANOG") & pos("SOX2") & ~pos("SOX17")
    yst = pos("AFP") & (pos("APOA1") | pos("GPC3") | pos("TTR"))
    cho = pos("CGA") & (pos("CGB3") | pos("CGB5"))
    imm = pos("PTPRC")
    germ = pos("DDX4") | pos("DAZL") | pos("SYCP3")
    return dict(file=name, n_cells=int(len(lib)), median_umi=float(np.median(lib)) if len(lib) else np.nan,
                raw_integer_counts=bool(is_int), markers_found=len(genes_present),
                seminoma_like=int(tum_sem.sum()), EC_like=int(tum_ec.sum()), YST_like=int(yst.sum()),
                chorio_like=int(cho.sum()), immune_PTPRC=int(imm.sum()), normal_germ_like=int(germ.sum()),
                pct_tumour_like=round(100 * float((tum_sem | tum_ec | yst | cho).mean()), 2) if len(lib) else np.nan)


def inspect_anndata(a, name):
    import scipy.sparse as sp
    a.var_names_make_unique()
    X = a.X
    lib = np.asarray(X.sum(axis=1)).ravel()
    sub = X[:, :200].data if sp.issparse(X) else np.asarray(X[:50, :200]).ravel()
    is_int = bool(np.allclose(sub, np.round(sub))) if len(sub) else False
    g = [x for x in ALL_MARKERS if x in a.var_names]
    M = a[:, g].X
    M = pd.DataFrame(M.toarray() if sp.issparse(M) else np.asarray(M), index=a.obs_names, columns=g)
    rows = [summarise_counts(name, M, lib, g, is_int)]
    # try to split multi-sample files by a sample column or barcode tag
    key = next((c for c in a.obs.columns if re.search(r"sample|donor|patient|orig|batch", c, re.I)), None)
    tags = a.obs[key].astype(str) if key else pd.Series([BC_RE.sub("", b).strip("-_.:") or "1" for b in a.obs_names],
                                                       index=a.obs_names)
    if 1 < tags.nunique() <= 40:
        for t in tags.unique():
            m = (tags == t).values
            rows.append(summarise_counts(f"{name} | {t}", M[m], lib[m], g, is_int))
    return rows


def inspect_text_matrix(path):
    """Stream a dense text matrix (genes x cells or cells x genes); keep marker rows and library sizes."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        head = fh.readline().rstrip("\n")
    sep = "\t" if head.count("\t") >= head.count(",") else ","
    tok = [t.strip('"') for t in head.split(sep)]
    genes_in_rows = sum(bool(BC_RE.search(t)) for t in tok[1:60]) > 20 or len(tok) > 3000
    lib, keep, cols, is_int = None, {}, None, True
    for chunk in pd.read_csv(path, sep=sep, index_col=0, chunksize=2000, low_memory=False):
        if genes_in_rows:
            v = chunk.values.astype(np.float32)
            lib = v.sum(0) if lib is None else lib + v.sum(0)
            idx = chunk.index.astype(str).str.replace(r"^(hg19_|GRCh38_+)", "", regex=True)
            for g, row in zip(idx, v):
                if g in ALL_MARKERS:
                    keep[g] = row
            cols = list(chunk.columns.astype(str))
        else:
            chunk.columns = chunk.columns.astype(str).str.replace(r"^(hg19_|GRCh38_+)", "", regex=True)
            v = chunk.values.astype(np.float32)
            s = v.sum(1)
            lib = s if lib is None else np.r_[lib, s]
            g = [x for x in ALL_MARKERS if x in chunk.columns]
            part = chunk[g]
            keep = part if not isinstance(keep, pd.DataFrame) else pd.concat([keep, part])
            cols = None
        is_int &= bool(np.allclose(v[:20, :50], np.round(v[:20, :50])))
    if genes_in_rows:
        M = pd.DataFrame(keep, index=cols)
    else:
        M = keep if isinstance(keep, pd.DataFrame) else pd.DataFrame()
    return M, np.asarray(lib), is_int


def inspect_sc_path(p):
    """Inspect one file or directory; returns list of summary rows."""
    p = Path(p)
    rows = []
    try:
        if p.is_dir() and any(p.glob("matrix.mtx*")):
            import scanpy as sc
            rows += inspect_anndata(sc.read_10x_mtx(p, var_names="gene_symbols", cache=False), str(p.relative_to(RAW.parent.parent)
                                                                                                     if RAW.parent.parent in p.parents else p))
        elif p.suffix == ".h5ad":
            import scanpy as sc
            a = sc.read_h5ad(p)
            if "counts" in a.layers:
                a.X = a.layers["counts"]
            rows += inspect_anndata(a, p.name)
        elif p.suffix == ".h5":
            import scanpy as sc
            rows += inspect_anndata(sc.read_10x_h5(p), p.name)
        elif re.search(r"\.(txt|tsv|csv)(\.gz)?$", p.name):
            est = p.stat().st_size * (6 if p.name.endswith(".gz") else 1) / 1024 ** 3
            if est > INSPECT_MAX_GB:
                return [dict(file=p.name, note=f"not parsed (≈{est:.1f} GB uncompressed); raise INSPECT_MAX_GB")]
            M, lib, is_int = inspect_text_matrix(p)
            rows.append(summarise_counts(p.name, M, lib, list(M.columns), is_int))
    except Exception as e:
        rows.append(dict(file=p.name, note=f"inspection failed: {str(e)[:150]}"))
    return rows


def extract_archives(folder):
    for f in list(folder.glob("*.tar")) + list(folder.glob("*.tar.gz")) + list(folder.glob("*.tgz")):
        out = folder / (f.name.split(".tar")[0] + "_extracted")
        if not out.exists():
            try:
                with tarfile.open(f) as t:
                    t.extractall(out)
                status(f"extract {f.name}", "ok")
            except Exception as e:
                status(f"extract {f.name}", "failed", e)
    for f in folder.glob("*.zip"):
        out = folder / (f.stem + "_extracted")
        if not out.exists():
            with zipfile.ZipFile(f) as z:
                z.extractall(out)


def candidate_sc_inputs(folder):
    """10x directories, h5/h5ad, and text matrices under a folder (deduplicated)."""
    folder = Path(folder)
    dirs = sorted({m.parent for m in folder.rglob("matrix.mtx*")})
    files = [f for f in folder.rglob("*") if f.is_file() and
             (f.suffix in (".h5", ".h5ad") or re.search(r"(matrix|count|umi|expr|raw|data).*\.(txt|tsv|csv)(\.gz)?$",
                                                        f.name, re.I))
             and not any(d in f.parents for d in dirs) and ".part" not in f.name]
    return dirs + sorted(files)


# %% =================================================================================
# 4. GEO DATASETS
# ====================================================================================
geo_rows, clin_rows, sc_rows = [], [], []
for acc, (role, kind, note) in GEO.items():
    log(f"===== {acc} ({kind}, {role})")
    out = RAW / acc
    out.mkdir(exist_ok=True)
    base = geo_url(acc)
    try:
        mfiles = [f for f in list_geo_dir(base + "matrix/") if "series_matrix" in f]
        sfiles = list_geo_dir(base + "suppl/")
        status(f"{acc} listing", "ok", f"{len(mfiles)} matrix, {len(sfiles)} suppl files")
    except Exception as e:
        mfiles, sfiles = [], []
        status(f"{acc} listing", "failed", e)

    # series matrix -> metadata + clinical fields
    meta = []
    for f in mfiles:
        path, st = download(base + "matrix/" + f, out / f, max_gb=2)
        status(f"{acc} {f}", st)
        if path:
            series, df = parse_series_matrix(path)
            if len(df):
                df.insert(0, "file", f)
                meta.append(df)
            title = series.get("title", [""])[0]
    meta = pd.concat(meta) if meta else pd.DataFrame()
    if len(meta):
        meta.to_csv(REP / f"{acc}_sample_metadata.csv")
        for c, info in clinical_fields(meta).items():
            clin_rows.append(dict(dataset=acc, field=c, **info))

    # supplementary files
    suppl_info = []
    for f in sfiles:
        size = remote_size(base + "suppl/" + f)
        suppl_info.append(dict(file=f, size_MB=round(size / 1024 ** 2, 1) if size else None))
    pd.DataFrame(suppl_info).to_csv(REP / f"{acc}_suppl_files.csv", index=False)
    if kind == "scRNA" and DOWNLOAD_SC_SUPPL and acc not in LOCAL:
        for f in sfiles:
            if re.search(r"(RAW\.tar|\.h5ad?$|matrix|count|umi|barcode|feature|gene|\.tar(\.gz)?$|\.mtx)", f, re.I):
                path, st = download(base + "suppl/" + f, out / f)
                status(f"{acc} {f}", st)
        extract_archives(out)
        for sub in list(out.glob("*_extracted")):
            extract_archives(sub)

    # scRNA inspection
    if kind == "scRNA":
        folder = LOCAL.get(acc, out)
        inputs = candidate_sc_inputs(folder)
        if acc == "GSE112013" and (folder / "GSE112013_normal_testis.h5ad").exists():
            inputs = [folder / "GSE112013_normal_testis.h5ad"]
        status(f"{acc} sc inputs", "ok" if inputs else "none found", f"{len(inputs)} file(s)/folder(s) in {folder}")
        for p in inputs:
            for r in inspect_sc_path(p):
                sc_rows.append(dict(dataset=acc, **r))

    rds = [f for f in sfiles if re.search(r"\.(rds|rda|RData|loom|seurat)", f, re.I)]
    geo_rows.append(dict(dataset=acc, role=role, kind=kind, note=note, n_samples=len(meta),
                         platforms=", ".join(sorted(meta["platform_id"].unique())) if "platform_id" in meta else "",
                         n_suppl_files=len(sfiles), suppl_total_GB=round(sum((x["size_MB"] or 0) for x in suppl_info) / 1024, 2),
                         R_only_files=", ".join(rds), local=str(LOCAL.get(acc, ""))))

GEO_SUM = pd.DataFrame(geo_rows)
CLIN = pd.DataFrame(clin_rows)
SC = pd.DataFrame(sc_rows)
GEO_SUM.to_csv(REP / "A_geo_datasets.csv", index=False)
CLIN.to_csv(REP / "B_clinical_fields_found.csv", index=False)
SC.to_csv(REP / "C_scRNA_inspection.csv", index=False)

# %% =================================================================================
# 5. TCGA-TGCT (UCSC Xena)
# ====================================================================================
log("===== TCGA-TGCT (UCSC Xena)")
tdir = RAW / "TCGA_TGCT"
tdir.mkdir(exist_ok=True)
TCGA = {}
for fname, url in XENA.items():
    path, st = download(url, tdir / fname, max_gb=3)
    status(f"TCGA {fname}", st)
    TCGA[fname] = path

tcga_sum = {}
try:
    expr = TCGA.get("TCGA_TGCT_HiSeqV2.gz")
    if expr:
        hdr = pd.read_csv(expr, sep="\t", nrows=0).columns[1:]
        tum = [s for s in hdr if s[13:15] == "01"]
        tcga_sum["expression_samples"] = len(hdr)
        tcga_sum["primary_tumours"] = len(tum)
    clin = pd.read_csv(TCGA["TCGA_TGCT_clinicalMatrix.tsv"], sep="\t", index_col=0) if TCGA.get("TCGA_TGCT_clinicalMatrix.tsv") else None
    surv = pd.read_csv(TCGA["TCGA_TGCT_survival.tsv"], sep="\t") if TCGA.get("TCGA_TGCT_survival.tsv") else None
    if clin is not None:
        hcol = next((c for c in clin.columns if re.search(r"histolog", c, re.I)), None)
        if hcol:
            tcga_sum["histology_counts"] = clin[hcol].value_counts().to_dict()
        for c in [c for c in clin.columns if re.search(r"igcccg|stage|risk|s_stage|serum", c, re.I)][:10]:
            tcga_sum[f"field:{c}"] = int(clin[c].notna().sum())
    if surv is not None:
        surv["patient"] = surv["_PATIENT"] if "_PATIENT" in surv else surv["sample"].str[:12]
        sp_ = surv.drop_duplicates("patient")
        for ep in ("OS", "DSS", "DFI", "PFI"):
            if ep in sp_:
                tcga_sum[f"{ep}_n"] = int(sp_[ep].notna().sum())
                tcga_sum[f"{ep}_events"] = int((sp_[ep] == 1).sum())
        if clin is not None and hcol:
            h = clin[hcol].groupby(clin.index.str[:12]).first()
            sp_["histology"] = sp_["patient"].map(h)
            if "PFI" in sp_:
                tcga_sum["PFI_events_by_histology"] = sp_.groupby("histology")["PFI"].agg(
                    lambda x: f"{int((x == 1).sum())}/{int(x.notna().sum())}").to_dict()
        sp_.to_csv(REP / "D_TCGA_survival_with_histology.csv", index=False)
    status("TCGA summary", "ok")
except Exception as e:
    status("TCGA summary", "failed", e)
json.dump(tcga_sum, open(REP / "D_TCGA_summary.json", "w"), indent=2, default=str)

# %% =================================================================================
# 6. FEASIBILITY FLAGS
# ====================================================================================
log("===== feasibility")


def flag(name, level, evidence, consequence):
    FLAGS.append(dict(check=name, level=level, evidence=evidence, consequence=consequence))


# F1 scRNA availability / format
for acc in [a for a, v in GEO.items() if v[1] == "scRNA"]:
    d = SC[(SC.dataset == acc) & ~SC["file"].astype(str).str.contains(r" \| ", regex=True)] if len(SC) else pd.DataFrame()
    ok = bool(len(d)) and "n_cells" in d and d["n_cells"].fillna(0).sum() > 0
    rawc = bool(d.get("raw_integer_counts", pd.Series(dtype=bool)).fillna(False).any()) if len(d) else False
    flag(f"F1 {acc} readable scRNA", "GO" if ok and rawc else "RISK" if ok else "NO-GO",
         f"cells={int(d['n_cells'].fillna(0).sum()) if ok else 0}; raw counts={rawc}; "
         f"R-only files: {GEO_SUM.set_index('dataset').loc[acc, 'R_only_files'] or 'none'}",
         "usable" if ok and rawc else "needs conversion from RDS/normalised data (R step) or manual download")

# F2 cross-histology tumour cells (the key risk)
SC_TOT = SC[~SC["file"].astype(str).str.contains(r" \| ", regex=True)] if len(SC) else SC   # file totals only
if len(SC_TOT) and "EC_like" in SC_TOT:
    ec = int(SC_TOT["EC_like"].fillna(0).sum())
    yst = int(SC_TOT["YST_like"].fillna(0).sum())
    sem = int(SC_TOT["seminoma_like"].fillna(0).sum())
    flag("F2 non-seminoma tumour cells in scRNA", "GO" if ec + yst >= 150 else "RISK" if ec + yst >= 30 else "NO-GO",
         f"seminoma-like {sem}, EC-like {ec}, YST-like {yst}, chorio-like {int(SC_TOT['chorio_like'].fillna(0).sum())} "
         "(marker-based, before QC)",
         "cross-histology TUMOUR states feasible" if ec + yst >= 150 else
         "non-seminoma tumour cells too few -> focus on microenvironment; tumour states from bulk (TCGA) only")

# F3 Korkola clinical outcome
for acc in ("GSE3218", "GSE10783"):
    c = CLIN[CLIN.dataset == acc] if len(CLIN) else pd.DataFrame()
    surv_f = c[c.field.str.contains(r"surv|status|vital|death|follow|month|year|os|outcome|relapse", case=False)] if len(c) else c
    risk_f = c[c.field.str.contains(r"igcccg|risk", case=False)] if len(c) else c
    flag(f"F3 {acc} clinical outcome", "GO" if len(surv_f) else "NO-GO",
         f"survival-like fields: {list(surv_f.field) or 'none'}; IGCCCG/risk fields: {list(risk_f.field) or 'none'}",
         "independent outcome validation possible" if len(surv_f) else
         "no outcome in GEO -> use TCGA PFI as primary endpoint; request Korkola clinical tables (paper supplement)")

# F4 TCGA events
pfi_e = tcga_sum.get("PFI_events")
flag("F4 TCGA-TGCT PFI events", "GO" if pfi_e and pfi_e >= 25 else "RISK" if pfi_e else "NO-GO",
     f"PFI events {pfi_e} / {tcga_sum.get('PFI_n')}; by histology {tcga_sum.get('PFI_events_by_histology')}",
     "Cox with ≤ events/10 covariates; use Firth penalisation" if pfi_e else "TCGA survival not available")

# F5 normal reference
nn = int(SC_TOT[SC_TOT.dataset.isin(["GSE120508", "GSE112013"])]["n_cells"].fillna(0).sum()) \
    if len(SC_TOT) and "n_cells" in SC_TOT else 0
flag("F5 normal testis reference", "GO" if nn > 3000 else "RISK", f"{nn} cells", "normal germ/somatic states for reference")

FL = pd.DataFrame(FLAGS)
FL.to_csv(REP / "E_feasibility_flags.csv", index=False)
pd.DataFrame(STATUS).to_csv(REP / "F_run_log.csv", index=False)

# %% =================================================================================
# 7. REPORT
# ====================================================================================
def md_table(df, cols=None, max_rows=60):
    if df is None or not len(df):
        return "_none_\n"
    d = df[cols] if cols else df
    d = d.head(max_rows).fillna("")
    out = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * d.shape[1]]
    out += ["| " + " | ".join(str(v).replace("|", "/")[:120] for v in r) + " |" for r in d.values]
    return "\n".join(out) + "\n"


rep = ["# STEP 0 — data audit and feasibility\n", f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}\n",
       "## Feasibility flags", md_table(FL),
       "## GEO datasets", md_table(GEO_SUM),
       "## scRNA inspection (marker-based tumour-like cell counts, before QC)",
       md_table(SC, [c for c in ["dataset", "file", "n_cells", "median_umi", "raw_integer_counts", "seminoma_like",
                                 "EC_like", "YST_like", "chorio_like", "immune_PTPRC", "normal_germ_like",
                                 "pct_tumour_like", "note"] if c in SC.columns]),
       "## Clinical fields found in GEO metadata", md_table(CLIN),
       "## TCGA-TGCT", "```json\n" + json.dumps(tcga_sum, indent=2, default=str) + "\n```\n",
       "## Failed / skipped items", md_table(pd.DataFrame(STATUS).query("state not in ['ok','exists','downloaded']")
                                             if STATUS else None)]
(REP / "STEP0_AUDIT.md").write_text("\n".join(rep))
print("\n".join(rep[:6]))
log(f"DONE -> {REP / 'STEP0_AUDIT.md'}")
