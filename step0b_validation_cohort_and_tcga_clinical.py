#!/usr/bin/env python
"""
STEP 0b  -  validation cohort (GSE99420) + clean TCGA-TGCT clinical table
===========================================================================
Project : single-cell-anchored tumour-immune ecosystem model of relapse in
          stage I testicular germ cell tumours (TCGA discovery, GSE99420 validation)

What this script does
  A. GSE99420 (Princess Margaret, stage I TGCT on surveillance, WG-DASL, FFPE)
     - downloads ONLY what is missing: series matrix + normalised matrix
       (the 96 MB IDAT tar is skipped unless GET_IDAT=True)
     - downloads the GPL14951 probe annotation, maps probes -> gene symbols
     - parses sample characteristics, detects relapse / time / histology fields
     - writes gene x sample matrix (log2) + clean clinical table
  B. TCGA-TGCT (files already downloaded by step 0 in data/raw/TCGA_TGCT)
     - FIXES the step-0 bug: histology now from `histological_type`
       (not the component-percentage columns)
     - stage, stage I flag, LVI, rete testis / tunica / spermatic cord invasion,
       tumour size, serum markers, percent embryonal carcinoma ... whatever exists
     - joins PFI / DFI (TCGA-CDR) and marks samples with RNA-seq
  C. GO / RISK / NO-GO flags for the final title + markdown report

Nothing that already exists on disk is downloaded again.
Run:  conda activate scvi-env ; python step0b_validation_cohort_and_tcga_clinical.py
"""
import gzip, io, json, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------ config
BASE = Path(os.environ.get("TGCT_BASE", "/home/arshia/testis project"))
PROJ = BASE / "TGCT_ecotype"
RAW = PROJ / "data" / "raw"
PROC = PROJ / "data" / "processed"
REP = PROJ / "step0b_report"
for d in (RAW, PROC, REP):
    d.mkdir(parents=True, exist_ok=True)

GET_IDAT = False          # True -> also fetch GSE99420_RAW.tar (96 MB IDATs)
ACC = "GSE99420"
GPL = "GPL14951"
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo"

GSE_FILES = {
    f"{ACC}_series_matrix.txt.gz": f"{GEO_FTP}/series/GSE99nnn/{ACC}/matrix/{ACC}_series_matrix.txt.gz",
    f"{ACC}_alldata_normalized_nonmediancentered.txt.gz":
        f"{GEO_FTP}/series/GSE99nnn/{ACC}/suppl/{ACC}_alldata_normalized_nonmediancentered.txt.gz",
}
if GET_IDAT:
    GSE_FILES[f"{ACC}_RAW.tar"] = f"{GEO_FTP}/series/GSE99nnn/{ACC}/suppl/{ACC}_RAW.tar"

# probe annotation: try in order until one works
GPL_URLS = [
    (f"{GPL}.annot.gz", f"{GEO_FTP}/platforms/GPL14nnn/{GPL}/annot/{GPL}.annot.gz"),
    (f"{GPL}_full_table.txt",
     f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GPL}&targ=self&form=text&view=data"),
    (f"{GPL}_family.soft.gz", f"{GEO_FTP}/platforms/GPL14nnn/{GPL}/soft/{GPL}_family.soft.gz"),
]

TCGA_DIR = RAW / "TCGA_TGCT"
LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


# ------------------------------------------------------------------ download helpers
def get(url, stream=False, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, stream=stream, timeout=120)
            if r.status_code == 200:
                return r
            log(f"  HTTP {r.status_code} {url}")
            if r.status_code in (404, 403):
                return None
        except requests.RequestException as e:
            log(f"  retry {i+1}: {e}")
        time.sleep(5 * (i + 1))
    return None


def download(url, dest: Path):
    """skip if file exists and is non-empty; resume-safe via .part file"""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  exists, skip: {dest.name}")
        return dest
    r = get(url, stream=True)
    if r is None:
        return None
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # GEO 'view=data' returns HTML when a table does not exist
    if tmp.stat().st_size < 2000 and b"<html" in tmp.read_bytes()[:500].lower():
        tmp.unlink()
        return None
    tmp.rename(dest)
    log(f"  downloaded {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def opener(p: Path):
    return gzip.open(p, "rt", errors="replace") if p.suffix == ".gz" else open(p, "rt", errors="replace")


# ------------------------------------------------------------------ GEO parsing
def parse_series_matrix(path: Path):
    """returns (sample_meta DataFrame indexed by GSM, expression table or None)"""
    meta, chars, table, in_tab = {}, [], [], False
    with opener(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_tab = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_tab = False
                continue
            if in_tab:
                table.append(line.split("\t"))
                continue
            if not line.startswith("!Sample_"):
                continue
            parts = [x.strip('"') for x in line.split("\t")]
            key = parts[0].replace("!Sample_", "")
            if key == "characteristics_ch1":
                chars.append(parts[1:])
            elif key not in meta:
                meta[key] = parts[1:]
            else:                                    # repeated key -> numbered
                k = 2
                while f"{key}_{k}" in meta:
                    k += 1
                meta[f"{key}_{k}"] = parts[1:]
    df = pd.DataFrame(meta)
    for row in chars:                                # "key: value" -> char_key column
        for i, v in enumerate(row):
            if ":" in v:
                k, val = v.split(":", 1)
                col = "char_" + re.sub(r"\W+", "_", k.strip().lower()).strip("_")
                df.loc[i, col] = val.strip()
            elif v:
                df.loc[i, "char_unlabelled"] = v
    df = df.set_index("geo_accession")
    expr = None
    if len(table) > 1:
        expr = pd.DataFrame(table[1:], columns=[c.strip('"') for c in table[0]])
        expr = expr.set_index(expr.columns[0])
        expr.index = expr.index.str.strip('"')
        expr = expr.apply(pd.to_numeric, errors="coerce")
        if expr.shape[0] == 0 or expr.notna().sum().sum() == 0:
            expr = None
    return df, expr


def load_gpl_annotation(path: Path):
    """probe ID -> gene symbol from .annot / full table / family soft"""
    with opener(path) as fh:
        lines = fh.read().splitlines()
    # locate the table
    start = None
    for i, l in enumerate(lines):
        if l.startswith("!platform_table_begin"):
            start = i + 1
            break
    if start is None:                                # plain table: first line starting with ID
        start = next(i for i, l in enumerate(lines) if l.split("\t")[0] in ("ID", '"ID"'))
    end = next((i for i, l in enumerate(lines[start:], start) if l.startswith("!platform_table_end")), len(lines))
    tab = pd.read_csv(io.StringIO("\n".join(lines[start:end])), sep="\t", dtype=str, low_memory=False)
    tab.columns = [c.strip('"') for c in tab.columns]
    sym_col = next((c for c in tab.columns if c.lower() in
                    ("gene symbol", "symbol", "gene_symbol", "ilmn_gene", "gene")), None)
    if sym_col is None:
        sym_col = next((c for c in tab.columns if "symbol" in c.lower()), None)
    if sym_col is None:
        raise ValueError(f"no symbol column in {path.name}: {list(tab.columns)[:20]}")
    ann = tab[["ID", sym_col]].dropna()
    ann.columns = ["probe", "symbol"]
    ann["symbol"] = ann["symbol"].str.split(r"\s*///\s*|\s*;\s*").str[0].str.strip()
    ann = ann[ann["symbol"].str.len() > 0]
    log(f"  GPL annotation from {path.name}: {len(ann)} probes with symbol (column '{sym_col}')")
    return ann.set_index("probe")["symbol"]


def read_normalised_matrix(path: Path, meta: pd.DataFrame):
    df = pd.read_csv(path, sep="\t", low_memory=False, compression="infer")
    df.columns = [str(c).strip('"').strip() for c in df.columns]
    id_col = next((c for c in df.columns if re.search(r"^(id_ref|probe|probe_id|id|probeid|target)", c, re.I)),
                  df.columns[0])
    df = df.set_index(id_col)
    keep = [c for c in df.columns if not re.search(r"detection|pval|p\.val|p-val|bead|stderr|nbeads|avg_nbeads", c, re.I)]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.notna().mean() > 0.5]
    # map column names -> GSM (via title / description / source fields)
    lut = {}
    for gsm, row in meta.iterrows():
        lut[gsm] = gsm
        for f in [c for c in meta.columns if re.match(r"(title|description|source_name)", c)]:
            v = str(row[f]).strip()
            if v and v != "nan":
                lut[v] = gsm
                lut[re.sub(r"\W+", "", v).lower()] = gsm
    mapped = {}
    for c in df.columns:
        cc = re.sub(r"\.(avg_signal|signal|avg)$", "", c, flags=re.I)
        g = lut.get(c) or lut.get(cc) or lut.get(re.sub(r"\W+", "", cc).lower())
        if g is None:                                 # substring match as last resort
            hits = [gsm for k, gsm in lut.items() if len(k) > 3 and (k in cc or cc in k)]
            g = hits[0] if len(set(hits)) == 1 else None
        mapped[c] = g
    n_ok = sum(v is not None for v in mapped.values())
    log(f"  normalised matrix: {df.shape[0]} probes x {df.shape[1]} columns; mapped to GSM: {n_ok}")
    if n_ok >= 0.8 * df.shape[1]:
        df = df[[c for c in df.columns if mapped[c]]].rename(columns=mapped)
    return df, n_ok


def to_log2(m: pd.DataFrame):
    q99 = np.nanpercentile(m.values, 99)
    if q99 > 50:
        log(f"  values look linear (99th pct {q99:.0f}) -> log2(x - min + 1)")
        m = np.log2(m - np.nanmin(m.values) + 1)
    else:
        log(f"  values already log-scale (99th pct {q99:.2f})")
    return m


def collapse_to_genes(m: pd.DataFrame, ann: pd.Series):
    m = m.loc[m.index.intersection(ann.index)].copy()
    m["symbol"] = ann.loc[m.index].values
    m["_mean"] = m.drop(columns="symbol").mean(axis=1)
    m = m.sort_values("_mean", ascending=False).drop_duplicates("symbol")   # most expressed probe per gene
    return m.drop(columns="_mean").set_index("symbol")


# ------------------------------------------------------------------ clinical harmonisation (GSE99420)
def harmonise_gse_clinical(meta: pd.DataFrame):
    txt_cols = [c for c in meta.columns if c.startswith("char_") or c in ("title", "source_name_ch1", "description")]
    out = pd.DataFrame(index=meta.index)
    out["title"] = meta.get("title")

    def find(patterns):
        for c in meta.columns:
            if c.startswith("char_") and re.search(patterns, c, re.I):
                return c
        return None

    c_rel = find(r"relapse|recur|status|outcome|event|group")
    c_time = find(r"time|month|follow|day|year|rfs|dfs")
    c_hist = find(r"histol|subtype|type|seminoma|tumou?r")
    c_stage = find(r"stage")
    c_lvi = find(r"lvi|lymph.*vasc|vascular|invasion")
    log(f"  GSE99420 field guess: relapse={c_rel}, time={c_time}, histology={c_hist}, stage={c_stage}, lvi={c_lvi}")

    def rel_code(v):
        v = str(v).lower()
        if re.search(r"non[-_ ]?relap|no[-_ ]?relap|norelap|\bnr\b|no recur|^no$|^0$|false|disease[- ]free", v):
            return 0
        if re.search(r"relap|recur|^yes$|^1$|true|\br\b", v):
            return 1
        return np.nan

    if c_rel:
        out["relapse"] = meta[c_rel].map(rel_code)
    else:   # fall back to title/description text
        joined = meta[[c for c in txt_cols if c in meta]].astype(str).agg(" ".join, axis=1)
        out["relapse"] = joined.map(rel_code)
    if c_time:
        out["time_raw"] = pd.to_numeric(meta[c_time].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
        out["time_field"] = c_time
    def hist_code(v):
        v = str(v).lower().strip()
        if v in ("ns", "nsgct", "nonseminoma"):
            return "NSGCT"
        if v in ("s", "se", "sem"):
            return "SE"
        if re.search(r"non[-_ ]?sem|nsgct|nonsem|embryonal|yolk|terat|chorio|mixed", v):
            return "NSGCT"
        if re.search(r"semin", v):
            return "SE"
        return np.nan
    if c_hist:
        out["histology"] = meta[c_hist].map(hist_code)
    else:
        joined = meta[[c for c in txt_cols if c in meta]].astype(str).agg(" ".join, axis=1)
        out["histology"] = joined.map(hist_code)
    for name, c in (("stage", c_stage), ("lvi", c_lvi)):
        if c:
            out[name] = meta[c]
    for c in meta.columns:            # keep every raw characteristic for manual checking
        if c.startswith("char_"):
            out["raw_" + c] = meta[c]
    return out


# ------------------------------------------------------------------ TCGA harmonisation
def find_cols(df, pattern, exclude=None):
    cols = [c for c in df.columns if re.search(pattern, c, re.I)]
    if exclude:
        cols = [c for c in cols if not re.search(exclude, c, re.I)]
    return cols


def tcga_histology(row_val):
    v = str(row_val).lower()
    if v in ("nan", "", "[not available]", "[unknown]", "[discrepancy]"):
        return np.nan
    if re.search(r"mixed|non[- ]?semin", v):
        return "NSGCT"
    if re.search(r"embryonal|yolk|terat|chorio|germ cell tumor, nos|mixed germ", v):
        return "NSGCT"
    if "semin" in v:
        return "SE"
    return "other"


def tcga_stage_group(v):
    v = str(v).upper().replace("STAGE", "").strip()
    if not v or v in ("NAN", "[NOT AVAILABLE]", "[UNKNOWN]", "[DISCREPANCY]"):
        return np.nan
    m = re.match(r"^(IV|III|II|I|0)", v)
    return m.group(1) if m else np.nan


def harmonise_tcga():
    res = {}
    f_clin = TCGA_DIR / "TCGA_TGCT_clinicalMatrix.tsv"
    f_surv = TCGA_DIR / "TCGA_TGCT_survival.tsv"
    f_gdc = TCGA_DIR / "TCGA-TGCT.clinical.tsv.gz"
    f_hiseq = TCGA_DIR / "TCGA_TGCT_HiSeqV2.gz"
    f_star = TCGA_DIR / "TCGA-TGCT.star_counts.tsv.gz"
    missing = [f.name for f in (f_clin, f_surv) if not f.exists()]
    if missing:
        log(f"  TCGA files missing {missing} -> run step0 first")
        return None, res
    clin = pd.read_csv(f_clin, sep="\t", index_col=0, low_memory=False)
    surv = pd.read_csv(f_surv, sep="\t")
    log(f"  clinicalMatrix: {clin.shape}; survival: {surv.shape}")

    # ---- histology (bug fix)
    hcol = "histological_type" if "histological_type" in clin else None
    if hcol is None:
        cand = find_cols(clin, r"histolog", exclude=r"percent|pct|_number|count")
        cand = [c for c in cand if clin[c].dtype == object and clin[c].nunique() < 30]
        hcol = cand[0] if cand else None
    log(f"  histology column -> {hcol}")
    out = pd.DataFrame(index=clin.index)
    out["patient"] = out.index.str[:12]
    out["sample_type_code"] = out.index.str[13:15]
    if hcol:
        out["histology_raw"] = clin[hcol]
        out["histology"] = clin[hcol].map(tcga_histology)

    # ---- stage
    st_cols = find_cols(clin, r"pathologic_stage|clinical_stage|^stage|ajcc.*stage", exclude=r"tumor_stage_t|_n$|_m$")
    stc = next((c for c in ["pathologic_stage", "clinical_stage"] + st_cols if c in clin), None)
    log(f"  stage column -> {stc}")
    if stc:
        out["stage_raw"] = clin[stc]
        out["stage_group"] = clin[stc].map(tcga_stage_group)
        out["stage_I"] = (out["stage_group"] == "I").astype(float).where(out["stage_group"].notna())
    for c in ("pathologic_T", "pathologic_N", "pathologic_M", "igcccg_stage"):
        if c in clin:
            out[c] = clin[c]

    # ---- pathological risk factors (whatever exists)
    rf_patterns = {
        "lvi": r"lymphovascular|vascular_invasion|lymph_vasc",
        "rete_testis": r"rete",
        "invasion_other": r"tunica|spermatic|epididym|hilar|invasion",
        "tumor_size": r"size|dimension|tumor_length",
        "pct_embryonal": r"embryonal",
        "serum_markers": r"serum|afp|hcg|beta|ldh|lactate",
        "cryptorchidism": r"cryptorch|undescend",
        "treatment": r"radiation|chemo|surveillance|therapy|treatment",
    }
    found = {}
    for key, pat in rf_patterns.items():
        cols = [c for c in find_cols(clin, pat) if c not in out.columns]
        found[key] = cols
        for c in cols:
            out[c] = clin[c]
    res["risk_factor_columns"] = found
    log(f"  risk-factor columns found: { {k: len(v) for k, v in found.items()} }")

    # ---- GDC clinical: extra fields if present (keyed by patient)
    if f_gdc.exists():
        gdc = pd.read_csv(f_gdc, sep="\t", low_memory=False)
        pid = next((c for c in gdc.columns if re.search(r"submitter_id$", c) and "sample" not in c), None)
        if pid:
            gdc["patient"] = gdc[pid].astype(str).str[:12]
            extra = [c for c in gdc.columns if re.search(
                r"lymphovascular|perineural|rete|tumor_size|tumor_largest|primary_diagnosis|morphology|ajcc_pathologic_stage|igcccg|serum|treatment_type|treatment_or_therapy",
                c, re.I)]
            if extra:
                g = gdc.groupby("patient")[extra].first().add_prefix("gdc_")
                out = out.join(g, on="patient")
                res["gdc_extra_columns"] = extra
                log(f"  GDC extra columns: {len(extra)}")
                pdx = "gdc_primary_diagnosis.diagnoses"
                if pdx in out and "histology" in out:
                    miss = out["histology"].isna()
                    out.loc[miss, "histology"] = out.loc[miss, pdx].map(tcga_histology)
                    out["histology_source"] = np.where(miss & out["histology"].notna(), "GDC_primary_diagnosis",
                                                       np.where(out["histology"].notna(), "histological_type", None))
                    log(f"  histology filled from GDC primary_diagnosis: {int((miss & out['histology'].notna()).sum())}")

    # ---- survival (TCGA-CDR)
    sc = surv.copy()
    sc["patient"] = sc["_PATIENT"] if "_PATIENT" in sc else sc["sample"].str[:12]
    keep = [c for c in ("PFI", "PFI.time", "DFI", "DFI.time", "OS", "OS.time", "DSS", "DSS.time") if c in sc]
    sc = sc.groupby("patient")[keep].first()
    out = out.join(sc, on="patient")

    # ---- RNA-seq availability
    for tag, f in (("has_HiSeqV2", f_hiseq), ("has_STAR", f_star)):
        if f.exists():
            with opener(f) as fh:
                hdr = fh.readline().rstrip("\n").split("\t")[1:]
            ids = set(h[:15] for h in hdr)
            out[tag] = out.index.str[:15].isin(ids)

    # one primary-tumour row per patient
    prim = out[out["sample_type_code"] == "01"].copy()
    prim = prim[~prim["patient"].duplicated()]
    prim.to_csv(PROC / "TCGA_TGCT_clinical_harmonised.csv")
    log(f"  primary tumours: {len(prim)} -> {PROC/'TCGA_TGCT_clinical_harmonised.csv'}")

    # ---- summaries
    def nn(s):   # non-missing, ignoring TCGA placeholders
        bad = s.isna() | s.astype(str).str.strip().str.lower().isin(
            ["nan", "none", "", "[not available]", "[unknown]", "[not evaluated]", "[discrepancy]", "[not applicable]"])
        return int((~bad).sum())

    res["n_primary"] = len(prim)
    res["histology_counts"] = prim["histology"].value_counts(dropna=False).to_dict() if "histology" in prim else None
    res["histology_raw_counts"] = prim["histology_raw"].value_counts(dropna=False).to_dict() if "histology_raw" in prim else None
    res["stage_counts"] = prim["stage_group"].value_counts(dropna=False).to_dict() if "stage_group" in prim else None
    for ep in ("PFI", "DFI", "OS"):
        if ep in prim:
            res[f"{ep}_all"] = f"{int((prim[ep]==1).sum())}/{int(prim[ep].notna().sum())}"
    if "stage_I" in prim:
        s1 = prim[prim["stage_I"] == 1]
        res["stage_I_n"] = len(s1)
        res["stage_I_with_RNA"] = int(s1.get("has_HiSeqV2", pd.Series(True, index=s1.index)).sum())
        res["stage_I_PFI_events"] = int((s1["PFI"] == 1).sum()) if "PFI" in s1 else None
        res["stage_I_PFI_events_by_histology"] = (s1.groupby("histology")["PFI"].agg(
            lambda x: f"{int((x==1).sum())}/{int(x.notna().sum())}").to_dict()
            if "PFI" in s1 and "histology" in s1 else None)
        avail = {}
        for key, cols in list(found.items()) + [("gdc", [c for c in prim.columns if c.startswith("gdc_")])]:
            for c in cols:
                c2 = c if c in s1 else None
                if c2:
                    avail[c2] = nn(s1[c2])
        res["stage_I_field_non_missing"] = dict(sorted(avail.items(), key=lambda x: -x[1]))
        # LVI vs relapse sanity table
        lvi_col = next((c for c in found.get("lvi", []) if nn(s1[c]) > 0), None)
        if lvi_col is None:
            lvi_col = next((c for c in prim.columns if c.startswith("gdc_") and "lymphovascular" in c and nn(s1[c]) > 0), None)
        if lvi_col and "PFI" in s1:
            res["stage_I_LVI_column"] = lvi_col
            res["stage_I_LVI_x_PFI"] = pd.crosstab(s1[lvi_col].astype(str), s1["PFI"]).to_dict()
            rr = s1.groupby(s1[lvi_col].astype(str))["PFI"].agg(lambda x: f"{(x==1).mean():.2f} ({int((x==1).sum())}/{int(x.notna().sum())})")
            res["stage_I_relapse_rate_by_LVI"] = rr.to_dict()
        if "pathologic_T" in s1 and "PFI" in s1:
            res["stage_I_relapse_rate_by_pT"] = s1.groupby(s1["pathologic_T"].astype(str).str[:2])["PFI"].agg(
                lambda x: f"{(x==1).mean():.2f} ({int((x==1).sum())}/{int(x.notna().sum())})").to_dict()
    return prim, res


# ================================================================== RUN
log("===== A. GSE99420")
gdir = RAW / ACC
gdir.mkdir(parents=True, exist_ok=True)
paths = {k: download(u, gdir / k) for k, u in GSE_FILES.items()}

gpl_ann, gse = None, {}
(RAW / "GPL").mkdir(parents=True, exist_ok=True)
for fname, url in GPL_URLS:
    p = download(url, RAW / "GPL" / fname)
    if p:
        try:
            gpl_ann = load_gpl_annotation(p)
            break
        except Exception as e:
            log(f"  annotation parse failed for {fname}: {e}")

meta = expr_sm = None
if paths.get(f"{ACC}_series_matrix.txt.gz"):
    meta, expr_sm = parse_series_matrix(paths[f"{ACC}_series_matrix.txt.gz"])
    log(f"  series matrix: {meta.shape[0]} samples, {meta.shape[1]} fields; "
        f"embedded table: {None if expr_sm is None else expr_sm.shape}")
    meta.to_csv(REP / "GSE99420_sample_meta_raw.csv")
    gse["n_samples"] = int(meta.shape[0])
    gse["characteristics"] = {c: meta[c].value_counts().head(8).to_dict() for c in meta.columns if c.startswith("char_")}
    gse["titles_example"] = meta["title"].head(10).tolist() if "title" in meta else None

expr = None
nm = paths.get(f"{ACC}_alldata_normalized_nonmediancentered.txt.gz")
if expr_sm is not None and expr_sm.shape[1] >= 0.9 * len(meta) and gpl_ann is not None \
        and expr_sm.index.isin(gpl_ann.index).mean() > 0.5:
    log("  using GEO-processed table embedded in series matrix (GSM columns, ID_REF = GPL IDs)")
    expr = expr_sm.loc[:, expr_sm.columns.isin(meta.index)]
    nm = None                                   # supplementary file kept only as backup
    Path(REP / "GSE99420_supplementary_header.txt").write_text(
        "\n".join(pd.read_csv(paths[f"{ACC}_alldata_normalized_nonmediancentered.txt.gz"], sep="\t", nrows=3).columns.astype(str)))
if nm is not None and meta is not None:
    try:
        expr, n_ok = read_normalised_matrix(nm, meta)
        gse["normalised_cols_mapped"] = n_ok
        if n_ok < 0.8 * expr.shape[1]:
            log("  WARNING: columns of normalised file could not be matched to GSM -> check REP/GSE99420_columns.txt")
            Path(REP / "GSE99420_columns.txt").write_text("\n".join(map(str, expr.columns)))
    except Exception as e:
        log(f"  normalised matrix failed: {e}")
if (expr is None or expr.shape[1] == 0) and expr_sm is not None:
    log("  using expression table embedded in series matrix")
    expr = expr_sm

if expr is not None and gpl_ann is not None:
    expr = to_log2(expr)
    hit = expr.index.isin(gpl_ann.index).mean()
    gse["probe_annotation_rate"] = round(float(hit), 3)
    g = collapse_to_genes(expr, gpl_ann)
    g.to_csv(PROC / "GSE99420_expr_gene_log2.csv.gz")
    gse["genes"] = int(g.shape[0])
    gse["samples_in_matrix"] = int(g.shape[1])
    log(f"  gene matrix {g.shape} -> {PROC/'GSE99420_expr_gene_log2.csv.gz'}")
    # marker sanity: should separate SE vs NSGCT and show immune genes
    sanity = [x for x in ("POU5F1", "NANOG", "SOX17", "KIT", "SOX2", "AFP", "CD3E", "CD8A", "CD163", "PTPRC", "IFNG")
              if x in g.index]
    gse["marker_genes_present"] = sanity
    gse["median_marker_expr"] = g.loc[sanity].median(axis=1).round(2).to_dict()
elif expr is not None:
    log("  no probe annotation -> expression saved at probe level")
    expr.to_csv(PROC / "GSE99420_expr_probe.csv.gz")

clin99 = None
if meta is not None:
    clin99 = harmonise_gse_clinical(meta)
    clin99.to_csv(PROC / "GSE99420_clinical.csv")
    gse["relapse_counts"] = clin99["relapse"].value_counts(dropna=False).to_dict()
    gse["histology_counts"] = clin99["histology"].value_counts(dropna=False).to_dict()
    if "relapse" in clin99 and "histology" in clin99:
        gse["relapse_by_histology"] = pd.crosstab(clin99["histology"].fillna("NA"),
                                                  clin99["relapse"].fillna(-1)).to_dict()
    gse["has_time_to_event"] = "time_raw" in clin99 and clin99["time_raw"].notna().sum() > 0
    gse["has_stage"] = "stage" in clin99
    gse["has_lvi"] = "lvi" in clin99
    # biological sanity check: labelled histology vs marker genes (SE: SOX17/KIT ; NSGCT: SOX2)
    gfile = PROC / "GSE99420_expr_gene_log2.csv.gz"
    if gfile.exists() and clin99["histology"].notna().any():
        gm = pd.read_csv(gfile, index_col=0)
        z = gm.sub(gm.mean(1), axis=0).div(gm.std(1) + 1e-9, axis=0)
        se_g = [x for x in ("SOX17", "KIT", "PRDM1") if x in z.index]
        ns_g = [x for x in ("SOX2", "DNMT3B", "AFP", "GPC3") if x in z.index]
        if se_g and ns_g:
            d = (z.loc[se_g].mean() - z.loc[ns_g].mean()).reindex(clin99.index)
            gse["SE_minus_NS_marker_score_by_label"] = d.groupby(clin99["histology"]).median().round(2).to_dict()
            pred = np.where(d > 0, "SE", "NSGCT")
            gse["histology_label_vs_marker_agreement"] = round(float((pred == clin99["histology"]).mean()), 3)

log("===== B. TCGA-TGCT clinical")
prim, tcga = harmonise_tcga()

# ================================================================== C. flags
FLAGS = []


def flag(name, level, detail, action):
    FLAGS.append((name, level, detail, action))
    log(f"  [{level}] {name}: {detail}")


log("===== C. flags")
rel = clin99["relapse"] if clin99 is not None else pd.Series(dtype=float)
n1, n0, nna = int((rel == 1).sum()), int((rel == 0).sum()), int(rel.isna().sum())
flag("V1 GSE99420 relapse labels",
     "GO" if n1 >= 20 and n0 >= 20 else "RISK" if n1 + n0 > 0 else "NO-GO",
     f"relapse={n1}, no relapse={n0}, unknown={nna}",
     "binary validation (AUC, logistic, Wilcoxon of score)" if n1 else
     "labels not in GEO -> inspect GSE99420_sample_meta_raw.csv; else take labels from Lewin 2018 supplement / contact authors")
flag("V2 GSE99420 time-to-relapse",
     "GO" if gse.get("has_time_to_event") else "RISK",
     f"time field present: {gse.get('has_time_to_event')}",
     "Cox/C-index in validation" if gse.get("has_time_to_event") else
     "validation as relapse vs no relapse only (AUC + logistic OR); case-control design => no absolute risk")
flag("V3 GSE99420 expression usable",
     "GO" if gse.get("genes", 0) > 10000 and gse.get("samples_in_matrix", 0) >= 50 else "RISK",
     f"genes={gse.get('genes')}, samples={gse.get('samples_in_matrix')}, probe annot rate={gse.get('probe_annotation_rate')}",
     "rank-based scores only (singscore / within-sample); no BayesPrism on DASL-FFPE")
flag("V4 GSE99420 histology",
     "GO" if len([k for k in gse.get("histology_counts", {}) if isinstance(k, str)]) == 2 else "RISK",
     f"{gse.get('histology_counts')}",
     "adjust validation for histology" if gse.get("histology_counts") else
     "infer SE vs NSGCT from markers (SOX17/KIT vs SOX2/AFP) and report as inferred")

s1n, s1e = tcga.get("stage_I_n"), tcga.get("stage_I_PFI_events")
flag("T1 TCGA stage I relapse events",
     "GO" if s1e and s1e >= 20 else "RISK" if s1e else "NO-GO",
     f"stage I n={s1n}, PFI events={s1e}, by histology={tcga.get('stage_I_PFI_events_by_histology')}",
     "Firth Cox, max ~2-3 covariates (score + histology + LVI)" if s1e else "check stage column")
lvi_n = 0
for c, v in (tcga.get("stage_I_field_non_missing") or {}).items():
    if re.search(r"lymphovascular|vascular", c, re.I):
        lvi_n = max(lvi_n, v)
flag("T2 TCGA LVI available in stage I",
     "GO" if s1n and lvi_n >= 0.7 * s1n else "RISK" if lvi_n else "NO-GO",
     f"LVI non-missing {lvi_n}/{s1n}",
     "incremental value over LVI (+histology): ΔC-index, LRT, decision curve" if lvi_n else
     "'adds to clinicopathological factors' claim limited to histology + T stage (pT1 vs pT2 ~ LVI proxy)")
pt_n = (tcga.get("stage_I_field_non_missing") or {}).get("pathologic_T", 0) if prim is not None else 0
if prim is not None and "pathologic_T" in prim and s1n:
    s1 = prim[prim["stage_I"] == 1]
    pt_n = int(s1["pathologic_T"].astype(str).str.match(r"^T\d").fillna(False).sum())
flag("T3 TCGA pT available in stage I", "GO" if s1n and pt_n >= 0.8 * s1n else "RISK",
     f"pT non-missing {pt_n}/{s1n}", "pT2+ (=LVI/rete/tunica) as fallback risk factor")
rete = [c for c, v in (tcga.get("stage_I_field_non_missing") or {}).items() if "rete" in c.lower() and v > 0]
flag("T4 TCGA rete testis invasion", "GO" if rete else "RISK", f"columns with data: {rete}",
     "use if present; otherwise state as limitation")

# ================================================================== report
lev = {"GO": "🟢", "RISK": "🟠", "NO-GO": "🔴"}
md = ["# STEP 0b report — validation cohort + TCGA clinical\n",
      f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n",
      "## Flags\n", "| flag | level | detail | action |", "|---|---|---|---|"]
md += [f"| {n} | {lev[l]} {l} | {d} | {a} |" for n, l, d, a in FLAGS]
md += ["\n## GSE99420\n", "```json\n" + json.dumps(gse, indent=2, default=str) + "\n```\n",
       "## TCGA-TGCT (primary tumours)\n", "```json\n" + json.dumps(tcga, indent=2, default=str) + "\n```\n",
       "## Outputs\n",
       f"- `{PROC/'GSE99420_expr_gene_log2.csv.gz'}`",
       f"- `{PROC/'GSE99420_clinical.csv'}`  (raw characteristics kept as raw_char_* for manual check)",
       f"- `{PROC/'TCGA_TGCT_clinical_harmonised.csv'}`",
       f"- `{REP/'GSE99420_sample_meta_raw.csv'}`\n",
       "## Log\n", "```", *LOG, "```"]
(REP / "STEP0b_REPORT.md").write_text("\n".join(md), encoding="utf-8")
log(f"report -> {REP/'STEP0b_REPORT.md'}")
