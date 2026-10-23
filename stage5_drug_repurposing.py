#!/usr/bin/env python
"""
STAGE 5  -  Drug repurposing (hypothesis-generating)
=====================================================
Targets are nominated from OUR data with pre-set rules, then mapped to drugs; nothing is tuned on outcome.

 1  target nomination
    T1 tumour-intrinsic: Tumour_SE_like / Tumour_EC_like vs ALL non-tumour cells of the atlas
       (logFC > 1, detected in >= 30 % of tumour cells and <= 10 % of others), confirmed in bulk:
       expressed in the TCGA tumour-specific expression from stage-2 deconvolution (median CPM >= 10)
    T2 stromal / immune-cold programme (P2, linked to chrX/12p dosage): locked stage-3 signature genes
    T3 MIF - CD74 / CXCR4 / CD44 axis (stage 4)
    safety flags: expression in normal testis cells (germ, Sertoli, Leydig; GSE112013) and in immune cells
 2  positive controls (must be recovered if the nomination works):
    CLDN6 (CAR-T / ADC trials in GCT), TNFRSF8 = CD30 (brentuximab vedotin, EC), POU5F1/NANOG (not druggable),
    KIT (imatinib tried in seminoma)
 3  drug lookup: DGIdb (GraphQL API, cached) + curated table for the MIF axis / controls (offline fallback)
 4  signature reversal: L1000CDS2 (LINCS) for (i) EC-like vs SE-like tumour programme, (ii) seminoma vs
    normal germ cells (cached; skipped gracefully if offline)
 5  per-state drug-target scores in the atlas (drug2cell-style mean target expression) and a ranked
    priority table: tumour specificity, bulk confirmation, approval status, safety, reversal evidence

Run:  conda activate scvi-env ; python stage5_drug_repurposing.py
"""
import os, re, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("TGCT_BASE", "/home/arshia/testis project"))
PROJ = BASE / "TGCT_ecotype"
S1, S2, S3, S4 = (PROJ / x for x in ("stage1_atlas", "stage2_tcga", "stage3_clinical", "stage4_mechanism"))
OUT = PROJ / "stage5_drugs"
CACHE = OUT / "cache"
for d in (OUT, CACHE, OUT / "figures"):
    d.mkdir(parents=True, exist_ok=True)
DGIDB_URL = "https://dgidb.org/api/graphql"
L1000_URL = "https://maayanlab.cloud/L1000CDS2/query"
MIF_AXIS = ["MIF", "CD74", "CXCR4", "CD44", "ACKR3"]
POS_CTRL = ["CLDN6", "TNFRSF8", "KIT", "POU5F1", "NANOG"]
# literature-curated gene-drug pairs (prior knowledge, NOT data-derived; flagged source="curated" in every table).
# approval status as of mid-2026 - verify before publication. ibudilast: approved in Japan only (asthma/stroke).
STRICT_DGIDB = True      # True: stop if DGIdb cannot be reached, so results never rest on the curated list alone
CURATED = [("MIF", "IBUDILAST", "inhibitor (approved JP only)", True), ("MIF", "IMALUMAB", "antibody", False),
           ("CD74", "MILATUZUMAB", "antibody", False), ("CXCR4", "PLERIXAFOR", "antagonist", True),
           ("CXCR4", "MAVORIXAFOR", "antagonist", True), ("CXCR4", "MOTIXAFORTIDE", "antagonist", True),
           ("CD44", "BIVATUZUMAB MERTANSINE", "antibody-drug conjugate", False),
           ("TNFRSF8", "BRENTUXIMAB VEDOTIN", "antibody-drug conjugate", True),
           ("KIT", "IMATINIB", "inhibitor", True), ("KIT", "AVAPRITINIB", "inhibitor", True),
           ("CLDN6", "BNT211 (CLDN6 CAR-T)", "cell therapy", False), ("CLDN6", "TORL-1-23 (CLDN6 ADC)", "antibody-drug conjugate", False)]
SEED = 0
LOG = []
PROV = {"dgidb": "ok", "l1000": {}}


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    LOG.append(s)


# ============================================================ 1. targets
def nominate_targets():
    import scanpy as sc
    f = OUT / "T_targets.csv"
    if f.exists():
        return pd.read_csv(f)
    a = sc.read_h5ad(S1 / "TGCT_atlas_stage1.h5ad")
    a.X = a.layers["lognorm"]
    st = a.obs["state"].astype(str)
    tum = (st == "Tumour_malignant").values
    cnt = a.layers["counts"]
    def det(g):
        return np.asarray((cnt[:, a.var_names.get_loc(g)] > 0).todense()).ravel() if g in a.var_names else np.zeros(a.n_obs, bool)
    ec = det("SOX2").astype(int) + det("DNMT3B")
    se = det("SOX17").astype(int) + det("KIT") + det("PRDM1")
    lab = st.where(~tum, np.where(ec > se, "Tumour_EC_like", "Tumour_SE_like"))
    bad = lab.str.contains("_other|_contaminant|Unresolved|Tumour_CNVneg|Tumour_GCT")
    a = a[~bad.values].copy()
    a.obs["grp"] = lab[~bad.values].values
    a.obs["grp2"] = np.where(a.obs["grp"].str.startswith("Tumour"), a.obs["grp"], "non_tumour")
    rows = []
    for g in ("Tumour_SE_like", "Tumour_EC_like"):
        if (a.obs["grp2"] == g).sum() < 20:
            continue
        sc.tl.rank_genes_groups(a, "grp2", groups=[g], reference="non_tumour", method="wilcoxon", pts=True, use_raw=False)
        de = sc.get.rank_genes_groups_df(a, g)
        de = de[(de["logfoldchanges"] > 1) & (de["pvals_adj"] < 1e-3)].copy()
        ref = (a.obs["grp2"] == "non_tumour").values
        Cg = a[:, de["names"].values].layers["counts"]
        de["pct_nz_reference"] = np.asarray((Cg[ref] > 0).mean(0)).ravel()
        if "pct_nz_group" not in de:
            de["pct_nz_group"] = np.asarray((Cg[(a.obs["grp2"] == g).values] > 0).mean(0)).ravel()
        de = de[(de["pct_nz_group"] >= 0.3) & (de["pct_nz_reference"] <= 0.1)]
        de = de[~de["names"].str.match(r"^(MT-|RP[SL]\d|AC\d|AL\d|LINC)")]
        for _, r in de.head(300).iterrows():
            rows.append(dict(gene=r["names"], source=f"T1_{g}", logFC=r["logfoldchanges"], pct_tumour=r["pct_nz_group"],
                             pct_other=r["pct_nz_reference"]))
    sig = json.load(open(S3 / "LOCKED_signatures.json"))
    for g in sig["P2_TME_stromal"]["up"]:
        rows.append(dict(gene=g, source="T2_stromal_programme"))
    for g in MIF_AXIS:
        rows.append(dict(gene=g, source="T3_MIF_axis"))
    T = pd.DataFrame(rows)
    # expression context: normal testis cells (safety) and immune cells
    normal = a.obs["site"].astype(str).eq("normal").values & a.obs["grp"].isin(["Germ_normal", "Sertoli", "Leydig"]).values
    immune = a.obs["compartment"].astype(str).isin(["T_NK", "Myeloid", "B_plasma"]).values
    tumour_m = a.obs["grp2"].str.startswith("Tumour").values
    genes = sorted(set(T["gene"]) | set(POS_CTRL))
    genes = [g for g in genes if g in a.var_names]
    X = a[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    ctx = pd.DataFrame({"mean_tumour": X[tumour_m].mean(0), "mean_normal_testis": X[normal].mean(0),
                        "pct_normal_testis": (X[normal] > 0).mean(0), "mean_immune": X[immune].mean(0)}, index=genes)
    # bulk confirmation: tumour-specific expression from stage-2 deconvolution
    tz = S2 / "tcga_tumour_expression.csv.gz"
    if tz.exists():
        Z = pd.read_csv(tz, index_col=0)
        ctx["TCGA_tumour_CPM_median"] = Z.median().reindex(genes)
    T = T.merge(ctx, left_on="gene", right_index=True, how="left")
    T["safety_flag_normal_testis"] = T["pct_normal_testis"] > 0.2
    T["bulk_confirmed"] = T.get("TCGA_tumour_CPM_median", pd.Series(np.nan, index=T.index)) >= 10
    T.to_csv(f, index=False)
    ctx.loc[[g for g in POS_CTRL if g in ctx.index]].to_csv(OUT / "T_positive_controls_expression.csv")
    pc = T[T["gene"].isin(POS_CTRL)][["gene", "source"]]
    log(f"  targets: {T['gene'].nunique()} genes ({T['source'].value_counts().to_dict()}); "
        f"positive controls nominated by T1: {sorted(set(pc['gene']))}")
    return T


# ============================================================ 2. DGIdb
def dgidb(genes):
    f = CACHE / "dgidb.json"
    cache = json.load(open(f)) if f.exists() else {}
    todo = [g for g in genes if g not in cache]
    if todo:
        import requests
        q = """query($names:[String!]!){ genes(names:$names){ nodes{ name interactions{ interactionScore
               interactionTypes{ type } drug{ name conceptId approved } } } } }"""
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            try:
                r = requests.post(DGIDB_URL, json={"query": q, "variables": {"names": chunk}}, timeout=120)
                r.raise_for_status()
                nodes = r.json()["data"]["genes"]["nodes"]
                got = {n["name"]: n["interactions"] for n in nodes}
                for g in chunk:
                    cache[g] = got.get(g, [])
            except Exception as e:
                msg = f"DGIdb query failed ({type(e).__name__}: {str(e)[:120]})"
                if STRICT_DGIDB:
                    raise SystemExit(f"\n!! {msg}\n   results would rest on the curated list only - fix the connection "
                                     f"or set STRICT_DGIDB = False (then the report is marked accordingly)")
                log(f"  {msg} - curated fallback only for {len(chunk)} genes")
                PROV["dgidb"] = "FAILED - curated list only"
                break
            time.sleep(0.5)
        json.dump(cache, open(f, "w"))
    rows = []
    for g in genes:
        for it in cache.get(g, []) or []:
            d = it.get("drug") or {}
            rows.append(dict(gene=g, drug=str(d.get("name", "")).upper(), approved=bool(d.get("approved")),
                             interaction=";".join(t["type"] for t in it.get("interactionTypes") or [] if t.get("type")),
                             dgidb_score=it.get("interactionScore"), source="DGIdb"))
    for g, drug, kind, appr in CURATED:
        if g in genes:
            rows.append(dict(gene=g, drug=drug, approved=appr, interaction=kind, dgidb_score=np.nan, source="curated"))
    D = pd.DataFrame(rows).drop_duplicates(["gene", "drug"])
    return D


# ============================================================ 3. L1000CDS2 reversal
def l1000_reverse(name, up, dn):
    f = CACHE / f"l1000_{name}.json"
    if f.exists():
        res = json.load(open(f))
    else:
        import requests
        payload = {"data": {"upGenes": up, "dnGenes": dn},
                   "config": {"aggravate": False, "searchMethod": "geneSet", "share": False, "combination": False,
                              "db-version": "latest"}, "metadata": []}
        try:
            r = requests.post(L1000_URL, json=payload, timeout=180)
            r.raise_for_status()
            res = r.json()
            json.dump(res, open(f, "w"))
        except Exception as e:
            log(f"  L1000CDS2 {name} failed ({type(e).__name__}: {str(e)[:80]}) - skipped")
            PROV["l1000"][name] = "FAILED"
            return pd.DataFrame()
    PROV["l1000"][name] = "ok (cached)" if f.exists() else "ok"
    top = res.get("topMeta", [])
    rows = [dict(query=name, rank=i + 1, perturbagen=str(t.get("pert_desc", "")).upper(), pert_id=t.get("pert_id"),
                 cell_line=t.get("cell_id"), dose=t.get("pert_dose"), time=t.get("pert_time"), score=t.get("score"))
            for i, t in enumerate(top)]
    return pd.DataFrame(rows)


def reversal_queries():
    out = []
    de = S4 / "B3_EC_vs_SE_tumour_DE.csv"
    if de.exists():
        d = pd.read_csv(de)
        d = d[~d["names"].str.match(r"^(MT-|RP[SL]\d|AC\d|AL\d|LINC)")]
        up = d[(d["logfoldchanges"] > 1) & (d["pvals_adj"] < 1e-3)].sort_values("scores", ascending=False)["names"].head(150).tolist()
        dn = d[(d["logfoldchanges"] < -1) & (d["pvals_adj"] < 1e-3)].sort_values("scores")["names"].head(150).tolist()
        if len(up) >= 20 and len(dn) >= 20:
            out.append(l1000_reverse("EC_like_vs_SE_like", up, dn))
    # seminoma vs normal germ cells (malignant state)
    import scanpy as sc
    a = sc.read_h5ad(S1 / "TGCT_atlas_stage1.h5ad", backed="r")
    st = a.obs["state"].astype(str)
    keep = (st == "Tumour_malignant") | (st == "Germ_normal")
    if (st == "Germ_normal").sum() >= 50 and (st == "Tumour_malignant").sum() >= 50:
        b = a[keep.values].to_memory()
        b.X = b.layers["lognorm"]
        b.obs["g"] = np.where(b.obs["state"].astype(str) == "Tumour_malignant", "tumour", "germ")
        sc.tl.rank_genes_groups(b, "g", groups=["tumour"], reference="germ", method="wilcoxon", use_raw=False)
        d = sc.get.rank_genes_groups_df(b, "tumour")
        d = d[~d["names"].str.match(r"^(MT-|RP[SL]\d|AC\d|AL\d|LINC)")]
        up = d[d["logfoldchanges"] > 1].head(150)["names"].tolist()
        dn = d[d["logfoldchanges"] < -1].sort_values("scores").head(150)["names"].tolist()
        out.append(l1000_reverse("tumour_vs_normal_germ", up, dn))
    R = pd.concat([x for x in out if len(x)], ignore_index=True) if any(len(x) for x in out) else pd.DataFrame()
    R.to_csv(OUT / "R_L1000CDS2_reversal.csv", index=False)
    return R


# ============================================================ 4. per-state drug scores + priority
def drug_state_scores(D):
    import scanpy as sc
    a = sc.read_h5ad(S1 / "TGCT_atlas_stage1.h5ad")
    a.X = a.layers["lognorm"]
    st = a.obs["state"].astype(str)
    keep = ~st.str.contains("_other|_contaminant|Unresolved|Tumour_CNVneg|Tumour_GCT")
    a = a[keep.values]
    st = st[keep.values]
    genes = sorted(set(D["gene"]) & set(a.var_names))
    X = a[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    M = pd.DataFrame(X, columns=genes).groupby(st.values).mean()
    S = {}
    for drug, g in D.groupby("drug")["gene"]:
        gg = [x for x in g if x in M.columns]
        if gg:
            S[drug] = M[gg].mean(1)
    S = pd.DataFrame(S)                           # states x drugs
    S.to_csv(OUT / "S_drug_target_score_by_state.csv")
    return S


def prioritise(T, D, S, R):
    tum = [s for s in S.index if s.startswith("Tumour")]
    normal = [s for s in S.index if s in ("Germ_normal", "Sertoli", "Leydig")]
    rest = [s for s in S.index if s not in tum + normal]
    rows = []
    for drug, g in D.groupby("drug"):
        genes = sorted(set(g["gene"]))
        t = T[T["gene"].isin(genes)]
        r = dict(drug=drug, targets=";".join(genes), n_targets=len(genes), approved=bool(g["approved"].any()),
                 interaction=";".join(sorted({x for x in g["interaction"].astype(str) if x and x != "nan"}))[:80],
                 target_sources=";".join(sorted(set(t["source"]))),
                 bulk_confirmed=bool(t["bulk_confirmed"].any()) if len(t) else False,
                 normal_testis_flag=bool(t["safety_flag_normal_testis"].any()) if len(t) else False)
        if drug in S.columns:
            r["score_tumour"] = S.loc[tum, drug].max() if tum else np.nan
            r["score_nontumour_max"] = S.loc[rest, drug].max() if rest else np.nan
            r["score_normal_testis"] = S.loc[normal, drug].max() if normal else np.nan
            r["tumour_specificity"] = r["score_tumour"] - r["score_nontumour_max"]
        r["L1000_reversal_hit"] = bool(len(R) and R["perturbagen"].str.contains(re.escape(drug.split(" ")[0]), na=False).any())
        rows.append(r)
    P = pd.DataFrame(rows)
    # priority = rank aggregation (higher better): specificity, approval, bulk confirmation, reversal; minus safety
    P["rank_spec"] = P["tumour_specificity"].rank(pct=True)
    P["priority"] = (P["rank_spec"].fillna(0) + 0.5 * P["approved"] + 0.5 * P["bulk_confirmed"] +
                     0.5 * P["L1000_reversal_hit"] - 0.5 * P["normal_testis_flag"])
    P = P.sort_values("priority", ascending=False)
    P.to_csv(OUT / "P_drug_priority.csv", index=False)
    return P



# ============================================================ 5. stricter evidence (added after first real run)
# DGIdb mixes curated mechanistic interactions with text-mined co-mentions (e.g. phenytoin-POU5F1, TNF blockers-ALPL).
# Only interactions with an explicit inhibitory / targeting type are kept for the main ranking.
# LINCS research codes -> INN used by DGIdb (verify: PD-0325901 = mirdametinib, TG101348 = fedratinib)
LINCS_SYNONYMS = {"PD-0325901": "MIRDAMETINIB", "TG101348": "FEDRATINIB", "AZD-6244": "SELUMETINIB",
                  "GSK-1120212": "TRAMETINIB", "BEZ-235": "DACTOLISIB", "PLX-4032": "VEMURAFENIB"}
TARGETING = r"inhibitor|antagonist|blocker|antibody|binder|conjugate|negative modulator|suppressor|inverse agonist|cleavage|degrader|vaccine|cell therapy|inhibitory"


def bulk_support(T):
    """bulk confirmation for every target: tumour-specific deconvolved expression OR bulk expression that tracks tumour content"""
    bc = S2 / "bayesprism_input" / "tcga_bulk_counts.csv.gz"
    tf = S2 / "tcga_type_fractions.csv"
    if not (bc.exists() and tf.exists()):
        return T
    B = pd.read_csv(bc, index_col=0)
    cpm = B.div(B.sum(1), axis=0) * 1e6
    tum = pd.read_csv(tf, index_col=0)["Tumour"].reindex(cpm.index)
    g = [x for x in T["gene"].unique() if x in cpm.columns]
    from scipy.stats import spearmanr
    rho = {x: spearmanr(cpm[x], tum, nan_policy="omit")[0] for x in g}
    T = T.drop(columns=[c for c in ("bulk_median_CPM", "rho_with_tumour_fraction") if c in T])
    T["bulk_median_CPM"] = T["gene"].map(cpm[g].median())
    T["rho_with_tumour_fraction"] = T["gene"].map(rho)
    T["bulk_confirmed"] = (T.get("TCGA_tumour_CPM_median", pd.Series(np.nan, index=T.index)) >= 10) | \
                          ((T["bulk_median_CPM"] >= 5) & (T["rho_with_tumour_fraction"] > 0.3))
    return T


def dgidb_drug_targets(drugs):
    """full target list of named drugs (targeting interactions only) - used to compare mechanisms"""
    f = CACHE / "dgidb_drugs.json"
    cache = json.load(open(f)) if f.exists() else {}
    todo = [d for d in drugs if d not in cache]
    if todo:
        import requests
        q = """query($names:[String!]!){ drugs(names:$names){ nodes{ name interactions{ gene{ name }
               interactionTypes{ type } } } } }"""
        for i in range(0, len(todo), 40):
            chunk = todo[i:i + 40]
            r = None
            for attempt in range(4):
                try:
                    r = requests.post(DGIDB_URL, json={"query": q, "variables": {"names": chunk}}, timeout=120)
                    r.raise_for_status()
                    break
                except Exception as e:
                    log(f"  DGIdb drug query retry {attempt + 1}/4: {type(e).__name__}")
                    r = None
                    time.sleep(5 * (attempt + 1))
            try:
                if r is None:
                    raise RuntimeError("no response after retries")
                for n in r.json()["data"]["drugs"]["nodes"]:
                    gs = [it["gene"]["name"] for it in n.get("interactions") or []
                          if re.search(TARGETING, ";".join(t["type"] for t in it.get("interactionTypes") or [] if t.get("type")), re.I)]
                    cache[str(n["name"]).upper()] = sorted(set(gs))
                for d in chunk:
                    cache.setdefault(d, [])
            except Exception as e:
                log(f"  DGIdb drug->target query failed: {type(e).__name__}: {str(e)[:80]}")
                break
            time.sleep(0.5)
        json.dump(cache, open(f, "w"))
    return {d: set(cache.get(d, [])) for d in drugs}


def convergence(Ps, R, D):
    """mechanisms supported by two independent lines: (a) drug targets a gene nominated from our data,
    (b) a LINCS compound that reverses our tumour signatures shares a molecular target with that drug"""
    if not len(R):
        return pd.DataFrame(), pd.DataFrame()
    R = R[~R["perturbagen"].isin(["-666", "", "NAN"]) & ~R["perturbagen"].str.startswith("BRD")].copy()
    both = R.groupby("perturbagen")["query"].nunique()
    R["pan_reverser"] = R["perturbagen"].map(both) > 1            # reverses both unrelated queries -> likely non-specific
    R["dgidb_name"] = R["perturbagen"].map(lambda c: LINCS_SYNONYMS.get(c, c))
    comp = sorted(R["perturbagen"].unique())
    cand = Ps[Ps["approved"]].head(150)["drug"].tolist()
    tg0 = dgidb_drug_targets(sorted(set(R["dgidb_name"])) + cand)
    tg = dict(tg0)
    for c in comp:
        tg[c] = tg0.get(LINCS_SYNONYMS.get(c, c), set())
    R["compound_targets"] = R["perturbagen"].map(lambda c: ";".join(sorted(tg.get(c, set()))[:12]))
    rows = []
    for c in comp:
        for d in cand:
            sh = tg.get(c, set()) & tg.get(d, set())
            if sh:
                rows.append(dict(lincs_compound=c, reverses=";".join(sorted(R.loc[R["perturbagen"] == c, "query"].unique())),
                                 pan_reverser=bool(R.loc[R["perturbagen"] == c, "pan_reverser"].any()),
                                 best_rank=int(R.loc[R["perturbagen"] == c, "rank"].min()),
                                 approved_drug=d, shared_targets=";".join(sorted(sh)),
                                 data_nominated_targets_of_drug=Ps.set_index("drug").loc[d, "targets"]))
    Cv = pd.DataFrame(rows)
    if len(Cv):
        Cv = Cv.sort_values(["pan_reverser", "best_rank"])
    return Cv, R

# ============================================================ RUN
if __name__ == "__main__":
    t0 = time.time()
    log("1 target nomination"); T = nominate_targets()
    log("2 drug lookup (DGIdb + curated)")
    D = dgidb(sorted(set(T["gene"]) | set(POS_CTRL)))
    D.to_csv(OUT / "D_gene_drug_interactions.csv", index=False)
    log(f"  {len(D)} gene-drug pairs, {D['drug'].nunique()} drugs ({int(D.drop_duplicates('drug')['approved'].sum())} approved)")
    T = bulk_support(T)
    T.to_csv(OUT / "T_targets.csv", index=False)
    log("3 LINCS L1000CDS2 reversal"); R = reversal_queries()
    log("4 scoring + priority (all DGIdb interactions - for completeness only)")
    S = drug_state_scores(D); P = prioritise(T, D, S, R)
    log("5 strict evidence: targeting interactions only + mechanism convergence")
    Ds = D[D["interaction"].astype(str).str.contains(TARGETING, case=False, regex=True) | (D["source"] == "curated")].copy()
    # ADC payload attribution (e.g. MMAE 'vedotin' drugs listed against TUBB isotypes) is not the antibody target -> drop
    adc = Ds["drug"].str.contains(r"VEDOTIN|MAFODOTIN|SORAVTANSINE|MERTANSINE|EMTANSINE|DERUXTECAN|GOVITECAN|OZOGAMICIN|TESIRINE", regex=True)
    payload_gene = Ds["gene"].str.match(r"^(TUBA|TUBB|TOP1$|TOP2)")
    Ds = Ds[~(adc & payload_gene)]
    # evidence threshold: DGIdb interaction score >= median of targeting interactions (curated pairs kept)
    thr = Ds.loc[Ds["source"] == "DGIdb", "dgidb_score"].astype(float).median()
    Ds = Ds[(Ds["dgidb_score"].astype(float) >= thr) | (Ds["source"] == "curated")]
    log(f"  strict filter: targeting type + no ADC-payload attribution + DGIdb score >= {thr:.3g}")
    Ss = drug_state_scores(Ds)
    Ps = prioritise(T, Ds, Ss, R)
    Ps.to_csv(OUT / "P_drug_priority_STRICT.csv", index=False)
    # mechanism-level view (one row per data-nominated target gene) - avoids lists of 20 drugs hitting one gene
    Tg = T.drop_duplicates("gene").set_index("gene")
    mech = []
    for g, dd in Ds.groupby("gene"):
        if g not in Tg.index:
            continue
        t = Tg.loc[g]
        mech.append(dict(target=g, nominated_by=";".join(sorted(T.loc[T["gene"] == g, "source"].unique())),
                         logFC_tumour=t.get("logFC"), pct_tumour=t.get("pct_tumour"), pct_other=t.get("pct_other"),
                         pct_normal_testis=t.get("pct_normal_testis"), bulk_confirmed=t.get("bulk_confirmed"),
                         rho_tumour_fraction=t.get("rho_with_tumour_fraction"),
                         n_approved=int(dd["approved"].sum()), approved_examples=";".join(dd.loc[dd["approved"], "drug"].head(5)),
                         n_clinical_or_tool=int((~dd["approved"]).sum())))
    M = pd.DataFrame(mech)
    if len(M):
        M = M.sort_values(["bulk_confirmed", "logFC_tumour"], ascending=False)
    M.to_csv(OUT / "M_mechanism_targets.csv", index=False)
    Cv, R2 = convergence(Ps, R, D)
    Cv.to_csv(OUT / "C_mechanism_convergence.csv", index=False)
    R2.to_csv(OUT / "R_L1000CDS2_reversal_annotated.csv", index=False)
    log(f"  strict: {len(Ds)} targeting gene-drug pairs ({Ds['drug'].nunique()} drugs); convergent pairs: {len(Cv)}")
    pc = pd.read_csv(OUT / "T_positive_controls_expression.csv", index_col=0) if (OUT / "T_positive_controls_expression.csv").exists() else pd.DataFrame()
    t1 = T[T["source"].str.startswith("T1")]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        top = Ps[Ps["approved"] & Ps["drug"].isin(Ss.columns)].head(25)["drug"].tolist()
        if top:
            H = Ss[top].T
            H = H.sub(H.mean(1), axis=0).div(H.std(1) + 1e-9, axis=0)
            fig, ax = plt.subplots(figsize=(max(8, 0.32 * H.shape[1]), 0.32 * len(top) + 2))
            im = ax.imshow(H.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
            ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
            ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=90, fontsize=7)
            plt.colorbar(im, ax=ax, label="target score (z across states)")
            fig.tight_layout(); fig.savefig(OUT / "figures" / "F_top_approved_drugs_by_state.png", dpi=200); plt.close(fig)
    except Exception as e:
        log(f"  figure error {e}")
    import hashlib
    inputs = [S1 / "TGCT_atlas_stage1.h5ad", S2 / "tcga_tumour_expression.csv.gz", S3 / "LOCKED_signatures.json",
              S4 / "B3_EC_vs_SE_tumour_DE.csv"]
    prov = ["| input (real outputs of stages 1-4) | exists | size MB | modified | md5 (first 1 MB) |", "|---|---|---|---|---|"]
    for f_ in inputs:
        if f_.exists():
            with open(f_, "rb") as fh:
                h = hashlib.md5(fh.read(1 << 20)).hexdigest()[:12]
            prov.append(f"| {f_.relative_to(PROJ)} | yes | {f_.stat().st_size / 1e6:.1f} | "
                        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(f_.stat().st_mtime))} | {h} |")
        else:
            prov.append(f"| {f_.relative_to(PROJ)} | **MISSING** | | | |")
    n_cur = int((D["source"] == "curated").sum()); n_dgi = int((D["source"] == "DGIdb").sum())
    md = ["# STAGE 5 report - drug repurposing (hypothesis-generating)", "",
          f"runtime {(time.time() - t0) / 60:.1f} min", "",
          "## Provenance", *prov, "",
          f"- DGIdb: **{PROV['dgidb']}** ({n_dgi} gene-drug pairs); curated literature pairs: {n_cur} (marked source=curated)",
          f"- L1000CDS2: {PROV['l1000'] or 'not run'}", "",
          "## Positive controls (should be tumour-high, low elsewhere)",
          (pc.round(3).to_markdown() if len(pc) else "n/a"), "",
          f"- positive controls recovered by the unbiased T1 nomination: "
          f"{sorted(set(t1['gene']) & set(POS_CTRL))}", "",
          "## Tumour-intrinsic targets (T1, top 30 by logFC)",
          t1.sort_values("logFC", ascending=False).head(30).round(3).to_markdown(index=False), "",
          "## MAIN - druggable targets nominated from the data (one row per target; strict interactions)",
          (M.head(30).round(3).to_markdown(index=False) if len(M) else "none"), "",
          "## drug priority, strict interactions (top 30)",
          Ps.head(30).drop(columns=["rank_spec"]).round(3).to_markdown(index=False), "",
          "## MAIN - mechanism convergence (data-nominated target drug x LINCS reverser sharing a target)",
          (Cv.head(30).to_markdown(index=False) if len(Cv) else "none"), "",
          "## L1000 reversers with their targets (pan_reverser = reverses both unrelated queries)",
          (R2.groupby("query").head(15)[["query", "rank", "perturbagen", "cell_line", "score", "pan_reverser", "compound_targets"]].to_markdown(index=False) if len(R2) else "n/a"), "",
          "## (for completeness) priority using ALL DGIdb interactions incl. text-mined - NOT for interpretation",
          P.head(15).drop(columns=["rank_spec"]).round(3).to_markdown(index=False), "",
          "## MIF axis drugs (targeting interactions; receptors sit on immune cells, so tumour specificity is expected to be <0)",
          Ps[Ps["targets"].str.contains("MIF|CD74|CXCR4|CD44|ACKR3", na=False)].head(15).drop(columns=["rank_spec"]).round(3).to_markdown(index=False), ""]
    if len(R):
        pass
    md += ["## Log", "```", *LOG, "```"]
    (OUT / "STAGE5_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log(f"done -> {OUT / 'STAGE5_REPORT.md'}")
