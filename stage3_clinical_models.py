#!/usr/bin/env python
"""
STAGE 3 / 4  -  Relapse models in stage I TGCT: TCGA discovery + GSE99420 independent validation
=================================================================================================
Pre-specified analysis (written to stage3_clinical/ANALYSIS_PLAN.json BEFORE any outcome is touched;
the plan file is never overwritten, so later runs are checked against it).

Predictors (continuous, z-scored)
  P1  TME immune/lymphoid programme score          (stage 2, histology-adjusted NMF)
  P2  TME vascular-stromal / M2 programme score    (stage 2, histology-adjusted NMF)
  P3  tumour EC-likeness = 1 - SE-likeness          (stage 2 deconvolution; molecular proxy of %EC)
  P4  IFNG-effector T-cell programme                (replication of Cell Death Discov 2025)
  P5  CD163+ macrophage fraction of the TME         (replication of Cell Death Discov 2025)

Discovery: TCGA-TGCT stage I, endpoint PFI (secondary DFI)
  M0 = histology + LVI ;  M1 = M0 + predictor
  HR per SD (95% CI), LRT M1 vs M0, Harrell C, bootstrap optimism-corrected C (200x), BH-FDR over P1-P5,
  predictor x histology interaction, sensitivity: + adjuvant radiotherapy, all stages + stage, DFI,
  decision-curve analysis at 3 years for the best predictor.

Validation: GSE99420 (30 relapse / 30 no relapse, 15+15 per histology, FFPE WG-DASL)
  gene signatures LOCKED in TCGA (within-histology correlation with each predictor, genes restricted to
  those measured in GSE99420), scored by singscore; AUC (bootstrap CI), logistic OR per SD adjusted for
  histology, within-histology Wilcoxon.

Run:  conda activate scvi-env ; pip install lifelines ; python stage3_clinical_models.py
"""
import os, re, json, time, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, norm
warnings.filterwarnings("ignore")

BASE = Path(os.environ.get("TGCT_BASE", "/home/arshia/testis project"))
PROJ = BASE / "TGCT_ecotype"
S2 = PROJ / "stage2_tcga"
PROC = PROJ / "data" / "processed"
OUT = PROJ / "stage3_clinical"
FIG = OUT / "figures"
for d in (OUT, FIG):
    d.mkdir(parents=True, exist_ok=True)
SEED = 0
N_BOOT = 200 if not os.environ.get("TGCT_TEST") else 20
SIG_SIZE = 50
DCA_DAYS = 3 * 365
LOG = []


def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    LOG.append(s)


def sha(f):
    return hashlib.md5(Path(f).read_bytes()).hexdigest()[:12] if Path(f).exists() else None


IFNG_SET = ["IFNG", "TNF", "CCL4", "CCL3", "XCL1", "XCL2", "CRTAM", "GZMB", "PRF1", "CXCL9", "CXCL10", "STAT1"]
CD163_SET = ["CD163", "MRC1", "F13A1", "STAB1", "LYVE1", "SIGLEC1", "MS4A4A", "FOLR2"]
IMMUNE_TYPES = r"^(CD4_|CD8_|Treg|T_|NK$|MAIT|B_|Plasma|Macro_|Mono_|cDC|mregDC|pDC)"


# ============================================================ data assembly
def assemble_tcga():
    cl = pd.read_csv(PROC / "TCGA_TGCT_clinical_harmonised.csv", index_col=0)
    cl.index = cl.index.str[:15]
    adj = pd.read_csv(S2 / "S5c_adjusted_ecotypes.csv", index_col=0)
    sel = pd.read_csv(S2 / "S5c_tumour_SE_likeness.csv", index_col=0)
    load = pd.read_csv(S2 / "S5c_programme_state_loadings.csv", index_col=0)      # states x programmes
    st = pd.read_csv(S2 / "tcga_state_fractions.csv", index_col=0)
    tme = pd.read_csv(S2 / "S5_tme_state_composition.csv", index_col=0)
    bulk = pd.read_csv(S2 / "bayesprism_input" / "tcga_bulk_counts.csv.gz", index_col=0)
    # name programmes by content: immune share of loadings
    imm = load.index.str.match(IMMUNE_TYPES)
    share = (load[imm].sum() / load.sum()).sort_values(ascending=False)
    p_imm, p_str = share.index[0], share.index[-1]
    log(f"  programme naming: immune = {p_imm} (immune loading share {share.iloc[0]:.2f}); "
        f"vascular-stromal = {p_str} ({share.iloc[-1]:.2f})")
    d = pd.DataFrame(index=adj.index)
    d["P1_TME_immune"] = adj[f"score_{p_imm}"]
    d["P2_TME_stromal"] = adj[f"score_{p_str}"]
    d["P3_EC_likeness"] = 1 - sel["SE_likeness"].reindex(d.index)
    # P4: IFNG programme (singscore on bulk) ; P5: CD163 macrophages as share of TME
    logx = log_cpm(bulk)
    d["P4_IFNG_programme"] = singscore(logx, IFNG_SET)[0].reindex(d.index)
    d["P5_CD163_macro"] = tme["Macro_CD163_M2"].reindex(d.index) if "Macro_CD163_M2" in tme else np.nan
    d["histology"] = adj["histology_used"]
    d["NSGCT"] = (d["histology"] == "NSGCT").astype(int)
    lvi = cl["lymphovascular_invasion_present"].astype(str).str.upper().reindex(d.index)
    d["LVI"] = lvi.map({"YES": 1, "NO": 0})
    pt = cl["pathologic_T"].astype(str).str[:2].reindex(d.index)
    d["pT2plus"] = pt.map(lambda x: 1 if x in ("T2", "T3", "T4") else (0 if x == "T1" else np.nan))
    d["stage_I"] = cl["stage_I"].reindex(d.index)
    d["stage_group"] = cl["stage_group"].reindex(d.index)
    rt = cl["radiation_therapy"].astype(str).str.upper().reindex(d.index) if "radiation_therapy" in cl else pd.Series(np.nan, index=d.index)
    d["adjuvant_RT"] = rt.map({"YES": 1, "NO": 0})
    for c in ("PFI", "PFI.time", "DFI", "DFI.time"):
        d[c] = cl[c].reindex(d.index) if c in cl else np.nan
    return d, logx, (p_imm, p_str)


def log_cpm(counts):
    cpm = counts.div(counts.sum(1), axis=0) * 1e6
    x = np.log2(cpm + 1)
    return x.loc[:, (cpm > 1).mean() > 0.2]


def singscore(expr_log, genes, down=None):
    """rank-based single-sample score (Foroutan 2018); bidirectional if `down` given"""
    R = expr_log.rank(axis=1, method="average")
    n = expr_log.shape[1]

    def one(gs):
        g = [x for x in gs if x in expr_log.columns]
        if len(g) < 5:
            return pd.Series(np.nan, index=expr_log.index), len(g)
        s = R[g].mean(1)
        lo, hi = (len(g) + 1) / 2, n - (len(g) - 1) / 2
        return (s - lo) / (hi - lo) - 0.5, len(g)
    up, nu = one(genes)
    if down is None:
        return up, nu
    dn, nd = one(down)
    return up - dn, nu + nd


# ============================================================ survival helpers
def cox(df, dur, ev, covs, penal=0.0):
    from lifelines import CoxPHFitter
    x = df[[dur, ev] + covs].dropna()
    for p in (penal, 0.05, 0.2):
        try:
            m = CoxPHFitter(penalizer=p).fit(x, dur, ev)
            return m, x, p
        except Exception:
            continue
    raise RuntimeError("Cox failed")


def cindex(m, x, dur, ev):
    from lifelines.utils import concordance_index
    return concordance_index(x[dur], -m.predict_partial_hazard(x), x[ev])


def boot_optimism(df, dur, ev, covs, n=N_BOOT, seed=SEED):
    """Harrell's bootstrap optimism correction of the C-index"""
    m, x, p = cox(df, dur, ev, covs)
    app = cindex(m, x, dur, ev)
    rng = np.random.default_rng(seed)
    opt = []
    for b in range(n):
        xb = x.iloc[rng.integers(0, len(x), len(x))]
        if xb[ev].sum() < 5:
            continue
        try:
            mb, _, _ = cox(xb, dur, ev, covs, penal=p)
            opt.append(cindex(mb, xb, dur, ev) - cindex(mb, x, dur, ev))
        except Exception:
            continue
    return app, app - np.mean(opt) if opt else np.nan


def lrt(m0, m1):
    from scipy.stats import chi2
    stat = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    df = len(m1.params_) - len(m0.params_)
    return stat, chi2.sf(stat, df)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def decision_curve(df, dur, ev, covs_list, t=DCA_DAYS, names=None):
    from lifelines import KaplanMeierFitter
    ths = np.arange(0.05, 0.51, 0.025)
    out = {"threshold": ths}
    km_all = KaplanMeierFitter().fit(df[dur], df[ev])
    risk_all = 1 - float(km_all.predict(t))
    out["treat_all"] = risk_all - (1 - risk_all) * ths / (1 - ths)
    out["treat_none"] = np.zeros_like(ths)
    for covs, nm in zip(covs_list, names):
        m, x, _ = cox(df, dur, ev, covs)
        risk = 1 - m.predict_survival_function(x, times=[t]).T.iloc[:, 0]
        nb = []
        for pt in ths:
            pos = risk >= pt
            if pos.sum() == 0:
                nb.append(0.0); continue
            km = KaplanMeierFitter().fit(x.loc[pos, dur], x.loc[pos, ev])
            r = 1 - float(km.predict(t))
            ppos = pos.mean()
            nb.append(r * ppos - (1 - r) * ppos * pt / (1 - pt))
        out[nm] = np.array(nb)
    return pd.DataFrame(out)


# ============================================================ signatures locked in TCGA
def within_hist_spearman(logx, score, hist):
    """mean of within-histology Spearman correlations, weighted by n (removes SE-vs-NSGCT confounding)"""
    rs, ws = [], []
    for h in hist.dropna().unique():
        idx = hist.index[(hist == h)].intersection(score.dropna().index)
        if len(idx) < 15:
            continue
        r = logx.loc[idx].rank().corrwith(score.loc[idx].rank())
        rs.append(r.fillna(0)); ws.append(len(idx))
    return sum(r * w for r, w in zip(rs, ws)) / sum(ws)


def lock_signatures(d, logx, genes_valid):
    f = OUT / "LOCKED_signatures.json"
    if f.exists():
        return json.load(open(f))
    x = logx.loc[:, logx.columns.isin(genes_valid)]
    sig = {}
    for p in ("P1_TME_immune", "P2_TME_stromal"):
        r = within_hist_spearman(x, d[p], d["histology"])
        sig[p] = {"up": r.sort_values(ascending=False).head(SIG_SIZE).index.tolist(), "down": None,
                  "min_rho": float(r.sort_values(ascending=False).iloc[SIG_SIZE - 1])}
    # EC-likeness: variation is meaningful mainly inside NSGCT
    ns = d.index[d["histology"] == "NSGCT"]
    r = x.loc[ns].rank().corrwith(d.loc[ns, "P3_EC_likeness"].rank()).dropna().sort_values()
    sig["P3_EC_likeness"] = {"up": r.tail(SIG_SIZE).index[::-1].tolist(), "down": r.head(SIG_SIZE).index.tolist(),
                             "min_rho": float(r.tail(SIG_SIZE).iloc[0])}
    sig["P4_IFNG_programme"] = {"up": IFNG_SET, "down": None}
    sig["P5_CD163_macro"] = {"up": CD163_SET, "down": None}
    json.dump(sig, open(f, "w"), indent=1)
    log(f"  signatures locked (TCGA, restricted to {len(genes_valid)} genes measured in GSE99420) -> {f.name}")
    return sig


def auc_ci(y, s, n=2000, seed=SEED):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    a = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], s[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def logit_or(y, x, adj):
    import statsmodels.api as smapi
    X = smapi.add_constant(pd.concat([x.rename("score"), adj], axis=1).astype(float))
    m = smapi.Logit(y.astype(float), X).fit(disp=0)
    b, se = m.params["score"], m.bse["score"]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues["score"]


# ============================================================ RUN
if __name__ == "__main__":
    PRED_ALL = ["P1_TME_immune", "P2_TME_stromal", "P3_EC_likeness", "P4_IFNG_programme", "P5_CD163_macro"]
    PRED = PRED_ALL
    plan_f = OUT / "ANALYSIS_PLAN.json"
    inputs = {str(p.relative_to(PROJ)): sha(p) for p in [S2 / "S5c_adjusted_ecotypes.csv", S2 / "S5c_tumour_SE_likeness.csv",
                                                          S2 / "tcga_state_fractions.csv", PROC / "TCGA_TGCT_clinical_harmonised.csv",
                                                          PROC / "GSE99420_expr_gene_log2.csv.gz"]}
    if not plan_f.exists():
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), predictors=PRED,
                       primary="TCGA stage I, PFI, Cox M0=histology+LVI vs M1=M0+predictor (per SD), BH-FDR over predictors",
                       secondary=["DFI", "all stages + stage", "+ adjuvant RT", "predictor x histology", "DCA 3y"],
                       validation="GSE99420: locked TCGA signatures, singscore, AUC + logistic OR adj. histology",
                       signature_size=SIG_SIZE, inputs_md5=inputs), open(plan_f, "w"), indent=1)
        log(f"  analysis plan written (pre-specification) -> {plan_f.name}")
    else:
        old = json.load(open(plan_f))
        if old.get("inputs_md5") != inputs:
            log("  !! inputs differ from those recorded in ANALYSIS_PLAN.json - report this deviation")

    d, logx, prog = assemble_tcga()
    d.to_csv(OUT / "S3_tcga_analysis_table.csv")
    for p in PRED:
        d[p + "_z"] = (d[p] - d[p].mean()) / d[p].std()
    s1 = d[(d["stage_I"] == 1) & d["PFI.time"].notna() & (d["PFI.time"] > 0)].copy()
    log(f"  TCGA stage I: n={len(s1)}, PFI events={int(s1['PFI'].sum())}, LVI known={int(s1['LVI'].notna().sum())}")

    # ---------------- primary analysis
    base = ["NSGCT", "LVI"]
    rows = []
    m0, x0, _ = cox(s1, "PFI.time", "PFI", base)
    c0_app, c0_opt = boot_optimism(s1, "PFI.time", "PFI", base)
    for p in PRED:
        cov = base + [p + "_z"]
        s = s1.dropna(subset=cov)
        if len(s) < 20 or s["PFI"].sum() < 5:
            log(f"  !! {p}: only {len(s)} complete cases - skipped")
            continue
        m0s, _, _ = cox(s, "PFI.time", "PFI", base)
        m1, x1, pen = cox(s, "PFI.time", "PFI", cov)
        st, plrt = lrt(m0s, m1)
        hr = m1.hazard_ratios_[p + "_z"]
        ci = np.exp(m1.confidence_intervals_.loc[p + "_z"].values)
        c1_app, c1_opt = boot_optimism(s, "PFI.time", "PFI", cov)
        c0s_app, c0s_opt = boot_optimism(s, "PFI.time", "PFI", base)
        # univariable and interaction
        mu, _, _ = cox(s, "PFI.time", "PFI", [p + "_z"])
        s = s.assign(inter=s[p + "_z"] * s["NSGCT"])
        mi, _, _ = cox(s, "PFI.time", "PFI", cov + ["inter"])
        rows.append(dict(predictor=p, n=len(s), events=int(s["PFI"].sum()),
                         HR_univ=mu.hazard_ratios_[p + "_z"], p_univ=mu.summary.loc[p + "_z", "p"],
                         HR_adj=hr, CI_low=ci[0], CI_high=ci[1], p_adj=m1.summary.loc[p + "_z", "p"],
                         LRT=st, p_LRT=plrt, C_M0=c0s_opt, C_M1=c1_opt, dC_optimism_corr=c1_opt - c0s_opt,
                         p_interaction_histology=mi.summary.loc["inter", "p"], penalizer=pen))
    prim = pd.DataFrame(rows)
    PRED = prim["predictor"].tolist()
    prim["q_LRT_BH"] = bh(prim["p_LRT"])
    prim.to_csv(OUT / "S3_primary_stageI_PFI.csv", index=False)
    log("  PRIMARY (stage I, PFI, adjusted for histology + LVI):\n" +
        prim[["predictor", "HR_adj", "CI_low", "CI_high", "p_LRT", "q_LRT_BH", "dC_optimism_corr"]].round(3).to_string(index=False))

    # ---------------- secondary / sensitivity
    sec = []
    allst = d[d["PFI.time"].notna() & (d["PFI.time"] > 0) & d["stage_group"].notna()].copy()
    allst["stage_II_III"] = (allst["stage_group"] != "I").astype(int)
    dfi = s1[s1["DFI.time"].notna() & (s1["DFI.time"] > 0)].copy()
    for p in PRED:
        z = p + "_z"
        for name, df_, dur, ev, cov in (
                ("DFI stage I", dfi, "DFI.time", "DFI", base + [z]),
                ("PFI all stages +stage", allst, "PFI.time", "PFI", base + ["stage_II_III", z]),
                ("PFI stage I +adjuvant RT", s1, "PFI.time", "PFI", base + ["adjuvant_RT", z]),
                ("PFI stage I +pT2+ (no LVI)", s1, "PFI.time", "PFI", ["NSGCT", "pT2plus", z]),
                ("PFI stage I NSGCT only", s1[s1["NSGCT"] == 1], "PFI.time", "PFI", ["LVI", z]),
                ("PFI stage I SE only", s1[s1["NSGCT"] == 0], "PFI.time", "PFI", ["LVI", z])):
            try:
                x = df_.dropna(subset=cov + [dur, ev])
                if x[ev].sum() < 5:
                    continue
                m, _, _ = cox(x, dur, ev, cov)
                ci = np.exp(m.confidence_intervals_.loc[z].values)
                sec.append(dict(analysis=name, predictor=p, n=len(x), events=int(x[ev].sum()),
                                HR=m.hazard_ratios_[z], CI_low=ci[0], CI_high=ci[1], p=m.summary.loc[z, "p"]))
            except Exception as e:
                sec.append(dict(analysis=name, predictor=p, error=str(e)[:60]))
    sec = pd.DataFrame(sec)
    sec.to_csv(OUT / "S3_secondary_sensitivity.csv", index=False)

    # ---------------- decision curve for the strongest predictor
    best = prim.sort_values("p_LRT").iloc[0]["predictor"]
    dca = decision_curve(s1.dropna(subset=base + [best + "_z"]), "PFI.time", "PFI",
                         [base, base + [best + "_z"]], names=["M0_histology_LVI", f"M1_+{best}"])
    dca.to_csv(OUT / "S3_decision_curve_3y.csv", index=False)

    # ---------------- figures: forest + KM (median split, within-histology standardised)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter
        fig, ax = plt.subplots(figsize=(6, 3))
        y = np.arange(len(prim))[::-1]
        ax.errorbar(prim["HR_adj"], y, xerr=[prim["HR_adj"] - prim["CI_low"], prim["CI_high"] - prim["HR_adj"]], fmt="o", color="k")
        ax.axvline(1, ls="--", c="grey"); ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(prim["predictor"])
        ax.set_xlabel("HR per SD (adj. histology + LVI), stage I PFI")
        fig.savefig(FIG / "S3_forest_primary.png", dpi=200, bbox_inches="tight"); plt.close(fig)
        fig, axes = plt.subplots(1, len(PRED), figsize=(4 * len(PRED), 3.4))
        for ax, p in zip(axes, PRED):
            s = s1.dropna(subset=[p])
            hi = s[p] > s.groupby("histology")[p].transform("median")
            for lab, m in (("high", hi), ("low", ~hi)):
                KaplanMeierFitter().fit(s.loc[m, "PFI.time"] / 365, s.loc[m, "PFI"], label=f"{lab} (n={m.sum()})").plot_survival_function(ax=ax, ci_show=False)
            ax.set_title(p, fontsize=9); ax.set_xlabel("years"); ax.set_ylim(0, 1.02)
        fig.tight_layout(); fig.savefig(FIG / "S3_KM_stageI_PFI.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for c in dca.columns[1:]:
            ax.plot(dca["threshold"], dca[c], label=c)
        ax.set_ylim(-0.05, max(0.05, dca.iloc[:, 1:].values.max() * 1.1)); ax.set_xlabel("threshold risk at 3 y"); ax.set_ylabel("net benefit"); ax.legend(fontsize=7)
        fig.savefig(FIG / "S3_DCA_3y.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        log(f"  figure error: {e}")

    # ---------------- validation GSE99420
    vx = pd.read_csv(PROC / "GSE99420_expr_gene_log2.csv.gz", index_col=0).T          # samples x genes
    vc = pd.read_csv(PROC / "GSE99420_clinical.csv", index_col=0).reindex(vx.index)
    sig = lock_signatures(d, logx, set(vx.columns))
    # sanity: locked signature reproduces the TCGA predictor
    sanity, val = [], []
    for p in PRED:
        sc_t, ng = singscore(logx, sig[p]["up"], sig[p]["down"])
        rho = spearmanr(sc_t.reindex(d.index), d[p], nan_policy="omit")[0]
        sc_v, ngv = singscore(vx, sig[p]["up"], sig[p]["down"])
        sanity.append(dict(predictor=p, genes_tcga=ng, genes_gse99420=ngv, rho_signature_vs_tcga_predictor=rho))
        y = vc["relapse"].astype(int)
        z = (sc_v - sc_v.mean()) / sc_v.std()
        a, lo, hi = auc_ci(y, z)
        orr, olo, ohi, pp = logit_or(y, z, pd.get_dummies(vc["histology"], drop_first=True).astype(float))
        w = {}
        for h in ("SE", "NSGCT"):
            m = vc["histology"] == h
            w[h] = mannwhitneyu(z[m & (y == 1)], z[m & (y == 0)]).pvalue
        val.append(dict(predictor=p, AUC=a, AUC_low=lo, AUC_high=hi, OR_per_SD_adj_hist=orr, OR_low=olo, OR_high=ohi,
                        p_logistic=pp, p_wilcox_SE=w["SE"], p_wilcox_NSGCT=w["NSGCT"],
                        expected_direction="HR>1" if prim.set_index("predictor").loc[p, "HR_adj"] > 1 else "HR<1"))
        vc[p] = z
    sanity = pd.DataFrame(sanity); val = pd.DataFrame(val)
    sanity.to_csv(OUT / "S3_signature_sanity.csv", index=False)
    val.to_csv(OUT / "S3_validation_GSE99420.csv", index=False)
    vc.to_csv(OUT / "S3_gse99420_scores.csv")
    log("  VALIDATION GSE99420:\n" + val[["predictor", "AUC", "AUC_low", "AUC_high", "OR_per_SD_adj_hist", "p_logistic"]].round(3).to_string(index=False))

    md = ["# STAGE 3 report - relapse models (stage I)", "",
          f"- programmes: immune = {prog[0]}, vascular-stromal = {prog[1]}",
          f"- TCGA stage I: n={len(s1)}, PFI events={int(s1['PFI'].sum())}; base model M0 = histology + LVI "
          f"(optimism-corrected C = {c0_opt:.3f})", "",
          "## Primary analysis (TCGA stage I, PFI)", prim.round(4).to_markdown(index=False), "",
          "## Secondary / sensitivity", sec.round(4).to_markdown(index=False), "",
          f"## Decision curve (3 y) - best predictor: {best}", dca.round(4).to_markdown(index=False), "",
          "## Locked signatures: reproduce TCGA predictor?", sanity.round(3).to_markdown(index=False), "",
          "## Independent validation GSE99420 (30 R / 30 NR)", val.round(4).to_markdown(index=False), "",
          "## Log", "```", *LOG, "```"]
    (OUT / "STAGE3_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log(f"done -> {OUT / 'STAGE3_REPORT.md'}")
