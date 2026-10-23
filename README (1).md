# TGCT tumour-microenvironment programmes

Code for the manuscript:

> **Histology-independent immune and stromal programmes in testicular germ cell tumours: single-cell-anchored deconvolution, spatial organisation and absence of prognostic value in stage I disease.**
> [Authors]. *Frontiers in Immunology* (submitted).

All analyses use public, de-identified data. No data are stored in this repository.

## Pipeline

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 0 | `step0_data_audit.py` | Downloads and audits the public datasets | `step0_audit/` |
| 0b | `step0b_validation_cohort_and_tcga_clinical.py` | Harmonises GSE99420 and TCGA-TGCT clinical data | `data/processed/` |
| 1 | `stage1_sc_atlas.py` | Single-cell atlas: QC, Scrublet, scVI, lineage annotation, inferCNV/12p malignant calls, fine states, deconvolution reference | `stage1_atlas/` |
| 2 | `stage2_tcga_ecosystem.py` | Two-pass multinomial deconvolution of TCGA-TGCT, leave-one-dataset-out benchmark, histology-adjusted consensus NMF programmes | `stage2_tcga/` |
| 3 | `stage3_clinical_models.py` | Pre-specified stage I relapse analysis (Cox, optimism-corrected C-index, decision curves) and GSE99420 validation with locked signatures | `stage3_clinical/` |
| 4 | `stage4_mechanism_spatial.py` | Visium spatial analysis, ligand-receptor scoring, LIANA+, TCGA copy-number/mutation associations | `stage4_mechanism/` |
| 5 | `stage5_drug_repurposing.py` | Target nomination, DGIdb 5.0 annotation, L1000CDS2 reversal (hypothesis-generating) | `stage5_drugs/` |
| Figures | `make_figures.py` | Figures 1-7 and the supplementary-table workbook | `figures_final/` |
| — | `software_versions.py` | Records package versions | `software_versions.txt` |

Run the scripts in order. Each stage reads only the outputs of earlier stages and writes a `STAGE*_REPORT.md`.

## Data

| Accession | Type | Use |
|---|---|---|
| GEO GSE228501 | scRNA-seq, 4 primary seminomas | Atlas |
| GEO GSE261811 | scRNA-seq, 4 primary TGCT | Atlas |
| GEO GSE197778 | scRNA-seq, primary, 2 lymph-node metastases, blood (1 patient) | Atlas, communication |
| GEO GSE112013 | scRNA-seq, normal adult testis | Atlas |
| GEO GSE229823 | 10x Visium, 3 seminoma sections | Spatial analysis |
| GEO GSE99420 | WG-DASL, stage I TGCT on surveillance | External validation |
| TCGA-TGCT (UCSC Xena) | RNA-seq, clinical, MC3, GISTIC2 | Discovery cohort |

## Setup

```bash
conda env create -f environment.yml
conda activate scvi-env
export TGCT_BASE="/path/to/project"     # folder that will contain TGCT_ecotype/
python step0_data_audit.py
```

A GPU is recommended for scVI (stage 1); the other stages run on CPU.

## Reproducibility

- Random seeds are fixed (`SEED = 0`).
- The relapse analysis plan is written to `stage3_clinical/ANALYSIS_PLAN.json`, with input checksums, before any survival model is fitted.
- Package versions used for the manuscript are listed in `environment.yml`.

## Citation

Please cite the manuscript above (details will be added on publication).

## Licence

See `LICENSE`.
