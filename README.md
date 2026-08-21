# QM640 Governance Analytics — Data-Driven Quality and Risk Analytics

Capstone project comparing governance-relevant change in the software QA/defect domain (AI-coding era) and the IT audit/disclosure domain (PCAOB inspection data), for QM640 at Walsh College.

## Repository Structure

```
qm640-governance-analytics/
├── data/
│   ├── raw/
│   │   ├── apache_jira_raw.csv                  # 30,733 real JIRA issues (Camel + Hadoop)
│   │   ├── pcaob_deficiencies_raw.csv           # 16,704 real PCAOB deficiency records
│   │   ├── nasa_promise_arff/                   # 13 real source ARFF files (8 used)
│   │   └── nasa_promise_combined_raw.csv        # 12,655 rows, NASA/PROMISE substitute
│   └── cleaned/
│       ├── qa_defect_dataset.csv                # Real JIRA data, era + resolution_time_days added
│       ├── audit_disclosure_dataset.csv         # Real PCAOB data, leaked fields removed
│       ├── nasa_promise_combined_clean.csv      # 12,639 rows, post dedup + imputation
│       └── rq1_real_mined_dataset.csv           # 1,804 real mined code-metric samples
├── notebooks/                                    # numbered in pipeline order, all executed
│   ├── 01_extract_apache_jira.ipynb              # Real JIRA extraction (documented)
│   ├── 02_extract_pcaob_sec_edgar.ipynb          # PCAOB done; SEC EDGAR still pending
│   ├── 03_data_cleaning.ipynb                    # Cleans both datasets; removes leaked PCAOB fields
│   ├── 04_nasa_promise_data_loading.ipynb        # Loads 8 real NASA/PROMISE ARFF files
│   ├── 05_nasa_promise_cleaning_eda_baseline_model.ipynb  # EDA + LogReg/RF baseline (5 figures inline)
│   ├── 06_nasa_promise_statistical_testing.ipynb # McNemar's test, bootstrap 95% CIs
│   ├── 07_rq1_code_metric_mining_analysis.ipynb  # REAL RQ1: mined Camel/Hadoop code metrics
│   ├── 08_rq2_era_analysis.ipynb                 # REAL RQ2: era-moderated regression, real JIRA data
│   ├── 09_rq3_pcaob_severity_model.ipynb         # REAL RQ3: PCAOB severity, leakage-corrected
│   ├── 10_rq4_remediation_time_model.ipynb       # SCAFFOLD ONLY -- needs SEC EDGAR data
│   └── 11_rq5_cross_domain_synthesis.ipynb       # SCAFFOLD ONLY -- needs RQ4's real result
├── figures/                                      # fig0 (pipeline) through fig8 (RQ1 mining)
├── reports/Daljeet_Kaur_Johar_QM640_Interim_Report.docx
└── synopsis/Daljeet_Kaur_Johar_QM640_Synopsis.docx
```

All 9 runnable notebooks (01-09) were actually executed -- every cell's real output,
including figures, is already saved inside the `.ipynb` files. Notebooks 10 and 11
are honestly marked as not-yet-runnable scaffolds (they print their own status
message explaining exactly what's missing, rather than erroring silently).

## What's Real vs. What's Still a Substitute or Pending

| Research Question | Status | Data Source |
|---|---|---|
| RQ1 (defect-proneness x era) | **Real** | Mined Camel/Hadoop code metrics (1,804 samples) |
| RQ2 (resolution time x era) | **Real** | Real Apache JIRA data (30,733 issues) |
| RQ3 (PCAOB severity) | **Real** | Real PCAOB data (16,704 records), leakage caught & fixed |
| RQ4 (remediation time) | **Pending** | Needs real SEC EDGAR extraction (not yet run) |
| RQ5 (cross-domain synthesis) | **Pending** | Needs RQ4's real result first |

The NASA/PROMISE dataset (notebooks 04-06) was the original interim-stage substitute
for RQ1 before the real code-metric mining pipeline (notebook 07) was built; it now
serves as a cross-validation reference rather than RQ1's primary evidence.

**Known labeling caveat (RQ1):** `defect_prone = 1` in the mined dataset means
"commit linked to a tracked, resolved JIRA issue," not a strict "confirmed bug"
label, since the JIRA extraction did not capture the issue-type field. See the
report's Limitations section for full detail.

## Reproducing the Analysis

```bash
pip install -r requirements.txt
jupyter notebook
# Run in order: 01 -> 02 (PCAOB half only; SEC EDGAR half is commented out) -> 03 ->
# 04 -> 05 -> 06 -> 07 -> 08 -> 09
# Notebook 07 will clone the real apache/camel and apache/hadoop repos on first run
# (large, full commit history) if they aren't already present under ../repos/.
```

## Key Real Results (Summary)

| Domain | Model | Test AUC | Test Accuracy |
|---|---|---|---|
| RQ1 (code-metric mining) | Random Forest | 0.730 | 0.734 |
| RQ2 (JIRA era regression) | OLS (era interactions) | -- | Cohen's f-squared = 0.0011 (negligible) |
| RQ3 (PCAOB severity) | Random Forest | 0.986 | 0.971 |
| NASA/PROMISE baseline | Random Forest | 0.731 | 0.752 |

McNemar's test (NASA/PROMISE LogReg vs. RF): chi-square = 26.28, p < .0001.
