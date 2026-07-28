# QM640 Governance Analytics — Data-Driven Quality and Risk Analytics

Capstone project comparing governance-relevant change in the software QA/defect domain (AI-coding era) and the IT audit/disclosure domain (PCAOB inspection data), for QM640 at Walsh College.

## Repository Structure

```
qm640-governance-analytics/
├── data/
│   ├── raw/nasa_promise_combined_raw.csv      # 12,655 rows, 8 NASA/PROMISE projects, as-loaded
│   └── clean/nasa_promise_combined_clean.csv  # 12,639 rows, post dedup + median imputation
├── notebooks/
│   ├── 1_data_loading.py                      # Loads 8 ARFF files, combines, saves raw CSV
│   ├── 2_cleaning_eda_modeling.py             # Cleaning, EDA, LogReg/RF baseline models, figures
│   └── 3_statistical_significance_testing.py  # McNemar's test, bootstrap 95% CIs
├── figures/                                    # fig0 (pipeline) through fig5 (feature importance)
├── reports/Daljeet_Kaur_Johar_QM640_Interim_Report.docx
└── synopsis/Daljeet_Kaur_Johar_QM640_Synopsis.docx
```

## Data Provenance

**QA/defect domain (this interim stage):** 8 real, peer-reviewed NASA/PROMISE software defect
datasets (CM1, JM1, KC1, KC3, MW1, PC1, PC3, PC4), originally released by Shepperd, Song, Sun,
and Mair (2014), mirrored for research reuse at
[github.com/klainfo/NASADefectDataset](https://github.com/klainfo/NASADefectDataset).

**Not yet included (planned for final report):**
- Apache JIRA issue data (timestamped, for the pre-AI vs. AI-coding era comparison in RQ1/RQ2)
- PCAOB Part I.A / Part I.B inspection datasets (RQ3)
- SEC EDGAR material-weakness filings, 2020+ (RQ4)

These three sources were identified (real download endpoints located) but could not be pulled
into the sandboxed analysis environment used to prepare the interim report. See the Interim
Report's "Limitations and Risks" and "Next Steps" sections for details.

## Reproducing the Analysis

```bash
pip install -r requirements.txt
python notebooks/1_data_loading.py            # requires the 8 .arff files (see note below)
python notebooks/2_cleaning_eda_modeling.py
python notebooks/3_statistical_significance_testing.py
```

Note: `1_data_loading.py` expects the 8 `.arff` files from the NASADefectDataset repo's
`CleanedData/MDP/D''/` folder to be present locally (clone that repo, or download the individual
files). `data/raw/` and `data/clean/` in this repo already contain the output of that step, so
you can skip straight to script 2 if you just want to reproduce the EDA/modeling/statistics.

## Key Preliminary Results (Interim, QA/Defect Domain)

| Model | Test AUC | Test Accuracy | Test F1 |
|---|---|---|---|
| Logistic Regression | 0.719 | 0.717 | 0.429 |
| Random Forest | 0.731 | 0.752 | 0.427 |

McNemar's test (LogReg vs. RF): χ² = 26.28, p < .0001 — significant per-module disagreement.
Bootstrap 95% CI on AUC difference (RF − LogReg): [-0.002, 0.027] — includes zero.
