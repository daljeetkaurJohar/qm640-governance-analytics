# QM640 Governance Analytics — Data-Driven Quality and Risk Analytics

Capstone project comparing governance-relevant change in the software QA/defect domain (AI-coding era) and the IT audit/disclosure domain (PCAOB inspection data and SEC EDGAR material-weakness disclosures), for QM640 at Walsh College.

**Status: all five research questions (RQ1–RQ5) are complete and executed.** This supersedes an earlier version of this README that described RQ4 and RQ5 as pending — that was out of date; the analysis for both was finished and is in the repository, just not previously documented here.

## Final, Authoritative Notebooks

The repository contains some earlier exploratory notebooks alongside final versions for a few research questions. **If you only run one set of notebooks per research question, run these:**

| RQ | Final notebook(s) | Notes |
|---|---|---|
| RQ1 | `RQ1_1_Camel.ipynb`, `RQ1_2_Hadoop.ipynb`, `RQ1_3_Kafka.ipynb`, `RQ1_4_Tika.ipynb`, `RQ1_5_EDA_Complete_Analysis.ipynb`, `RQ1_6_Interpretation_and_Robustness_UPDATED.ipynb` | Use the `_UPDATED` version of notebook 6, not `RQ1_6_Interpretation_and_Robustness.ipynb` (superseded). |
| RQ2 | `RQ2_1_JIRA_Extraction.ipynb`, `RQ2__full_reassignments.ipynb`, `RQ2_5_Interpretation_and_Robustness.ipynb` | `RQ2__full_reassignments.ipynb` is the full-population (N=27,385) extraction; `RQ2_2_Reassignments.ipynb` and `RQ2_3_Analysis.ipynb` are earlier, partial-sample versions (N=3,000), kept for history only. |
| RQ3 | `RQ3_1_PCAOB_Extraction.ipynb`, `RQ3_2_Model_Training_v2.ipynb`, `RQ3_3_SHAP_v2.ipynb`, `RQ3_5_Interpretation_and_Robustness.ipynb` | Use the `_v2` versions of notebooks 2 and 3, not the originals (superseded). `RQ3_5` is the final robustness pass (complete-case, temporal holdout, feature ablation) and is the source of the report's headline RQ3 numbers. |
| RQ4 | `RQ4_1_Analysis.ipynb`, `RQ4_2_EDA.ipynb` | |
| RQ5 | `RQ5_1_Point_Estimate_Synthesis.ipynb`, `RQ5_2_Bootstrap_CI.ipynb`, `RQ5_3_EDA.ipynb` | **Known issue:** both RQ5 notebooks still group Camel and Hadoop as one combined project (via `rq1_refined_with_real_issue_type.csv`), predating the RQ1 four-project correction above. They produce a qualitatively similar but not identical result to the report's primary RQ5 figures (see the report's Methodological Rigor and Reproducibility section for the exact numbers and the discrepancy). **These two notebooks need to be re-run against the four separate `*_real_mined_dataset.csv` files before they can be treated as the authoritative RQ5 source.** |

Superseded/exploratory notebooks kept for history (do not treat as authoritative): `RQ1_6_Interpretation_and_Robustness.ipynb` (pre-`_UPDATED`), `RQ2_2_Reassignments.ipynb`, `RQ2_3_Analysis.ipynb`, `RQ2_4_EDA.ipynb`, `RQ2_External_validation.ipynb`, `RQ3_2_Model_Training.ipynb` (pre-`_v2`), `RQ3_3_SHAP.ipynb` (pre-`_v2`), `RQ3_4_EDA.ipynb`.

The NASA/PROMISE cross-validation baseline notebooks (`04_nasa_promise_data_loading.ipynb`, `05_nasa_promise_cleaning_eda_baseline_model.ipynb`, `06_nasa_promise_statistical_testing.ipynb`) are unaffected by any of the above and remain as originally run.

## Repository Structure

```
qm640-governance-analytics/
├── data/
│   ├── raw/
│   │   ├── apache_jira_raw.csv                  # 30,733 JIRA issues (Camel + Hadoop)
│   │   ├── pcaob_deficiencies_raw.csv           # PCAOB deficiency records, pre-cleaning
│   │   ├── pcaob_part1a_raw.csv / pcaob_part1b_raw.csv
│   │   ├── num_reassignments_FULL.csv           # Full-population JIRA changelog extraction
│   │   ├── nasa_promise_arff/                   # Source ARFF files for the NASA/PROMISE baseline
│   │   └── nasa_promise_combined_raw.csv
│   └── cleaned/
│       ├── camel_real_mined_dataset.csv         # RQ1: N=12,598, strict Bug-type label
│       ├── hadoop_real_mined_dataset.csv        # RQ1: N=6,038, strict Bug-type label
│       ├── kafka_real_mined_dataset.csv         # RQ1: N=433, broad defect_prone label
│       ├── tika_real_mined_dataset.csv          # RQ1: N=395, broad defect_prone label
│       ├── rq1_real_mined_dataset.csv           # RQ1: earlier curated subsample (N=1,804), superseded
│       ├── apache_jira_raw.csv                  # RQ2 primary analysis input
│       ├── num_reassignments_FULL.csv           # RQ2: full-population reassignment counts
│       ├── audit_disclosure_dataset.csv         # RQ3: PCAOB data, leaked fields removed (N=17,077)
│       ├── rq3_shap_importance_v2.csv           # RQ3: SHAP feature importance, final version
│       ├── rq4_real_sec_edgar_dataset_FINAL.csv # RQ4: 113 verified SEC EDGAR material-weakness records
│       ├── rq4_real_results_corrected.json      # RQ4: censoring-aware sensitivity results
│       ├── nasa_promise_combined_clean.csv      # NASA/PROMISE baseline, post dedup + imputation
│       └── *.json, *_progress.csv, *_checkpoint.csv  # intermediate pipeline artifacts
├── notebooks/                                    # see "Final, Authoritative Notebooks" above
├── figures/                                      # fig0 (pipeline) through fig9, by research question
├── docs/
│   ├── data_dictionary.md
│   ├── Daljeet_Kaur_Johar_QM640_Synopsis.docx
│   └── Daljeet_Kaur_Johar_QM640_Interim_Report.docx
└── reports/
    └── Daljeet_Kaur_Johar_QM640_Final_Report.docx   # the final capstone report
```

## What's in Each Research Question

| Research Question | Status | Data Source | Headline Result |
|---|---|---|---|
| RQ1 (defect-proneness x era) | **Complete** | Mined code metrics, 4 Apache projects (Camel, Hadoop, Kafka, Tika; N=19,464 total) | Era-moderation significant in Camel and Hadoop, not in Kafka or Tika; all effect sizes negligible (f² < 0.02) |
| RQ2 (resolution time x era) | **Complete** | Apache JIRA (N=30,733; reassignment sub-analysis N=27,385) | Priority x era interactions significant but practically negligible (f²=0.0011); reassignment x era interaction is specification-sensitive (significant in 3 of 4 model specifications tested, not significant in the primary raw-scale model) |
| RQ3 (PCAOB severity classification) | **Complete** | PCAOB inspection data (N=17,077), leakage caught and corrected | Logistic Regression (96.46% accuracy, AUC 0.987) outperforms Random Forest on every metric; robust across complete-case, temporal-holdout, and feature-ablation checks |
| RQ4 (remediation time) | **Complete** | SEC EDGAR material-weakness disclosures, individually verified (N=113) | Cox model (all 113 cases, censoring-aware) is primary: concordance=0.604, no individually significant predictor. An observed-case-only OLS sensitivity model (N=61) has R²=0.200 but adjusted R²=0.021 and a non-significant overall F-test — treated as a sensitivity check, not the primary finding. |
| RQ5 (cross-domain synthesis) | **Complete, with caveats** | Reuses RQ1 and RQ4 results | Point-estimate gap (0.055) is under the 0.10 equivalence threshold, but only 43.5% of 1,000 bootstrap resamples support equivalence, and the two domains' effect sizes come from different model families (logistic pseudo-R² vs. OLS R²), so the comparison is exploratory rather than confirmatory. See the "known issue" under RQ5 above regarding notebook versioning. |

## Reproducing the Analysis

1. Clone the repository and `pip install -r requirements.txt` (pandas, numpy, scikit-learn, statsmodels, lifelines, shap, lizard, scipy).
2. Run the notebooks listed under "Final, Authoritative Notebooks" above, in the order listed per research question.
3. All raw and cleaned data referenced by those notebooks is included in `data/raw/` and `data/cleaned/` — no external API calls are required to reproduce the reported numbers (the original extraction from JIRA/PCAOB/SEC EDGAR is documented in the `_1_Extraction` / `_1_Analysis` notebooks for provenance, but re-running extraction from live APIs is not necessary to reproduce results).
4. The final report (`reports/Daljeet_Kaur_Johar_QM640_Final_Report.docx`) documents, in its "Methodological Rigor and Reproducibility" section, every discrepancy found between earlier drafts and the notebooks actually in this repository, and states which notebook output each reported figure traces back to.

## Known Open Items

- The RQ5 bootstrap notebook (`RQ5_2_Bootstrap_CI.ipynb`) needs to be re-run against the four separate RQ1 project files (see table above) rather than the combined Camel/Hadoop file, so that its output is consistent with the RQ1 four-project analysis it depends on.
- Kafka and Tika (RQ1) use a broader defect-proneness label than Camel and Hadoop, since strict Bug-type labeling was not extended to those two projects; this is disclosed as a limitation in the final report.
