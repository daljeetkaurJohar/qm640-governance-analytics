# QM640 Data Analytics Capstone: Governance Risk Analytics

**Author:** Daljeet Kaur Johar | Walsh College | QM640 Data Analytics Capstone

## Overview
This repository contains the data, code, and analysis for a capstone study examining
whether established governance analytics practices — software defect prediction and
IT-audit risk modeling — hold under two real, current conditions: the generative-AI
coding era, and the PCAOB's newly released (April 2025) machine-readable inspection
data. The study concludes with a fifth, cross-domain question comparing the magnitude
of change found in both domains.

## Data Sources (all real, publicly available — no synthetic or Kaggle data)
- **Apache Software Foundation JIRA** (https://issues.apache.org/jira) — real, resolved software issues
- **NASA/PROMISE Software Defect Prediction Repository** — cross-validation reference
- **PCAOB Inspection Datasets** (https://pcaobus.org/oversight/inspections/firm-inspection-reports) — real audit deficiency records, 2018–present
- **SEC EDGAR Full-Text Search** (https://www.sec.gov/edgar/search/) — real material weakness disclosures, 2020+

## Repository Structure
```
qm640-governance-analytics/
├── data/
│   ├── raw/                                  # untouched downloads from original sources
│   └── cleaned/                               # processed, analysis-ready datasets
├── notebooks/
│   ├── 01_extract_apache_jira.py             # pulls real JIRA issue data via REST API
│   ├── 02_extract_pcaob_sec_edgar.py         # pulls real PCAOB + SEC EDGAR data
│   ├── 03_data_cleaning.py                   # cleans/merges raw -> cleaned datasets
│   ├── 04_rq1_rq2_era_analysis.py            # moderated regression: pre/post AI-coding era
│   ├── 05_rq3_pcaob_severity_model.py        # classification: Part I.A vs I.B
│   ├── 06_rq4_remediation_time_model.py      # regression/survival: remediation time
│   └── 07_rq5_cross_domain_synthesis.py      # compares effect sizes across domains
├── docs/
│   ├── synopsis.pdf                          # full capstone synopsis
│   └── data_dictionary.md                    # field-by-field dictionary (Tables 1-2)
└── README.md
```

## Reproducing This Study
1. Clone this repository: `git clone https://github.com/[username]/qm640-governance-analytics.git`
2. Install dependencies: `pip install requests pandas numpy scikit-learn scipy lifelines statsmodels`
3. Run notebooks in `/notebooks` in numbered order (01 → 07)
4. Raw data is pulled live from the sources listed above — see each script's docstring for API details

## Research Questions
See `docs/synopsis.pdf` for the full synopsis, including all 5 research questions,
hypotheses, sample size calculations, and analytic approach. Summary:

- **RQ1/RQ2** (Apache JIRA + NASA-PROMISE): Has the relationship between code/process
  metrics and defect-proneness/resolution-time changed between the pre-AI-coding era
  (before 2023-01-01) and the AI-assisted-coding era (2023-01-01 onward)?
- **RQ3** (PCAOB): Can ML classify deficiency severity (Part I.A vs I.B) using PCAOB's
  newly released machine-readable datasets?
- **RQ4** (SEC EDGAR): Do modern ML models identify different remediation-time drivers
  than pre-2018 traditional-statistics research (Mojtahedi & Zhou, 2024)?
- **RQ5**: Is the magnitude of change in RQ1/RQ2 comparable to the magnitude of change
  in RQ4, suggesting a shared "digital-era governance disruption" across both domains?

## License
Code in this repository is released under the MIT License. Data usage follows each
original source's own public-data terms (Apache Software Foundation, NASA/PROMISE,
PCAOB, and U.S. SEC EDGAR).
