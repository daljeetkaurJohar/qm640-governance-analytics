# Data Dictionary

This document describes every field used in the analysis, split by dataset.
Both datasets are real, downloaded directly from their original public sources
(no synthetic or Kaggle-hosted data is used anywhere in this study).

---

## Table 1 — Software QA / Defect Dataset (Apache JIRA / NASA-PROMISE)

**Source:** https://issues.apache.org/jira (real-time API) and
https://promise.site.uottawa.ca/SERepository/datasets-page.html (cross-validation)
**File:** `data/cleaned/qa_defect_dataset.csv`

| Variable | Description | Type / Scale |
|---|---|---|
| issue_id | Unique JIRA issue/defect identifier | Categorical (ID) |
| project_name | Apache project (e.g., CAMEL, HADOOP) | Categorical |
| loc | Lines of code in the affected module | Continuous |
| cyclomatic_complexity | McCabe complexity of the affected module | Continuous |
| prior_defect_count | Number of prior defects logged against the module | Continuous |
| test_coverage_pct | % of module covered by automated tests, where available | Continuous (0–100) |
| priority | JIRA-assigned issue priority | Ordinal |
| component | Functional component/module of the project | Categorical |
| num_comments | Number of comments on the issue thread | Continuous |
| num_reassignments | Number of times the issue was reassigned | Continuous |
| resolution_time_days | Days from issue creation to resolution (**RQ2 target**) | Continuous |
| defect_prone | Whether the module was reclassified as defect-prone (**RQ1 target**) | Binary |
| era | Pre-AI-coding (resolved before 2023-01-01) vs. AI-coding era (resolved 2023-01-01 onward) — RQ1/RQ2 moderator | Binary |

---

## Table 2 — IT Audit / Disclosure Dataset (PCAOB / SEC EDGAR)

**Source:** https://pcaobus.org/oversight/inspections/firm-inspection-reports (PCAOB) and
https://www.sec.gov/edgar/search/ (SEC EDGAR)
**File:** `data/cleaned/audit_disclosure_dataset.csv`

| Variable | Description | Type / Scale |
|---|---|---|
| deficiency_id | PCAOB inspection deficiency record identifier | Categorical (ID) |
| inspection_year | Year of the PCAOB inspection report | Continuous |
| firm_network_category | PCAOB firm classification (e.g., Global Network Firm, Non-Affiliate) | Categorical |
| audit_area | Audit area affected by the deficiency (e.g., revenue, ICFR) | Categorical |
| standard_cited | PCAOB auditing standard most directly cited | Categorical |
| severity_part | Part I.A (more severe) vs. Part I.B classification (**RQ3 target**) | Binary |
| company_cik | SEC Central Index Key of the related public company (where matched) | Categorical (ID) |
| industry_sic | SIC industry code of the disclosing company | Categorical |
| weakness_category | Category of disclosed material weakness (e.g., IT general controls, revenue recognition) | Categorical |
| disclosure_date | Date material weakness was first disclosed (8-K Item 4.02 / 10-K) | Date |
| remediation_confirmed_date | Date a subsequent filing confirmed remediation | Date |
| remediation_days | Days from disclosure to confirmed remediation (**RQ4 target**) | Continuous |

---

## Notes on Data Cleaning
- Duplicate issue/deficiency records are removed based on `issue_id` / `deficiency_id`.
- Records with missing target variables (`defect_prone`, `resolution_time_days`,
  `severity_part`, `remediation_days`) are excluded from the respective RQ's model
  but retained in the raw file for transparency.
- `era` is derived directly from each record's real resolution timestamp — no
  manual labeling or AI-authorship detection is used.
- `remediation_days` is computed only for weaknesses with a confirmed remediation
  filing; unremediated weaknesses at data cutoff are treated as right-censored in
  the survival model (RQ4).
