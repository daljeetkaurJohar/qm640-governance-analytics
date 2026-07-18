"""
03_data_cleaning.py

Cleans and merges the raw extracted data into the two analysis-ready datasets
described in docs/data_dictionary.md:
    - data/cleaned/qa_defect_dataset.csv
    - data/cleaned/audit_disclosure_dataset.csv

Usage:
    python 03_data_cleaning.py
"""

import pandas as pd
import numpy as np

ERA_CUTOFF = "2023-01-01"  # boundary between pre-AI-coding and AI-assisted-coding eras


# ---------------------------------------------------------------------------
# QA / Defect dataset (Apache JIRA + code-metric mining)
# ---------------------------------------------------------------------------
def clean_qa_dataset():
    df = pd.read_csv("../data/raw/apache_jira_raw.csv")

    # Drop duplicate issues
    df = df.drop_duplicates(subset="issue_id")

    # Parse dates
    df["resolution_date"] = pd.to_datetime(df["resolution_date"], errors="coerce")
    df = df.dropna(subset=["resolution_date"])

    # Derive era directly from the real resolution timestamp
    df["era"] = np.where(df["resolution_date"] < ERA_CUTOFF, "pre_ai", "ai_era")

    # Compute resolution_time_days from real created/resolution timestamps
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df["resolution_time_days"] = (df["resolution_date"] - df["created"]).dt.days
    df = df[df["resolution_time_days"] >= 0]  # drop any data-quality anomalies

    # NOTE: loc, cyclomatic_complexity, prior_defect_count, test_coverage_pct, and
    # defect_prone must be merged in here from a separate repo-mining step (e.g.,
    # using `lizard` against the real git history for each issue's associated commit).
    # Placeholder columns are added below so the schema matches the data dictionary;
    # replace this block with your real merged values before modeling.
    for col in ["loc", "cyclomatic_complexity", "prior_defect_count",
                "test_coverage_pct", "defect_prone"]:
        if col not in df.columns:
            df[col] = np.nan

    df.to_csv("../data/cleaned/qa_defect_dataset.csv", index=False)
    print(f"Cleaned QA/defect dataset: {len(df)} real records "
          f"({(df['era'] == 'pre_ai').sum()} pre-AI-era, "
          f"{(df['era'] == 'ai_era').sum()} AI-era)")
    return df


# ---------------------------------------------------------------------------
# Audit / Disclosure dataset (PCAOB + SEC EDGAR)
# ---------------------------------------------------------------------------
def clean_audit_dataset():
    pcaob = pd.read_csv("../data/raw/pcaob_deficiencies_raw.csv")
    pcaob = pcaob.drop_duplicates(subset="deficiency_id")

    # Standardize severity_part to binary (1 = Part I.A, 0 = Part I.B)
    if "severity_part" in pcaob.columns:
        pcaob["severity_part"] = pcaob["severity_part"].apply(
            lambda x: 1 if str(x).strip().upper() in ("I.A", "1.A", "IA") else 0
        )

    pcaob.to_csv("../data/cleaned/audit_disclosure_dataset.csv", index=False)
    print(f"Cleaned audit/disclosure dataset: {len(pcaob)} real PCAOB records")

    # SEC EDGAR remediation-timeline data would be merged in separately once
    # matched to company CIK and disclosure/remediation dates -- see data
    # dictionary for the remediation_days field definition.
    return pcaob


def main():
    clean_qa_dataset()
    clean_audit_dataset()


if __name__ == "__main__":
    main()
