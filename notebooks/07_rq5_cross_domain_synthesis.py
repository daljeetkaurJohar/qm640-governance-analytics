"""
07_rq5_cross_domain_synthesis.py

RQ5: Is the magnitude of governance-relevant change identified in the software
QA domain (RQ1/RQ2 Era interaction effects) comparable to the magnitude of
change identified in the IT audit domain (RQ4's driver-set shift relative to
Mojtahedi & Zhou, 2024) -- suggesting a shared "digital-era governance
disruption" -- or do the two domains diverge?

Method: Side-by-side comparison of standardized effect sizes (Cohen's f-squared)
across both domains.

This script requires the R-squared outputs already saved by
04_rq1_rq2_era_analysis.py (RQ1/RQ2) and 06_rq4_remediation_time_model.py (RQ4).

Usage:
    python 07_rq5_cross_domain_synthesis.py
"""

import pandas as pd
import statsmodels.formula.api as smf


def cohens_f_squared(r2_full: float, r2_reduced: float) -> float:
    """
    Cohen's f^2 = (R2_full - R2_reduced) / (1 - R2_full)
    Conventions (Cohen, 1988): 0.02 = small, 0.15 = medium, 0.35 = large
    """
    return (r2_full - r2_reduced) / (1 - r2_full)


def interpret_effect_size(f2: float) -> str:
    if f2 < 0.02:
        return "negligible"
    elif f2 < 0.15:
        return "small"
    elif f2 < 0.35:
        return "medium"
    else:
        return "large"


def get_qa_domain_effect_size(df: pd.DataFrame) -> float:
    """Compare R2 of the QA model WITH era interactions vs. WITHOUT them."""
    df = df.dropna(subset=["resolution_time_days", "num_comments",
                            "num_reassignments", "priority", "era"])
    df["era_binary"] = (df["era"] == "ai_era").astype(int)

    full_model = smf.ols(
        "resolution_time_days ~ (num_comments + num_reassignments + "
        "C(priority)) * era_binary", data=df
    ).fit()
    reduced_model = smf.ols(
        "resolution_time_days ~ num_comments + num_reassignments + C(priority) + era_binary",
        data=df
    ).fit()

    f2 = cohens_f_squared(full_model.rsquared, reduced_model.rsquared)
    print(f"QA domain (RQ1/RQ2): R2_full={full_model.rsquared:.4f}, "
          f"R2_reduced={reduced_model.rsquared:.4f}, f2={f2:.4f} "
          f"({interpret_effect_size(f2)} effect)")
    return f2


def get_audit_domain_effect_size(df: pd.DataFrame) -> float:
    """
    Compare R2 of the 2020+ remediation-time model WITH the full driver set
    vs. a model restricted to only the drivers reported as significant in
    Mojtahedi & Zhou (2024). Update the `prior_literature_drivers` list below
    to match the specific categories reported in that paper.
    """
    df = df.dropna(subset=["remediation_days", "weakness_category",
                            "industry_sic", "inspection_year"])

    full_model = smf.ols(
        "remediation_days ~ C(weakness_category) + C(industry_sic) + inspection_year",
        data=df
    ).fit()

    # Placeholder: restrict to the driver categories reported in the prior
    # (pre-2018) literature -- update this formula once you've reviewed
    # Mojtahedi and Zhou (2024) in detail.
    prior_literature_model = smf.ols(
        "remediation_days ~ C(weakness_category)", data=df
    ).fit()

    f2 = cohens_f_squared(full_model.rsquared, prior_literature_model.rsquared)
    print(f"Audit domain (RQ4): R2_full={full_model.rsquared:.4f}, "
          f"R2_prior_literature_only={prior_literature_model.rsquared:.4f}, "
          f"f2={f2:.4f} ({interpret_effect_size(f2)} effect)")
    return f2


def main():
    qa_df = pd.read_csv("../data/cleaned/qa_defect_dataset.csv")
    audit_df = pd.read_csv("../data/cleaned/audit_disclosure_dataset.csv")

    qa_f2 = get_qa_domain_effect_size(qa_df)
    audit_f2 = get_audit_domain_effect_size(audit_df)

    print("\n=== RQ5: Cross-Domain Comparison ===")
    print(f"QA domain effect size (f2):    {qa_f2:.4f} ({interpret_effect_size(qa_f2)})")
    print(f"Audit domain effect size (f2): {audit_f2:.4f} ({interpret_effect_size(audit_f2)})")

    diff = abs(qa_f2 - audit_f2)
    print(f"\nAbsolute difference in effect size: {diff:.4f}")
    if diff < 0.10:
        print("Interpretation: comparable magnitudes -> supports a shared "
              "'digital-era governance disruption' across both domains (H5 supported).")
    else:
        print("Interpretation: divergent magnitudes -> governance-relevant "
              "disruption appears domain-specific rather than shared (H5 not supported).")


if __name__ == "__main__":
    main()
