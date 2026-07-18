"""
06_rq4_remediation_time_model.py

RQ4: Among companies disclosing a material weakness since 2020, do modern ML
models identify different remediation-time drivers than the traditional
statistical methods used in Mojtahedi and Zhou (2024), which analyzed
pre-2018 data?

Method: Multiple linear regression on remediation_days, supplemented by a Cox
proportional hazards survival model (handles right-censoring for weaknesses
not yet remediated at data cutoff).

Usage:
    python 06_rq4_remediation_time_model.py
"""

import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter


def run_regression(df: pd.DataFrame):
    model = smf.ols(
        "remediation_days ~ C(weakness_category) + C(industry_sic) + inspection_year",
        data=df
    ).fit()
    print("=== RQ4: Multiple Regression (remediation_days) ===")
    print(model.summary())
    return model


def run_survival_model(df: pd.DataFrame):
    """
    event_observed = 1 if remediation_confirmed_date is present (event occurred),
    0 if right-censored (still unremediated at data cutoff).
    """
    surv_df = df.copy()
    surv_df["event_observed"] = surv_df["remediation_confirmed_date"].notna().astype(int)
    surv_df["duration"] = surv_df["remediation_days"].fillna(
        (pd.Timestamp("2026-07-17") - pd.to_datetime(surv_df["disclosure_date"])).dt.days
    )

    cols = ["duration", "event_observed", "weakness_category", "industry_sic"]
    surv_df = surv_df.dropna(subset=cols)
    surv_df = pd.get_dummies(surv_df[cols], columns=["weakness_category", "industry_sic"],
                              drop_first=True)

    cph = CoxPHFitter()
    cph.fit(surv_df, duration_col="duration", event_col="event_observed")
    print("\n=== RQ4: Cox Proportional Hazards Model (time-to-remediation) ===")
    cph.print_summary()
    print(f"\nConcordance index (C-statistic): {cph.concordance_index_:.3f}")
    return cph


def compare_to_prior_literature(model):
    """
    Narrative comparison against Mojtahedi and Zhou (2024), which used
    thematic/association-rule methods on 2014-2018 data. This is a manual,
    qualitative step -- read the significant drivers identified by `model`
    and compare them against the categories reported in that paper's findings.
    """
    print("\n=== Comparison to Mojtahedi & Zhou (2024) ===")
    print("Significant drivers in this 2020+ ML model:")
    print(model.pvalues[model.pvalues < 0.05])
    print("\nCompare the above against the weakness categories and industry")
    print("associations reported in Mojtahedi and Zhou (2024) to assess H4.")


def main():
    df = pd.read_csv("../data/cleaned/audit_disclosure_dataset.csv")
    reg_model = run_regression(df)
    run_survival_model(df)
    compare_to_prior_literature(reg_model)


if __name__ == "__main__":
    main()
