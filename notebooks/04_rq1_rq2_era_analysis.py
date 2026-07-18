"""
04_rq1_rq2_era_analysis.py

RQ1: Has the relationship between code/process metrics and defect-proneness
     changed between the pre-AI-coding era and the AI-assisted-coding era?
RQ2: Has the relationship between issue/process characteristics and defect
     resolution time changed across those same two eras?

Method: Moderated logistic regression (RQ1) and moderated multiple linear
regression (RQ2), each with Era x predictor interaction terms.

Usage:
    python 04_rq1_rq2_era_analysis.py
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_rq1(df: pd.DataFrame):
    """Moderated logistic regression: defect_prone ~ metrics * era"""
    df = df.dropna(subset=["defect_prone", "loc", "cyclomatic_complexity",
                            "prior_defect_count", "test_coverage_pct", "era"])
    df["era_binary"] = (df["era"] == "ai_era").astype(int)

    formula = (
        "defect_prone ~ (loc + cyclomatic_complexity + prior_defect_count + "
        "test_coverage_pct) * era_binary"
    )
    model = smf.logit(formula, data=df).fit()
    print("=== RQ1: Moderated Logistic Regression (defect-proneness) ===")
    print(model.summary())

    # H1 is supported if any Era interaction term is significant (p < .05)
    interaction_terms = [t for t in model.pvalues.index if ":era_binary" in t]
    significant = [t for t in interaction_terms if model.pvalues[t] < 0.05]
    print(f"\nSignificant Era interaction terms (p < .05): {significant}")
    return model


def run_rq2(df: pd.DataFrame):
    """Moderated multiple regression: resolution_time_days ~ process factors * era"""
    df = df.dropna(subset=["resolution_time_days", "num_comments",
                            "num_reassignments", "priority", "era"])
    df["era_binary"] = (df["era"] == "ai_era").astype(int)

    formula = (
        "resolution_time_days ~ (num_comments + num_reassignments + "
        "C(priority)) * era_binary"
    )
    model = smf.ols(formula, data=df).fit()
    print("\n=== RQ2: Moderated Multiple Regression (resolution time) ===")
    print(model.summary())

    interaction_terms = [t for t in model.pvalues.index if ":era_binary" in t]
    significant = [t for t in interaction_terms if model.pvalues[t] < 0.05]
    print(f"\nSignificant Era interaction terms (p < .05): {significant}")
    return model


def main():
    df = pd.read_csv("../data/cleaned/qa_defect_dataset.csv")
    rq1_model = run_rq1(df)
    rq2_model = run_rq2(df)

    # Save R-squared values for use in RQ5's cross-domain effect-size comparison
    with open("../data/cleaned/rq1_rq2_effect_sizes.txt", "w") as f:
        f.write(f"RQ2 R-squared (full model with interactions): {rq2_model.rsquared}\n")


if __name__ == "__main__":
    main()
