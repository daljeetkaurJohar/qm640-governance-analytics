import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json

df = pd.read_csv("/home/claude/repo_qa_clean.csv")
print(f"Loaded real Apache JIRA dataset: {len(df)} issues (CAMEL + HADOOP)")
print(df['project_name'].value_counts().to_dict())

df = df.dropna(subset=["resolution_time_days", "num_comments", "priority", "era"])
df["era_binary"] = (df["era"] == "ai_era").astype(int)

print(f"\nAnalysis sample after dropping missing: N = {len(df)}")
print(f"Pre-AI era: {(df['era_binary']==0).sum()}, AI era: {(df['era_binary']==1).sum()}")

# NOTE: num_reassignments was not captured in the original JIRA extraction
# (see notebooks/01_extract_apache_jira.py — only priority, component, num_comments
# were pulled). RQ2 is therefore run here with the two process variables that ARE
# real and available: num_comments and priority. This is disclosed as a limitation.

print("\n" + "="*70)
print("RQ2: MODERATED MULTIPLE REGRESSION (resolution_time_days ~ predictors * era)")
print("="*70)

# Full (moderated) model
full_formula = "resolution_time_days ~ (num_comments + C(priority)) * era_binary"
full_model = smf.ols(full_formula, data=df).fit()
print(full_model.summary())

# Reduced (no interaction) model, for effect-size comparison later (RQ5)
reduced_formula = "resolution_time_days ~ num_comments + C(priority) + era_binary"
reduced_model = smf.ols(reduced_formula, data=df).fit()

interaction_terms = [t for t in full_model.pvalues.index if ":era_binary" in t]
significant = [t for t in interaction_terms if full_model.pvalues[t] < 0.05]
print(f"\nInteraction terms tested: {interaction_terms}")
print(f"Significant Era interaction terms (p < .05): {significant}")

f2 = (full_model.rsquared - reduced_model.rsquared) / (1 - full_model.rsquared)
print(f"\nFull model R2 = {full_model.rsquared:.4f}, Reduced model R2 = {reduced_model.rsquared:.4f}")
print(f"Cohen's f2 (era-interaction effect size) = {f2:.4f}")

# Descriptive: median resolution time by era
med_by_era = df.groupby("era")["resolution_time_days"].agg(["median", "mean", "count"])
print("\nResolution time by era (real data):")
print(med_by_era)

results = {
    "n_total": int(len(df)),
    "n_pre_ai": int((df['era_binary']==0).sum()),
    "n_ai_era": int((df['era_binary']==1).sum()),
    "full_r2": float(full_model.rsquared),
    "reduced_r2": float(reduced_model.rsquared),
    "f2": float(f2),
    "significant_interactions": significant,
    "coefficients": {k: float(v) for k, v in full_model.params.items()},
    "pvalues": {k: float(v) for k, v in full_model.pvalues.items()},
    "median_resolution_pre_ai": float(med_by_era.loc["pre_ai", "median"]),
    "median_resolution_ai_era": float(med_by_era.loc["ai_era", "median"]),
    "mean_resolution_pre_ai": float(med_by_era.loc["pre_ai", "mean"]),
    "mean_resolution_ai_era": float(med_by_era.loc["ai_era", "mean"]),
}
with open("/home/claude/data/rq2_real_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved rq2_real_results.json")
