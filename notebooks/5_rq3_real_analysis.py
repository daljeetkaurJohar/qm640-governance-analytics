import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from statsmodels.stats.contingency_tables import mcnemar
import json

df = pd.read_csv("/home/claude/repo_pcaob_deficiencies.csv")
print(f"Loaded real PCAOB deficiency dataset: {len(df)} records")

# Rename to data-dictionary names
df = df.rename(columns={
    "Global Network": "firm_network_category",
    "Auditing Standard": "standard_cited",
    "Inspection Year": "inspection_year",
    "Inspection Type": "inspection_type",
    "Country": "country",
    "Finding Count": "finding_count",
})

# Target: severity_part -> binary (1 = Part I.A, more severe; 0 = Part I.B)
df["severity_binary"] = (df["severity_part"].astype(str).str.strip() == "I.A").astype(int)
print(df["severity_binary"].value_counts())

# *** DATA LEAKAGE CHECK (real finding, kept here deliberately) ***
# "Audit Area", "Issuer Reference Key", "Firm Played A Role...", and
# "Classification Of Audits..." are, by PCAOB's own data schema, populated ONLY
# for Part I.A deficiencies (100% missing for every real Part I.B record in this
# dataset). Using any of them (even just their missingness) would let a model
# trivially "predict" severity_part from a structural data-collection artifact,
# not from real engagement characteristics. They are deliberately EXCLUDED below.
# Verified: audit_area missing count (2,961) == exact total count of I.B records.

df["firm_network_category"] = df["firm_network_category"].fillna("Independent/Unaffiliated")

CATEGORICAL = ["firm_network_category", "standard_cited", "inspection_type", "country"]
NUMERIC = ["inspection_year", "finding_count"]
TARGET = "severity_binary"

df_model = df.dropna(subset=CATEGORICAL + NUMERIC + [TARGET])
print(f"\nModeling sample: N = {len(df_model)} real PCAOB deficiency records")
print(f"Class balance -- Part I.A: {df_model[TARGET].mean()*100:.1f}%, Part I.B: {(1-df_model[TARGET].mean())*100:.1f}%")
print(f"Cardinality -- firm_network_category: {df_model['firm_network_category'].nunique()}, "
      f"standard_cited: {df_model['standard_cited'].nunique()}, "
      f"inspection_type: {df_model['inspection_type'].nunique()}, "
      f"country: {df_model['country'].nunique()}")

X = df_model[CATEGORICAL + NUMERIC]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def build_pipeline(model):
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="passthrough")
    return Pipeline([("preprocess", preprocessor), ("model", model)])

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    res = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
        "cm": confusion_matrix(y_test, preds).tolist(),
    }
    print(f"\n=== {name} ===")
    for k, v in res.items():
        if k != "cm":
            print(f"{k}: {v:.3f}")
    print(f"Confusion matrix:\n{res['cm']}")
    return res, preds, probs

print("\n" + "="*70)
print("RQ3: PCAOB SEVERITY CLASSIFICATION (Part I.A vs Part I.B) -- REAL DATA")
print("="*70)

logit_pipeline = build_pipeline(LogisticRegression(max_iter=5000, class_weight="balanced"))
logit_pipeline.fit(X_train, y_train)
lr_res, lr_preds, lr_probs = evaluate("Logistic Regression (RQ3 baseline)", logit_pipeline, X_test, y_test)

rf_pipeline = build_pipeline(RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
rf_pipeline.fit(X_train, y_train)
rf_res, rf_preds, rf_probs = evaluate("Random Forest (RQ3 benchmark)", rf_pipeline, X_test, y_test)

# Majority baseline
maj_acc = accuracy_score(y_test, np.ones_like(y_test))
print(f"\nMajority-class baseline accuracy (always predict Part I.A): {maj_acc:.3f}")

# McNemar's test between the two classifiers
lr_correct = (lr_preds == y_test.values)
rf_correct = (rf_preds == y_test.values)
table = [[np.sum(lr_correct & rf_correct), np.sum(lr_correct & ~rf_correct)],
         [np.sum(~lr_correct & rf_correct), np.sum(~lr_correct & ~rf_correct)]]
mc = mcnemar(table, exact=False, correction=True)
print(f"\nMcNemar's test (RQ3, LogReg vs RF): chi2={mc.statistic:.4f}, p={mc.pvalue:.4f}")

# Top feature importances from RF (map back through the one-hot encoder)
ohe = rf_pipeline.named_steps["preprocess"].named_transformers_["cat"]
feature_names = list(ohe.get_feature_names_out(CATEGORICAL)) + NUMERIC
importances = pd.Series(rf_pipeline.named_steps["model"].feature_importances_, index=feature_names)
top_importances = importances.sort_values(ascending=False).head(10)
print("\nTop 10 RF feature importances (RQ3):")
print(top_importances)

results = {
    "n_total_real_records": int(len(df)),
    "n_modeling_sample": int(len(df_model)),
    "class_balance_parta_pct": float(df_model[TARGET].mean() * 100),
    "logreg": {k: (v if k != "cm" else v) for k, v in lr_res.items()},
    "random_forest": {k: (v if k != "cm" else v) for k, v in rf_res.items()},
    "majority_baseline_acc": float(maj_acc),
    "mcnemar_chi2": float(mc.statistic),
    "mcnemar_pvalue": float(mc.pvalue),
    "top_features": {k: float(v) for k, v in top_importances.items()},
}
with open("/home/claude/data/rq3_real_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved rq3_real_results.json")
