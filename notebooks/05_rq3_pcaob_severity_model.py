"""
05_rq3_pcaob_severity_model.py

RQ3: Using the PCAOB's newly released (April 2025) machine-readable inspection
datasets, can a supervised ML classification model accurately predict whether a
deficiency is classified at the more severe Part I.A level versus Part I.B?

Method: Logistic regression, benchmarked against a Random Forest classifier.

Usage:
    python 05_rq3_pcaob_severity_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)


CATEGORICAL_FEATURES = ["firm_network_category", "audit_area", "standard_cited"]
NUMERIC_FEATURES = ["inspection_year"]
TARGET = "severity_part"


def build_pipeline(model):
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1:        {f1_score(y_test, preds):.3f}")
    print(f"AUC:       {roc_auc_score(y_test, probs):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, preds)}")


def main():
    df = pd.read_csv("../data/cleaned/audit_disclosure_dataset.csv")
    df = df.dropna(subset=CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET])

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logit_pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    logit_pipeline.fit(X_train, y_train)
    evaluate("Logistic Regression (RQ3 baseline)", logit_pipeline, X_test, y_test)

    rf_pipeline = build_pipeline(RandomForestClassifier(n_estimators=300, random_state=42))
    rf_pipeline.fit(X_train, y_train)
    evaluate("Random Forest (RQ3 benchmark)", rf_pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
