import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.impute import SimpleImputer

pd.set_option('display.width', 120)
plt.style.use('seaborn-v0_8-whitegrid')
OUT = "/home/claude/data"

raw = pd.read_csv(f"{OUT}/combined_raw.csv")
FEATURES = ['BRANCH_COUNT', 'CYCLOMATIC_COMPLEXITY', 'DESIGN_COMPLEXITY', 'ESSENTIAL_COMPLEXITY',
            'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY', 'HALSTEAD_EFFORT', 'HALSTEAD_ERROR_EST',
            'HALSTEAD_LENGTH', 'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME',
            'LOC_BLANK', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS', 'LOC_EXECUTABLE', 'LOC_TOTAL',
            'NUM_OPERANDS', 'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS', 'NUM_UNIQUE_OPERATORS']

# ---------- 1. DATA CLEANING ----------
print("="*70)
print("DATA CLEANING LOG")
print("="*70)

n_start = len(raw)
missing_counts = raw[FEATURES].isna().sum()
missing_pct = (missing_counts / n_start * 100).round(2)
print("\nMissing values per feature (top 5):")
print(missing_pct.sort_values(ascending=False).head(5))
total_missing_cells = raw[FEATURES].isna().sum().sum()
print(f"\nTotal missing cells: {total_missing_cells} / {n_start*len(FEATURES)} ({total_missing_cells/(n_start*len(FEATURES))*100:.3f}%)")

# Duplicates (exact duplicate rows across feature set, within same project)
dupes = raw.duplicated(subset=FEATURES+['project']).sum()
print(f"Duplicate rows (same project, identical metrics): {dupes} ({dupes/n_start*100:.2f}%)")

# Drop exact duplicates
clean = raw.drop_duplicates(subset=FEATURES+['project']).copy()
n_after_dedup = len(clean)
print(f"Rows after removing duplicates: {n_after_dedup} (removed {n_start-n_after_dedup})")

# Impute missing with median (per-feature)
imputer = SimpleImputer(strategy='median')
clean[FEATURES] = imputer.fit_transform(clean[FEATURES])

# Outlier detection via IQR rule, reported not removed (kept, common in defect literature since extreme values are legitimate large modules)
outlier_summary = {}
for col in FEATURES:
    q1, q3 = clean[col].quantile(0.25), clean[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = ((clean[col] < lower) | (clean[col] > upper)).sum()
    outlier_summary[col] = n_out
outlier_series = pd.Series(outlier_summary).sort_values(ascending=False)
print("\nOutlier counts (IQR rule, top 5 features):")
print(outlier_series.head(5))
print(f"Note: Outliers retained (large/complex modules are legitimate high-risk cases in defect prediction, not data errors).")

n_final = len(clean)
print(f"\nFinal cleaned N = {n_final} (from {n_start} raw records, {n_start-n_final} removed as exact duplicates, 0 rows dropped for missingness — median-imputed instead)")

clean.to_csv(f"{OUT}/combined_clean.csv", index=False)

# ---------- 2. DESCRIPTIVE STATISTICS ----------
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS (key metrics)")
print("="*70)
key_metrics = ['LOC_TOTAL', 'CYCLOMATIC_COMPLEXITY', 'LOC_COMMENTS', 'HALSTEAD_VOLUME', 'BRANCH_COUNT']
desc = clean[key_metrics].describe().round(2)
print(desc)

# defect rate by project after cleaning
defect_by_project = clean.groupby('project')['Defective_bin'].agg(['count','mean'])
defect_by_project['mean'] = (defect_by_project['mean']*100).round(2)
defect_by_project.columns = ['n_modules', 'defect_rate_pct']
print("\nDefect rate by project (post-cleaning):")
print(defect_by_project)

# ---------- 3. FIGURES ----------
# Figure: defect rate by project
fig, ax = plt.subplots(figsize=(8,4.5))
d = defect_by_project.sort_values('defect_rate_pct', ascending=False)
ax.bar(d.index, d['defect_rate_pct'], color='#3B6E8F')
ax.set_ylabel('Defect rate (%)')
ax.set_xlabel('Project')
ax.set_title('Figure 1. Defect Rate by NASA/PROMISE Project')
for i, v in enumerate(d['defect_rate_pct']):
    ax.text(i, v+0.3, f"{v:.1f}%", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_defect_rate_by_project.png", dpi=150)
plt.close()

# Figure: boxplot LOC_TOTAL by defective status
fig, ax = plt.subplots(figsize=(7,4.5))
data_box = [clean[clean.Defective_bin==0]['LOC_TOTAL'].clip(upper=clean['LOC_TOTAL'].quantile(0.95)),
            clean[clean.Defective_bin==1]['LOC_TOTAL'].clip(upper=clean['LOC_TOTAL'].quantile(0.95))]
ax.boxplot(data_box, labels=['Not Defective','Defective'], patch_artist=True,
           boxprops=dict(facecolor='#A9C6D6'))
ax.set_ylabel('LOC_TOTAL (95th pct capped for display)')
ax.set_title('Figure 2. Module Size (LOC) by Defect Status')
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_loc_by_defect.png", dpi=150)
plt.close()

# Figure: boxplot cyclomatic complexity by defective status
fig, ax = plt.subplots(figsize=(7,4.5))
data_box2 = [clean[clean.Defective_bin==0]['CYCLOMATIC_COMPLEXITY'].clip(upper=clean['CYCLOMATIC_COMPLEXITY'].quantile(0.95)),
             clean[clean.Defective_bin==1]['CYCLOMATIC_COMPLEXITY'].clip(upper=clean['CYCLOMATIC_COMPLEXITY'].quantile(0.95))]
ax.boxplot(data_box2, labels=['Not Defective','Defective'], patch_artist=True,
           boxprops=dict(facecolor='#D6A9A9'))
ax.set_ylabel('Cyclomatic Complexity (95th pct capped)')
ax.set_title('Figure 3. Cyclomatic Complexity by Defect Status')
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_complexity_by_defect.png", dpi=150)
plt.close()

print("\nSaved figures: fig1_defect_rate_by_project.png, fig2_loc_by_defect.png, fig3_complexity_by_defect.png")

# ---------- 4. MODELING ----------
print("\n" + "="*70)
print("MODELING: DEFECT-PRONENESS CLASSIFICATION")
print("="*70)

X = clean[FEATURES].values
y = clean['Defective_bin'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train defect rate: {y_train.mean()*100:.2f}%, Test defect rate: {y_test.mean()*100:.2f}%")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# Logistic Regression
logreg = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(logreg, X_train_s, y_train, cv=cv5, scoring='roc_auc')
logreg.fit(X_train_s, y_train)
pred_lr = logreg.predict(X_test_s)
proba_lr = logreg.predict_proba(X_test_s)[:,1]

results['LogisticRegression'] = {
    'cv_auc_mean': cv_scores_lr.mean(), 'cv_auc_std': cv_scores_lr.std(),
    'accuracy': accuracy_score(y_test, pred_lr),
    'precision': precision_score(y_test, pred_lr),
    'recall': recall_score(y_test, pred_lr),
    'f1': f1_score(y_test, pred_lr),
    'auc': roc_auc_score(y_test, proba_lr),
    'cm': confusion_matrix(y_test, pred_lr)
}

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=cv5, scoring='roc_auc')
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:,1]

results['RandomForest'] = {
    'cv_auc_mean': cv_scores_rf.mean(), 'cv_auc_std': cv_scores_rf.std(),
    'accuracy': accuracy_score(y_test, pred_rf),
    'precision': precision_score(y_test, pred_rf),
    'recall': recall_score(y_test, pred_rf),
    'f1': f1_score(y_test, pred_rf),
    'auc': roc_auc_score(y_test, proba_rf),
    'cm': confusion_matrix(y_test, pred_rf)
}

for name, r in results.items():
    print(f"\n--- {name} ---")
    print(f"5-fold CV AUC (train): {r['cv_auc_mean']:.3f} +/- {r['cv_auc_std']:.3f}")
    print(f"Test Accuracy:  {r['accuracy']:.3f}")
    print(f"Test Precision: {r['precision']:.3f}")
    print(f"Test Recall:    {r['recall']:.3f}")
    print(f"Test F1:        {r['f1']:.3f}")
    print(f"Test AUC:       {r['auc']:.3f}")
    print(f"Confusion Matrix [[TN FP],[FN TP]]:\n{r['cm']}")

# majority baseline
maj_pred = np.zeros_like(y_test)
print(f"\nMajority-class baseline accuracy: {accuracy_score(y_test, maj_pred):.3f} (always predict 'not defective')")

# Feature importance (RF)
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 8 features by Random Forest importance:")
print(importances.head(8).round(4))

# Figure: ROC curves
fig, ax = plt.subplots(figsize=(6,6))
for name, r, proba in [('Logistic Regression', results['LogisticRegression'], proba_lr),
                        ('Random Forest', results['RandomForest'], proba_rf)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Chance')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Figure 4. ROC Curves - Defect-Proneness Classifiers (Test Set)')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_roc_curves.png", dpi=150)
plt.close()

# Figure: feature importance
fig, ax = plt.subplots(figsize=(8,5))
top8 = importances.head(8).sort_values()
ax.barh(top8.index, top8.values, color='#3B6E8F')
ax.set_xlabel('Random Forest Feature Importance')
ax.set_title('Figure 5. Top Predictors of Defect-Proneness')
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_feature_importance.png", dpi=150)
plt.close()

print("\nSaved figures: fig4_roc_curves.png, fig5_feature_importance.png")

# Save numeric results for report-writing
import json
summary = {
    'n_raw': int(n_start),
    'n_dupes_removed': int(n_start - n_after_dedup),
    'n_final': int(n_final),
    'total_missing_cells': int(total_missing_cells),
    'total_cells': int(n_start*len(FEATURES)),
    'overall_defect_rate': float(clean['Defective_bin'].mean()),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'results': {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else float(vv)) for kk, vv in v.items()} for k, v in results.items()},
    'majority_baseline_acc': float(accuracy_score(y_test, maj_pred)),
    'top_features': importances.head(8).round(4).to_dict(),
    'defect_by_project': defect_by_project.to_dict('index'),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved summary.json")
