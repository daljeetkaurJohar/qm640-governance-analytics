import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
import json

OUT = "/home/claude/data"
clean = pd.read_csv(f"{OUT}/combined_clean.csv")

FEATURES = ['BRANCH_COUNT', 'CYCLOMATIC_COMPLEXITY', 'DESIGN_COMPLEXITY', 'ESSENTIAL_COMPLEXITY',
            'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY', 'HALSTEAD_EFFORT', 'HALSTEAD_ERROR_EST',
            'HALSTEAD_LENGTH', 'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME',
            'LOC_BLANK', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS', 'LOC_EXECUTABLE', 'LOC_TOTAL',
            'NUM_OPERANDS', 'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS', 'NUM_UNIQUE_OPERATORS']

X = clean[FEATURES].values
y = clean['Defective_bin'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
logreg.fit(X_train_s, y_train)
pred_lr = logreg.predict(X_test_s)
proba_lr = logreg.predict_proba(X_test_s)[:, 1]

rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

# ---------- McNemar's Test (Step 4: "Statistical Significance Testing") ----------
print("="*70)
print("McNEMAR'S TEST: LogReg vs Random Forest (paired predictions, same test set)")
print("="*70)

lr_correct = (pred_lr == y_test)
rf_correct = (pred_rf == y_test)

# 2x2 contingency table: [both correct/LR correct RF wrong], [RF correct LR wrong / both wrong]
n_both_correct = np.sum(lr_correct & rf_correct)
n_lr_only = np.sum(lr_correct & ~rf_correct)   # LR correct, RF wrong
n_rf_only = np.sum(~lr_correct & rf_correct)   # RF correct, LR wrong
n_both_wrong = np.sum(~lr_correct & ~rf_correct)

table = [[n_both_correct, n_lr_only],
         [n_rf_only, n_both_wrong]]

print(f"Both correct: {n_both_correct}, LR-only-correct: {n_lr_only}, RF-only-correct: {n_rf_only}, Both wrong: {n_both_wrong}")

result = mcnemar(table, exact=False, correction=True)
print(f"McNemar chi-square statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4f}")
if result.pvalue < 0.05:
    print("=> Statistically significant difference between LogReg and RF classification patterns (p < .05)")
else:
    print("=> No statistically significant difference between LogReg and RF classification patterns (p >= .05)")

# ---------- Bootstrap Confidence Intervals (Step 4: "Confidence Intervals") ----------
print("\n" + "="*70)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (2,000 resamples of test set)")
print("="*70)

rng = np.random.RandomState(42)
n_boot = 2000
n_test = len(y_test)

boot_acc_lr, boot_auc_lr = [], []
boot_acc_rf, boot_auc_rf = [], []

for i in range(n_boot):
    idx = rng.randint(0, n_test, n_test)
    yt = y_test[idx]
    if len(np.unique(yt)) < 2:
        continue
    boot_acc_lr.append(accuracy_score(yt, pred_lr[idx]))
    boot_auc_lr.append(roc_auc_score(yt, proba_lr[idx]))
    boot_acc_rf.append(accuracy_score(yt, pred_rf[idx]))
    boot_auc_rf.append(roc_auc_score(yt, proba_rf[idx]))

def ci(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

lr_acc_ci = ci(boot_acc_lr)
lr_auc_ci = ci(boot_auc_lr)
rf_acc_ci = ci(boot_acc_rf)
rf_auc_ci = ci(boot_auc_rf)

print(f"Logistic Regression - Accuracy: {np.mean(boot_acc_lr):.3f} [95% CI: {lr_acc_ci[0]:.3f}, {lr_acc_ci[1]:.3f}]")
print(f"Logistic Regression - AUC:      {np.mean(boot_auc_lr):.3f} [95% CI: {lr_auc_ci[0]:.3f}, {lr_auc_ci[1]:.3f}]")
print(f"Random Forest       - Accuracy: {np.mean(boot_acc_rf):.3f} [95% CI: {rf_acc_ci[0]:.3f}, {rf_acc_ci[1]:.3f}]")
print(f"Random Forest       - AUC:      {np.mean(boot_auc_rf):.3f} [95% CI: {rf_auc_ci[0]:.3f}, {rf_auc_ci[1]:.3f}]")

# Bootstrap difference in AUC (paired, same resamples) -> CI on the difference
boot_auc_diff = np.array(boot_auc_rf) - np.array(boot_auc_lr)
diff_ci = ci(boot_auc_diff)
print(f"\nPaired bootstrap difference in AUC (RF - LogReg): {np.mean(boot_auc_diff):.4f} [95% CI: {diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
if diff_ci[0] > 0 or diff_ci[1] < 0:
    print("=> 95% CI excludes zero: RF's AUC advantage over LogReg is statistically meaningful")
else:
    print("=> 95% CI includes zero: RF's AUC advantage over LogReg is NOT statistically distinguishable from chance")

stats_summary = {
    "mcnemar_table": [[int(x) for x in row] for row in table],
    "mcnemar_chi2": float(result.statistic),
    "mcnemar_pvalue": float(result.pvalue),
    "lr_acc_mean": float(np.mean(boot_acc_lr)), "lr_acc_ci": [float(x) for x in lr_acc_ci],
    "lr_auc_mean": float(np.mean(boot_auc_lr)), "lr_auc_ci": [float(x) for x in lr_auc_ci],
    "rf_acc_mean": float(np.mean(boot_acc_rf)), "rf_acc_ci": [float(x) for x in rf_acc_ci],
    "rf_auc_mean": float(np.mean(boot_auc_rf)), "rf_auc_ci": [float(x) for x in rf_auc_ci],
    "auc_diff_mean": float(np.mean(boot_auc_diff)), "auc_diff_ci": [float(x) for x in diff_ci],
}
with open(f"{OUT}/stats_summary.json", "w") as f:
    json.dump(stats_summary, f, indent=2)
print("\nSaved stats_summary.json")
