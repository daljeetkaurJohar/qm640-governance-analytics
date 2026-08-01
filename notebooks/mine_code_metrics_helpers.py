import subprocess
import re
import time
import random
import pandas as pd
import lizard

random.seed(42)

def get_commits_for_issue(repo_dir, issue_id):
    """Return list of commit SHAs whose message references this issue_id."""
    result = subprocess.run(
        ["git", "-C", repo_dir, "log", "--all", "--format=%H %s", "--grep", issue_id, "-i"],
        capture_output=True, text=True, timeout=30
    )
    shas = []
    for line in result.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and issue_id in parts[1]:
            shas.append(parts[0])
    return shas

def get_changed_java_files(repo_dir, sha, main_source_only=True):
    result = subprocess.run(
        ["git", "-C", repo_dir, "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
        capture_output=True, text=True, timeout=30
    )
    files = [f for f in result.stdout.splitlines() if f.endswith(".java")]
    if main_source_only:
        files = [f for f in files if "/test/" not in f and "/generated/" not in f]
    return files

def get_commit_date(repo_dir, sha):
    result = subprocess.run(
        ["git", "-C", repo_dir, "show", "-s", "--format=%cI", sha],
        capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()

def analyze_commit_metrics(repo_dir, sha, files):
    """Aggregate real lizard metrics across all changed .java files in a commit."""
    total_nloc, total_complexity_sum, total_functions, total_tokens = 0, 0, 0, 0
    files_analyzed = 0
    for path in files:
        result = subprocess.run(
            ["git", "-C", repo_dir, "show", f"{sha}:{path}"],
            capture_output=True, text=True, timeout=20
        )
        if result.returncode != 0 or not result.stdout.strip():
            continue
        try:
            analysis = lizard.analyze_file.analyze_source_code(path, result.stdout)
        except Exception:
            continue
        total_nloc += analysis.nloc
        total_functions += len(analysis.function_list)
        total_tokens += analysis.token_count
        for fn in analysis.function_list:
            total_complexity_sum += fn.cyclomatic_complexity
        files_analyzed += 1
    if files_analyzed == 0 or total_functions == 0:
        return None
    return {
        "loc": total_nloc,
        "cyclomatic_complexity": total_complexity_sum / total_functions,
        "num_functions": total_functions,
        "num_files_changed": files_analyzed,
        "token_count": total_tokens,
    }


if __name__ == "__main__":
    REPO = "/home/claude/repos/camel_blobless"
    test_issues = ["CAMEL-24319", "CAMEL-24291", "CAMEL-24268", "CAMEL-24288", "CAMEL-24318"]

    t0 = time.time()
    rows = []
    for issue_id in test_issues:
        shas = get_commits_for_issue(REPO, issue_id)
        if not shas:
            continue
        sha = shas[0]  # first/earliest referencing commit
        files = get_changed_java_files(REPO, sha)
        if not files:
            continue
        metrics = analyze_commit_metrics(REPO, sha, files)
        if metrics:
            metrics["issue_id"] = issue_id
            metrics["sha"] = sha
            rows.append(metrics)
    elapsed = time.time() - t0
    print(f"Processed {len(test_issues)} issues in {elapsed:.2f}s -> {elapsed/len(test_issues):.2f}s/issue")
    print(pd.DataFrame(rows))
