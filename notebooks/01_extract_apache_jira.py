"""
01_extract_apache_jira.py

Pulls real, resolved issue data from the Apache Software Foundation's public JIRA
instance via its REST API. No authentication is required for public projects.

Output: data/raw/apache_jira_raw.csv

NOTE: This script pulls issue/process metadata (priority, component, comments,
resolution dates) directly from the JIRA API. Code-level metrics referenced in the
data dictionary (loc, cyclomatic_complexity, test_coverage_pct, prior_defect_count)
are NOT available from the JIRA API itself -- they require a separate repository-
mining step against the project's real git history (e.g., using the `lizard` or
`radon` Python packages to compute complexity/LOC on the specific commit that
closed each issue). See 03_data_cleaning.py for where these are merged in.

Usage:
    python 01_extract_apache_jira.py
"""

import requests
import pandas as pd
import time

JIRA_BASE_URL = "https://issues.apache.org/jira/rest/api/2/search"

# Choose one or more large, long-running Apache projects with issue histories
# spanning both before and after 2023-01-01 (needed for the era comparison in RQ1/RQ2).
PROJECTS = ["CAMEL", "HADOOP"]

PAGE_SIZE = 100  # JIRA API max per request is typically 100


def fetch_project_issues(project_key: str) -> list:
    """Fetch all resolved issues for a given Apache project."""
    all_issues = []
    start_at = 0

    jql = f'project={project_key} AND resolution=Fixed ORDER BY resolutiondate ASC'
    fields = "created,resolutiondate,priority,components,comment,summary,status"

    while True:
        params = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": PAGE_SIZE,
            "fields": fields,
        }
        response = requests.get(JIRA_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        issues = data.get("issues", [])
        if not issues:
            break

        for issue in issues:
            f = issue["fields"]
            all_issues.append({
                "issue_id": issue["key"],
                "project_name": project_key,
                "created": f.get("created"),
                "resolution_date": f.get("resolutiondate"),
                "priority": (f.get("priority") or {}).get("name"),
                "component": ", ".join(c["name"] for c in f.get("components", [])),
                "num_comments": (f.get("comment") or {}).get("total", 0),
            })

        start_at += PAGE_SIZE
        print(f"  {project_key}: fetched {len(all_issues)} issues so far...")

        # Be polite to the public API
        time.sleep(0.5)

        if start_at >= data.get("total", 0):
            break

    return all_issues


def main():
    all_rows = []
    for project in PROJECTS:
        print(f"Fetching {project}...")
        all_rows.extend(fetch_project_issues(project))

    df = pd.DataFrame(all_rows)
    df.to_csv("../data/raw/apache_jira_raw.csv", index=False)
    print(f"\nSaved {len(df)} real issues to data/raw/apache_jira_raw.csv")
    print(df.head())


if __name__ == "__main__":
    main()
