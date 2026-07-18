"""
02_extract_pcaob_sec_edgar.py

Pulls real audit-deficiency data from PCAOB's official downloadable inspection
datasets, and real material-weakness disclosure data from SEC EDGAR's Full-Text
Search system.

Output:
    data/raw/pcaob_deficiencies_raw.csv
    data/raw/sec_edgar_disclosures_raw.json

Usage:
    python 02_extract_pcaob_sec_edgar.py
"""

import requests
import pandas as pd
import json
import time

# ---------------------------------------------------------------------------
# PART 1: PCAOB inspection datasets
# ---------------------------------------------------------------------------
# PCAOB publishes bulk CSV/XML/JSON files directly -- no API call needed.
# Visit https://pcaobus.org/oversight/inspections/firm-inspection-reports,
# locate the "Inspection Data" downloadable file section, and download the
# current Part I.A / Part I.B deficiency dataset manually, or automate the
# direct file download if a stable direct link is published for the file you need.

def load_pcaob_csv(local_path: str) -> pd.DataFrame:
    """
    Load a manually downloaded PCAOB deficiency CSV into a DataFrame.
    Download the file first from:
    https://pcaobus.org/oversight/inspections/firm-inspection-reports
    """
    df = pd.read_csv(local_path)
    df.to_csv("../data/raw/pcaob_deficiencies_raw.csv", index=False)
    print(f"Loaded {len(df)} real PCAOB deficiency records.")
    return df


# ---------------------------------------------------------------------------
# PART 2: SEC EDGAR Full-Text Search (material weakness disclosures, 2020+)
# ---------------------------------------------------------------------------
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# SEC requires a descriptive User-Agent identifying you -- this is mandatory,
# not optional, per SEC's fair-access policy.
HEADERS = {
    "User-Agent": "Daljeet Kaur Johar daljeetkaur07@gmail.com"
}


def search_material_weakness_filings(form_type: str, start_date: str, end_date: str,
                                      max_pages: int = 10) -> list:
    """
    Search SEC EDGAR full-text search for material weakness disclosures.
    form_type: e.g. "8-K" or "10-K"
    start_date / end_date: "YYYY-MM-DD"
    """
    all_hits = []
    for page in range(max_pages):
        params = {
            "q": '"material weakness"',
            "forms": form_type,
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": page * 10,
        }
        response = requests.get(EDGAR_SEARCH_URL, params=params, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            print(f"  Stopped at page {page}, status {response.status_code}")
            break

        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        all_hits.extend(hits)
        print(f"  {form_type}: fetched {len(all_hits)} filings so far...")
        time.sleep(0.3)  # respect SEC's rate limits

    return all_hits


def main():
    print("STEP 1: PCAOB data")
    print("  -> Manually download the PCAOB Part I.A/I.B CSV from:")
    print("     https://pcaobus.org/oversight/inspections/firm-inspection-reports")
    print("  -> Then call load_pcaob_csv('path/to/downloaded_file.csv')")

    print("\nSTEP 2: SEC EDGAR material weakness disclosures (2020-2026)")
    filings_8k = search_material_weakness_filings("8-K", "2020-01-01", "2026-07-17")
    filings_10k = search_material_weakness_filings("10-K", "2020-01-01", "2026-07-17")

    all_filings = filings_8k + filings_10k
    with open("../data/raw/sec_edgar_disclosures_raw.json", "w") as f:
        json.dump(all_filings, f, indent=2)

    print(f"\nSaved {len(all_filings)} real SEC filings to "
          f"data/raw/sec_edgar_disclosures_raw.json")


if __name__ == "__main__":
    main()
