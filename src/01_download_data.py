"""Download all raw data for the project from their official sources.

This is the entry point of the pipeline. It downloads:
  - BRFSS annual survey files (2011-2019) from the CDC, as zipped .XPT, and
    extracts them into data/raw/.
  - UKCPR National Welfare Data (state-year minimum wage + controls) from the
    University of Kentucky Center for Poverty Research, into data/raw/.

Raw data is downloaded rather than committed to the repo because the BRFSS
files are far too large for GitHub, and pulling from source keeps the project
reproducible: anyone who clones the repo runs this script to regenerate the
exact raw inputs. Files that already exist are skipped.

Output: data/raw/LLCP<year>.XPT  (2011-2019) and data/raw/ukcpr_welfare_data.xlsx
"""

from pathlib import Path
import zipfile
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

BRFSS_YEARS = range(2011, 2020)
BRFSS_URL = "https://www.cdc.gov/brfss/annual_data/{y}/files/LLCP{y}XPT.zip"
UKCPR_URL = ("https://ukcpr.uky.edu/sites/default/files/2026-02/"
             "ukcpr_national_welfare_data_1980_2024_jan26update.xlsx")


def stream_download(url: str, dest: Path) -> None:
    """Stream a file to disk, skipping if it already exists."""
    if dest.exists():
        print(f"  already present: {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_brfss() -> None:
    print("BRFSS (CDC):")
    for year in BRFSS_YEARS:
        # Skip if the .XPT for this year is already extracted (names may have a trailing space).
        if any(str(year) in p.name and p.name.strip().upper().endswith(".XPT")
               for p in RAW.iterdir()):
            print(f"  {year}: already extracted")
            continue
        zip_path = RAW / f"LLCP{year}.zip"
        stream_download(BRFSS_URL.format(y=year), zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(RAW)
        print(f"  {year}: extracted")


def download_ukcpr() -> None:
    print("UKCPR National Welfare Data:")
    stream_download(UKCPR_URL, RAW / "ukcpr_welfare_data.xlsx")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    download_brfss()
    download_ukcpr()
    print("\nAll raw data ready in data/raw/")


if __name__ == "__main__":
    main()
