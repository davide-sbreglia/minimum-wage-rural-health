"""Build a harmonized BRFSS panel (2011-2019) from the raw .XPT files.

Keeps the project's variables, renames them to stable names, and stacks all
years into one long panel with a `year` column. Reads in chunks to stay within
Colab RAM (files have 342 columns, ~400-500k rows each).

Cross-year harmonization: all key variables share the same BRFSS name across
2011-2019 EXCEPT sex (SEX 2011-2017, SEX1 2018, SEXVAR 2019) -> `sex`.
(Race is deliberately not included: no single race variable spans all 9 years
cleanly; it will be added later only if the analysis requires it.)

Input:  data/raw/LLCP<year>.XPT
Output: data/processed/brfss_panel.parquet
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
YEARS = range(2011, 2020)

STABLE_VARS = {
    "_STATE": "state",
    "MSCODE": "geo_msa",
    "GENHLTH": "gen_health",
    "MENTHLTH": "ment_health_days",
    "PHYSHLTH": "phys_health_days",
    "_EDUCAG": "education",
    "_LLCPWT": "weight",
    "_AGEG5YR": "age_group",   # age in 5-year groups (1=18-24 ... 13=80+), 14=missing
    "IMONTH": "interview_month",  # interview month, for month fixed effects
}

SEX_VAR_BY_YEAR = {y: "SEX" for y in range(2011, 2018)}
SEX_VAR_BY_YEAR[2018] = "SEX1"
SEX_VAR_BY_YEAR[2019] = "SEXVAR"


def xpt_path_for(year: int) -> Path:
    matches = [p for p in RAW_DIR.iterdir()
               if str(year) in p.name and p.name.strip().upper().endswith(".XPT")]
    if not matches:
        raise FileNotFoundError(f"No .XPT found for {year} in {RAW_DIR}")
    return matches[0]


def load_year(year: int) -> pd.DataFrame:
    sex_var = SEX_VAR_BY_YEAR[year]
    keep = list(STABLE_VARS.keys()) + [sex_var]
    pieces = []
    for chunk in pd.read_sas(xpt_path_for(year), format="xport", chunksize=50_000):
        pieces.append(chunk[keep])
    df = pd.concat(pieces, ignore_index=True)
    rename_map = dict(STABLE_VARS)
    rename_map[sex_var] = "sex"
    df = df.rename(columns=rename_map)
    df["year"] = year
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in YEARS:
        df_year = load_year(year)
        print(f"  {year}: {len(df_year):>7,} rows", flush=True)
        frames.append(df_year)
    panel = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "brfss_panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"\nPanel built: {len(panel):,} rows x {panel.shape[1]} columns")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
