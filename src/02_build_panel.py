"""Build a harmonized BRFSS panel (2011-2019) from the raw .XPT files.

Keeps the project's variables, renames them to stable names, and stacks all
years into one long panel. Reads in chunks to stay within Colab RAM.

Cross-year harmonization:
  - sex: SEX (2011-2017), SEX1 (2018), SEXVAR (2019) -> `sex`
  - race: built from two sources with DIFFERENT codings:
        _IMPRACE (imputed race, available all years except 2015-2016)
        _RACEGR3 (race groups, used only for 2015-2016)
    Both are mapped to Sigaud's 5 categories: white, black, asian, hispanic, other.
    NOTE: _RACEGR3 does not separate Asians (folded into 'other'), so 'asian'
    is identifiable in all years EXCEPT 2015-2016. Asians are a small share, so
    impact is minor; this is documented as a data limitation.

Input:  data/raw/LLCP<year>.XPT
Output: data/processed/brfss_panel.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np

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
    "_AGEG5YR": "age_group",
    "IMONTH": "interview_month",
}

SEX_VAR_BY_YEAR = {y: "SEX" for y in range(2011, 2018)}
SEX_VAR_BY_YEAR[2018] = "SEX1"
SEX_VAR_BY_YEAR[2019] = "SEXVAR"

# Map each source variable's codes to Sigaud's 5 race categories.
# _IMPRACE: 1=white, 2=black, 3=asian, 4=AmInd/AKNative, 5=hispanic, 6=other/multi
IMPRACE_MAP = {1: "white", 2: "black", 3: "asian", 4: "other",
               5: "hispanic", 6: "other"}
# _RACEGR3: 1=white, 2=black, 3=other-nonhisp, 4=multiracial, 5=hispanic, 9=missing
RACEGR3_MAP = {1: "white", 2: "black", 3: "other", 4: "other",
               5: "hispanic", 9: np.nan}


def xpt_path_for(year: int) -> Path:
    matches = [p for p in RAW_DIR.iterdir()
               if str(year) in p.name and p.name.strip().upper().endswith(".XPT")]
    if not matches:
        raise FileNotFoundError(f"No .XPT found for {year} in {RAW_DIR}")
    return matches[0]


def race_var_for(year: int, cols: set) -> tuple[str, dict]:
    """Pick the race source variable for a year and its code->category map."""
    if "_IMPRACE" in cols:
        return "_IMPRACE", IMPRACE_MAP
    return "_RACEGR3", RACEGR3_MAP


def load_year(year: int) -> pd.DataFrame:
    sex_var = SEX_VAR_BY_YEAR[year]
    # Peek at columns to choose the race variable for this year.
    cols0 = set(next(pd.read_sas(xpt_path_for(year), format="xport", chunksize=1)).columns)
    race_var, race_map = race_var_for(year, cols0)

    keep = list(STABLE_VARS.keys()) + [sex_var, race_var]
    pieces = []
    for chunk in pd.read_sas(xpt_path_for(year), format="xport", chunksize=50_000):
        pieces.append(chunk[keep])
    df = pd.concat(pieces, ignore_index=True)

    rename_map = dict(STABLE_VARS)
    rename_map[sex_var] = "sex"
    df = df.rename(columns=rename_map)

    # Harmonize race onto Sigaud's categories
    df["race"] = df[race_var].map(race_map)
    df = df.drop(columns=[race_var])

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
