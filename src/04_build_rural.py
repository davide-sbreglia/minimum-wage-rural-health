"""Construct the rural/urban indicator from BRFSS metropolitan status (MSCODE).

MSCODE is the only geographic variable available consistently across 2011-2019
(_METSTAT and _URBSTAT exist only from 2018). We collapse its categories into a
binary metropolitan/non-metropolitan split, which operationalizes the
rural/urban axis used by Sigaud et al. (2022).

Decisions (all documented for the paper's data section):
  - Drop non-state territories: Puerto Rico (66), Guam (72), Virgin Islands (78).
    These lack a metropolitan classification and fall outside US state minimum
    wage policy, which is the treatment.
  - MSCODE coding (CDC BRFSS codebook):
        1 = in the center city of an MSA
        2 = outside center city, in the county containing the center city
        3 = in a county of the MSA, no center city
        5 = not in an MSA
        4 = appears only 2011-2013, <0.2% of records -> set to NaN
    rural = 1 if MSCODE == 5 ; rural = 0 if MSCODE in {1,2,3} ; else NaN
  - Rows with missing geography (NaN MSCODE) are kept but have rural = NaN; they
    are excluded only in geography-specific analyses. The rural share among
    geo-identified respondents is stable over time (30-36%), so this attrition
    affects statistical power, not composition.

Input:  data/processed/brfss_panel_clean.parquet
Output: data/processed/brfss_panel_geo.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
IN_PATH = PROCESSED_DIR / "brfss_panel_clean.parquet"
OUT_PATH = PROCESSED_DIR / "brfss_panel_geo.parquet"

NON_STATE_TERRITORIES = [66, 72, 78]  # Guam, Puerto Rico, US Virgin Islands


def build_rural(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary `rural` column from MSCODE; drop non-state territories."""
    df = df[~df["state"].isin(NON_STATE_TERRITORIES)].copy()

    # Map MSCODE -> rural indicator. Codes 1,2,3 = metropolitan (urban),
    # code 5 = non-metropolitan (rural), code 4 and anything else -> NaN.
    rural = np.select(
        condlist=[df["geo_msa"] == 5, df["geo_msa"].isin([1, 2, 3])],
        choicelist=[1, 0],
        default=np.nan,
    )
    df["rural"] = rural
    return df


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    n_before = len(df)

    df = build_rural(df)

    df.to_parquet(OUT_PATH, index=False)

    n_after = len(df)
    n_geo = df["rural"].notna().sum()
    print(f"Rows before: {n_before:,}")
    print(f"Rows after dropping territories: {n_after:,} "
          f"(removed {n_before - n_after:,})")
    print(f"\nGeo-identified rows (rural not NaN): {n_geo:,} "
          f"({100*n_geo/n_after:.1f}%)")
    print("\nRural indicator distribution (geo-identified only):")
    print(df["rural"].value_counts(dropna=False).rename({0.0: "urban", 1.0: "rural"}))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
