"""Recode BRFSS special missing-value codes into proper values and NaN.

BRFSS uses numeric sentinel codes instead of empty cells. This script converts
those codes for the project's variables, following the CDC BRFSS codebooks.
No rows are dropped: missing values are marked NaN; row filtering happens later.

Recoding (per CDC BRFSS codebooks):
  ment_health_days / phys_health_days (1-30 valid): 88->0, 77/99->NaN
  gen_health (1-5 valid):                            7/9 -> NaN
  sex (1=male, 2=female):                            7/9 -> NaN
  age_group (_AGEG5YR, 1-13 valid):                  14  -> NaN ("Don't know/Refused/Missing")
  education, geo_msa, state, year, weight, interview_month: handled elsewhere / clean.

Input:  data/processed/brfss_panel.parquet
Output: data/processed/brfss_panel_clean.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
IN_PATH = PROCESSED_DIR / "brfss_panel.parquet"
OUT_PATH = PROCESSED_DIR / "brfss_panel_clean.parquet"


def recode_health_days(series: pd.Series) -> pd.Series:
    """88->0, 77/99->NaN for BRFSS 'days in past 30' variables."""
    return series.replace({88: 0, 77: np.nan, 99: np.nan})


def recode_dk_refused(series: pd.Series) -> pd.Series:
    """7='don't know', 9='refused' -> NaN."""
    return series.replace({7: np.nan, 9: np.nan})


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    n_before = len(df)

    df["ment_health_days"] = recode_health_days(df["ment_health_days"])
    df["phys_health_days"] = recode_health_days(df["phys_health_days"])
    df["gen_health"] = recode_dk_refused(df["gen_health"])
    df["sex"] = recode_dk_refused(df["sex"])
    df["age_group"] = df["age_group"].replace({14: np.nan})  # 14 = DK/Refused/Missing

    df.to_parquet(OUT_PATH, index=False)

    print(f"Rows: {n_before:,} (unchanged; no rows dropped)\n")
    print("Missing values (NaN) after recoding:")
    for col in ["gen_health", "ment_health_days", "phys_health_days", "sex", "age_group"]:
        n_na = df[col].isna().sum()
        print(f"  {col:18s} {n_na:>9,}  ({100*n_na/n_before:.1f}%)")
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
