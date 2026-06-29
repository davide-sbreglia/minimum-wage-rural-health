"""Recode BRFSS special missing-value codes into proper values and NaN.

BRFSS does not use empty cells for non-substantive answers; it uses numeric
sentinel codes. This script converts those codes for the project's variables,
following the CDC BRFSS codebooks. It does NOT drop any rows: missing values
are marked as NaN and row-level filtering is deferred to the analysis stage.

Recoding (per CDC BRFSS codebooks):
  ment_health_days / phys_health_days  (valid range 1-30):
      88 -> 0     ("None" = zero bad-health days)
      77 -> NaN   ("Don't know / Not sure")
      99 -> NaN   ("Refused")
  gen_health  (valid range 1-5):
      7  -> NaN   ("Don't know / Not sure")
      9  -> NaN   ("Refused")
  sex  (1 = male, 2 = female):
      7  -> NaN   ("Don't know / Not sure")
      9  -> NaN   ("Refused")
  education, geo_msa, state, year: handled elsewhere or already clean.

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
    """Recode a BRFSS 'days in past 30' variable (88->0, 77/99->NaN)."""
    return series.replace({88: 0, 77: np.nan, 99: np.nan})


def recode_dk_refused(series: pd.Series) -> pd.Series:
    """Recode BRFSS variables where 7='don't know' and 9='refused' -> NaN."""
    return series.replace({7: np.nan, 9: np.nan})


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    n_before = len(df)

    df["ment_health_days"] = recode_health_days(df["ment_health_days"])
    df["phys_health_days"] = recode_health_days(df["phys_health_days"])
    df["gen_health"] = recode_dk_refused(df["gen_health"])
    df["sex"] = recode_dk_refused(df["sex"])

    df.to_parquet(OUT_PATH, index=False)

    print(f"Rows: {n_before:,} (unchanged; no rows dropped)\n")
    print("Missing values (NaN) after recoding:")
    for col in ["gen_health", "ment_health_days", "phys_health_days", "sex"]:
        n_na = df[col].isna().sum()
        print(f"  {col:18s} {n_na:>9,}  ({100*n_na/n_before:.1f}%)")
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
