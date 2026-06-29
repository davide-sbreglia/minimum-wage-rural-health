"""Build the analysis sample following Sigaud et al. (2022) / Sigaud (2021).

Starting from the merged analytic dataset, this script applies the two sample
definitions that characterize Sigaud's study:

  1. Low-education filter: keep respondents with a high-school education or less,
     i.e. _EDUCAG in {1, 2} (less than high school, or high-school graduate).
     These are the workers most exposed to the minimum wage. _EDUCAG == 9
     ("don't know / refused") is treated as missing and excluded.

  2. Dichotomized health outcomes (Sigaud's main specification, following
     Zhao et al. 2018):
       - poor_mental  = 1 if ment_health_days >= 14, else 0
       - poor_physical= 1 if phys_health_days >= 14, else 0
     We also add the standard self-rated-health binary used widely in the
     literature:
       - fair_poor_health = 1 if gen_health in {4, 5} (fair or poor), else 0
     Each binary is NaN where the underlying health variable is missing.

This is kept as a separate, explicit step (not folded into cleaning) so the
sample restriction is transparent and auditable.

Input:  data/processed/analytic_dataset.parquet
Output: data/processed/sample_sigaud.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")


def main() -> None:
    df = pd.read_parquet(PROCESSED / "analytic_dataset.parquet")
    n_start = len(df)

    # --- Tag 1: low-education filter -------------------------------------
    # Treat 9 (don't know/refused) as missing, then keep {1, 2}.
    df["education"] = df["education"].replace({9: np.nan})
    sample = df[df["education"].isin([1, 2])].copy()
    n_lowedu = len(sample)

    # --- Tag 2: dichotomized health outcomes -----------------------------
    # Binary "significant burden" = 14+ poor days (NaN preserved where input is NaN).
    sample["poor_mental"] = np.where(
        sample["ment_health_days"].isna(), np.nan,
        (sample["ment_health_days"] >= 14).astype(float))
    sample["poor_physical"] = np.where(
        sample["phys_health_days"].isna(), np.nan,
        (sample["phys_health_days"] >= 14).astype(float))
    # Self-rated health binary: fair or poor (gen_health 4 or 5).
    sample["fair_poor_health"] = np.where(
        sample["gen_health"].isna(), np.nan,
        sample["gen_health"].isin([4, 5]).astype(float))

    sample.to_parquet(PROCESSED / "sample_sigaud.parquet", index=False)

    # --- Report ----------------------------------------------------------
    print(f"Starting rows (all education): {n_start:,}")
    print(f"After low-education filter {{1,2}}: {n_lowedu:,} "
          f"({100*n_lowedu/n_start:.1f}%)")
    print("\nDichotomized outcome prevalence (share = 1):")
    for col in ["poor_mental", "poor_physical", "fair_poor_health"]:
        print(f"  {col:18s} {sample[col].mean():.3f}  "
              f"(missing: {sample[col].isna().sum():,})")
    print("\nSample by sex x rural (geo-identified only):")
    geo = sample.dropna(subset=["rural"])
    tab = geo.groupby(["sex", "rural"]).size().unstack()
    tab.columns = ["urban", "rural"]
    tab.index = ["male", "female"]
    print(tab.to_string())

    print(f"\nSaved to: {PROCESSED / 'sample_sigaud.parquet'}")


if __name__ == "__main__":
    main()
