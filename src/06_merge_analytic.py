"""Merge the BRFSS panel with the minimum-wage treatment into an analytic dataset.

Joins each respondent to the effective real minimum wage (and controls) of their
state and survey year, via state FIPS + year. The lagged treatment (mw_real_lag1)
is the regressor of interest, following Sigaud's one-year-lag specification.

Input:  data/processed/brfss_panel_geo.parquet      (respondents + health + rural)
        data/processed/treatment_min_wage.parquet   (state-year treatment + controls)
Output: data/processed/analytic_dataset.parquet
"""

from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")


def main() -> None:
    geo = pd.read_parquet(PROCESSED / "brfss_panel_geo.parquet")
    mw = pd.read_parquet(PROCESSED / "treatment_min_wage.parquet")

    n_before = len(geo)

    # Left join on state + year: keep every respondent, attach their state-year treatment.
    merged = geo.merge(
        mw,
        left_on=["state", "year"],
        right_on=["state_fips", "year"],
        how="left",
        validate="many_to_one",  # many respondents -> one state-year row; errors if not
    )

    # Sanity: every respondent should have matched a treatment row.
    n_unmatched = merged["mw_real_lag1"].isna().sum() - geo["state"].isna().sum()
    merged = merged.drop(columns=["state_fips"])  # redundant after merge

    merged.to_parquet(PROCESSED / "analytic_dataset.parquet", index=False)

    print(f"Respondents before merge: {n_before:,}")
    print(f"Respondents after merge:  {len(merged):,}  (should be identical)")
    print(f"\nColumns: {list(merged.columns)}")
    print(f"\nRows missing lagged treatment: {merged['mw_real_lag1'].isna().sum():,}")
    print("\nSample (one state, both rural and urban):")
    cols = ["state", "year", "rural", "gen_health", "ment_health_days",
            "mw_real_lag1", "unemployment_rate"]
    print(merged[merged["state"] == 6][cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
