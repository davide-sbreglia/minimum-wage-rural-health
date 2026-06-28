"""Build the minimum-wage treatment series (state x year) following Sigaud et al.

Treatment construction (per Sigaud et al. 2022 / Sigaud 2021 thesis):
  1. Effective minimum wage = max(state minimum wage, federal minimum wage).
     Some states (e.g. GA, WY) set a state minimum below the federal floor;
     federally covered workers still face the $7.25 federal rate, so the max
     is the wage that actually binds.
  2. Convert nominal to real 2019 dollars using the annual CPI-U.
  3. Apply a one-year lag: health in year t is modeled as responding to the
     effective real minimum wage in year t-1 (Horn et al. 2017; Lenhart 2019).

Source: University of Kentucky Center for Poverty Research, National Welfare
Data (1980-2024). State minimum wage, federal minimum wage, and the project's
control variables all come from this single source, matching Sigaud.

Input:  data/raw/ukcpr_welfare_data.xlsx  (downloaded from UKCPR)
Output: data/processed/treatment_min_wage.parquet  (state_fips, year, mw_real_lag1, ...)
"""

from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
UKCPR_PATH = RAW / "ukcpr_welfare_data.xlsx"

YEAR_MIN, YEAR_MAX = 2011, 2019

# Annual CPI-U (US city average, all items, 1982-84=100), BLS series CUUR0000SA0.
# Used to deflate nominal dollars to real 2019 dollars: real = nominal * (CPI_2019 / CPI_year).
# We need one extra earlier year (2010) because of the one-year lag.
CPI_U = {
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957,
    2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120,
    2018: 251.107, 2019: 255.657,
}
CPI_BASE = CPI_U[2019]  # express everything in 2019 dollars

# Control variables to carry along (same source/definitions as Sigaud).
CONTROL_COLS = {
    "AFDC/TANF benefit for 4-person family": "tanf_4person",
    "FS/SNAP Benefit for 4-person family": "snap_4person",
    "State EITC Rate": "state_eitc_rate",
    "Unemployment rate": "unemployment_rate",
    "Personal income": "personal_income",
}


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(UKCPR_PATH, sheet_name="Data")

    # Keep one extra prior year (2010) so the lag is defined for 2011.
    df = df[(df["year"] >= YEAR_MIN - 1) & (df["year"] <= YEAR_MAX)].copy()

    # 1. Effective nominal minimum wage = max(state, federal)
    df["mw_effective_nom"] = df[["State Minimum Wage", "Federal Minimum Wage"]].max(axis=1)

    # 2. Deflate to real 2019 dollars
    df["cpi"] = df["year"].map(CPI_U)
    df["mw_effective_real"] = df["mw_effective_nom"] * (CPI_BASE / df["cpi"])

    # Rename controls
    df = df.rename(columns=CONTROL_COLS)

    keep = ["state_fips", "year", "mw_effective_nom", "mw_effective_real"] + list(CONTROL_COLS.values())
    out = df[keep].sort_values(["state_fips", "year"]).copy()

    # 3. One-year lag of the real effective minimum wage (within each state)
    out["mw_real_lag1"] = out.groupby("state_fips")["mw_effective_real"].shift(1)

    # Drop the helper year 2010 (kept only to define the 2011 lag)
    out = out[out["year"] >= YEAR_MIN].copy()

    out.to_parquet(PROCESSED / "treatment_min_wage.parquet", index=False)

    print(f"Treatment built: {len(out)} state-year rows ({YEAR_MIN}-{YEAR_MAX})")
    print(f"States: {out['state_fips'].nunique()}")
    print("\nReal effective minimum wage (2019 $) — describe:")
    print(out["mw_effective_real"].describe().round(2).to_string())
    print("\nLagged treatment missing (expected 0 after dropping 2010):",
          out["mw_real_lag1"].isna().sum())
    print("\nExample — a state over time (FIPS 6 = California):")
    print(out[out["state_fips"] == 6][["year", "mw_effective_nom", "mw_effective_real", "mw_real_lag1"]].to_string(index=False))


if __name__ == "__main__":
    main()
