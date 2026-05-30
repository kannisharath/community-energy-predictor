"""
data_preprocessing.py
---------------------
Community Energy Access Predictor
Author: Sharath Chandra Reddy Karrepu
Organization: Community Dreams Foundation
Role: Data Scientist

Description:
    Generates synthetic but realistic community-level dataset,
    performs cleaning, preprocessing, and feature engineering.
    In a real project, replace generate_dataset() with actual
    data ingestion from CSV / API / database.
"""

import pandas as pd
import numpy as np
import os

# ── Reproducibility ──────────────────────────────────────────────
np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────
N_COMMUNITIES = 1500
OUTPUT_PATH = "data/communities_clean.csv"
RAW_PATH = "data/communities_raw.csv"


def generate_dataset(n: int = N_COMMUNITIES) -> pd.DataFrame:
    """
    Simulate community-level socioeconomic & energy features.
    Features are inspired by publicly available World Bank and
    US EIA datasets on energy access and community development.
    """
    regions = ["Southeast", "Midwest", "Southwest", "Northeast", "West"]
    community_types = ["Rural", "Suburban", "Urban"]

    data = {
        "community_id": [f"COM-{i:04d}" for i in range(1, n + 1)],
        "region": np.random.choice(regions, n),
        "community_type": np.random.choice(
            community_types, n, p=[0.45, 0.35, 0.20]
        ),
        "population": np.random.randint(500, 50000, n),
        "median_household_income_usd": np.random.normal(42000, 12000, n).clip(15000, 120000),
        "poverty_rate_pct": np.random.beta(2, 6, n) * 60,          # 0–60 %
        "unemployment_rate_pct": np.random.beta(2, 8, n) * 30,     # 0–30 %
        "avg_monthly_energy_bill_usd": np.random.normal(130, 35, n).clip(40, 400),
        "renewable_energy_adoption_pct": np.random.beta(1.5, 5, n) * 100,
        "grid_reliability_score": np.random.uniform(1, 10, n),     # 1=poor, 10=excellent
        "distance_to_nearest_grid_km": np.random.exponential(15, n).clip(0, 200),
        "solar_irradiance_kwh_m2": np.random.normal(4.5, 1.2, n).clip(1.5, 7.5),
        "avg_household_size": np.random.normal(2.8, 0.7, n).clip(1.5, 6.5),
        "pct_households_without_electricity": np.random.beta(1.2, 8, n) * 40,
        "community_org_count": np.random.poisson(3, n),
        "prior_energy_program_participation": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "internet_access_pct": np.random.normal(72, 18, n).clip(10, 100),
        "education_index": np.random.uniform(0, 1, n),             # 0=low, 1=high
    }

    df = pd.DataFrame(data)

    # ── Introduce realistic missing values ───────────────────────
    for col in ["avg_monthly_energy_bill_usd", "renewable_energy_adoption_pct",
                "internet_access_pct", "grid_reliability_score"]:
        mask = np.random.rand(n) < 0.05       # ~5 % missing
        df.loc[mask, col] = np.nan

    # ── Introduce a few duplicates ───────────────────────────────
    dup_idx = np.random.choice(df.index, size=20, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, impute missing values, and fix dtypes."""
    print(f"Raw shape      : {df.shape}")

    # 1. Drop duplicates
    df = df.drop_duplicates(subset="community_id")
    print(f"After dedup    : {df.shape}")

    # 2. Impute numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed '{col}' → median = {median_val:.2f}")

    # 3. Ensure correct dtypes
    df["prior_energy_program_participation"] = df[
        "prior_energy_program_participation"
    ].astype(int)

    print(f"Missing values : {df.isnull().sum().sum()}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that improve model signal."""

    # Energy burden = monthly bill as % of monthly income
    df["energy_burden_pct"] = (
        df["avg_monthly_energy_bill_usd"] /
        (df["median_household_income_usd"] / 12)
    ) * 100

    # Vulnerability index (higher = more at-risk community)
    df["vulnerability_index"] = (
        df["poverty_rate_pct"] * 0.35 +
        df["unemployment_rate_pct"] * 0.25 +
        df["pct_households_without_electricity"] * 0.25 +
        df["energy_burden_pct"] * 0.15
    )

    # Solar potential flag
    df["high_solar_potential"] = (df["solar_irradiance_kwh_m2"] >= 5.0).astype(int)

    # Encode categoricals
    df = pd.get_dummies(df, columns=["region", "community_type"], drop_first=False)

    # ── Target variable ──────────────────────────────────────────
    # Binary: will a community benefit from an energy intervention?
    # Defined as: vulnerability_index > 25th percentile AND
    #             grid_reliability_score < median
    threshold_vuln = df["vulnerability_index"].quantile(0.60)
    threshold_grid = df["grid_reliability_score"].median()

    df["needs_energy_intervention"] = (
        (df["vulnerability_index"] > threshold_vuln) &
        (df["grid_reliability_score"] < threshold_grid)
    ).astype(int)

    pos_rate = df["needs_energy_intervention"].mean()
    print(f"Target positive rate: {pos_rate:.1%}")
    return df


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 55)
    print("  STEP 1 – Generating synthetic community dataset")
    print("=" * 55)
    raw_df = generate_dataset()
    raw_df.to_csv(RAW_PATH, index=False)
    print(f"Raw data saved → {RAW_PATH}")

    print("\n" + "=" * 55)
    print("  STEP 2 – Cleaning data")
    print("=" * 55)
    clean_df = clean_data(raw_df)

    print("\n" + "=" * 55)
    print("  STEP 3 – Feature engineering")
    print("=" * 55)
    final_df = feature_engineering(clean_df)

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nProcessed data saved → {OUTPUT_PATH}")
    print(f"Final shape: {final_df.shape}")
    print("\nSample (5 rows):")
    print(final_df[["community_id", "vulnerability_index",
                     "energy_burden_pct", "needs_energy_intervention"]].head())


if __name__ == "__main__":
    main()
