"""
eda_visualization.py
--------------------
Community Energy Access Predictor
Author: Sharath Chandra Reddy Karrepu
Organization: Community Dreams Foundation

Description:
    Exploratory Data Analysis – generates charts saved to outputs/
    so they can be embedded in the GitHub README and reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
})

DATA_PATH = "data/communities_clean.csv"
OUT_DIR = "outputs"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


# ── 1. Distribution of vulnerability index ────────────────────────
def plot_vulnerability_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["vulnerability_index"], bins=40, kde=True,
                 color="#2E86AB", ax=ax)
    ax.axvline(df["vulnerability_index"].quantile(0.60),
               color="#E84855", linestyle="--", linewidth=1.8,
               label="60th percentile (intervention threshold)")
    ax.set_title("Distribution of Community Vulnerability Index")
    ax.set_xlabel("Vulnerability Index")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    path = f"{OUT_DIR}/01_vulnerability_distribution.png"
    fig.savefig(path)
    plt.close()
    print(f"Saved → {path}")


# ── 2. Energy burden by community type ───────────────────────────
def plot_energy_burden_by_type(df: pd.DataFrame):
    # Recover community_type from one-hot columns
    type_cols = [c for c in df.columns if c.startswith("community_type_")]
    if type_cols:
        df = df.copy()
        df["community_type"] = df[type_cols].idxmax(axis=1).str.replace(
            "community_type_", ""
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = df.groupby("community_type")["energy_burden_pct"].median().sort_values(
        ascending=False
    ).index
    sns.boxplot(data=df, x="community_type", y="energy_burden_pct",
                order=order, palette="Set2", ax=ax)
    ax.set_title("Energy Burden (%) by Community Type")
    ax.set_xlabel("Community Type")
    ax.set_ylabel("Energy Burden (% of Income)")
    plt.tight_layout()
    path = f"{OUT_DIR}/02_energy_burden_by_type.png"
    fig.savefig(path)
    plt.close()
    print(f"Saved → {path}")


# ── 3. Correlation heatmap ────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    key_cols = [
        "vulnerability_index", "energy_burden_pct", "poverty_rate_pct",
        "unemployment_rate_pct", "grid_reliability_score",
        "renewable_energy_adoption_pct", "median_household_income_usd",
        "pct_households_without_electricity", "solar_irradiance_kwh_m2",
        "needs_energy_intervention",
    ]
    key_cols = [c for c in key_cols if c in numeric_df.columns]
    corr = numeric_df[key_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = f"{OUT_DIR}/03_correlation_heatmap.png"
    fig.savefig(path)
    plt.close()
    print(f"Saved → {path}")


# ── 4. Target class balance ───────────────────────────────────────
def plot_class_balance(df: pd.DataFrame):
    counts = df["needs_energy_intervention"].value_counts()
    labels = ["No Intervention Needed", "Needs Intervention"]
    colors = ["#52B788", "#E84855"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10, str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Target Variable Class Balance")
    ax.set_ylabel("Number of Communities")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    path = f"{OUT_DIR}/04_class_balance.png"
    fig.savefig(path)
    plt.close()
    print(f"Saved → {path}")


# ── 5. Income vs vulnerability scatter ───────────────────────────
def plot_income_vs_vulnerability(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = df["needs_energy_intervention"].map({0: "#52B788", 1: "#E84855"})
    ax.scatter(df["median_household_income_usd"], df["vulnerability_index"],
               c=colors, alpha=0.4, s=12, edgecolors="none")
    ax.set_xlabel("Median Household Income (USD)")
    ax.set_ylabel("Vulnerability Index")
    ax.set_title("Income vs Vulnerability Index\n(Red = Needs Intervention)")
    # Legend proxy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#52B788',
               markersize=8, label='No Intervention'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E84855',
               markersize=8, label='Needs Intervention'),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    path = f"{OUT_DIR}/05_income_vs_vulnerability.png"
    fig.savefig(path)
    plt.close()
    print(f"Saved → {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} communities, {df.shape[1]} features\n")

    print("Generating EDA charts...")
    plot_vulnerability_distribution(df)
    plot_energy_burden_by_type(df)
    plot_correlation_heatmap(df)
    plot_class_balance(df)
    plot_income_vs_vulnerability(df)

    print("\nAll charts saved to outputs/")


if __name__ == "__main__":
    main()
