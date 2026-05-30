"""
predict.py
----------
Community Energy Access Predictor
Author: Sharath Chandra Reddy Karrepu
Organization: Community Dreams Foundation

Description:
    Loads the saved best model and runs inference on new data.
    Accepts a CSV file path as CLI argument, outputs predictions.

Usage:
    python src/predict.py --input data/new_communities.csv
    python src/predict.py --demo        # runs on 5 synthetic examples
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os


MODEL_PATH = "models/best_model.pkl"
DATA_PATH  = "data/communities_clean.csv"   # used to get column list


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at '{MODEL_PATH}'.")
        print("Run 'python src/model_training.py' first.")
        sys.exit(1)
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], artifact["model_name"]


def get_expected_features() -> list:
    """Return feature list consistent with training."""
    df = pd.read_csv(DATA_PATH)
    drop = ["community_id", "needs_energy_intervention"]
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


def make_demo_input(feature_cols: list) -> pd.DataFrame:
    """Create 5 synthetic communities for demo purposes."""
    np.random.seed(99)
    n = 5
    demo = {col: np.random.rand(n) for col in feature_cols}

    # Force clear high-risk profile on row 0
    idx_vuln = feature_cols.index("vulnerability_index") if "vulnerability_index" in feature_cols else None
    idx_grid = feature_cols.index("grid_reliability_score") if "grid_reliability_score" in feature_cols else None
    if idx_vuln is not None:
        demo["vulnerability_index"][0] = 45.0   # very high
    if idx_grid is not None:
        demo["grid_reliability_score"][0] = 2.0  # poor grid

    return pd.DataFrame(demo)


def predict(pipe, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X = df[[c for c in feature_cols if c in df.columns]]
    y_pred  = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]

    results = df.copy()
    results["prediction"] = y_pred
    results["intervention_probability"] = y_proba.round(4)
    results["recommendation"] = results["prediction"].map({
        0: "✅ Stable – monitor quarterly",
        1: "🔴 Priority – schedule energy intervention"
    })
    return results


def main():
    parser = argparse.ArgumentParser(description="Community Energy Intervention Predictor")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--demo",  action="store_true", help="Run on 5 synthetic demo communities")
    args = parser.parse_args()

    pipe, model_name = load_model()
    feature_cols = get_expected_features()

    print(f"\nModel loaded: {model_name}")
    print(f"Features expected: {len(feature_cols)}\n")

    if args.demo or (not args.input):
        print("Running DEMO on 5 synthetic communities...\n")
        df_input = make_demo_input(feature_cols)
    else:
        if not os.path.exists(args.input):
            print(f"ERROR: File not found – {args.input}")
            sys.exit(1)
        df_input = pd.read_csv(args.input)
        print(f"Loaded {len(df_input)} communities from {args.input}\n")

    results = predict(pipe, df_input, feature_cols)

    display_cols = [c for c in ["community_id", "vulnerability_index",
                                 "grid_reliability_score", "poverty_rate_pct",
                                 "intervention_probability", "recommendation"]
                    if c in results.columns]
    print(results[display_cols].to_string(index=False))

    out_path = "outputs/predictions.csv"
    os.makedirs("outputs", exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nPredictions saved → {out_path}")


if __name__ == "__main__":
    main()
