"""
model_training.py
-----------------
Community Energy Access Predictor
Author: Sharath Chandra Reddy Karrepu
Organization: Community Dreams Foundation

Description:
    Trains and evaluates multiple classification models to predict
    which communities need an energy intervention.
    Saves the best model to models/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# ── Paths ─────────────────────────────────────────────────────────
DATA_PATH   = "data/communities_clean.csv"
MODEL_DIR   = "models"
OUT_DIR     = "outputs"

# ── Target & features ─────────────────────────────────────────────
TARGET = "needs_energy_intervention"
DROP_COLS = ["community_id", TARGET]


def load_and_split(test_size: float = 0.20):
    df = pd.read_csv(DATA_PATH)

    # Drop non-feature columns
    feature_cols = [c for c in df.columns
                    if c not in DROP_COLS and df[c].dtype != object]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, feature_cols


def build_pipelines() -> dict:
    """Return a dict of model name → sklearn Pipeline."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08,
                max_depth=4, random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42))
        ]),
    }


def evaluate_models(pipelines: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Cross-validate all models and return summary table."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, pipe in pipelines.items():
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "CV AUC (mean)": cv_scores.mean().round(4),
            "CV AUC (std)":  cv_scores.std().round(4),
            "Test AUC":      round(test_auc, 4),
        })
        print(f"  {name:<22} CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
              f"  |  Test AUC: {test_auc:.4f}")

    return pd.DataFrame(results).sort_values("Test AUC", ascending=False)


def save_best_model(pipelines: dict, results_df: pd.DataFrame, X_train, y_train):
    best_name = results_df.iloc[0]["Model"]
    best_pipe  = pipelines[best_name]
    best_pipe.fit(X_train, y_train)   # refit on full train set

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = f"{MODEL_DIR}/best_model.pkl"
    joblib.dump({"pipeline": best_pipe, "model_name": best_name}, path)
    print(f"\nBest model: '{best_name}' saved → {path}")
    return best_name, best_pipe


def plot_roc_curves(pipelines: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2E86AB", "#E84855", "#52B788", "#F4A261"]

    for (name, pipe), color in zip(pipelines.items(), colors):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = f"{OUT_DIR}/06_roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_confusion_matrix(best_pipe, X_test, y_test, best_name: str):
    y_pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Intervention", "Needs Intervention"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {best_name}")
    plt.tight_layout()
    path = f"{OUT_DIR}/07_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_feature_importance(best_pipe, feature_cols: list, best_name: str):
    clf = best_pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        print("Feature importance not available for this model type.")
        return

    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top20 = importances.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    top20.plot(kind="barh", color="#2E86AB", ax=ax)
    ax.set_title(f"Top 20 Feature Importances – {best_name}")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = f"{OUT_DIR}/08_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  STEP 1 – Loading & splitting data")
    print("=" * 55)
    X_train, X_test, y_train, y_test, feature_cols = load_and_split()

    print("\n" + "=" * 55)
    print("  STEP 2 – Cross-validating models")
    print("=" * 55)
    pipelines = build_pipelines()
    results_df = evaluate_models(pipelines, X_train, X_test, y_train, y_test)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 55)
    print("  STEP 3 – Saving best model")
    print("=" * 55)
    best_name, best_pipe = save_best_model(pipelines, results_df, X_train, y_train)

    print("\n" + "=" * 55)
    print("  STEP 4 – Classification report (best model)")
    print("=" * 55)
    y_pred = best_pipe.predict(X_test)
    print(classification_report(y_test, y_pred,
          target_names=["No Intervention", "Needs Intervention"]))

    print("=" * 55)
    print("  STEP 5 – Generating evaluation charts")
    print("=" * 55)
    plot_roc_curves(pipelines, X_test, y_test)
    plot_confusion_matrix(best_pipe, X_test, y_test, best_name)
    plot_feature_importance(best_pipe, feature_cols, best_name)

    print("\nAll done!")


if __name__ == "__main__":
    main()
