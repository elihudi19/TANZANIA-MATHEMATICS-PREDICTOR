"""
train_model.py
--------------
Trains two models (Logistic Regression and Random Forest) to predict
Form Four NECTA Mathematics outcomes for schools in the Mwanza Region.

Outputs (saved into ./model_artifacts/):
    - logistic_model.pkl     -> full sklearn Pipeline (preprocessing + Logistic Regression)
    - random_forest_model.pkl-> full sklearn Pipeline (preprocessing + Random Forest)
    - feature_config.pkl     -> encoding maps + feature order (used by app.py)
    - metrics.json           -> accuracy, precision, recall, F1, confusion matrix for both models
    - confusion_matrix_logreg.png
    - confusion_matrix_rf.png

Run:
    python train_model.py
"""

import json
import os
import sys
import traceback

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed, safe for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_PATH = "Mwanza_Dataset.csv"
OUTPUT_DIR = "model_artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["Teacher-to-student ratio", "Attendance", "Mock_Score"]
CATEGORICAL_ENCODED_FEATURES = ["School_Type_Encoded", "Has_Book_Encoded"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_ENCODED_FEATURES
TARGET = "NECTA result"

# Ordinal mapping for the mock exam grade (A is best, F is worst)
MOCK_GRADE_MAP = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

# School Type: Private = 0 (baseline), Government = 1
# (Chosen deliberately so coefficients are interpretable and to avoid the
#  Simpson's Paradox reversal seen when Government was used as baseline.)
SCHOOL_TYPE_MAP = {"Private": 0, "Government": 1}

# Mathematics Books ownership: Not Own Book = 0, Own a Book = 1
BOOK_MAP = {"Not Own Book": 0, "Own a Book": 1}


def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and engineer the encoded feature columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find dataset at '{path}'.")

    df = pd.read_csv(path)

    required_cols = [
        "School Type",
        "Teacher-to-student ratio",
        "Attendance",
        "Mathematics Books",
        "Mock result",
        "NECTA result",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Drop rows with any missing values in required columns rather than crash
    before = len(df)
    df = df.dropna(subset=required_cols).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[warning] Dropped {dropped} rows with missing values.")

    # Validate categorical values before mapping, to avoid silent NaNs
    unknown_school = set(df["School Type"].unique()) - set(SCHOOL_TYPE_MAP)
    if unknown_school:
        raise ValueError(f"Unexpected 'School Type' values found: {unknown_school}")

    unknown_book = set(df["Mathematics Books"].unique()) - set(BOOK_MAP)
    if unknown_book:
        raise ValueError(f"Unexpected 'Mathematics Books' values found: {unknown_book}")

    unknown_mock = set(df["Mock result"].unique()) - set(MOCK_GRADE_MAP)
    if unknown_mock:
        raise ValueError(f"Unexpected 'Mock result' values found: {unknown_mock}")

    df["School_Type_Encoded"] = df["School Type"].map(SCHOOL_TYPE_MAP)
    df["Has_Book_Encoded"] = df["Mathematics Books"].map(BOOK_MAP)
    df["Mock_Score"] = df["Mock result"].map(MOCK_GRADE_MAP)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Scale the numeric/ordinal columns (helps Logistic Regression coefficients
    stay well-behaved) while passing the already-binary encoded columns through.
    Random Forest does not need scaling but is unaffected by it, so we reuse
    the same preprocessor for both models for consistency.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("passthrough", FunctionTransformer(validate=False), CATEGORICAL_ENCODED_FEATURES),
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    """Compute accuracy, precision, recall, F1 and confusion matrix."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


def save_confusion_matrix_plot(cm, title, filename):
    """Save a labeled confusion matrix heatmap using matplotlib only (no seaborn dependency)."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail (0)", "Pass (1)"])
    ax.set_yticklabels(["Fail (0)", "Pass (1)"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[saved] {filename}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and preparing data...")
    df = load_and_prepare_data(DATA_PATH)
    print(f"Loaded {len(df)} rows after cleaning.")

    X = df[FEATURE_ORDER]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ----------------------------------------------------------------- #
    # Logistic Regression
    # ----------------------------------------------------------------- #
    logreg_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    logreg_pipeline.fit(X_train, y_train)
    logreg_metrics = evaluate_model("Logistic Regression", logreg_pipeline, X_test, y_test)

    # ----------------------------------------------------------------- #
    # Random Forest
    # ----------------------------------------------------------------- #
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest", rf_pipeline, X_test, y_test)

    # ----------------------------------------------------------------- #
    # Save confusion matrix plots
    # ----------------------------------------------------------------- #
    save_confusion_matrix_plot(
        logreg_metrics["confusion_matrix"],
        "Logistic Regression - Confusion Matrix",
        os.path.join(OUTPUT_DIR, "confusion_matrix_logreg.png"),
    )
    save_confusion_matrix_plot(
        rf_metrics["confusion_matrix"],
        "Random Forest - Confusion Matrix",
        os.path.join(OUTPUT_DIR, "confusion_matrix_rf.png"),
    )

    # ----------------------------------------------------------------- #
    # Extract logistic regression coefficients for the suggestion engine
    # ----------------------------------------------------------------- #
    logreg_clf = logreg_pipeline.named_steps["classifier"]
    coefficients = dict(zip(FEATURE_ORDER, logreg_clf.coef_[0].tolist()))

    rf_clf = rf_pipeline.named_steps["classifier"]
    rf_importances = dict(zip(FEATURE_ORDER, rf_clf.feature_importances_.tolist()))

    # ----------------------------------------------------------------- #
    # Save models + config
    # ----------------------------------------------------------------- #
    joblib.dump(logreg_pipeline, os.path.join(OUTPUT_DIR, "logistic_model.pkl"))
    joblib.dump(rf_pipeline, os.path.join(OUTPUT_DIR, "random_forest_model.pkl"))

    feature_config = {
        "feature_order": FEATURE_ORDER,
        "numeric_features": NUMERIC_FEATURES,
        "school_type_map": SCHOOL_TYPE_MAP,
        "book_map": BOOK_MAP,
        "mock_grade_map": MOCK_GRADE_MAP,
        "logreg_coefficients": coefficients,
        "rf_feature_importances": rf_importances,
    }
    joblib.dump(feature_config, os.path.join(OUTPUT_DIR, "feature_config.pkl"))

    all_metrics = {
        "logistic_regression": logreg_metrics,
        "random_forest": rf_metrics,
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nAll artifacts saved to:", OUTPUT_DIR)
    print("Done.")


def train_and_save():
    """
    Callable entry point (used by app.py as an auto-train fallback when the
    model_artifacts/ folder is missing, e.g. on a fresh Streamlit Cloud
    deploy where the folder wasn't committed to git). Reuses the exact same
    logic as running `python train_model.py` from the command line.
    """
    main()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR] Training failed:", str(exc))
        traceback.print_exc()
        sys.exit(1)
