
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_FILE = DATA_DIR / "train_features_full.tsv"
VALIDATION_FILE = DATA_DIR / "validation_features_full.tsv"
MODEL_FILE = OUTPUT_DIR / "full_lightgbm_model.pkl"

FEATURE_COLUMNS = [
    "name_exact",
    "address_exact",
    "country_match",
    "name_ratio",
    "name_token_set_ratio",
    "address_ratio",
    "address_token_set_ratio",
    "name_length_diff",
    "address_length_diff",
]

USE_COLUMNS = FEATURE_COLUMNS + ["label"]

DTYPES = {
    "name_exact": "int8",
    "address_exact": "int8",
    "country_match": "int8",
    "name_ratio": "float32",
    "name_token_set_ratio": "float32",
    "address_ratio": "float32",
    "address_token_set_ratio": "float32",
    "name_length_diff": "float32",
    "address_length_diff": "float32",
    "label": "int8",
}


def load_features(file_path):
    print(f"\nLoading: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        usecols=USE_COLUMNS,
        dtype=DTYPES,
    )

    print(f"Loaded rows: {len(df):,}")
    print(f"Positive labels: {(df['label'] == 1).sum():,}")
    print(f"Negative labels: {(df['label'] == 0).sum():,}")

    return df


def calculate_f05(precision, recall):
    denominator = 0.25 * precision + recall

    if denominator == 0:
        return 0.0

    return 1.25 * precision * recall / denominator


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(VALIDATION_FILE)

    if MODEL_FILE.exists():
        raise FileExistsError(
            f"Model already exists: {MODEL_FILE}\n"
            "Rename it before running this script again."
        )

    # Load the full training dataset.
    train_df = load_features(TRAIN_FILE)

    # Load the full validation dataset.
    validation_df = load_features(VALIDATION_FILE)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df["label"]

    # Release the original DataFrames where possible.
    del train_df
    del validation_df

    print("\nStarting full LightGBM training...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    print("\nTraining completed.")

    print("\nGenerating validation probabilities...")
    probabilities = model.predict_proba(X_validation)[:, 1]

    print("\nValidation results:")
    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'F0.5':>10}"
    )

    thresholds = [0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95]

    best_threshold = None
    best_f05 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int8)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f05 = calculate_f05(precision, recall)

        print(
            f"{threshold:>10.2f} "
            f"{precision:>12.4f} "
            f"{recall:>10.4f} "
            f"{f05:>10.4f}"
        )

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    print(f"\nBest tested threshold: {best_threshold:.2f}")
    print(f"Best validation F0.5: {best_f05:.4f}")

    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_FILE,
    )

    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()