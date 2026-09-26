import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "output/full_lightgbm_model.pkl"
DATA_PATH = "dataset/processed/validation_features_full.tsv"

# Load model
saved = joblib.load(MODEL_PATH)

model = saved["model"]
feature_columns = saved["feature_columns"]

# Use strings as dictionary keys to avoid floating-point key errors
thresholds = [round(float(t), 3) for t in np.arange(0.80, 0.991, 0.005)]

results = {
    str(t): {
        "tp": 0,
        "fp": 0,
        "fn": 0
    }
    for t in thresholds
}

print("Loading validation data and testing thresholds...\n")

for chunk in pd.read_csv(
    DATA_PATH,
    sep="\t",
    chunksize=100000
):
    X = chunk[feature_columns]
    y = chunk["label"].astype(int).to_numpy()

    probabilities = model.predict_proba(X)[:, 1]

    for t in thresholds:
        predictions = probabilities >= t

        tp = np.sum(predictions & (y == 1))
        fp = np.sum(predictions & (y == 0))
        fn = np.sum(~predictions & (y == 1))

        results[str(t)]["tp"] += int(tp)
        results[str(t)]["fp"] += int(fp)
        results[str(t)]["fn"] += int(fn)

best_threshold = None
best_f05 = -1
best_precision = 0
best_recall = 0

print("--- Threshold Sweep ---")

for t in thresholds:
    data = results[str(t)]

    tp = data["tp"]
    fp = data["fp"]
    fn = data["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F0.5
    beta_squared = 0.25

    f05 = (
        (1 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
        if (beta_squared * precision + recall) > 0
        else 0
    )

    print(
        f"Threshold={t:.3f} | "
        f"Precision={precision:.4%} | "
        f"Recall={recall:.4%} | "
        f"F0.5={f05:.5f}"
    )

    if f05 > best_f05:
        best_f05 = f05
        best_threshold = t
        best_precision = precision
        best_recall = recall

print("\n--- BEST THRESHOLD ---")
print(f"Threshold : {best_threshold:.3f}")
print(f"Precision : {best_precision:.4%}")
print(f"Recall    : {best_recall:.4%}")
print(f"F0.5      : {best_f05:.5f}")