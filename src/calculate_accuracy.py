import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, confusion_matrix

MODEL_PATH = "output/full_lightgbm_model.pkl"
DATA_PATH = "dataset/processed/validation_features_full.tsv"
THRESHOLD = 0.95

saved = joblib.load(MODEL_PATH)

model = saved["model"]
feature_columns = saved["feature_columns"]

correct = 0
total = 0
tp = fp = tn = fn = 0

for chunk in pd.read_csv(DATA_PATH, sep="\t", chunksize=100000):
    X = chunk[feature_columns]
    y = chunk["label"].astype(int)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn_c, fp_c, fn_c, tp_c = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    tn += tn_c
    fp += fp_c
    fn += fn_c
    tp += tp_c
    total += len(y)

accuracy = (tp + tn) / total
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f05 = (
    1.25 * precision * recall / (0.25 * precision + recall)
    if precision + recall else 0
)

print("\n--- Validation Metrics ---")
print(f"Total pairs: {total:,}")
print(f"Accuracy: {accuracy:.4%}")
print(f"Precision: {precision:.4%}")
print(f"Recall: {recall:.4%}")
print(f"F0.5 Score: {f05:.4f}")
print(f"True Positives: {tp:,}")
print(f"True Negatives: {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")