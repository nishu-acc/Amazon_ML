
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    classification_report,
)

DATA_FILE = Path("dataset/processed/train_features_sample.tsv")
MODEL_FILE = Path("output/baseline_model.pkl")

# Load the sample feature dataset
df = pd.read_csv(DATA_FILE, sep="\t")
print("Rows loaded:", len(df))

# Keep source IDs for grouping, but don't use them as model features
groups = df["source1_entity_id"]

X = df.drop(columns=["label", "source1_entity_id"])
y = df["label"]

# Split by source entity to prevent leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

shared_entities = (
    set(groups.iloc[train_idx])
    & set(groups.iloc[val_idx])
)

print("Shared source entities:", len(shared_entities))

# Train the model
model = LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

print("\nTraining LightGBM...")
model.fit(X_train, y_train)

# Evaluate different thresholds
probabilities = model.predict_proba(X_val)[:, 1]

print("\n--- Threshold Comparison ---")

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f05 = fbeta_score(
        y_val, predictions, beta=0.5, zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F0.5: {f05:.4f}"
    )

# Show report at threshold 0.90
predictions = (probabilities >= 0.90).astype(int)

print("\n--- Classification Report (threshold = 0.90) ---")
print(classification_report(y_val, predictions, zero_division=0))

# Save model
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved to: {MODEL_FILE}")