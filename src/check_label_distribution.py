import pandas as pd
from pathlib import Path

path = Path("dataset/processed/train_pairs.tsv")

label_counts = {"0": 0, "1": 0}
total_rows = 0

for chunk in pd.read_csv(
    path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=["label"],
    chunksize=500_000,
):
    counts = chunk["label"].value_counts()

    for label in label_counts:
        label_counts[label] += int(counts.get(label, 0))

    total_rows += len(chunk)
    print(f"Processed {total_rows:,} rows")

print("\n--- Label Distribution ---")
print("Total pairs:", f"{total_rows:,}")
print("Matches (1):", f"{label_counts['1']:,}")
print("Non-matches (0):", f"{label_counts['0']:,}")

if total_rows:
    print("Match percentage:", round(label_counts["1"] / total_rows * 100, 2), "%")
