import pandas as pd
from pathlib import Path

path = Path("dataset/processed/train_pairs.tsv")

df = pd.read_csv(
    path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=5,
)

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.to_string(index=False))