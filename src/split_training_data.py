import pandas as pd
from pathlib import Path

INPUT_FILE = Path("dataset/processed/train_pairs.tsv")
OUTPUT_DIR = Path("dataset/processed")

TRAIN_FILE = OUTPUT_DIR / "train_split.tsv"
VAL_FILE = OUTPUT_DIR / "validation_split.tsv"

CHUNK_SIZE = 500_000
VALIDATION_PERCENT = 10

if TRAIN_FILE.exists() or VAL_FILE.exists():
    raise FileExistsError(
        "Split files already exist. Rename or remove them before rerunning."
    )

train_rows = 0
val_rows = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE,
):
    # Assign each unique source entity consistently to one split.
    entity_hash = pd.util.hash_pandas_object(
        chunk["source1_entity_id"],
        index=False,
    )

    is_validation = (
        entity_hash % 100 < VALIDATION_PERCENT
    )

    train_chunk = chunk.loc[~is_validation]
    val_chunk = chunk.loc[is_validation]

    train_chunk.to_csv(
        TRAIN_FILE,
        sep="\t",
        index=False,
        mode="a",
        header=not TRAIN_FILE.exists(),
    )

    val_chunk.to_csv(
        VAL_FILE,
        sep="\t",
        index=False,
        mode="a",
        header=not VAL_FILE.exists(),
    )

    train_rows += len(train_chunk)
    val_rows += len(val_chunk)

    print(
        f"Processed {train_rows + val_rows:,} rows | "
        f"Train: {train_rows:,} | Validation: {val_rows:,}"
    )

print("\n--- Split completed ---")
print("Training rows:", f"{train_rows:,}")
print("Validation rows:", f"{val_rows:,}")
print("Training file:", TRAIN_FILE)
print("Validation file:", VAL_FILE)