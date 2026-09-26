from pathlib import Path
import sqlite3

from candidate_generation_experimental import (
    initialize_database,
    source_paths,
    count_blocks,
    build_membership_index,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "dataset" / "normalized"
DATABASE = ROOT / "output" / "candidate_blocking_experimental_train.sqlite"

SPLIT = "train"

paths = source_paths(DATA_ROOT, SPLIT)

print("Building training blocking database...")
print(f"Database: {DATABASE}")

connection = initialize_database(
    DATABASE,
    SPLIT,
    reset=True,
    confirm_destructive=True,
)

try:
    count_blocks(connection, paths)
    build_membership_index(connection, paths)

    print("\nTraining blocking database completed successfully.")
    print(f"Saved to: {DATABASE}")

finally:
    connection.close()