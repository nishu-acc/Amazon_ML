import pandas as pd
from pathlib import Path

files = {
    "Original": Path("output/candidate_pairs_backup.tsv"),
    "Experimental": Path("output/candidate_pairs_experimental.tsv"),
}

for name, path in files.items():
    if not path.exists():
        print(f"\n{name}: File not found: {path}")
        continue

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    counts = df["candidate_entity_ids"].apply(
        lambda x: len(set(x.split(",")) - {""})
    )

    print(f"\n--- {name} ---")
    print("Source1 rows:", len(df))
    print("Rows with candidates:", (counts > 0).sum())
    print("Rows without candidates:", (counts == 0).sum())
    print("Total candidate pairs:", counts.sum())

    print(
        "Duplicate Source1 IDs:",
        df["source1_entity_id"].duplicated().sum()
    )