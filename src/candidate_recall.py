import pandas as pd

GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"
VALIDATION_FEATURES = "dataset/processed/validation_features_full.tsv"

print("Step 1: Loading validation Source1 IDs...")

validation_ids = set()

for chunk in pd.read_csv(
    VALIDATION_FEATURES,
    sep="\t",
    usecols=["source1_entity_id"],
    chunksize=200000
):
    validation_ids.update(chunk["source1_entity_id"].dropna().astype(str))

print(f"Validation Source1 entities: {len(validation_ids):,}")

print("\nStep 2: Reading ground truth...")

total_gt_pairs = 0
validation_gt_entities = 0

for chunk in pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    chunksize=100000
):
    chunk["source1_entity_id"] = chunk["source1_entity_id"].astype(str)

    matched = chunk[
        chunk["source1_entity_id"].isin(validation_ids)
    ]

    validation_gt_entities += len(matched)

    for ids in matched["matched_entity_ids"].dropna():
        ids = str(ids).strip()

        if ids:
            total_gt_pairs += len([
                x for x in ids.split(",")
                if x.strip()
            ])

print(f"Validation GT entities: {validation_gt_entities:,}")
print(f"Total ground-truth matches: {total_gt_pairs:,}")

candidate_positive_pairs = 382011

candidate_recall = (
    candidate_positive_pairs / total_gt_pairs
    if total_gt_pairs > 0
    else 0
)

print("\n--- CANDIDATE RECALL ---")
print(f"Candidate true-positive pairs : {candidate_positive_pairs:,}")
print(f"Ground-truth positive pairs   : {total_gt_pairs:,}")
print(f"Candidate Recall              : {candidate_recall:.4%}")