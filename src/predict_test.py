
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"

SOURCE1_FILE = TEST_DIR / "test_source1.tsv"
SOURCE2_FILE = TEST_DIR / "test_source2.tsv"
SOURCE3_FILE = TEST_DIR / "test_source3.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs_experimental.tsv"

MODEL_FILE = OUTPUT_DIR / "full_lightgbm_model.pkl"
DB_FILE = OUTPUT_DIR / "test_entity_lookup.sqlite"
OUTPUT_FILE = OUTPUT_DIR / "test_predictions_1k.tsv"

LIMIT = 1000
BATCH_SIZE = 10_000
THRESHOLD = 0.95

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


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def build_lookup_database():
    if DB_FILE.exists():
        raise FileExistsError(
            f"Lookup database already exists: {DB_FILE}\n"
            "Rename it before running again."
        )

    print("Building disk-backed entity lookup database...")

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY,
                business_name TEXT,
                business_address TEXT,
                country TEXT
            )
        """)

        for file_path in [SOURCE2_FILE, SOURCE3_FILE]:
            print(f"Indexing {file_path.name}...")

            for chunk in pd.read_csv(
                file_path,
                sep="\t",
                chunksize=BATCH_SIZE,
                dtype=str,
                keep_default_na=False,
            ):
                records = [
                    (
                        row.entity_id,
                        row.business_name,
                        row.business_address,
                        row.country,
                    )
                    for row in chunk.itertuples(index=False)
                ]

                conn.executemany(
                    """
                    INSERT INTO entities
                    (entity_id, business_name, business_address, country)
                    VALUES (?, ?, ?, ?)
                    """,
                    records,
                )

        conn.execute(
            "CREATE INDEX idx_entity_id ON entities(entity_id)"
        )

    print("Lookup database created.")


def make_features(source, candidate):
    name1 = clean_text(source["business_name"])
    name2 = clean_text(candidate["business_name"])

    address1 = clean_text(source["business_address"])
    address2 = clean_text(candidate["business_address"])

    country1 = clean_text(source["country"])
    country2 = clean_text(candidate["country"])

    return {
        "source1_entity_id": source["entity_id"],
        "candidate_entity_id": candidate["entity_id"],
        "name_exact": int(name1 == name2 and name1 != ""),
        "address_exact": int(
            address1 == address2 and address1 != ""
        ),
        "country_match": int(
            country1 == country2 and country1 != ""
        ),
        "name_ratio": fuzz.ratio(name1, name2),
        "name_token_set_ratio": fuzz.token_set_ratio(
            name1, name2
        ),
        "address_ratio": fuzz.ratio(address1, address2),
        "address_token_set_ratio": fuzz.token_set_ratio(
            address1, address2
        ),
        "name_length_diff": abs(len(name1) - len(name2)),
        "address_length_diff": abs(
            len(address1) - len(address2)
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        SOURCE1_FILE,
        SOURCE2_FILE,
        SOURCE3_FILE,
        CANDIDATE_FILE,
        MODEL_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}\n"
            "Rename it before running again."
        )

    build_lookup_database()

    print("Loading model...")
    saved = joblib.load(MODEL_FILE)
    model = saved["model"]

    print(f"Using threshold: {THRESHOLD}")

    source1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t",
        nrows=LIMIT,
        dtype=str,
        keep_default_na=False,
    )

    candidates = pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        nrows=LIMIT,
        dtype=str,
        keep_default_na=False,
    )

    if len(source1) != len(candidates):
        raise ValueError(
            "Source1 and candidate rows do not align."
        )

    if not source1["entity_id"].equals(
        candidates["source1_entity_id"]
    ):
        raise ValueError(
            "Source1 IDs do not match candidate file order."
        )

    all_features = []
    missing_candidate_ids = 0
    total_pairs = 0

    with sqlite3.connect(DB_FILE) as conn:
        for i, row in enumerate(
            source1.itertuples(index=False), start=1
        ):
            candidate_text = candidates.iloc[i - 1][
                "candidate_entity_ids"
            ]

            if not candidate_text:
                continue

            candidate_ids = candidate_text.split(",")

            for start in range(0, len(candidate_ids), 900):
                batch_ids = candidate_ids[start:start + 900]
                placeholders = ",".join("?" for _ in batch_ids)

                query = f"""
                    SELECT entity_id, business_name,
                           business_address, country
                    FROM entities
                    WHERE entity_id IN ({placeholders})
                """

                found = conn.execute(
                    query, batch_ids
                ).fetchall()

                entity_map = {
                    item[0]: {
                        "entity_id": item[0],
                        "business_name": item[1],
                        "business_address": item[2],
                        "country": item[3],
                    }
                    for item in found
                }

                for candidate_id in batch_ids:
                    candidate = entity_map.get(candidate_id)

                    if candidate is None:
                        missing_candidate_ids += 1
                        continue

                    source = {
                        "entity_id": row.entity_id,
                        "business_name": row.business_name,
                        "business_address": row.business_address,
                        "country": row.country,
                    }

                    all_features.append(
                        make_features(source, candidate)
                    )
                    total_pairs += 1

            if i % 100 == 0:
                print(
                    f"Processed {i:,}/{len(source1):,} Source1 records"
                )

    print(f"\nCandidate pairs found: {total_pairs:,}")
    print(f"Missing candidate IDs: {missing_candidate_ids:,}")

    if not all_features:
        raise ValueError("No candidate pairs could be matched to entities.")

    feature_df = pd.DataFrame(all_features)

    print("Scoring candidate pairs...")
    probabilities = model.predict_proba(
        feature_df[FEATURE_COLUMNS]
    )[:, 1]

    feature_df["match_probability"] = probabilities
    feature_df["predicted_match"] = (
        probabilities >= THRESHOLD
    ).astype(int)

    feature_df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    print("\nPrediction test completed.")
    print(f"Source1 records: {len(source1):,}")
    print(f"Candidate pairs scored: {len(feature_df):,}")
    print(
        "Predicted matches:",
        int(feature_df["predicted_match"].sum()),
    )
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()