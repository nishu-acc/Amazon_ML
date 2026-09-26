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
OUTPUT_FILE = OUTPUT_DIR / "matching_results.tsv"

READ_CHUNK_SIZE = 1000
MAX_SOURCE1_ROWS = None # Safe sample first; set to None for the full run later.
DB_INSERT_CHUNK_SIZE = 10000
SQL_BATCH_SIZE = 900
THRESHOLD = 0.95

FEATURE_COLUMNS = [
    "name_exact", "address_exact", "country_match", "name_ratio",
    "name_token_set_ratio", "address_ratio", "address_token_set_ratio",
    "name_length_diff", "address_length_diff",
]


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def build_lookup_database():
    if DB_FILE.exists():
        print(f"Using existing lookup database: {DB_FILE}")
        return

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
                file_path, sep="\t", chunksize=DB_INSERT_CHUNK_SIZE,
                dtype=str, keep_default_na=False
            ):
                records = [
                    (r.entity_id, r.business_name, r.business_address, r.country)
                    for r in chunk.itertuples(index=False)
                ]
                conn.executemany("""
                    INSERT INTO entities
                    (entity_id, business_name, business_address, country)
                    VALUES (?, ?, ?, ?)
                """, records)
    print("Lookup database created.")


def make_features(source, candidate):
    name1 = clean_text(source["business_name"])
    name2 = clean_text(candidate["business_name"])
    address1 = clean_text(source["business_address"])
    address2 = clean_text(candidate["business_address"])
    country1 = clean_text(source["country"])
    country2 = clean_text(candidate["country"])
    return {
        "name_exact": int(name1 == name2 and name1 != ""),
        "address_exact": int(address1 == address2 and address1 != ""),
        "country_match": int(country1 == country2 and country1 != ""),
        "name_ratio": fuzz.ratio(name1, name2),
        "name_token_set_ratio": fuzz.token_set_ratio(name1, name2),
        "address_ratio": fuzz.ratio(address1, address2),
        "address_token_set_ratio": fuzz.token_set_ratio(address1, address2),
        "name_length_diff": abs(len(name1) - len(name2)),
        "address_length_diff": abs(len(address1) - len(address2)),
    }


def fetch_candidates(conn, candidate_ids):
    entity_map = {}
    for start in range(0, len(candidate_ids), SQL_BATCH_SIZE):
        ids = candidate_ids[start:start + SQL_BATCH_SIZE]
        if not ids:
            continue
        placeholders = ",".join("?" for _ in ids)
        query = f"""
            SELECT entity_id, business_name, business_address, country
            FROM entities WHERE entity_id IN ({placeholders})
        """
        for entity_id, name, address, country in conn.execute(query, ids):
            entity_map[entity_id] = {
                "entity_id": entity_id,
                "business_name": name or "",
                "business_address": address or "",
                "country": country or "",
            }
    return entity_map


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [SOURCE1_FILE, SOURCE2_FILE, SOURCE3_FILE, CANDIDATE_FILE, MODEL_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT_FILE}")

    build_lookup_database()
    print("Loading model...")
    model = joblib.load(MODEL_FILE)["model"]
    print(f"Using threshold: {THRESHOLD}")

    src_reader = pd.read_csv(
        SOURCE1_FILE, sep="\t", chunksize=READ_CHUNK_SIZE, nrows=MAX_SOURCE1_ROWS,
        dtype=str, keep_default_na=False
    )
    cand_reader = pd.read_csv(
        CANDIDATE_FILE, sep="\t", chunksize=READ_CHUNK_SIZE, nrows=MAX_SOURCE1_ROWS,
        dtype=str, keep_default_na=False
    )

    total_sources = total_candidate_ids = total_missing = total_matches = 0

    with sqlite3.connect(DB_FILE) as conn, open(
        OUTPUT_FILE, "w", encoding="utf-8", newline=""
    ) as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        sentinel = object()
        for chunk_no, pair in enumerate(zip(src_reader, cand_reader), start=1):
            src_chunk, cand_chunk = pair
            if len(src_chunk) != len(cand_chunk):
                raise ValueError(f"Chunk {chunk_no}: row counts do not align.")
            if not src_chunk["entity_id"].equals(cand_chunk["source1_entity_id"]):
                raise ValueError(f"Chunk {chunk_no}: Source1 IDs do not align.")

            for src, cand_row in zip(
                src_chunk.itertuples(index=False), cand_chunk.itertuples(index=False)
            ):
                source = {
                    "entity_id": src.entity_id,
                    "business_name": src.business_name,
                    "business_address": src.business_address,
                    "country": src.country,
                }
                text_ids = cand_row.candidate_entity_ids
                ids = text_ids.split(",") if text_ids else []
                total_candidate_ids += len(ids)
                entity_map = fetch_candidates(conn, ids)

                features, ordered_ids = [], []
                for candidate_id in ids:
                    candidate = entity_map.get(candidate_id)
                    if candidate is None:
                        total_missing += 1
                        continue
                    features.append(make_features(source, candidate))
                    ordered_ids.append(candidate_id)

                matched_ids = []
                if features:
                    feature_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
                    probabilities = model.predict_proba(feature_df)[:, 1]
                    matched_ids = [
                        cid for cid, prob in zip(ordered_ids, probabilities)
                        if prob >= THRESHOLD
                    ]

                total_matches += len(matched_ids)
                out.write(f"{src.entity_id}\t{','.join(matched_ids)}\n")
                total_sources += 1

            out.flush()
            print(
                f"Processed {total_sources:,} Source1 records; "
                f"predicted matches: {total_matches:,}"
            )

    # Check that both files have the same number of chunks/rows.
    src_extra = next(src_reader, None)
    cand_extra = next(cand_reader, None)
    if src_extra is not None or cand_extra is not None:
        raise ValueError("Source1 and candidate files have different numbers of rows.")

    print("\nFull prediction completed.")
    print(f"Source1 records processed: {total_sources:,}")
    print(f"Candidate IDs read: {total_candidate_ids:,}")
    print(f"Missing candidate IDs: {total_missing:,}")
    print(f"Predicted matches: {total_matches:,}")
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
