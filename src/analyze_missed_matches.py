import sqlite3
from collections import Counter
from pathlib import Path
import re
import pandas as pd

GROUND_TRUTH = Path("dataset/train/train_ground_truth.tsv")
VALIDATION_FEATURES = "dataset/processed/validation_split.tsv"
SOURCE1 = Path("dataset/normalized/train/train_source1.tsv")
SOURCE2 = Path("dataset/normalized/train/train_source2.tsv")
SOURCE3 = Path("dataset/normalized/train/train_source3.tsv")
DB_PATH = Path("output/candidate_blocking_experimental.sqlite")

CHUNK_SIZE = 200_000
MIN_TOKEN_LENGTH = 3
MAX_EXACT_NAME_BLOCK_SIZE = 200
MAX_TOKEN_BLOCK_SIZE = 300
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def tokens(value):
    return {t for t in TOKEN_PATTERN.findall(clean(value)) if len(t) >= MIN_TOKEN_LENGTH}


print("Step 1: Loading survived positive validation pairs...")
survived = set()
validation_ids = set()
for chunk in pd.read_csv(VALIDATION_FEATURES, sep="\t", dtype=object, keep_default_na=False,
                         usecols=["source1_entity_id", "candidate_entity_id", "label"],
                         chunksize=CHUNK_SIZE):
    validation_ids.update(chunk["source1_entity_id"].astype(str))
    pos = chunk[chunk["label"].astype(str) == "1"]
    survived.update(zip(pos["source1_entity_id"].astype(str), pos["candidate_entity_id"].astype(str)))

print(f"Validation Source1 IDs: {len(validation_ids):,}")
print(f"Survived positive pairs: {len(survived):,}")

print("\nStep 2: Loading ground-truth pairs...")
ground_truth_pairs = []
for chunk in pd.read_csv(GROUND_TRUTH, sep="\t", dtype=object, keep_default_na=False,
                         chunksize=CHUNK_SIZE):
    chunk["source1_entity_id"] = chunk["source1_entity_id"].astype(str)
    chunk = chunk[chunk["source1_entity_id"].isin(validation_ids)]
    for row in chunk.itertuples(index=False):
        sid = str(row.source1_entity_id)
        matched = str(row.matched_entity_ids)
        if matched:
            for cid in matched.split(","):
                cid = cid.strip()
                if cid:
                    ground_truth_pairs.append((sid, cid))

print(f"Ground-truth pairs: {len(ground_truth_pairs):,}")
missed = [pair for pair in ground_truth_pairs if pair not in survived]
print(f"Missed ground-truth pairs: {len(missed):,}")

needed_s1 = {sid for sid, _ in missed}
needed_candidates = {cid for _, cid in missed}


def load_records(path, needed_ids):
    records = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False, chunksize=CHUNK_SIZE):
        chunk["entity_id"] = chunk["entity_id"].astype(str)
        part = chunk[chunk["entity_id"].isin(needed_ids)]
        for row in part.itertuples(index=False):
            records[str(row.entity_id)] = (
                clean(row.country),
                clean(row.normalized_business_name),
                clean(row.normalized_business_address),
            )
    return records


print("\nStep 3: Loading normalized Source1 records...")
s1_records = load_records(SOURCE1, needed_s1)
print(f"Source1 records loaded: {len(s1_records):,}")

print("\nStep 4: Loading normalized Source2/3 records...")
candidate_records = {}
candidate_records.update(load_records(SOURCE2, needed_candidates))
candidate_records.update(load_records(SOURCE3, needed_candidates))
print(f"Matched candidate records loaded: {len(candidate_records):,}")

print("\nStep 5: Collecting block keys used by missed matches...")
needed_blocks = set()
pair_info = []

for sid, cid in missed:
    s1 = s1_records.get(sid)
    target = candidate_records.get(cid)
    if not s1 or not target:
        pair_info.append((sid, cid, None, None, set(), set()))
        continue

    country = s1[0]
    s1_name, s1_addr = s1[1], s1[2]
    t_name, t_addr = target[1], target[2]
    shared_name = tokens(s1_name) & tokens(t_name)
    shared_addr = tokens(s1_addr) & tokens(t_addr)
    exact = bool(s1_name and t_name and s1_name == t_name)

    if exact:
        needed_blocks.add(("exact_name", country, s1_name))
    for token in shared_name:
        needed_blocks.add(("name_token", country, token))
    for token in shared_addr:
        needed_blocks.add(("address_token", country, token))

    pair_info.append((sid, cid, country, exact, shared_name, shared_addr))

print(f"Unique block keys to inspect: {len(needed_blocks):,}")

print("\nStep 6: Reading block counts from SQLite...")
conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TEMP TABLE needed_blocks (
        block_type TEXT NOT NULL,
        country TEXT NOT NULL,
        block_key TEXT NOT NULL,
        PRIMARY KEY(block_type, country, block_key)
    ) WITHOUT ROWID
""")
conn.executemany("INSERT OR IGNORE INTO needed_blocks VALUES (?, ?, ?)", needed_blocks)
counts = {
    (row[0], row[1], row[2]): int(row[3])
    for row in conn.execute("""
        SELECT n.block_type, n.country, n.block_key, c.block_count
        FROM needed_blocks n
        LEFT JOIN block_counts c
          ON c.block_type = n.block_type
         AND c.country = n.country
         AND c.block_key = n.block_key
    """) if row[3] is not None
}
conn.close()
print(f"Block counts found: {len(counts):,}")

stats = Counter()
for sid, cid, country, exact, shared_name, shared_addr in pair_info:
    if country is None:
        stats["missing_source_record"] += 1
        continue

    exact_oversized = False
    if exact:
        s1_name = s1_records[sid][1]
        exact_count = counts.get(("exact_name", country, s1_name))
        if exact_count is not None and exact_count > MAX_EXACT_NAME_BLOCK_SIZE:
            exact_oversized = True

    oversized_name = [t for t in shared_name if counts.get(("name_token", country, t), 0) > MAX_TOKEN_BLOCK_SIZE]
    oversized_addr = [t for t in shared_addr if counts.get(("address_token", country, t), 0) > MAX_TOKEN_BLOCK_SIZE]

    stats["missed"] += 1
    if exact_oversized:
        stats["exact_name_oversized"] += 1
    if oversized_name:
        stats["has_oversized_name_token"] += 1
    if oversized_addr:
        stats["has_oversized_address_token"] += 1
    if len(oversized_name) >= 2:
        stats["two_or_more_oversized_name_tokens"] += 1
    if len(oversized_addr) >= 2:
        stats["two_or_more_oversized_address_tokens"] += 1
    if oversized_name and oversized_addr:
        stats["oversized_name_plus_address"] += 1
    if exact_oversized or len(oversized_name) >= 2 or len(oversized_addr) >= 2 or (oversized_name and oversized_addr):
        stats["potentially_recoverable_by_new_blocks"] += 1
    if not exact_oversized and not oversized_name and not oversized_addr:
        stats["no_oversized_shared_block"] += 1


def pct(n, d):
    return f"{n / d:.2%}" if d else "0.00%"

print("\n--- MISSED MATCH ANALYSIS ---")
missed_n = stats["missed"]
print(f"Missed matches: {missed_n:,}")
print(f"Exact-name oversized: {stats['exact_name_oversized']:,} ({pct(stats['exact_name_oversized'], missed_n)})")
print(f"Has oversized name token: {stats['has_oversized_name_token']:,} ({pct(stats['has_oversized_name_token'], missed_n)})")
print(f"Has oversized address token: {stats['has_oversized_address_token']:,} ({pct(stats['has_oversized_address_token'], missed_n)})")
print(f"2+ oversized name tokens: {stats['two_or_more_oversized_name_tokens']:,} ({pct(stats['two_or_more_oversized_name_tokens'], missed_n)})")
print(f"2+ oversized address tokens: {stats['two_or_more_oversized_address_tokens']:,} ({pct(stats['two_or_more_oversized_address_tokens'], missed_n)})")
print(f"Oversized name + address token: {stats['oversized_name_plus_address']:,} ({pct(stats['oversized_name_plus_address'], missed_n)})")
print(f"Potentially recoverable by new blocks: {stats['potentially_recoverable_by_new_blocks']:,} ({pct(stats['potentially_recoverable_by_new_blocks'], missed_n)})")
print(f"No oversized shared block: {stats['no_oversized_shared_block']:,} ({pct(stats['no_oversized_shared_block'], missed_n)})")
