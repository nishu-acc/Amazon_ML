"""Build labeled Source 1-to-reference training pairs from candidate IDs."""

from collections import Counter
import argparse
import csv
from pathlib import Path
import sqlite3
import sys
import tempfile

import pandas as pd


DEFAULT_CHUNK_SIZE = 100_000
OUTPUT_COLUMNS = (
	"source1_entity_id",
	"candidate_entity_id",
	"source1_business_name",
	"source1_business_address",
	"source1_country",
	"candidate_business_name",
	"candidate_business_address",
	"candidate_country",
	"label",
)
REFERENCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def read_chunks(path: Path, chunk_size: int):
	"""Read a TSV in bounded-size pandas chunks."""
	return pd.read_csv(path, sep="\t", chunksize=chunk_size, dtype=object, keep_default_na=False)


def create_database(database_path: Path) -> sqlite3.Connection:
	"""Create temporary disk-backed tables for references and labels."""
	connection = sqlite3.connect(database_path)
	connection.executescript(
		"""
		CREATE TABLE source_records (
			entity_id TEXT PRIMARY KEY,
			source_type TEXT NOT NULL,
			business_name TEXT NOT NULL,
			business_address TEXT NOT NULL,
			country TEXT NOT NULL
		);
		CREATE TABLE ground_truth (
			source1_entity_id TEXT NOT NULL,
			matched_entity_id TEXT NOT NULL,
			PRIMARY KEY (source1_entity_id, matched_entity_id)
		);
		CREATE TABLE seen_pairs (
			source1_entity_id TEXT NOT NULL,
			candidate_entity_id TEXT NOT NULL,
			PRIMARY KEY (source1_entity_id, candidate_entity_id)
		);
		"""
	)
	return connection


def load_source_records(connection: sqlite3.Connection, path: Path, source_type: str, chunk_size: int) -> Counter:
	"""Stream one source file into SQLite and report duplicate/missing fields."""
	stats = Counter()
	insert_sql = "INSERT OR IGNORE INTO source_records VALUES (?, ?, ?, ?, ?)"
	for chunk in read_chunks(path, chunk_size):
		missing_columns = [column for column in REFERENCE_COLUMNS if column not in chunk.columns]
		if missing_columns:
			raise ValueError(f"{path} is missing columns: {', '.join(missing_columns)}")
		rows = []
		for row in chunk.itertuples(index=False):
			values = dict(zip(chunk.columns, row))
			entity_id = str(values["entity_id"])
			if not entity_id:
				stats["missing_reference_ids"] += 1
				continue
			rows.append((
				entity_id,
				source_type,
				str(values["business_name"]),
				str(values["business_address"]),
				str(values["country"]),
			))
		connection.executemany(insert_sql, rows)
		connection.commit()
		stats["reference_rows_read"] += len(chunk)
	stats["duplicate_reference_ids"] = stats["reference_rows_read"] - connection.execute(
		"SELECT COUNT(*) FROM source_records WHERE source_type=?", (source_type,)
	).fetchone()[0] - stats["missing_reference_ids"]
	return stats


def load_ground_truth(connection: sqlite3.Connection, path: Path, chunk_size: int) -> Counter:
	"""Stream comma-separated ground-truth matches into a deduplicated table."""
	stats = Counter()
	insert_sql = "INSERT OR IGNORE INTO ground_truth VALUES (?, ?)"
	for chunk in read_chunks(path, chunk_size):
		if "source1_entity_id" not in chunk.columns or "matched_entity_ids" not in chunk.columns:
			raise ValueError(f"{path} must contain source1_entity_id and matched_entity_ids")
		rows = []
		for row in chunk.itertuples(index=False):
			values = dict(zip(chunk.columns, row))
			source1_id = str(values["source1_entity_id"])
			matched_ids = [item.strip() for item in str(values["matched_entity_ids"]).split(",") if item.strip()]
			stats["ground_truth_rows"] += 1
			stats["true_match_pairs_read"] += len(matched_ids)
			rows.extend((source1_id, matched_id) for matched_id in matched_ids)
		connection.executemany(insert_sql, rows)
		connection.commit()
	stats["duplicate_ground_truth_pairs"] = stats["true_match_pairs_read"] - connection.execute(
		"SELECT COUNT(*) FROM ground_truth"
	).fetchone()[0]
	return stats


def fetch_record(
	connection: sqlite3.Connection,
	entity_id: str,
	allowed_source_types: tuple[str, ...] | None = None,
) -> tuple[str, str, str] | None:
	"""Fetch business fields for a source record."""
	query = "SELECT business_name, business_address, country FROM source_records WHERE entity_id=?"
	parameters: tuple[object, ...] = (entity_id,)
	if allowed_source_types:
		placeholders = ",".join("?" for _ in allowed_source_types)
		query += f" AND source_type IN ({placeholders})"
		parameters += allowed_source_types
	row = connection.execute(query, parameters).fetchone()
	return tuple(row) if row else None


def validate_ground_truth_coverage(
	connection: sqlite3.Connection,
	source1_id: str,
	candidate_ids: set[str],
	stats: Counter,
) -> None:
	"""Count true matches omitted from the candidate list for one Source 1 row."""
	true_ids = {
		row[0]
		for row in connection.execute(
			"SELECT matched_entity_id FROM ground_truth WHERE source1_entity_id=?",
			(source1_id,),
		)
	}
	missing_ids = true_ids - candidate_ids
	stats["ground_truth_matches_missing_from_candidates"] += len(missing_ids)
	for missing_id in sorted(missing_ids):
		if len(stats["missing_ground_truth_examples"]) < 10:
			stats["missing_ground_truth_examples"].append((source1_id, missing_id))


def validate_missing_candidate_rows(connection: sqlite3.Connection, stats: Counter) -> None:
	"""Report true matches for Source 1 records absent from the candidate file."""
	missing_rows = connection.execute(
		"""
		SELECT ground_truth.source1_entity_id, ground_truth.matched_entity_id
		FROM ground_truth
		LEFT JOIN candidate_source1_ids
			ON candidate_source1_ids.source1_entity_id = ground_truth.source1_entity_id
		WHERE candidate_source1_ids.source1_entity_id IS NULL
		"""
	).fetchall()
	stats["source1_records_missing_from_candidates"] = len({row[0] for row in missing_rows})
	stats["ground_truth_matches_missing_from_candidate_rows"] = len(missing_rows)
	stats["missing_candidate_row_examples"] = missing_rows[:10]


def build_pairs(
	connection: sqlite3.Connection,
	candidate_path: Path,
	output_path: Path,
	chunk_size: int,
) -> Counter:
	"""Stream candidate rows, enrich them, label them, and validate them."""
	stats = Counter()
	stats["missing_ground_truth_examples"] = []
	stats["missing_candidate_examples"] = []
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with candidate_path.open("r", encoding="utf-8-sig", newline="") as input_file, output_path.open(
		"w", encoding="utf-8", newline=""
	) as output_file:
		reader = csv.DictReader(input_file, delimiter="\t")
		if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
			raise ValueError("Candidate file must contain source1_entity_id and candidate_entity_ids columns")
		writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
		writer.writeheader()
		for row in reader:
			source1_id = row["source1_entity_id"]
			raw_candidates = row["candidate_entity_ids"] or ""
			candidate_ids = [candidate_id.strip() for candidate_id in raw_candidates.split(",") if candidate_id.strip()]
			unique_candidate_ids = set(candidate_ids)
			stats["candidate_rows_read"] += 1
			connection.execute("INSERT OR IGNORE INTO candidate_source1_ids VALUES (?)", (source1_id,))
			stats["duplicate_candidate_pairs"] += len(candidate_ids) - len(unique_candidate_ids)
			if not unique_candidate_ids:
				stats["empty_candidate_rows"] += 1
			source1_record = fetch_record(connection, source1_id, ("source1",))
			if source1_record is None:
				stats["missing_source1_ids"] += 1
				source1_record = ("", "", "")
			valid_candidate_ids = set()
			for candidate_id in sorted(unique_candidate_ids):
				candidate_record = fetch_record(connection, candidate_id, ("source2", "source3"))
				if candidate_record is None:
					stats["missing_candidate_ids"] += 1
					if len(stats["missing_candidate_examples"]) < 10:
						stats["missing_candidate_examples"].append((source1_id, candidate_id))
					continue
				valid_candidate_ids.add(candidate_id)
				inserted = connection.execute(
					"INSERT OR IGNORE INTO seen_pairs VALUES (?, ?)",
					(source1_id, candidate_id),
				).rowcount
				if inserted == 0:
					stats["duplicate_candidate_pairs"] += 1
					continue
				stats["valid_candidate_ids"] += 1
				label = int(
					connection.execute(
						"SELECT 1 FROM ground_truth WHERE source1_entity_id=? AND matched_entity_id=?",
						(source1_id, candidate_id),
					).fetchone()
					is not None
				)
				if label not in (0, 1):
					stats["invalid_labels"] += 1
				writer.writerow(dict(zip(OUTPUT_COLUMNS, (source1_id, candidate_id, *source1_record, *candidate_record, label))))
				stats["pair_rows_written"] += 1
			validate_ground_truth_coverage(connection, source1_id, valid_candidate_ids, stats)
			if stats["pair_rows_written"] % 10_000 == 0:
				connection.commit()
	connection.commit()
	validate_missing_candidate_rows(connection, stats)
	return stats


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source1", type=Path, default=Path("dataset/normalized/train/train_source1.tsv"))
	parser.add_argument("--source2", type=Path, default=Path("dataset/normalized/train/train_source2.tsv"))
	parser.add_argument("--source3", type=Path, default=Path("dataset/normalized/train/train_source3.tsv"))
	parser.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
	parser.add_argument("--candidate-pairs", type=Path, default=Path("output/train_candidate_pairs.tsv"))
	parser.add_argument("--output", type=Path, default=Path("dataset/processed/train_pairs.tsv"))
	parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
	parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	root = Path(__file__).resolve().parents[1]
	paths = {
		name: path if path.is_absolute() else root / path
		for name, path in {
			"source1": args.source1,
			"source2": args.source2,
			"source3": args.source3,
			"ground_truth": args.ground_truth,
			"candidate_pairs": args.candidate_pairs,
			"output": args.output,
		}.items()
	}
	missing = [path for path in paths.values() if not path.exists() and path != paths["output"]]
	if missing:
		print(f"ERROR: missing input files: {', '.join(map(str, missing))}", file=sys.stderr)
		return 1
	if args.chunk_size <= 0:
		print("ERROR: chunk size must be positive", file=sys.stderr)
		return 1
	if paths["output"].exists() and not args.overwrite:
		print(f"ERROR: output already exists: {paths['output']}; use --overwrite to replace it", file=sys.stderr)
		return 1
	try:
		with tempfile.TemporaryDirectory(prefix="ber_pair_dataset_") as temp_name:
			database_path = Path(temp_name) / "pair_dataset.sqlite"
			connection = create_database(database_path)
			try:
				connection.execute(
					"CREATE TABLE candidate_source1_ids (source1_entity_id TEXT PRIMARY KEY)"
				)
				for source_type in ("source1", "source2", "source3"):
					load_source_records(connection, paths[source_type], source_type, args.chunk_size)
				load_ground_truth(connection, paths["ground_truth"], args.chunk_size)
				temp_output = Path(temp_name) / "train_pairs.tsv"
				stats = build_pairs(connection, paths["candidate_pairs"], temp_output, args.chunk_size)
			finally:
				connection.close()
			paths["output"].parent.mkdir(parents=True, exist_ok=True)
			temp_output.replace(paths["output"])
		print(f"Wrote labeled pairs to {paths['output']}")
		print(f"Validation summary: {dict(stats)}")
		return 0
	except (OSError, pd.errors.ParserError, sqlite3.Error, ValueError) as error:
		print(f"ERROR: pair dataset creation failed: {error}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
