"""Generate country-aware blocked candidate pairs with resumable checkpoints."""

from collections import Counter
import argparse
import csv
from pathlib import Path
import re
import sqlite3
import sys
from typing import Iterator

import pandas as pd


CHUNK_SIZE = 100_000
MIN_TOKEN_LENGTH = 3
MAX_EXACT_NAME_BLOCK_SIZE = 100
MAX_NAME_TOKEN_BLOCK_SIZE = 50
MAX_ADDRESS_TOKEN_BLOCK_SIZE = 100
MAX_CANDIDATES_PER_ENTITY = 5_000
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
BLOCKING_DATABASE = Path("output/candidate_blocking.sqlite")


def source_paths(data_root: Path, split: str) -> dict[str, Path]:
	"""Return normalized source paths for one split."""
	return {name: data_root / split / f"{split}_{name}.tsv" for name in ("source1", "source2", "source3")}


def clean_value(value: object) -> str:
	"""Convert a nullable TSV value into a stable comparison key."""
	if value is None or pd.isna(value):
		return ""
	return str(value).strip().casefold()


def tokens(value: object) -> tuple[str, ...]:
	"""Return meaningful alphanumeric tokens without discarding numbers."""
	return tuple(sorted({token for token in TOKEN_PATTERN.findall(clean_value(value)) if len(token) >= MIN_TOKEN_LENGTH}))


def source_blocks(country_value: object, name_value: object, address_value: object) -> Iterator[tuple[str, str, str]]:
	"""Yield the existing country-aware blocking keys for one row."""
	country = clean_value(country_value)
	if not country:
		return
	name = clean_value(name_value)
	if name:
		yield "exact_name", country, name
		for token in tokens(name):
			yield "name_token", country, token
	for token in tokens(address_value):
		yield "address_token", country, token


def read_chunks(path: Path) -> Iterator[pd.DataFrame]:
	"""Read one normalized TSV in bounded-size chunks."""
	return pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE, dtype=object, keep_default_na=False)


def estimate_rows(path: Path) -> int:
	"""Estimate rows with a streaming newline count, without loading data."""
	rows = 0
	with path.open("rb") as input_file:
		while block := input_file.read(8 * 1024 * 1024):
			rows += block.count(b"\n")
	return max(0, rows - 1)


def column_positions(chunk: pd.DataFrame) -> dict[str, int | None]:
	"""Locate fields once per chunk so row processing stays tuple-based."""
	return {
		column: chunk.columns.get_loc(column) if column in chunk.columns else None
		for column in ("entity_id", "country", "normalized_business_name", "normalized_business_address")
	}


def row_blocks(row: tuple[object, ...], positions: dict[str, int | None]) -> set[tuple[str, str, str]]:
	"""Extract blocking keys from a tuple without creating a pandas Series."""
	def value(column: str) -> object:
		position = positions[column]
		return row[position] if position is not None else ""

	return set(source_blocks(value("country"), value("normalized_business_name"), value("normalized_business_address")))


def checkpoint(
	connection: sqlite3.Connection,
	stage: str,
	source_type: str,
	chunk_number: int,
	rows_processed: int,
	output_offset: int = 0,
) -> None:
	"""Persist one completed chunk inside the caller's transaction."""
	connection.execute(
		"""
		INSERT INTO checkpoints(stage, source_type, chunk_number, rows_processed, output_offset, status, updated_at)
		VALUES (?, ?, ?, ?, ?, 'complete', datetime('now'))
		ON CONFLICT(stage, source_type) DO UPDATE SET
			chunk_number=excluded.chunk_number,
			rows_processed=excluded.rows_processed,
			output_offset=excluded.output_offset,
			status=excluded.status,
			updated_at=excluded.updated_at
		""",
		(stage, source_type, chunk_number, rows_processed, output_offset),
	)


def completed_chunk(connection: sqlite3.Connection, stage: str, source_type: str) -> int:
	"""Return the last transactionally completed chunk for a stage."""
	row = connection.execute(
		"SELECT chunk_number FROM checkpoints WHERE stage=? AND source_type=? AND status='complete'",
		(stage, source_type),
	).fetchone()
	return int(row[0]) if row else 0


def completed_state(connection: sqlite3.Connection, stage: str, source_type: str) -> tuple[int, int]:
	"""Return completed chunk and output offset for a stage."""
	row = connection.execute(
		"SELECT chunk_number, output_offset FROM checkpoints WHERE stage=? AND source_type=? AND status='complete'",
		(stage, source_type),
	).fetchone()
	return (int(row[0]), int(row[1])) if row else (0, 0)


def initialize_database(database_path: Path, split: str, reset: bool, confirm_destructive: bool) -> sqlite3.Connection:
	"""Open or create a resumable database without implicit deletion."""
	database_path.parent.mkdir(parents=True, exist_ok=True)
	if reset:
		if not confirm_destructive:
			raise RuntimeError("--reset-database requires --confirm-destructive")
		if database_path.exists():
			database_path.unlink()
	elif database_path.exists():
		try:
			probe = sqlite3.connect(database_path)
			result = probe.execute("PRAGMA integrity_check").fetchone()
			tables = {row[0] for row in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")}
			probe.close()
			if result != ("ok",):
				raise RuntimeError(f"SQLite integrity check failed: {result}")
			if not {"metadata", "checkpoints"}.issubset(tables):
				raise RuntimeError(
					f"Existing database {database_path} has no checkpoint metadata. "
					"Use --reset-database --confirm-destructive only after reviewing it."
				)
		except sqlite3.DatabaseError as error:
			raise RuntimeError(
				f"Existing database is unusable: {database_path}. "
				"Use --reset-database --confirm-destructive only after reviewing it."
			) from error

	connection = sqlite3.connect(database_path)
	connection.execute("PRAGMA journal_mode=WAL")
	connection.execute("PRAGMA synchronous=FULL")
	connection.execute("PRAGMA foreign_keys=ON")
	connection.executescript(
		"""
		CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
		CREATE TABLE IF NOT EXISTS checkpoints (
			stage TEXT NOT NULL,
			source_type TEXT NOT NULL,
			chunk_number INTEGER NOT NULL,
			rows_processed INTEGER NOT NULL,
			output_offset INTEGER NOT NULL DEFAULT 0,
			status TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			PRIMARY KEY(stage, source_type)
		);
		CREATE TABLE IF NOT EXISTS block_counts (
			block_type TEXT NOT NULL,
			country TEXT NOT NULL,
			block_key TEXT NOT NULL,
			block_count INTEGER NOT NULL,
			PRIMARY KEY(block_type, country, block_key)
		);
		CREATE TABLE IF NOT EXISTS block_members (
			block_type TEXT NOT NULL,
			country TEXT NOT NULL,
			block_key TEXT NOT NULL,
			source_type TEXT NOT NULL,
			entity_id TEXT NOT NULL,
			PRIMARY KEY(block_type, country, block_key, source_type, entity_id)
		) WITHOUT ROWID;
		CREATE TEMP TABLE IF NOT EXISTS requested_blocks (
			block_type TEXT NOT NULL,
			country TEXT NOT NULL,
			block_key TEXT NOT NULL,
			PRIMARY KEY(block_type, country, block_key)
		);
		"""
	)
	stored_split = connection.execute("SELECT value FROM metadata WHERE key='split'").fetchone()
	if stored_split and stored_split[0] != split:
		connection.close()
		raise RuntimeError(f"Database {database_path} belongs to split {stored_split[0]!r}, not {split!r}")
	connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('split', ?)", (split,))
	connection.commit()
	return connection


def progress(stage: str, source_type: str, rows: int, total: int | None = None) -> None:
	"""Log bounded progress with an estimate when a total is available."""
	if total:
		remaining = max(0, total - rows)
		estimate = f"estimated remaining rows: {remaining:,}"
	else:
		estimate = "estimated remaining rows: unavailable"
	print(f"[{stage}] {source_type}: rows processed {rows:,}; {estimate}")


def count_blocks(connection: sqlite3.Connection, paths: dict[str, Path]) -> None:
	"""Count Source 2/3 blocks, checkpointing each committed chunk."""
	upsert = """
		INSERT INTO block_counts(block_type, country, block_key, block_count) VALUES (?, ?, ?, ?)
		ON CONFLICT(block_type, country, block_key) DO UPDATE SET block_count=block_count + excluded.block_count
	"""
	for source_type in ("source2", "source3"):
		last_chunk = completed_chunk(connection, "count", source_type)
		rows = 0
		total = estimate_rows(paths[source_type])
		for chunk_number, chunk in enumerate(read_chunks(paths[source_type]), start=1):
			rows += len(chunk)
			if chunk_number <= last_chunk:
				progress("count-resume-skip", source_type, rows, total)
				continue
			positions = column_positions(chunk)
			counts: Counter[tuple[str, str, str]] = Counter()
			for row in chunk.itertuples(index=False, name=None):
				counts.update(row_blocks(row, positions))
			try:
				connection.execute("BEGIN")
				connection.executemany(upsert, ((*block, count) for block, count in counts.items()))
				checkpoint(connection, "count", source_type, chunk_number, rows)
				connection.commit()
			except Exception:
				connection.rollback()
				raise
			progress("count", source_type, rows, total)


def eligible_blocks(connection: sqlite3.Connection, blocks: set[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
	"""Keep the existing strategy-specific block size limits unchanged."""
	if not blocks:
		return set()
	connection.execute("DELETE FROM requested_blocks")
	connection.executemany("INSERT INTO requested_blocks VALUES (?, ?, ?)", blocks)
	rows = connection.execute(
		"""
		SELECT requested.block_type, requested.country, requested.block_key
		FROM requested_blocks AS requested JOIN block_counts AS counts USING (block_type, country, block_key)
		WHERE counts.block_count <= CASE requested.block_type
		WHEN 'exact_name' THEN ? WHEN 'name_token' THEN ? WHEN 'address_token' THEN ? END
		""",
		(MAX_EXACT_NAME_BLOCK_SIZE, MAX_NAME_TOKEN_BLOCK_SIZE, MAX_ADDRESS_TOKEN_BLOCK_SIZE),
	).fetchall()
	return {tuple(row) for row in rows}


def build_membership_index(connection: sqlite3.Connection, paths: dict[str, Path]) -> None:
	"""Index eligible Source 2/3 memberships with chunk transactions."""
	insert_member = "INSERT OR IGNORE INTO block_members VALUES (?, ?, ?, ?, ?)"
	for source_type in ("source2", "source3"):
		last_chunk = completed_chunk(connection, "index", source_type)
		rows = 0
		total = estimate_rows(paths[source_type])
		for chunk_number, chunk in enumerate(read_chunks(paths[source_type]), start=1):
			rows += len(chunk)
			if chunk_number <= last_chunk:
				progress("index-resume-skip", source_type, rows, total)
				continue
			positions = column_positions(chunk)
			row_data = []
			all_blocks: set[tuple[str, str, str]] = set()
			for row in chunk.itertuples(index=False, name=None):
				blocks = row_blocks(row, positions)
				entity_position = positions["entity_id"]
				entity_id = str(row[entity_position]) if entity_position is not None else ""
				row_data.append((entity_id, blocks))
				all_blocks.update(blocks)
			try:
				connection.execute("BEGIN")
				allowed = eligible_blocks(connection, all_blocks)
				connection.executemany(
					insert_member,
					((*block, source_type, entity_id) for entity_id, blocks in row_data for block in blocks & allowed),
				)
				checkpoint(connection, "index", source_type, chunk_number, rows)
				connection.commit()
			except Exception:
				connection.rollback()
				raise
			progress("index", source_type, rows, total)


def candidate_ids_for_row(
	connection: sqlite3.Connection,
	blocks: set[tuple[str, str, str]],
	allowed: set[tuple[str, str, str]],
) -> list[str]:
	"""Fetch and deduplicate candidates from eligible blocks."""
	candidates: set[str] = set()
	for block in blocks & allowed:
		rows = connection.execute(
			"SELECT entity_id FROM block_members WHERE block_type=? AND country=? AND block_key=?", block
		).fetchall()
		candidates.update(row[0] for row in rows)
	return sorted(candidates)[:MAX_CANDIDATES_PER_ENTITY]


def generate_pairs(
	connection: sqlite3.Connection,
	path: Path,
	output_path: Path,
	overwrite_output: bool,
) -> None:
	"""Generate pairs into a resumable temp file and atomically finalize it."""
	if output_path.exists() and not overwrite_output:
		raise RuntimeError(f"Output exists: {output_path}. Use --overwrite-output --confirm-destructive.")
	if output_path.exists():
		output_path.unlink()
	temp_path = output_path.with_name(output_path.name + ".tmp")
	last_chunk, offset = completed_state(connection, "generate", "source1")
	if overwrite_output:
		connection.execute("DELETE FROM checkpoints WHERE stage='generate' AND source_type='source1'")
		connection.commit()
		last_chunk, offset = 0, 0
	if offset and not temp_path.exists():
		raise RuntimeError(f"Checkpoint expects temp output, but it is missing: {temp_path}")
	if temp_path.exists() and offset == 0 and not overwrite_output:
		raise RuntimeError(f"Uncheckpointed temp output exists: {temp_path}; inspect before replacing it.")
	if temp_path.exists() and temp_path.stat().st_size < offset:
		raise RuntimeError(f"Temp output is shorter than its checkpoint: {temp_path}")
	temp_path.parent.mkdir(parents=True, exist_ok=True)
	total = estimate_rows(path)
	rows_processed = 0
	with temp_path.open("a+", encoding="utf-8", newline="") as output_file:
		output_file.truncate(offset)
		output_file.seek(0, 2)
		if offset == 0:
			csv.writer(output_file, delimiter="\t", lineterminator="\n").writerow(
				("source1_entity_id", "candidate_entity_ids")
			)
		for chunk_number, chunk in enumerate(read_chunks(path), start=1):
			rows_processed += len(chunk)
			if chunk_number <= last_chunk:
				continue
			positions = column_positions(chunk)
			rows = []
			all_blocks: set[tuple[str, str, str]] = set()
			for row in chunk.itertuples(index=False, name=None):
				blocks = row_blocks(row, positions)
				entity_position = positions["entity_id"]
				entity_id = str(row[entity_position]) if entity_position is not None else ""
				rows.append((entity_id, blocks))
				all_blocks.update(blocks)
			writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
			try:
				connection.execute("BEGIN")
				allowed = eligible_blocks(connection, all_blocks)
				for entity_id, blocks in rows:
					writer.writerow((entity_id, ",".join(candidate_ids_for_row(connection, blocks, allowed))))
				output_file.flush()
				new_offset = output_file.tell()
				checkpoint(connection, "generate", "source1", chunk_number, rows_processed, new_offset)
				connection.commit()
			except Exception:
				connection.rollback()
				raise
			progress("generate", "source1", rows_processed, total)
	if not temp_path.exists():
		raise RuntimeError("Candidate temp file was not created")
	if output_path.exists() and not overwrite_output:
		raise RuntimeError(f"Refusing to replace existing output: {output_path}")
	temp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--split", choices=("test", "train"), default="test")
	parser.add_argument("--data-root", type=Path, default=Path("dataset/normalized"))
	parser.add_argument("--database", type=Path, default=BLOCKING_DATABASE)
	parser.add_argument("--reset-database", action="store_true")
	parser.add_argument("--overwrite-output", action="store_true")
	parser.add_argument("--confirm-destructive", action="store_true")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	root = Path(__file__).resolve().parents[1]
	data_root = args.data_root if args.data_root.is_absolute() else root / args.data_root
	paths = source_paths(data_root, args.split)
	missing = [path for path in paths.values() if not path.exists()]
	if missing:
		print(f"ERROR: missing normalized files: {', '.join(map(str, missing))}", file=sys.stderr)
		return 1
	connection: sqlite3.Connection | None = None
	try:
		connection = initialize_database(root / args.database, args.split, args.reset_database, args.confirm_destructive)
		count_blocks(connection, paths)
		build_membership_index(connection, paths)
		output = root / "output" / ("candidate_pairs.tsv" if args.split == "test" else "train_candidate_pairs.tsv")
		generate_pairs(connection, paths["source1"], output, args.overwrite_output and args.confirm_destructive)
		print(f"Completed {args.split} candidate generation: {output}")
		return 0
	except (OSError, pd.errors.ParserError, sqlite3.Error, ValueError, RuntimeError) as error:
		print(f"ERROR: candidate generation failed safely: {error}", file=sys.stderr)
		return 1
	finally:
		if connection is not None:
			connection.close()


if __name__ == "__main__":
	raise SystemExit(main())
