"""Normalize business names and addresses in the local entity-resolution data."""

from pathlib import Path
import re
import unicodedata

import pandas as pd


CHUNK_SIZE = 100_000
WHITESPACE_PATTERN = re.compile(r"\s+")
SOURCE_FILES = (
	Path("dataset/train/train_source1.tsv"),
	Path("dataset/train/train_source2.tsv"),
	Path("dataset/train/train_source3.tsv"),
	Path("dataset/test/test_source1.tsv"),
	Path("dataset/test/test_source2.tsv"),
	Path("dataset/test/test_source3.tsv"),
)

# Standardize common typographic variants without deleting meaningful text.
PUNCTUATION_TRANSLATION = str.maketrans(
	{
		"\u2018": "'",
		"\u2019": "'",
		"\u201c": '"',
		"\u201d": '"',
		"\u2013": "-",
		"\u2014": "-",
		"\u2212": "-",
		"\u2026": "...",
		"\u00a0": " ",
	}
)


def normalize_value(value: object) -> str:
	"""Normalize one value while representing missing values as empty strings."""
	if value is None or pd.isna(value):
		return ""
	text = unicodedata.normalize("NFKC", str(value))
	text = text.translate(PUNCTUATION_TRANSLATION).lower()
	return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_text(values: pd.Series) -> pd.Series:
	"""Normalize text values while representing missing values as empty strings."""
	return values.map(normalize_value)


def process_file(project_root: Path, relative_path: Path) -> bool:
	"""Normalize one TSV in chunks and write it to the normalized directory."""
	input_path = project_root / relative_path
	output_path = project_root / "dataset/normalized" / Path(*relative_path.parts[1:])
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if not input_path.exists():
		print(f"ERROR: missing input file: {input_path}")
		return False

	output_path.unlink(missing_ok=True)
	rows_processed = 0
	try:
		chunks = pd.read_csv(
			input_path,
			sep="\t",
			chunksize=CHUNK_SIZE,
			dtype=object,
			keep_default_na=False,
		)
		for chunk_number, chunk in enumerate(chunks, start=1):
			for source_column, normalized_column in (
				("business_name", "normalized_business_name"),
				("business_address", "normalized_business_address"),
			):
				if source_column in chunk.columns:
					chunk[normalized_column] = normalize_text(chunk[source_column])
				else:
					print(f"WARNING: {input_path} has no {source_column!r} column")
					chunk[normalized_column] = ""

			chunk.to_csv(
				output_path,
				sep="\t",
				index=False,
				mode="a",
				header=chunk_number == 1,
				lineterminator="\n",
			)
			rows_processed += len(chunk)
			print(f"Processed {input_path}: chunk {chunk_number}, {rows_processed:,} rows")
	except Exception as error:
		output_path.unlink(missing_ok=True)
		print(f"ERROR: unable to normalize {input_path}: {error}")
		return False

	print(f"Wrote {output_path} ({rows_processed:,} rows)")
	return True


def main() -> None:
	"""Normalize all six source files without changing the originals."""
	project_root = Path(__file__).resolve().parents[1]
	successful = 0
	for relative_path in SOURCE_FILES:
		print(f"Starting {relative_path}...")
		if process_file(project_root, relative_path):
			successful += 1
	print(f"Normalization complete: {successful}/{len(SOURCE_FILES)} files processed successfully.")


if __name__ == "__main__":
	main()
