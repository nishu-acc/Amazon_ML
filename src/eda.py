"""Generate a lightweight exploratory data analysis report for the TSV data."""

from pathlib import Path

import pandas as pd


SOURCE_FILES = (
	Path("dataset/train/train_source1.tsv"),
	Path("dataset/train/train_source2.tsv"),
	Path("dataset/train/train_source3.tsv"),
	Path("dataset/test/test_source1.tsv"),
	Path("dataset/test/test_source2.tsv"),
	Path("dataset/test/test_source3.tsv"),
)
GROUND_TRUTH_FILE = Path("dataset/train/train_ground_truth.tsv")
REPORT_FILE = Path("reports/eda_report.txt")


def format_series(series: pd.Series) -> str:
	"""Format a value-count series as readable indented lines."""
	if series.empty:
		return "  (none)"
	return "\n".join(f"  {value!r}: {count}" for value, count in series.items())


def empty_value_count(dataframe: pd.DataFrame, column: str) -> int | None:
	"""Count both missing and whitespace-only values in a column."""
	if column not in dataframe.columns:
		return None
	return int(dataframe[column].fillna("").astype(str).str.strip().eq("").sum())


def report_source_file(path: Path, dataframe: pd.DataFrame) -> list[str]:
	"""Return the required EDA details for one source file."""
	lines = [
		f"File: {path.as_posix()}",
		f"Rows: {len(dataframe)}",
		f"Columns: {', '.join(map(str, dataframe.columns))}",
		"Missing values by column:",
		format_series(dataframe.isna().sum()),
	]

	if "entity_id" in dataframe.columns:
		duplicate_mask = dataframe["entity_id"].duplicated(keep=False)
		duplicate_ids = dataframe.loc[duplicate_mask, "entity_id"].drop_duplicates()
		lines.extend(
			[
				f"Duplicate entity ID rows: {int(duplicate_mask.sum())}",
				f"Duplicate entity IDs: {', '.join(map(str, duplicate_ids)) or '(none)'}",
			]
		)
	else:
		lines.append("Duplicate entity IDs: unavailable (missing 'entity_id' column)")

	if "country" in dataframe.columns:
		country_values = dataframe["country"].fillna("<missing>").astype(str).str.strip()
		country_values = country_values.mask(country_values.eq(""), "<empty>")
		lines.extend(["Country distribution:", format_series(country_values.value_counts(dropna=False))])
	else:
		lines.append("Country distribution: unavailable (missing 'country' column)")

	for column in ("business_name", "business_address"):
		count = empty_value_count(dataframe, column)
		if count is None:
			lines.append(f"Empty {column} values: unavailable (column not found)")
		else:
			lines.append(f"Empty {column} values: {count}")

	return lines


def report_ground_truth(path: Path, dataframe: pd.DataFrame) -> list[str]:
	"""Return the required match statistics for the ground-truth file."""
	lines = [
		f"File: {path.as_posix()}",
		f"Total rows: {len(dataframe)}",
	]

	if "matched_entity_ids" not in dataframe.columns:
		lines.append("Empty matched_entity_ids: unavailable (column not found)")
		lines.append("Match counts per Source 1 entity: unavailable (column not found)")
		return lines

	matches = dataframe["matched_entity_ids"].fillna("").astype(str).str.strip()
	empty_matches = int(matches.eq("").sum())
	match_counts = matches.map(lambda value: 0 if not value else len(value.split(",")))
	lines.extend(
		[
			f"Empty matched_entity_ids: {empty_matches}",
			"Match count distribution per Source 1 entity:",
			format_series(match_counts.value_counts().sort_index()),
		]
	)
	return lines


def load_file(path: Path, report_lines: list[str]) -> pd.DataFrame | None:
	"""Load a TSV if it exists, recording problems for the final report."""
	if not path.exists():
		report_lines.extend([f"MISSING FILE: {path.as_posix()}", ""])
		return None

	try:
		return pd.read_csv(path, sep="\t")
	except Exception as error:
		report_lines.extend([f"UNABLE TO READ FILE: {path.as_posix()}", f"Reason: {error}", ""])
		return None


def main() -> None:
	"""Load the local datasets and write the EDA report."""
	project_root = Path(__file__).resolve().parents[1]
	report_path = project_root / REPORT_FILE
	report_lines = [
		"Business Entity Resolution - Exploratory Data Analysis",
		"========================================================",
		"",
	]

	for relative_path in SOURCE_FILES:
		path = project_root / relative_path
		dataframe = load_file(path, report_lines)
		if dataframe is not None:
			report_lines.extend(report_source_file(relative_path, dataframe))
			report_lines.append("")

	report_lines.extend(["Ground Truth", "------------"])
	ground_truth = load_file(project_root / GROUND_TRUTH_FILE, report_lines)
	if ground_truth is not None:
		report_lines.extend(report_ground_truth(GROUND_TRUTH_FILE, ground_truth))
		report_lines.append("")

	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text("\n".join(report_lines), encoding="utf-8")
	print(f"EDA report written to {report_path}")


if __name__ == "__main__":
	main()
