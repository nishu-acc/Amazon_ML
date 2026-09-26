
import argparse
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"

INPUT_FILES = {
    "train": DATA_DIR / "train_split.tsv",
    "validation": DATA_DIR / "validation_split.tsv",
}

OUTPUT_FILES = {
    "train": DATA_DIR / "train_features_full.tsv",
    "validation": DATA_DIR / "validation_features_full.tsv",
}

CHUNK_SIZE = 10_000

REQUIRED_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "source1_business_name",
    "candidate_business_name",
    "source1_business_address",
    "candidate_business_address",
    "source1_country",
    "candidate_country",
    "label",
]


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def create_features(df):
    features = []

    for row in df.itertuples(index=False):
        name1 = clean_text(row.source1_business_name)
        name2 = clean_text(row.candidate_business_name)

        address1 = clean_text(row.source1_business_address)
        address2 = clean_text(row.candidate_business_address)

        country1 = clean_text(row.source1_country)
        country2 = clean_text(row.candidate_country)

        features.append({
            "source1_entity_id": row.source1_entity_id,
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
            "label": row.label,
        })

    return pd.DataFrame(features)


def main():
    parser = argparse.ArgumentParser(
        description="Generate features in chunks."
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    args = parser.parse_args()

    input_file = INPUT_FILES[args.split]
    output_file = OUTPUT_FILES[args.split]

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    if output_file.exists():
        raise FileExistsError(
            f"Output already exists: {output_file}\n"
            "Rename or remove it before running again."
        )

    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Split: {args.split}")
    print(f"Chunk size: {args.chunk_size}")

    total_rows = 0
    first_chunk = True

    try:
        for chunk in pd.read_csv(
            input_file,
            sep="\t",
            chunksize=args.chunk_size,
            dtype={"source1_entity_id": str},
        ):
            missing = set(REQUIRED_COLUMNS) - set(chunk.columns)

            if missing:
                raise ValueError(
                    f"Missing columns: {sorted(missing)}"
                )

            if args.limit is not None:
                remaining = args.limit - total_rows

                if remaining <= 0:
                    break

                chunk = chunk.iloc[:remaining]

            if chunk.empty:
                break

            features = create_features(chunk)

            features.to_csv(
                output_file,
                sep="\t",
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk,
            )

            first_chunk = False
            total_rows += len(chunk)

            print(f"Processed {total_rows:,} rows")

            if args.limit is not None and total_rows >= args.limit:
                break

    except Exception:
        if output_file.exists():
            output_file.unlink()
        raise

    if total_rows == 0:
        raise ValueError("No rows were processed.")

    print("\nFeature generation completed.")
    print(f"Total rows: {total_rows:,}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()