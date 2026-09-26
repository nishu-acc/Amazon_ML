
import argparse
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


INPUT_FILE = Path("dataset/processed/train_split.tsv")
OUTPUT_FILE = Path("dataset/processed/train_features_sample.tsv")

CHUNK_SIZE = 100_000


def clean(value):
    return str(value).strip().casefold()


def similarity(a, b, method):
    a = clean(a)
    b = clean(b)

    if not a or not b:
        return 0.0

    if method == "ratio":
        return fuzz.ratio(a, b)

    return fuzz.token_set_ratio(a, b)


def create_features(df):
    features = pd.DataFrame()
    features["source1_entity_id"] = df["source1_entity_id"]
    

    # Exact matching features
    features["name_exact"] = (
        df["source1_business_name"].str.casefold().str.strip()
        == df["candidate_business_name"].str.casefold().str.strip()
    ).astype("int8")

    features["address_exact"] = (
        df["source1_business_address"].str.casefold().str.strip()
        == df["candidate_business_address"].str.casefold().str.strip()
    ).astype("int8")

    features["country_match"] = (
        df["source1_country"].str.casefold().str.strip()
        == df["candidate_country"].str.casefold().str.strip()
    ).astype("int8")

    # String similarity features
    features["name_ratio"] = [
        similarity(a, b, "ratio")
        for a, b in zip(
            df["source1_business_name"],
            df["candidate_business_name"],
        )
    ]

    features["name_token_set_ratio"] = [
        similarity(a, b, "token")
        for a, b in zip(
            df["source1_business_name"],
            df["candidate_business_name"],
        )
    ]

    features["address_ratio"] = [
        similarity(a, b, "ratio")
        for a, b in zip(
            df["source1_business_address"],
            df["candidate_business_address"],
        )
    ]

    features["address_token_set_ratio"] = [
        similarity(a, b, "token")
        for a, b in zip(
            df["source1_business_address"],
            df["candidate_business_address"],
        )
    ]

    # Length difference features
    features["name_length_diff"] = (
        df["source1_business_name"].str.len()
        - df["candidate_business_name"].str.len()
    ).abs()

    features["address_length_diff"] = (
        df["source1_business_address"].str.len()
        - df["candidate_business_address"].str.len()
    ).abs()

    features["label"] = df["label"].astype("int8")

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. Rename it before rerunning."
        )

    processed = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        remaining = args.limit - processed

        if remaining <= 0:
            break

        chunk = chunk.iloc[:remaining]
        features = create_features(chunk)

        features.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=not OUTPUT_FILE.exists(),
        )

        processed += len(chunk)
        print(f"Processed {processed:,} rows")

    print(f"\nFeature generation completed: {OUTPUT_FILE}")
    print(f"Total rows: {processed:,}")
    print("Feature columns:", list(features.columns))


if __name__ == "__main__":
    main()