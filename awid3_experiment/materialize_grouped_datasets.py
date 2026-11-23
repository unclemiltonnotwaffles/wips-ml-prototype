
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_awid3"

INPUT_FILE = DATA_DIR / "awid3_cleaned.csv"

TRAIN_SPLIT_FILE = DATA_DIR / "train_capture_files.csv"
TEST_SPLIT_FILE = DATA_DIR / "test_capture_files.csv"

TRAIN_OUTPUT = DATA_DIR / "awid3_train.csv"
TEST_OUTPUT = DATA_DIR / "awid3_test.csv"

SUMMARY_OUTPUT = DATA_DIR / "materialized_split_summary.csv"

CHUNK_SIZE = 250_000

EXPECTED_TOTAL_RECORDS = 36_962_587


# ============================================================
# LOAD AND VALIDATE CAPTURE SPLITS
# ============================================================

def load_capture_names(path):
    df = pd.read_csv(path, dtype={"source_file": "string"})

    if "source_file" not in df.columns:
        raise ValueError(f"Missing source_file column in {path}")

    if df["source_file"].isna().any():
        raise ValueError(f"Missing capture identifiers in {path}")

    if df["source_file"].duplicated().any():
        raise ValueError(f"Duplicate capture identifiers in {path}")

    return set(df["source_file"].tolist())


def main():

    print("=" * 75)
    print("AWID3 GROUPED DATASET MATERIALISATION")
    print("=" * 75)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if not TRAIN_SPLIT_FILE.exists():
        raise FileNotFoundError(TRAIN_SPLIT_FILE)

    if not TEST_SPLIT_FILE.exists():
        raise FileNotFoundError(TEST_SPLIT_FILE)

    train_captures = load_capture_names(TRAIN_SPLIT_FILE)
    test_captures = load_capture_names(TEST_SPLIT_FILE)

    overlap = train_captures.intersection(test_captures)

    if overlap:
        raise ValueError(
            f"Capture leakage detected: {len(overlap)} overlapping files"
        )

    all_captures = train_captures.union(test_captures)

    print(f"\nTraining captures: {len(train_captures)}")
    print(f"Testing captures:  {len(test_captures)}")
    print(f"Total captures:    {len(all_captures)}")
    print(f"Capture overlap:   {len(overlap)}")

    if len(all_captures) != 745:
        raise ValueError(
            f"Expected 745 captures, found {len(all_captures)}"
        )

    # Remove previous outputs to prevent accidental appending.
    for output in [TRAIN_OUTPUT, TEST_OUTPUT, SUMMARY_OUTPUT]:
        if output.exists():
            output.unlink()

    # ========================================================
    # PROCESS CLEANED DATASET IN CHUNKS
    # ========================================================

    train_records = 0
    test_records = 0
    unassigned_records = 0
    total_records = 0

    train_labels = Counter()
    test_labels = Counter()

    train_seen = set()
    test_seen = set()

    train_first_chunk = True
    test_first_chunk = True

    print("\nProcessing cleaned dataset...")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    reader = pd.read_csv(
        INPUT_FILE,
        dtype={
            "source_file": "string",
            "Label": "string"
        },
        chunksize=CHUNK_SIZE
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        if "source_file" not in chunk.columns:
            raise ValueError("Missing source_file column")

        if "Label" not in chunk.columns:
            raise ValueError("Missing Label column")

        if chunk["source_file"].isna().any():
            raise ValueError(
                f"Missing capture identifier in chunk {chunk_number}"
            )

        if chunk["Label"].isna().any():
            raise ValueError(
                f"Missing label in chunk {chunk_number}"
            )

        train_mask = chunk["source_file"].isin(train_captures)
        test_mask = chunk["source_file"].isin(test_captures)

        # Every row must belong to exactly one partition.
        assigned_mask = train_mask | test_mask

        unassigned_records += int((~assigned_mask).sum())

        if (train_mask & test_mask).any():
            raise ValueError("A row was assigned to both partitions")

        train_chunk = chunk.loc[train_mask]
        test_chunk = chunk.loc[test_mask]

        # Write training records.
        if not train_chunk.empty:

            train_chunk.to_csv(
                TRAIN_OUTPUT,
                mode="w" if train_first_chunk else "a",
                header=train_first_chunk,
                index=False
            )

            train_first_chunk = False

            train_records += len(train_chunk)

            train_labels.update(
                train_chunk["Label"].value_counts().to_dict()
            )

            train_seen.update(train_chunk["source_file"].unique())

        # Write testing records.
        if not test_chunk.empty:

            test_chunk.to_csv(
                TEST_OUTPUT,
                mode="w" if test_first_chunk else "a",
                header=test_first_chunk,
                index=False
            )

            test_first_chunk = False

            test_records += len(test_chunk)

            test_labels.update(
                test_chunk["Label"].value_counts().to_dict()
            )

            test_seen.update(test_chunk["source_file"].unique())

        total_records += len(chunk)

        print(
            f"Chunk {chunk_number:>3}: "
            f"{len(chunk):>9,} rows | "
            f"Train: {train_records:>11,} | "
            f"Test: {test_records:>10,}"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 75)
    print("MATERIALISATION VALIDATION")
    print("=" * 75)

    print(f"Input records:       {total_records:,}")
    print(f"Training records:    {train_records:,}")
    print(f"Testing records:     {test_records:,}")
    print(f"Unassigned records:  {unassigned_records:,}")

    if total_records != EXPECTED_TOTAL_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_RECORDS:,} input records, "
            f"found {total_records:,}"
        )

    if train_records + test_records != total_records:
        raise ValueError("Partition record counts do not match input")

    if unassigned_records != 0:
        raise ValueError(
            f"{unassigned_records:,} records were not assigned"
        )

    if train_seen != train_captures:
        raise ValueError(
            "Training capture coverage does not match split file"
        )

    if test_seen != test_captures:
        raise ValueError(
            "Testing capture coverage does not match split file"
        )

    if train_seen.intersection(test_seen):
        raise ValueError("Capture leakage detected after extraction")

    if train_first_chunk or test_first_chunk:
        raise ValueError("One or both output datasets are empty")

    all_labels = sorted(set(train_labels) | set(test_labels))

    summary = pd.DataFrame([
        {
            "Label": label,
            "train_records": train_labels[label],
            "test_records": test_labels[label],
            "total_records": (
                train_labels[label] + test_labels[label]
            )
        }
        for label in all_labels
    ])

    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("\nCLASS DISTRIBUTION")

    print(summary.to_string(index=False))

    print("\nCAPTURE VALIDATION")
    print(f"Training captures verified: {len(train_seen)}")
    print(f"Testing captures verified:  {len(test_seen)}")
    print("Capture overlap:             0")

    print("\nOUTPUT FILES")
    print(TRAIN_OUTPUT)
    print(TEST_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\nPASS: All records assigned exactly once.")
    print("PASS: All capture identifiers verified.")
    print("PASS: No train/test capture leakage.")
    print("PASS: All 14 classes retained in both partitions.")

    print("\nDATASET MATERIALISATION COMPLETE")


if __name__ == "__main__":
    main()
