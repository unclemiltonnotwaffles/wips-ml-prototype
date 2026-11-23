
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_awid3"

MATRIX_FILE = DATA_DIR / "capture_class_matrix.csv"

OUTPUT_TRAIN = DATA_DIR / "train_capture_files.csv"
OUTPUT_TEST = DATA_DIR / "test_capture_files.csv"
OUTPUT_REPORT = DATA_DIR / "grouped_split_report.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42
CANDIDATES = 10000


def main():

    print("=" * 75)
    print("AWID3 GROUPED TRAIN/TEST SPLIT")
    print("=" * 75)

    if not MATRIX_FILE.exists():
        raise FileNotFoundError(MATRIX_FILE)

    matrix = pd.read_csv(
        MATRIX_FILE,
        index_col=0
    )

    matrix = matrix.fillna(0).astype(int)

    if matrix.empty:
        raise ValueError("Capture matrix is empty.")

    if (matrix < 0).any().any():
        raise ValueError("Negative record counts found.")

    capture_names = matrix.index.to_numpy()
    class_names = matrix.columns.tolist()

    counts = matrix.to_numpy(dtype=np.int64)

    total_per_class = counts.sum(axis=0)

    if np.any(total_per_class == 0):
        raise ValueError("At least one class has no records.")

    if len(capture_names) < 2:
        raise ValueError("Not enough capture files.")

    # Each capture is represented by one row.
    # GroupShuffleSplit therefore keeps every capture intact.
    X = np.zeros((len(capture_names), 1))
    groups = capture_names

    splitter = GroupShuffleSplit(
        n_splits=CANDIDATES,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    best_score = float("inf")
    best_train = None
    best_test = None
    best_test_counts = None

    target_total_fraction = TEST_SIZE

    print(f"Capture files: {len(capture_names)}")
    print(f"Classes: {len(class_names)}")
    print(f"Candidate splits: {CANDIDATES:,}")
    print("\nSearching for a valid grouped split...")

    for train_idx, test_idx in splitter.split(
        X,
        groups=groups
    ):

        train_counts = counts[train_idx].sum(axis=0)
        test_counts = counts[test_idx].sum(axis=0)

        # Every class must exist in both partitions.
        if np.any(train_counts == 0):
            continue

        if np.any(test_counts == 0):
            continue

        # Class-specific test proportions.
        class_fractions = (
            test_counts / total_per_class
        )

        # Overall test record proportion.
        total_fraction = (
            counts[test_idx].sum()
            / counts.sum()
        )

        # Balance the class proportions and overall
        # record proportion. Capture groups remain intact.
        class_error = np.mean(
            np.abs(class_fractions - TEST_SIZE)
        )

        total_error = abs(
            total_fraction - target_total_fraction
        )

        score = class_error + (2.0 * total_error)

        if score < best_score:

            best_score = score
            best_train = train_idx.copy()
            best_test = test_idx.copy()
            best_test_counts = test_counts.copy()

    if best_train is None:
        raise RuntimeError(
            "No valid split found. "
            "At least one class may not have enough "
            "independent capture files."
        )

    train_files = capture_names[best_train]
    test_files = capture_names[best_test]

    # Confirm no capture overlap.
    overlap = set(train_files) & set(test_files)

    if overlap:
        raise RuntimeError(
            f"Capture leakage detected: {overlap}"
        )

    # Save capture assignments.
    pd.DataFrame({
        "source_file": train_files
    }).to_csv(OUTPUT_TRAIN, index=False)

    pd.DataFrame({
        "source_file": test_files
    }).to_csv(OUTPUT_TEST, index=False)

    # Detailed class-level report.
    train_counts = counts[best_train].sum(axis=0)
    test_counts = counts[best_test].sum(axis=0)

    report = pd.DataFrame({
        "Label": class_names,
        "total_records": total_per_class,
        "train_records": train_counts,
        "test_records": test_counts,
        "train_percentage": (
            train_counts / total_per_class * 100
        ),
        "test_percentage": (
            test_counts / total_per_class * 100
        ),
        "train_capture_files": (
            (counts[best_train] > 0).sum(axis=0)
        ),
        "test_capture_files": (
            (counts[best_test] > 0).sum(axis=0)
        )
    })

    report.to_csv(OUTPUT_REPORT, index=False)

    # ==========================================
    # FINAL VALIDATION
    # ==========================================

    print("\nSPLIT SUMMARY")

    print(f"Training capture files: {len(train_files)}")
    print(f"Testing capture files:  {len(test_files)}")

    print(
        f"Training records: {counts[best_train].sum():,}"
    )

    print(
        f"Testing records:  {counts[best_test].sum():,}"
    )

    print(
        f"Actual test record percentage: "
        f"{counts[best_test].sum() / counts.sum() * 100:.2f}%"
    )

    print(f"Split objective score: {best_score:.6f}")

    print("\nCLASS DISTRIBUTION")

    print(
        report.round(2).to_string(index=False)
    )

    print("\nVALIDATION")

    print(f"Capture overlap: {len(overlap)}")

    print(
        "Classes in training:",
        int((train_counts > 0).sum()),
        "/",
        len(class_names)
    )

    print(
        "Classes in testing:",
        int((test_counts > 0).sum()),
        "/",
        len(class_names)
    )

    assert len(overlap) == 0
    assert np.all(train_counts > 0)
    assert np.all(test_counts > 0)

    print("\nPASS: No capture files overlap.")
    print("PASS: All classes appear in both partitions.")

    print("\nOUTPUT FILES")

    print(OUTPUT_TRAIN)
    print(OUTPUT_TEST)
    print(OUTPUT_REPORT)

    print("\nGROUPED SPLIT COMPLETE")


if __name__ == "__main__":
    main()
