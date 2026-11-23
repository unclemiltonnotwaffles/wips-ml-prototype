from pathlib import Path
import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    os.environ.get(
        "AWID3_RAW_DIR",
        r"C:\Users\chiwo\.cache\kagglehub\datasets\suumia\awid3-dataset\versions\1\CSV"
    )
).expanduser().resolve()

OUTPUT_DIR = Path(__file__).resolve().parent / "processed_awid3"

OUTPUT_FILE = OUTPUT_DIR / "awid3_cleaned.csv"
REPORT_FILE = OUTPUT_DIR / "cleaning_report.csv"
LABEL_REPORT_FILE = OUTPUT_DIR / "label_distribution.csv"

CHUNK_SIZE = 100_000

MIN_VALID_FEATURES = 3


# ============================================================
# FEATURES SELECTED FOR PREPROCESSING
# ============================================================

FEATURES = [
    "frame.len",
    "frame.time_delta",
    "radiotap.channel.freq",
    "radiotap.datarate",
    "wlan.fc.ds",
    "wlan.fc.protected",
    "wlan.fc.type",
    "wlan.fc.retry",
    "wlan.fc.subtype",
    "wlan_radio.channel",
    "wlan_radio.signal_dbm",
    "ip.proto",
    "ip.ttl",
    "tcp.dstport",
    "tcp.srcport",
    "udp.dstport",
    "udp.srcport",
]

LABEL_COLUMN = "Label"

OUTPUT_COLUMNS = [
    "source_file",
    *FEATURES,
    LABEL_COLUMN,
]


# ============================================================
# LABEL NORMALIZATION
# ============================================================

LABEL_MAP = {
    "SDDP": "SSDP",
    "Kr00K": "Kr00k",
}


# ============================================================
# DATA CONVERSION
# ============================================================

def convert_numeric_feature(raw, feature):
    """
    Convert a selected feature to numeric values.

    Special handling:
        wlan.fc.ds contains hexadecimal strings such as
        0x00000002, which pd.to_numeric() cannot parse directly.

    Returns:
        numeric: converted numeric Series
        invalid: number of non-empty values that failed conversion
    """

    raw = raw.astype("string").str.strip()
    raw = raw.replace("", pd.NA)

    # --------------------------------------------------------
    # Special case: wlan.fc.ds hexadecimal values
    # --------------------------------------------------------

    if feature == "wlan.fc.ds":

        hex_mask = raw.str.fullmatch(
            r"0[xX][0-9A-Fa-f]+",
            na=False
        )

        # Convert ordinary numeric values first.
        numeric = pd.to_numeric(
            raw.where(~hex_mask),
            errors="coerce"
        )

        # Convert hexadecimal strings to decimal integers.
        numeric.loc[hex_mask] = raw.loc[hex_mask].map(
            lambda value: int(value, 16)
        )

        numeric = pd.to_numeric(
            numeric,
            errors="coerce"
        )

    # --------------------------------------------------------
    # Standard numeric conversion for other features
    # --------------------------------------------------------

    else:

        numeric = pd.to_numeric(
            raw,
            errors="coerce"
        )

    # Count non-empty values that could not be converted.
    invalid = raw.notna() & numeric.isna()

    invalid_count = int(invalid.sum())

    return numeric, invalid_count


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def prepare_dataset():

    print("=" * 75)
    print("AWID3 DATASET PREPROCESSING")
    print("=" * 75)

    # --------------------------------------------------------
    # Validate dataset location
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_DIR}"
        )

    # --------------------------------------------------------
    # Find all CSV files recursively
    # --------------------------------------------------------

    csv_files = sorted(
        DATASET_DIR.rglob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            f"No CSV files found under:\n{DATASET_DIR}"
        )

    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"CSV files found: {len(csv_files)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Remove previous outputs to avoid accidental appending
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    if REPORT_FILE.exists():
        REPORT_FILE.unlink()

    if LABEL_REPORT_FILE.exists():
        LABEL_REPORT_FILE.unlink()

    # --------------------------------------------------------
    # Initialize statistics
    # --------------------------------------------------------

    stats = {
        "files_found": len(csv_files),
        "files_processed": 0,
        "input_rows": 0,
        "missing_label_rows": 0,
        "insufficient_feature_rows": 0,
        "output_rows": 0,
    }

    for feature in FEATURES:

        stats[f"invalid_{feature}"] = 0

    label_counts = {}

    output_header_written = False

    # --------------------------------------------------------
    # Process each CSV file
    # --------------------------------------------------------

    for file_index, file_path in enumerate(csv_files, start=1):

        print(
            f"\n[{file_index}/{len(csv_files)}] "
            f"Processing: {file_path.name}"
        )

        stats["files_processed"] += 1

        file_rows = 0

        # Read file in chunks.
        try:

            for chunk in pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
                low_memory=True,
                encoding="utf-8",
                on_bad_lines="warn",
            ):

                file_rows += len(chunk)

                stats["input_rows"] += len(chunk)

                # ------------------------------------------------
                # Ensure required columns exist
                # ------------------------------------------------

                if LABEL_COLUMN not in chunk.columns:

                    raise ValueError(
                        f"Label column '{LABEL_COLUMN}' missing in:\n"
                        f"{file_path}"
                    )

                for feature in FEATURES:

                    if feature not in chunk.columns:

                        chunk[feature] = pd.NA

                # ------------------------------------------------
                # Keep only selected features and label
                # ------------------------------------------------

                chunk = chunk[
                    FEATURES + [LABEL_COLUMN]
                ].copy()

                # ------------------------------------------------
                # Remove rows without labels
                # ------------------------------------------------

                labels = (
                    chunk[LABEL_COLUMN]
                    .astype("string")
                    .str.strip()
                )

                missing = (
                    labels.isna()
                    | labels.eq("")
                )

                missing_count = int(missing.sum())

                stats["missing_label_rows"] += missing_count

                chunk = chunk.loc[~missing].copy()

                if chunk.empty:

                    continue

                # ------------------------------------------------
                # Normalize known label variants
                # ------------------------------------------------

                chunk[LABEL_COLUMN] = (
                    chunk[LABEL_COLUMN]
                    .astype("string")
                    .str.strip()
                    .replace(LABEL_MAP)
                )

                # ------------------------------------------------
                # Convert selected features to numeric
                # ------------------------------------------------

                for feature in FEATURES:

                    numeric, invalid_count = convert_numeric_feature(
                        chunk[feature],
                        feature
                    )

                    stats[f"invalid_{feature}"] += invalid_count

                    chunk[feature] = numeric

                # ------------------------------------------------
                # Require at least three usable model features
                # ------------------------------------------------

                usable_count = (
                    chunk[FEATURES]
                    .notna()
                    .sum(axis=1)
                )

                insufficient = (
                    usable_count < MIN_VALID_FEATURES
                )

                insufficient_count = int(insufficient.sum())

                stats["insufficient_feature_rows"] += (
                    insufficient_count
                )

                chunk = chunk.loc[~insufficient].copy()

                if chunk.empty:

                    continue

                # ------------------------------------------------
                # Track class distribution
                # ------------------------------------------------

                counts = chunk[LABEL_COLUMN].value_counts()

                for label, count in counts.items():

                    label_counts[label] = (
                        label_counts.get(label, 0)
                        + int(count)
                    )

                # ------------------------------------------------
                # Add source filename
                # ------------------------------------------------

                chunk.insert(
                    0,
                    "source_file",
                    file_path.name
                )

                # ------------------------------------------------
                # Arrange output columns
                # ------------------------------------------------

                chunk = chunk[OUTPUT_COLUMNS]

                # ------------------------------------------------
                # Append cleaned data to output
                # ------------------------------------------------

                chunk.to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    index=False,
                    header=not output_header_written,
                )

                output_header_written = True

                stats["output_rows"] += len(chunk)

        except Exception as error:

            print(
                f"\nERROR processing {file_path.name}: {error}"
            )

            raise

        print(
            f"    Input rows processed: {file_rows:,}"
        )

    # ============================================================
    # GENERATE REPORTS
    # ============================================================

    print("\n" + "=" * 75)
    print("PREPROCESSING COMPLETE")
    print("=" * 75)

    # ------------------------------------------------------------
    # Cleaning report
    # ------------------------------------------------------------

    report = pd.DataFrame(
        list(stats.items()),
        columns=["metric", "value"]
    )

    report.to_csv(
        REPORT_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # Label distribution report
    # ------------------------------------------------------------

    label_report = pd.DataFrame(
        list(label_counts.items()),
        columns=["Label", "count"]
    )

    if not label_report.empty:

        label_report = label_report.sort_values(
            "count",
            ascending=False
        )

        label_report["percentage"] = (
            label_report["count"]
            / label_report["count"].sum()
            * 100
        )

    label_report.to_csv(
        LABEL_REPORT_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # Display statistics
    # ------------------------------------------------------------

    for key, value in stats.items():

        if isinstance(value, int):

            print(
                f"{key:35}: {value:,}"
            )

        else:

            print(
                f"{key:35}: {value}"
            )

    print("\nLABEL DISTRIBUTION")

    if not label_report.empty:

        print(
            label_report.to_string(index=False)
        )

    print("\nOUTPUT FILES")

    print(f"Cleaned dataset : {OUTPUT_FILE}")
    print(f"Cleaning report: {REPORT_FILE}")
    print(f"Label report  : {LABEL_REPORT_FILE}")

    print("\n" + "=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    prepare_dataset()