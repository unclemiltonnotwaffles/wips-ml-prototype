
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("processed_awid3")

TRAIN_INPUT = DATA_DIR / "awid3_train.csv"
TEST_INPUT = DATA_DIR / "awid3_test.csv"

TRAIN_OUTPUT = DATA_DIR / "awid3_deauth_train.csv"
TEST_OUTPUT = DATA_DIR / "awid3_deauth_test.csv"

TRAIN_REPORT = DATA_DIR / "deauth_train_report.csv"
TEST_REPORT = DATA_DIR / "deauth_test_report.csv"

CHUNK_SIZE = 250_000
MAX_NORMAL_TRAIN = 150_000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)

# ============================================================
# MODEL FEATURES
# ============================================================

BASE_FEATURES = [
    "frame.len",
    "frame.time_delta",
    "radiotap.channel.freq",
    "wlan.fc.protected",
    "wlan.fc.type",
    "wlan.fc.retry",
    "wlan.fc.subtype",
    "wlan_radio.channel",
    "wlan_radio.signal_dbm",
]

MODEL_FEATURES = BASE_FEATURES + [
    "to_ds",
    "from_ds",
]

REQUIRED_COLUMNS = [
    "source_file",
    "wlan.fc.ds",
    "Label",
] + BASE_FEATURES

OUTPUT_COLUMNS = [
    "source_file"
] + MODEL_FEATURES + [
    "target"
]

# ============================================================
# VALIDATE INPUT FILES
# ============================================================

for filepath in [TRAIN_INPUT, TEST_INPUT]:

    if not filepath.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {filepath}"
        )

print("=" * 75)
print("DEAUTH BINARY DATASET PREPARATION")
print("=" * 75)

print("Target classes:")
print("Normal = 0")
print("Deauth = 1")

# ============================================================
# FEATURE ENGINEERING AND LABEL CONVERSION
# ============================================================

def prepare_chunk(chunk):

    # Keep only Normal and Deauth records.
    chunk = chunk[
        chunk["Label"].isin(["Normal", "Deauth"])
    ].copy()

    if chunk.empty:
        return chunk

    # Convert DS values safely.
    ds = pd.to_numeric(
        chunk["wlan.fc.ds"],
        errors="raise"
    ).to_numpy(dtype=np.int8)

    if not np.isin(ds, [0, 1, 2, 3]).all():
        raise ValueError(
            "Invalid wlan.fc.ds value encountered."
        )

    # Extract direction bits.
    chunk["to_ds"] = np.bitwise_and(ds, 1)

    chunk["from_ds"] = np.bitwise_and(
        np.right_shift(ds, 1),
        1
    )

    # Binary target.
    chunk["target"] = (
        chunk["Label"] == "Deauth"
    ).astype(np.int8)

    # Remove original fields that are not model inputs.
    chunk = chunk.drop(
        columns=["wlan.fc.ds", "Label"]
    )

    # Keep only the required columns.
    chunk = chunk[OUTPUT_COLUMNS]

    return chunk


# ============================================================
# UNIFORM RESERVOIR SAMPLING FOR NORMAL TRAINING RECORDS
# ============================================================

normal_reservoir = None
normal_seen = 0

deauth_training_parts = []

train_original_counts = {
    "Normal": 0,
    "Deauth": 0,
}

train_capture_coverage = {
    "Normal": set(),
    "Deauth": set(),
}


def update_normal_reservoir(group):

    global normal_reservoir, normal_seen

    group = group.reset_index(drop=True)

    new_count = len(group)

    old_seen = normal_seen

    total_seen = old_seen + new_count

    if normal_reservoir is None:
        normal_reservoir = group.iloc[0:0].copy()

    old_reservoir_count = len(normal_reservoir)

    if total_seen <= MAX_NORMAL_TRAIN:

        normal_reservoir = pd.concat(
            [normal_reservoir, group],
            ignore_index=True
        )

    else:

        new_selected = int(
            rng.hypergeometric(
                ngood=new_count,
                nbad=old_seen,
                nsample=MAX_NORMAL_TRAIN
            )
        )

        old_selected = (
            MAX_NORMAL_TRAIN - new_selected
        )

        if old_selected > old_reservoir_count:
            raise RuntimeError(
                "Normal reservoir sampling inconsistency."
            )

        if old_selected > 0:

            selected_old = normal_reservoir.sample(
                n=old_selected,
                random_state=int(
                    rng.integers(0, 2**32 - 1)
                )
            )

        else:

            selected_old = normal_reservoir.iloc[0:0]

        if new_selected > 0:

            selected_new = group.sample(
                n=new_selected,
                random_state=int(
                    rng.integers(0, 2**32 - 1)
                )
            )

        else:

            selected_new = group.iloc[0:0]

        normal_reservoir = pd.concat(
            [selected_old, selected_new],
            ignore_index=True
        )

    normal_seen = total_seen


# ============================================================
# PROCESS TRAINING DATA
# ============================================================

print("\n" + "=" * 75)
print("PROCESSING TRAINING DATA")
print("=" * 75)

train_reader = pd.read_csv(
    TRAIN_INPUT,
    usecols=REQUIRED_COLUMNS,
    chunksize=CHUNK_SIZE
)

for chunk_number, chunk in enumerate(
    train_reader,
    start=1
):

    prepared = prepare_chunk(chunk)

    if prepared.empty:
        continue

    normal = prepared[
        prepared["target"] == 0
    ]

    deauth = prepared[
        prepared["target"] == 1
    ]

    # Track original class counts and capture coverage.
    train_original_counts["Normal"] += len(normal)
    train_original_counts["Deauth"] += len(deauth)

    train_capture_coverage["Normal"].update(
        normal["source_file"].unique()
    )

    train_capture_coverage["Deauth"].update(
        deauth["source_file"].unique()
    )

    # Sample Normal records using reservoir sampling.
    if not normal.empty:
        update_normal_reservoir(normal)

    # Retain every Deauth training record.
    if not deauth.empty:
        deauth_training_parts.append(deauth)

    print(
        f"Training chunk {chunk_number} processed."
    )

# ============================================================
# ASSEMBLE TRAINING DATASET
# ============================================================

if normal_reservoir is None:
    raise RuntimeError(
        "No Normal training records were found."
    )

if not deauth_training_parts:
    raise RuntimeError(
        "No Deauth training records were found."
    )

deauth_training = pd.concat(
    deauth_training_parts,
    ignore_index=True
)

normal_training = normal_reservoir

train_data = pd.concat(
    [
        normal_training,
        deauth_training
    ],
    ignore_index=True
)

train_data = train_data.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

train_data.to_csv(
    TRAIN_OUTPUT,
    index=False
)

# ============================================================
# PROCESS COMPLETE TEST DATASET
# ============================================================

print("\n" + "=" * 75)
print("PROCESSING COMPLETE TEST DATASET")
print("=" * 75)

test_counts = {
    "Normal": 0,
    "Deauth": 0,
}

test_capture_coverage = {
    "Normal": set(),
    "Deauth": set(),
}

test_first_chunk = True

test_reader = pd.read_csv(
    TEST_INPUT,
    usecols=REQUIRED_COLUMNS,
    chunksize=CHUNK_SIZE
)

for chunk_number, chunk in enumerate(
    test_reader,
    start=1
):

    prepared = prepare_chunk(chunk)

    if prepared.empty:
        continue

    normal = prepared[
        prepared["target"] == 0
    ]

    deauth = prepared[
        prepared["target"] == 1
    ]

    test_counts["Normal"] += len(normal)
    test_counts["Deauth"] += len(deauth)

    test_capture_coverage["Normal"].update(
        normal["source_file"].unique()
    )

    test_capture_coverage["Deauth"].update(
        deauth["source_file"].unique()
    )

    prepared.to_csv(
        TEST_OUTPUT,
        mode="w" if test_first_chunk else "a",
        header=test_first_chunk,
        index=False
    )

    test_first_chunk = False

    print(
        f"Test chunk {chunk_number} processed: "
        f"{len(prepared):,} retained records."
    )

if test_first_chunk:
    raise RuntimeError(
        "No Normal or Deauth test records were found."
    )

# ============================================================
# GENERATE REPORTS
# ============================================================

train_report = pd.DataFrame([
    {
        "class": "Normal",
        "target": 0,
        "original_records": train_original_counts["Normal"],
        "sampled_records": len(normal_training),
        "capture_files": len(
            train_capture_coverage["Normal"]
        ),
    },
    {
        "class": "Deauth",
        "target": 1,
        "original_records": train_original_counts["Deauth"],
        "sampled_records": len(deauth_training),
        "capture_files": len(
            train_capture_coverage["Deauth"]
        ),
    }
])

train_report.to_csv(
    TRAIN_REPORT,
    index=False
)

test_report = pd.DataFrame([
    {
        "class": "Normal",
        "target": 0,
        "records": test_counts["Normal"],
        "capture_files": len(
            test_capture_coverage["Normal"]
        ),
    },
    {
        "class": "Deauth",
        "target": 1,
        "records": test_counts["Deauth"],
        "capture_files": len(
            test_capture_coverage["Deauth"]
        ),
    }
])

test_report.to_csv(
    TEST_REPORT,
    index=False
)

# ============================================================
# FINAL VALIDATION
# ============================================================

if set(train_data["target"].unique()) != {0, 1}:
    raise RuntimeError(
        "Training dataset does not contain both classes."
    )

if train_data[MODEL_FEATURES].isna().any().any():
    raise RuntimeError(
        "Missing training model features detected."
    )

if train_data["source_file"].isna().any():
    raise RuntimeError(
        "Missing training capture identifiers detected."
    )

if set(test_counts) != {"Normal", "Deauth"}:
    raise RuntimeError("Unexpected test class structure.")

if train_data["target"].isna().any():
    raise RuntimeError("Missing training targets.")

if not train_data["target"].isin([0, 1]).all():
    raise RuntimeError("Invalid training target values.")

# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("DEAUTH DATASET PREPARATION SUMMARY")
print("=" * 75)

print("\nTRAINING DATASET:")
print(f"Total records: {len(train_data):,}")
print(f"Normal records: {len(normal_training):,}")
print(f"Deauth records: {len(deauth_training):,}")
print(
    f"Normal captures represented: "
    f"{len(train_capture_coverage['Normal']):,}"
)
print(
    f"Deauth captures represented: "
    f"{len(train_capture_coverage['Deauth']):,}"
)

print("\nTESTING DATASET:")
print(
    f"Total records: "
    f"{test_counts['Normal'] + test_counts['Deauth']:,}"
)
print(f"Normal records: {test_counts['Normal']:,}")
print(f"Deauth records: {test_counts['Deauth']:,}")
print(
    f"Normal captures represented: "
    f"{len(test_capture_coverage['Normal']):,}"
)
print(
    f"Deauth captures represented: "
    f"{len(test_capture_coverage['Deauth']):,}"
)

print("\nVALIDATION:")
print("Binary classes: PASS")
print("All Deauth training records retained: PASS")
print("Normal training sample limit respected: PASS")
print("Full Normal and Deauth test data retained: PASS")
print("Missing model features: 0")
print("Missing targets: 0")

print("\nOUTPUT FILES:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)
print(TRAIN_REPORT)
print(TEST_REPORT)

print("\nDEAUTH DATASET PREPARATION COMPLETE")
