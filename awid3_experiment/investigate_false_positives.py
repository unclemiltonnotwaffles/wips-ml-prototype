
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("processed_awid3/awid3_deauth_test.csv")
MODEL_PATH = Path("models/deauth_random_forest.joblib")
OUTPUT_DIR = Path("results")

CHUNK_SIZE = 250_000
THRESHOLD = 0.50
NORMAL_SAMPLE_PER_CHUNK = 1_000

FEATURES = [
    "frame.len",
    "frame.time_delta",
    "radiotap.channel.freq",
    "wlan.fc.protected",
    "wlan.fc.type",
    "wlan.fc.retry",
    "wlan.fc.subtype",
    "wlan_radio.channel",
    "wlan_radio.signal_dbm",
    "to_ds",
    "from_ds",
]

REQUIRED_COLUMNS = ["source_file", "target"] + FEATURES


# ============================================================
# INITIALISATION
# ============================================================

print("=" * 75)
print("DEAUTH FALSE-POSITIVE INVESTIGATION")
print("=" * 75)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading trained model...")
model = joblib.load(MODEL_PATH)

false_positive_chunks = []
normal_sample_chunks = []

total_records = 0
total_false_positives = 0
total_correct_normals = 0
chunk_number = 0

print("Model loaded.")
print(f"Classification threshold: {THRESHOLD}")
print("\nReading test dataset in chunks...")


# ============================================================
# PROCESS TEST DATA
# ============================================================

for chunk in pd.read_csv(
    DATASET_PATH,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1
    total_records += len(chunk)

    missing = set(REQUIRED_COLUMNS) - set(chunk.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    X = chunk[FEATURES]
    y = chunk["target"].astype(int)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # False positive:
    # Actual Normal (0), predicted Deauth (1)
    fp_mask = (y.to_numpy() == 0) & (predictions == 1)

    # Correctly classified Normal records
    tn_mask = (y.to_numpy() == 0) & (predictions == 0)

    if fp_mask.any():

        fp = chunk.loc[fp_mask, REQUIRED_COLUMNS].copy()

        fp["predicted_probability"] = probabilities[fp_mask]
        fp["predicted_class"] = 1
        fp["record_type"] = "False Positive"

        false_positive_chunks.append(fp)

        total_false_positives += len(fp)

    if tn_mask.any():

        tn = chunk.loc[tn_mask, REQUIRED_COLUMNS].copy()

        sample_size = min(
            NORMAL_SAMPLE_PER_CHUNK,
            len(tn)
        )

        if sample_size > 0:

            sample = tn.sample(
                n=sample_size,
                random_state=42 + chunk_number
            ).copy()

            sample["predicted_probability"] = probabilities[
                tn_mask
            ][sample.index.to_numpy() - chunk.index.to_numpy()[0]] if False else 0.0

            # Store the actual model probability for each sampled row.
            probability_series = pd.Series(
                probabilities[tn_mask],
                index=tn.index
            )

            sample["predicted_probability"] = probability_series.loc[
                sample.index
            ]

            sample["predicted_class"] = 0
            sample["record_type"] = "Correct Normal"

            normal_sample_chunks.append(sample)

        total_correct_normals += int(tn_mask.sum())

    print(
        f"Processed chunk {chunk_number}: "
        f"{total_records:,} records | "
        f"{total_false_positives:,} false positives"
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

if not false_positive_chunks:
    raise RuntimeError("No false positives were found.")

false_positives = pd.concat(
    false_positive_chunks,
    ignore_index=True
)

normal_sample = pd.concat(
    normal_sample_chunks,
    ignore_index=True
)

print("\nProcessing complete.")

print(f"Total test records: {total_records:,}")
print(f"False positives: {len(false_positives):,}")
print(f"Correctly classified Normal records: {total_correct_normals:,}")
print(f"Normal comparison sample: {len(normal_sample):,}")


# ============================================================
# 1. SAVE FALSE-POSITIVE RECORDS
# ============================================================

fp_path = OUTPUT_DIR / "false_positive_records.csv"

false_positives.to_csv(
    fp_path,
    index=False
)


# ============================================================
# 2. FALSE POSITIVES BY CAPTURE FILE
# ============================================================

capture_summary = (
    false_positives
    .groupby("source_file")
    .agg(
        false_positive_count=("target", "size"),
        mean_deauth_probability=("predicted_probability", "mean"),
        maximum_deauth_probability=("predicted_probability", "max"),
        minimum_deauth_probability=("predicted_probability", "min"),
    )
    .sort_values(
        "false_positive_count",
        ascending=False
    )
    .reset_index()
)

capture_path = OUTPUT_DIR / "false_positive_by_capture.csv"

capture_summary.to_csv(
    capture_path,
    index=False
)


# ============================================================
# 3. COMPARE FALSE POSITIVES AGAINST NORMAL RECORDS
# ============================================================

comparison = pd.concat(
    [false_positives, normal_sample],
    ignore_index=True
)

summary_rows = []

for record_type, group in comparison.groupby("record_type"):

    for feature in FEATURES:

        summary_rows.append({
            "record_type": record_type,
            "feature": feature,
            "count": group[feature].count(),
            "mean": group[feature].mean(),
            "median": group[feature].median(),
            "std": group[feature].std(),
            "minimum": group[feature].min(),
            "maximum": group[feature].max(),
        })

feature_summary = pd.DataFrame(summary_rows)

feature_path = OUTPUT_DIR / "false_positive_feature_summary.csv"

feature_summary.to_csv(
    feature_path,
    index=False
)


# ============================================================
# 4. SAVE COMPARISON SAMPLE
# ============================================================

sample_path = OUTPUT_DIR / "normal_comparison_sample.csv"

normal_sample.to_csv(
    sample_path,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("FALSE-POSITIVE INVESTIGATION COMPLETE")
print("=" * 75)

print("\nTOP 15 CAPTURE FILES BY FALSE-POSITIVE COUNT:")

print(
    capture_summary.head(15).to_string(index=False)
)

print("\nFALSE-POSITIVE PROBABILITY DISTRIBUTION:")

print(
    false_positives["predicted_probability"].describe().round(5)
)

print("\nOUTPUT FILES:")

print(fp_path)
print(capture_path)
print(feature_path)
print(sample_path)

print("\nInvestigation data successfully generated.")
