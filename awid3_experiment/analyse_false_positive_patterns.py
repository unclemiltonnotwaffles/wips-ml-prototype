
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = Path("processed_awid3/awid3_deauth_test.csv")
FP_PATH = Path("results/false_positive_records.csv")
NORMAL_SAMPLE_PATH = Path("results/normal_comparison_sample.csv")

OUTPUT_DIR = Path("results")
CHUNK_SIZE = 250_000

print("=" * 75)
print("FALSE-POSITIVE PATTERN ANALYSIS")
print("=" * 75)


# ============================================================
# LOAD INVESTIGATION RESULTS
# ============================================================

print("\nLoading false-positive records...")

fp = pd.read_csv(FP_PATH)
normal_sample = pd.read_csv(NORMAL_SAMPLE_PATH)

print(f"False positives loaded: {len(fp):,}")
print(f"Normal comparison sample: {len(normal_sample):,}")


# ============================================================
# COUNT NORMAL RECORDS PER CAPTURE IN COMPLETE TEST SET
# ============================================================

print("\nCounting actual Normal records per capture...")

normal_counts = {}

for chunk_number, chunk in enumerate(
    pd.read_csv(
        TEST_PATH,
        usecols=["source_file", "target"],
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    normal = chunk[chunk["target"] == 0]

    counts = normal["source_file"].value_counts()

    for filename, count in counts.items():
        normal_counts[filename] = (
            normal_counts.get(filename, 0) + int(count)
        )

    print(f"Processed test chunk {chunk_number}")


normal_counts_df = pd.DataFrame(
    list(normal_counts.items()),
    columns=["source_file", "normal_records"]
)


# ============================================================
# FALSE-POSITIVE RATE PER CAPTURE
# ============================================================

fp_counts = (
    fp.groupby("source_file")
    .size()
    .reset_index(name="false_positives")
)

capture_analysis = normal_counts_df.merge(
    fp_counts,
    on="source_file",
    how="left"
)

capture_analysis["false_positives"] = (
    capture_analysis["false_positives"]
    .fillna(0)
    .astype(int)
)

capture_analysis["false_positive_rate"] = (
    capture_analysis["false_positives"]
    / capture_analysis["normal_records"]
)

capture_analysis["false_positive_percentage"] = (
    capture_analysis["false_positive_rate"] * 100
)

capture_analysis = capture_analysis.sort_values(
    "false_positive_rate",
    ascending=False
)

capture_analysis.to_csv(
    OUTPUT_DIR / "false_positive_capture_rates.csv",
    index=False
)


# ============================================================
# FRAME TYPE AND SUBTYPE DISTRIBUTIONS
# ============================================================

print("\nAnalysing frame type and subtype patterns...")

features = [
    "wlan.fc.type",
    "wlan.fc.subtype",
]

for feature in features:

    fp_distribution = (
        fp[feature]
        .value_counts(normalize=True)
        .mul(100)
        .rename("false_positive_percentage")
    )

    normal_distribution = (
        normal_sample[feature]
        .value_counts(normalize=True)
        .mul(100)
        .rename("normal_sample_percentage")
    )

    comparison = pd.concat(
        [fp_distribution, normal_distribution],
        axis=1
    ).fillna(0)

    comparison["percentage_point_difference"] = (
        comparison["false_positive_percentage"]
        - comparison["normal_sample_percentage"]
    )

    comparison = comparison.sort_values(
        "false_positive_percentage",
        ascending=False
    )

    output_name = feature.replace(".", "_")

    comparison.to_csv(
        OUTPUT_DIR / f"false_positive_{output_name}_distribution.csv"
    )

    print(f"\n{feature.upper()} DISTRIBUTION:")
    print(comparison.round(3).to_string())


# ============================================================
# COMBINED FRAME TYPE / SUBTYPE COMPARISON
# ============================================================

fp_pairs = (
    fp.groupby(["wlan.fc.type", "wlan.fc.subtype"])
    .size()
    .reset_index(name="false_positive_count")
)

normal_pairs = (
    normal_sample.groupby(["wlan.fc.type", "wlan.fc.subtype"])
    .size()
    .reset_index(name="normal_sample_count")
)

pair_comparison = fp_pairs.merge(
    normal_pairs,
    on=["wlan.fc.type", "wlan.fc.subtype"],
    how="outer"
).fillna(0)

pair_comparison["false_positive_percentage"] = (
    pair_comparison["false_positive_count"]
    / len(fp) * 100
)

pair_comparison["normal_sample_percentage"] = (
    pair_comparison["normal_sample_count"]
    / len(normal_sample) * 100
)

pair_comparison["percentage_point_difference"] = (
    pair_comparison["false_positive_percentage"]
    - pair_comparison["normal_sample_percentage"]
)

pair_comparison = pair_comparison.sort_values(
    "false_positive_count",
    ascending=False
)

pair_comparison.to_csv(
    OUTPUT_DIR / "false_positive_type_subtype_comparison.csv",
    index=False
)


# ============================================================
# FOCUS ON ROGUEAP RECORDS
# ============================================================

rogue_fp = fp[
    fp["source_file"].isin([
        "RogueAP_36.csv",
        "RogueAP_37.csv"
    ])
]

print("\nROGUEAP FALSE-POSITIVE SUMMARY:")

print(f"RogueAP false positives: {len(rogue_fp):,}")

if len(rogue_fp) > 0:

    print("\nRogueAP frame type/subtype combinations:")

    print(
        rogue_fp.groupby(
            ["wlan.fc.type", "wlan.fc.subtype"]
        )
        .size()
        .sort_values(ascending=False)
        .head(20)
        .to_string()
    )

    print("\nRogueAP feature statistics:")

    print(
        rogue_fp[
            [
                "frame.len",
                "frame.time_delta",
                "wlan.fc.type",
                "wlan.fc.subtype",
                "wlan.fc.protected",
                "wlan.fc.retry",
                "wlan_radio.signal_dbm",
            ]
        ]
        .describe()
        .round(4)
        .to_string()
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 75)
print("TOP 15 CAPTURES BY FALSE-POSITIVE RATE")
print("=" * 75)

print(
    capture_analysis[
        [
            "source_file",
            "normal_records",
            "false_positives",
            "false_positive_percentage",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nOUTPUT FILES:")
print("results/false_positive_capture_rates.csv")
print("results/false_positive_wlan_fc_type_distribution.csv")
print("results/false_positive_wlan_fc_subtype_distribution.csv")
print("results/false_positive_type_subtype_comparison.csv")

print("\nPATTERN ANALYSIS COMPLETE")
