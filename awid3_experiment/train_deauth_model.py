
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("processed_awid3")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

TRAIN_FILE = DATA_DIR / "awid3_deauth_train.csv"
TEST_FILE = DATA_DIR / "awid3_deauth_test.csv"

MODEL_FILE = MODEL_DIR / "deauth_random_forest.joblib"
METRICS_FILE = RESULTS_DIR / "deauth_baseline_metrics.csv"
REPORT_FILE = RESULTS_DIR / "deauth_classification_report.csv"
IMPORTANCE_FILE = RESULTS_DIR / "deauth_feature_importance.csv"

CHUNK_SIZE = 250_000

RANDOM_STATE = 42

# ============================================================
# MODEL FEATURES
# ============================================================

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

TARGET = "target"

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# VERIFY INPUT FILES
# ============================================================

for filepath in [TRAIN_FILE, TEST_FILE]:

    if not filepath.exists():
        raise FileNotFoundError(filepath)

print("=" * 75)
print("DEAUTH RANDOM FOREST TRAINING")
print("=" * 75)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_data = pd.read_csv(TRAIN_FILE)

X_train = train_data[FEATURES]
y_train = train_data[TARGET].astype("int8")

if set(y_train.unique()) != {0, 1}:
    raise ValueError(
        "Training dataset must contain both classes."
    )

if X_train.isna().any().any():
    raise ValueError(
        "Missing training feature values detected."
    )

if not np.isfinite(
    X_train.to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "Non-finite training feature values detected."
    )

print(f"Training records: {len(train_data):,}")
print(f"Normal records: {(y_train == 0).sum():,}")
print(f"Deauth records: {(y_train == 1).sum():,}")

# ============================================================
# INITIALISE RANDOM FOREST
# ============================================================

print("\nInitialising Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETE")

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE,
    compress=3
)

print(f"\nModel saved: {MODEL_FILE}")

# ============================================================
# EVALUATE COMPLETE TEST DATASET IN CHUNKS
# ============================================================

print("\n" + "=" * 75)
print("EVALUATING COMPLETE TEST DATASET")
print("=" * 75)

y_true_parts = []
y_score_parts = []

total_test_records = 0

test_reader = pd.read_csv(
    TEST_FILE,
    usecols=FEATURES + [TARGET],
    chunksize=CHUNK_SIZE
)

for chunk_number, chunk in enumerate(
    test_reader,
    start=1
):

    X_test_chunk = chunk[FEATURES]

    y_test_chunk = chunk[TARGET].to_numpy(
        dtype=np.int8
    )

    if X_test_chunk.isna().any().any():
        raise ValueError(
            f"Missing features in test chunk {chunk_number}"
        )

    # Obtain probability of Deauth class.
    probabilities = model.predict_proba(
        X_test_chunk
    )[:, 1]

    y_true_parts.append(y_test_chunk)

    y_score_parts.append(
        probabilities.astype(np.float32)
    )

    total_test_records += len(chunk)

    print(
        f"Evaluated chunk {chunk_number}: "
        f"{total_test_records:,} records processed."
    )

# ============================================================
# COMBINE PREDICTIONS
# ============================================================

y_true = np.concatenate(y_true_parts)

y_scores = np.concatenate(y_score_parts)

# Initial operating threshold.
THRESHOLD = 0.50

y_pred = (
    y_scores >= THRESHOLD
).astype(np.int8)

# ============================================================
# CALCULATE METRICS
# ============================================================

print("\nCalculating evaluation metrics...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Deauth"],
    output_dict=True,
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

average_precision = average_precision_score(
    y_true,
    y_scores
)

roc_auc = roc_auc_score(
    y_true,
    y_scores
)

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([
    {
        "metric": "test_records",
        "value": total_test_records
    },
    {
        "metric": "threshold",
        "value": THRESHOLD
    },
    {
        "metric": "true_negatives",
        "value": tn
    },
    {
        "metric": "false_positives",
        "value": fp
    },
    {
        "metric": "false_negatives",
        "value": fn
    },
    {
        "metric": "true_positives",
        "value": tp
    },
    {
        "metric": "balanced_accuracy",
        "value": balanced_accuracy
    },
    {
        "metric": "deauth_precision",
        "value": report["Deauth"]["precision"]
    },
    {
        "metric": "deauth_recall",
        "value": report["Deauth"]["recall"]
    },
    {
        "metric": "deauth_f1",
        "value": report["Deauth"]["f1-score"]
    },
    {
        "metric": "false_positive_rate",
        "value": false_positive_rate
    },
    {
        "metric": "average_precision",
        "value": average_precision
    },
    {
        "metric": "roc_auc",
        "value": roc_auc
    }
])

metrics.to_csv(
    METRICS_FILE,
    index=False
)

# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    REPORT_FILE
)

# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("DEAUTH CLASSIFICATION RESULTS")
print("=" * 75)

print("\nCONFUSION MATRIX:")
print(cm)

print("\nInterpretation:")
print(f"True Negatives:  {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives:  {tp:,}")

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Deauth"],
        zero_division=0
    )
)

print(f"Balanced Accuracy: {balanced_accuracy:.6f}")
print(f"Average Precision: {average_precision:.6f}")
print(f"ROC AUC: {roc_auc:.6f}")
print(f"False Positive Rate: {false_positive_rate:.8f}")

print("\nFEATURE IMPORTANCE:")

print(importance.to_string(index=False))

print("\nOUTPUT FILES:")
print(MODEL_FILE)
print(METRICS_FILE)
print(REPORT_FILE)
print(IMPORTANCE_FILE)

print("\nDEAUTH MODEL TRAINING AND EVALUATION COMPLETE")
