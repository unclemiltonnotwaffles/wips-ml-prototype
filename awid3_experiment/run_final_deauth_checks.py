r"""
Final validation and source-context audit for the AWID3 binary Deauthentication WIDS.

Run from the project root:
python run_final_deauth_checks.py --raw-dir "C:\path\to\AWID3\CSV"

Inputs expected:
  processed_awid3/awid3_deauth_train.csv
  processed_awid3/awid3_deauth_test.csv
  models/deauth_random_forest.joblib

Outputs:
  results/validation_threshold_sweep.csv
  results/final_test_metrics.csv
  results/false_positive_records.csv
  results/rogueap_original_context.csv
  results/rogueap_match_quality.csv

The final test set is never used to select a threshold. The validation split is
grouped by source_file. The raw RogueAP audit uses exact rounded feature
signatures and reports duplicate/ambiguous matches rather than pretending
that feature-only matches identify a unique source packet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

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
GROUP = "source_file"
CHUNK_SIZE = 250_000
RANDOM_STATE = 42
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

RAW_CONTEXT_CANDIDATES = [
    "frame.number", "frame.time", "frame.time_epoch",
    "wlan.sa", "wlan.ta", "wlan.da", "wlan.ra",
    "wlan.seq", "wlan.frag", "wlan.fixed.reason_code",
    "wlan.fixed.reason_code_tree", "radiotap.dbm_antsignal",
    "Label",
]

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

def metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "records": int(len(y_true)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "deauth_precision": float(precision_score(y_true, pred, zero_division=0)),
        "deauth_recall": float(recall_score(y_true, pred, zero_division=0)),
        "deauth_f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
    }

def validate_training_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(FEATURES + [TARGET, GROUP])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype("int8")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Training data must contain target classes 0 and 1.")
    if df[FEATURES].isna().any().any():
        raise ValueError("Training data contains missing model features.")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=np.float64)).all():
        raise ValueError("Training data contains non-finite model features.")
    if df[GROUP].isna().any():
        raise ValueError("Missing source_file group values.")
    return df

def run_grouped_validation(train_path: Path, results_dir: Path) -> None:
    print("\n[1/4] Capture-grouped validation threshold sweep")
    df = validate_training_data(train_path)
    X = df[FEATURES]
    y = df[TARGET].to_numpy(dtype=np.int8)
    groups = df[GROUP].astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=500, test_size=0.20, random_state=RANDOM_STATE)
    chosen = None
    for tr_idx, va_idx in splitter.split(X, y, groups):
        if len(np.unique(y[tr_idx])) == 2 and len(np.unique(y[va_idx])) == 2:
            # Prefer validation containing at least two independent Deauth captures.
            val_groups = df.iloc[va_idx].loc[df.iloc[va_idx][TARGET] == 1, GROUP].nunique()
            if val_groups >= 2:
                chosen = (tr_idx, va_idx)
                break
    if chosen is None:
        raise RuntimeError(
            "Could not create a valid grouped validation split with both classes "
            "and at least two Deauth capture groups. Do not fall back to a row-random split."
        )

    tr_idx, va_idx = chosen
    train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
    print(f"Training rows: {len(train_df):,}; validation rows: {len(val_df):,}")
    print(f"Training capture groups: {train_df[GROUP].nunique()}; "
          f"validation capture groups: {val_df[GROUP].nunique()}")
    print("Validation Deauth capture groups:",
          val_df.loc[val_df[TARGET] == 1, GROUP].nunique())

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df[TARGET].astype(int))
    scores = model.predict_proba(val_df[FEATURES])[:, list(model.classes_).index(1)]
    y_val = val_df[TARGET].to_numpy(dtype=np.int8)

    rows = []
    for threshold in THRESHOLDS:
        row = metric_row(y_val, scores, threshold)
        row["validation_capture_groups"] = int(val_df[GROUP].nunique())
        row["validation_deauth_capture_groups"] = int(
            val_df.loc[val_df[TARGET] == 1, GROUP].nunique()
        )
        rows.append(row)

    out = results_dir / "validation_threshold_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved:", out)
    print(pd.DataFrame(rows)[[
        "threshold", "deauth_precision", "deauth_recall",
        "deauth_f1", "false_positives", "false_positive_rate",
    ]].to_string(index=False))
    print("Thresholds are reported for review; none is automatically promoted.")

def evaluate_final_test(test_path: Path, model_path: Path, results_dir: Path) -> pd.DataFrame:
    print("\n[2/4] Final untouched test evaluation at baseline threshold 0.50")
    model = joblib.load(model_path)
    required = FEATURES + [TARGET, GROUP]
    y_parts, score_parts = [], []
    fp_parts = []
    total = 0

    for number, chunk in enumerate(
        pd.read_csv(test_path, usecols=required, chunksize=CHUNK_SIZE), start=1
    ):
        X = chunk[FEATURES]
        if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Invalid feature values in test chunk {number}.")
        y = chunk[TARGET].to_numpy(dtype=np.int8)
        scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
        pred = (scores >= 0.50).astype(np.int8)
        fp_mask = (y == 0) & (pred == 1)

        if fp_mask.any():
            fp = chunk.loc[fp_mask, [GROUP] + FEATURES].copy()
            fp["predicted_probability"] = scores[fp_mask]
            fp["threshold"] = 0.50
            fp_parts.append(fp)

        y_parts.append(y)
        score_parts.append(scores.astype(np.float32))
        total += len(chunk)
        print(f"Test chunk {number}: {total:,} records")

    y_true = np.concatenate(y_parts)
    scores = np.concatenate(score_parts)
    metrics = metric_row(y_true, scores, 0.50)
    metrics["evaluation_partition"] = "untouched_capture_grouped_test"
    metrics["threshold_source"] = "fixed_baseline_0.50_not_tuned_on_test"
    metrics_df = pd.DataFrame([metrics])
    metrics_path = results_dir / "final_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fp_df = pd.concat(fp_parts, ignore_index=True) if fp_parts else pd.DataFrame(
        columns=[GROUP] + FEATURES + ["predicted_probability", "threshold"]
    )
    fp_path = results_dir / "false_positive_records.csv"
    fp_df.to_csv(fp_path, index=False)

    print("Saved:", metrics_path)
    print("Saved:", fp_path)
    print(metrics_df.to_string(index=False))
    print(f"False-positive records exported: {len(fp_df):,}")
    return fp_df

def parse_ds(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    is_hex = s.str.match(r"(?i)^0x[0-9a-f]+$", na=False)
    out = pd.to_numeric(s.where(~is_hex), errors="coerce")
    if is_hex.any():
        out.loc[is_hex] = s.loc[is_hex].map(lambda v: int(v, 16))
    return out

def rounded_key_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        values = pd.to_numeric(df[c], errors="coerce")
        out[c] = values.round(6).map(lambda x: "NA" if pd.isna(x) else f"{x:.6f}")
    return out

def audit_rogueap_raw(raw_dir: Path, fp_df: pd.DataFrame, results_dir: Path) -> None:
    print("\n[3/4] Matching RogueAP false positives to original capture rows")
    targets = {"RogueAP_36.csv", "RogueAP_37.csv"}
    raw_files = {p.name: p for p in raw_dir.rglob("*.csv") if p.name in targets}
    missing = targets - set(raw_files)
    if missing:
        raise FileNotFoundError(f"Could not locate original capture files: {sorted(missing)}")

    rogue_fp = fp_df[fp_df[GROUP].isin(targets)].copy()
    if rogue_fp.empty:
        print("No RogueAP false positives in this run; raw matching skipped.")
        return

    raw_feature_cols = [
        "frame.len", "frame.time_delta", "radiotap.channel.freq",
        "wlan.fc.protected", "wlan.fc.type", "wlan.fc.retry",
        "wlan.fc.subtype", "wlan_radio.channel", "wlan_radio.signal_dbm",
    ]
    required_raw = raw_feature_cols + ["wlan.fc.ds"]
    context_found = []
    candidate_frames = []
    quality_rows = []

    # Construct deterministic feature signatures. The DS direction bits are
    # derived from the original wlan.fc.ds field, exactly as in preprocessing.
    fp_keys = rounded_key_frame(rogue_fp, FEATURES)
    fp_keys.insert(0, GROUP, rogue_fp[GROUP].values)
    fp_keys["signature_id"] = (
        fp_keys[GROUP].astype(str) + "|" +
        fp_keys[FEATURES].astype(str).agg("|".join, axis=1)
    )
    fp_lookup = fp_keys[[GROUP, "signature_id"]].copy()

    for filename, path in raw_files.items():
        header = pd.read_csv(path, nrows=0).columns.tolist()
        absent = [c for c in required_raw if c not in header]
        if absent:
            raise KeyError(f"{filename} missing raw feature columns: {absent}")
        context = [c for c in RAW_CONTEXT_CANDIDATES if c in header]
        context_found.extend(context)

        keys_for_file = fp_keys.loc[fp_keys[GROUP] == filename].copy()
        key_set = set(keys_for_file["signature_id"])
        matched_count = 0
        candidate_count = 0

        for chunk_no, raw in enumerate(pd.read_csv(
            path, chunksize=50_000, low_memory=False, dtype_backend="numpy_nullable"
        ), start=1):
            raw = raw.copy()
            raw["wlan.fc.ds"] = parse_ds(raw["wlan.fc.ds"])
            raw["to_ds"] = np.bitwise_and(raw["wlan.fc.ds"].fillna(-1).astype("int64"), 1)
            raw["from_ds"] = np.bitwise_and(
                np.right_shift(raw["wlan.fc.ds"].fillna(-1).astype("int64"), 1), 1
            )
            for c in raw_feature_cols:
                raw[c] = pd.to_numeric(raw[c], errors="coerce")
            raw[FEATURES] = raw[FEATURES].apply(pd.to_numeric, errors="coerce")
            raw_keys = rounded_key_frame(raw, FEATURES)
            raw_keys.insert(0, GROUP, filename)
            raw_keys["signature_id"] = (
                raw_keys[GROUP].astype(str) + "|" +
                raw_keys[FEATURES].astype(str).agg("|".join, axis=1)
            )
            mask = raw_keys["signature_id"].isin(key_set)
            if not mask.any():
                continue

            hits = raw.loc[mask, context + raw_feature_cols + ["wlan.fc.ds", "to_ds", "from_ds"]].copy()
            hits.insert(0, GROUP, filename)
            hits.insert(1, "raw_chunk", chunk_no)
            hits.insert(2, "raw_chunk_row", hits.index)
            hits["signature_id"] = raw_keys.loc[mask, "signature_id"].values
            candidate_frames.append(hits)
            matched_count += hits["signature_id"].nunique()
            candidate_count += len(hits)

        quality_rows.append({
            "source_file": filename,
            "false_positive_signatures": int(len(key_set)),
            "matched_signatures": int(matched_count),
            "unmatched_signatures": int(len(key_set) - matched_count),
            "raw_candidate_rows": int(candidate_count),
            "ambiguous_signatures": 0,
        })

    if candidate_frames:
        candidates = pd.concat(candidate_frames, ignore_index=True)
        multiplicity = candidates.groupby("signature_id").size().rename("candidate_rows_for_signature")
        candidates = candidates.join(multiplicity, on="signature_id")
        candidates["match_is_ambiguous"] = candidates["candidate_rows_for_signature"] > 1
        candidates.to_csv(results_dir / "rogueap_original_context.csv", index=False)
        ambiguity = candidates.groupby("source_file").agg(
            candidate_rows=("signature_id", "size"),
            ambiguous_candidate_rows=("match_is_ambiguous", "sum"),
            unique_signatures=("signature_id", "nunique"),
        ).reset_index()
        quality = pd.DataFrame(quality_rows).drop(columns=["ambiguous_signatures"]).merge(
            ambiguity, on="source_file", how="left"
        )
    else:
        quality = pd.DataFrame(quality_rows)
        quality["candidate_rows"] = 0
        quality["ambiguous_candidate_rows"] = 0
        quality["unique_signatures"] = 0

    quality.to_csv(results_dir / "rogueap_match_quality.csv", index=False)
    print("Original context fields found:", sorted(set(context_found)))
    print("Saved:", results_dir / "rogueap_original_context.csv")
    print("Saved:", results_dir / "rogueap_match_quality.csv")
    print(quality.to_string(index=False))
    print("Important: repeated feature signatures can map to multiple original packets. "
          "Use frame/time/address fields and the ambiguity report before drawing conclusions.")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path,
                        help="Root folder containing the original AWID3 CSV captures.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd(),
                        help="Project root; defaults to the current directory.")
    args = parser.parse_args()

    root = args.project_dir.resolve()
    data_dir = root / "processed_awid3"
    results_dir = root / "results"
    model_dir = root / "models"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "awid3_deauth_train.csv"
    test_path = data_dir / "awid3_deauth_test.csv"
    model_path = model_dir / "deauth_random_forest.joblib"
    for path in (train_path, test_path, model_path):
        require_file(path)
    if not args.raw_dir.is_dir():
        raise NotADirectoryError(args.raw_dir)

    print("=" * 78)
    print("FINAL AWID3 DEAUTHENTICATION WIDS VALIDATION")
    print("=" * 78)
    run_grouped_validation(train_path, results_dir)
    fp_df = evaluate_final_test(test_path, model_path, results_dir)
    audit_rogueap_raw(args.raw_dir.resolve(), fp_df, results_dir)

    manifest = {
        "experiment": "AWID3 binary Normal-vs-Deauth packet classifier",
        "test_threshold": 0.50,
        "test_threshold_tuned_on_test": False,
        "raw_dataset_directory": str(args.raw_dir.resolve()),
        "outputs": [
            "validation_threshold_sweep.csv",
            "final_test_metrics.csv",
            "false_positive_records.csv",
            "rogueap_original_context.csv",
            "rogueap_match_quality.csv",
        ],
    }
    (results_dir / "final_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\n[4/4] COMPLETE")
    print("Manifest:", results_dir / "final_validation_manifest.json")
    print("Do not change the final-test threshold based on these test metrics.")

if __name__ == "__main__":
    main()
