"""Train and evaluate the WIPS Random Forest on time-separated windows."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from . import config
from .features import FEATURE_COLUMNS, build_window_features, train_test_split_time_ordered


def train_model(df, verbose: bool = True) -> dict:
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    features = build_window_features(df, config.WINDOW_SECONDS, config.DEAUTH_FLOOD_MIN)
    train, test = train_test_split_time_ordered(features, config.TEST_FRACTION)
    y_train, y_test = train["label"].astype(int), test["label"].astype(int)
    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training partition must contain both normal (0) and attack (1) windows. "
                         "Use a dataset with both classes represented before the split.")
    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary (0=normal, 1=attack)")
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Feature matrix contains missing values; refusing to train")

    clf = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS, max_depth=config.MAX_DEPTH,
        class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    metrics = {
        "evaluation": "chronological_holdout_by_global_window",
        "test_fraction": config.TEST_FRACTION,
        "window_seconds": config.WINDOW_SECONDS,
        "feature_columns": FEATURE_COLUMNS,
        "n_windows": int(len(features)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_labels": ["normal", "attack"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "feature_importances": dict(zip(FEATURE_COLUMNS,
                                        clf.feature_importances_.round(6).tolist())),
        "random_state": config.RANDOM_STATE,
        "model": "RandomForestClassifier",
    }
    if verbose:
        print("[train] chronological holdout report")
        print(classification_report(y_test, predictions, labels=[0, 1],
                                    target_names=["normal", "attack"], zero_division=0))
        print(f"[train] accuracy={metrics['accuracy']:.4f} "
              f"precision={metrics['precision']:.4f} "
              f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
    _save_confusion_matrix(y_test, predictions, config.CONFUSION_MATRIX_PATH)
    joblib.dump(clf, config.MODEL_PATH)
    Path(config.METRICS_PATH).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _save_confusion_matrix(y_true, y_pred, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["normal", "attack"])
    ax.set_yticks([0, 1], labels=["normal", "attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — chronological holdout")
    threshold = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
