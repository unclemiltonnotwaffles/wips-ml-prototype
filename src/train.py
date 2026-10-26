"""
train.py — trains the Random Forest, evaluates it, saves the artefacts.

Called by run_demo.py and run_awid3.py. Kept import-friendly so tests can
reuse `train_model()` without spawning a process.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")            # headless: we only save a PNG
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from . import config
from .features import (
    FEATURE_COLUMNS, build_window_features, train_test_split_time_ordered,
)

def train_model(df, verbose: bool = True) -> dict:
    """Full pipeline: windows → split → train → evaluate → save."""
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # ── 1. Features ──────────────────────────────────────────────────────────
    feats = build_window_features(
        df,
        window_seconds=config.WINDOW_SECONDS,
        deauth_flood_min=config.DEAUTH_FLOOD_MIN,
    )
    if verbose:
        print(f"[train] {len(feats):,} windows, "
              f"{int(feats['label'].sum()):,} labelled attack")

    train, test = train_test_split_time_ordered(feats, config.TEST_FRACTION)
    X_train, y_train = train[FEATURE_COLUMNS].values, train["label"].values
    X_test,  y_test  = test[FEATURE_COLUMNS].values,  test["label"].values

    # ── 2. Model ─────────────────────────────────────────────────────────────
    # WHY RandomForest?
    #   • No feature scaling needed (trees split on thresholds).
    #   • Robust to heavy-tailed counts (frame_count, deauth_count).
    #   • class_weight="balanced" handles 1:20 attack:normal imbalance.
    #   • feature_importances_ lets us explain the model in the viva.
    clf = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "attack_windows_test": int(y_test.sum()),
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if verbose:
        print("\n[train] classification report")
        print(classification_report(y_test, y_pred,
                                    labels=[0, 1],
                                    target_names=["normal", "attack"],
                                    zero_division=0))

        print(f"[train] accuracy = {metrics['accuracy']:.4f}")

    # ── 4. Feature importances ───────────────────────────────────────────────
    importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.round(4).tolist()))
    metrics["feature_importances"] = importances
    if verbose:
        print("[train] top features:")
        for k, v in sorted(importances.items(), key=lambda kv: -kv[1])[:5]:
            print(f"        {k:<15} {v:.3f}")

    # ── 5. Confusion matrix picture ──────────────────────────────────────────
    _save_confusion_matrix(y_test, y_pred, config.CONFUSION_MATRIX_PATH)

    # ── 6. Persist ───────────────────────────────────────────────────────────
    joblib.dump(clf, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"[train] saved model → {config.MODEL_PATH}")
        print(f"[train] saved metrics → {config.METRICS_PATH}")
        print(f"[train] saved confusion matrix → {config.CONFUSION_MATRIX_PATH}")

    return metrics

def _save_confusion_matrix(y_true, y_pred, path: str) -> None:
    """Write a clean, poster-ready confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "attack"])
    ax.set_yticklabels(["normal", "attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (held-out windows)")

    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
