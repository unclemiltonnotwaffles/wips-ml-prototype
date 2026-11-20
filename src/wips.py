"""Offline WIPS prototype detector.

This module scores precomputed feature windows and writes alerts plus a
persistent *proposed* MAC blocklist. It does not control an access point,
wireless controller, or other clients' traffic. Enforcement must be supplied
by a separately reviewed network-management integration.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS

_MAC_RE = re.compile(r"^(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$")


class WIPSDetector:
    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        blocklist_path: str = config.BLOCKLIST_PATH,
        alerts_path: str = config.ALERTS_PATH,
        block_after_alerts: int = config.BLOCK_AFTER_ALERTS,
        dry_run: bool = True,
        model=None,
    ):
        if block_after_alerts < 1:
            raise ValueError("block_after_alerts must be >= 1")
        self.model = model if model is not None else joblib.load(model_path)
        self.block_after_alerts = int(block_after_alerts)
        # Retained for compatibility with older callers. No host firewall
        # commands are executed by this offline CSV prototype.
        self.dry_run = bool(dry_run)
        self.blocklist_path = Path(blocklist_path)
        self.alerts_path = Path(alerts_path)
        self.blocklist_path.parent.mkdir(parents=True, exist_ok=True)
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        self.blocklist = self._load_blocklist()
        self._strikes: dict[str, int] = defaultdict(int)

    def _load_blocklist(self) -> set[str]:
        if not self.blocklist_path.exists():
            return set()
        try:
            data = json.loads(self.blocklist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read blocklist {self.blocklist_path}: {exc}") from exc
        if not isinstance(data, list) or any(not isinstance(x, str) for x in data):
            raise ValueError("Blocklist must be a JSON array of MAC address strings")
        invalid = [x for x in data if not _MAC_RE.fullmatch(x)]
        if invalid:
            raise ValueError(f"Blocklist contains invalid MAC addresses: {invalid[:3]}")
        return {x.lower() for x in data}

    def process_windows(self, feature_df: pd.DataFrame, verbose: bool = True) -> list[dict]:
        """Score rows and return only alerts created during this invocation."""
        if feature_df.empty:
            return []
        missing = sorted(set(FEATURE_COLUMNS) - set(feature_df.columns))
        if missing:
            raise ValueError(f"Missing model feature columns: {missing}")
        if not feature_df.index.nlevels >= 1:
            raise ValueError("Expected an index containing source MAC")
        X = feature_df[FEATURE_COLUMNS]
        if X.isna().any().any():
            raise ValueError("Feature data contains NaN values")
        preds = self.model.predict(X)
        classes = list(getattr(self.model, "classes_", [0, 1]))
        if 1 not in classes:
            attack_probs = [0.0] * len(feature_df)
        elif hasattr(self.model, "predict_proba"):
            raw = self.model.predict_proba(X)
            attack_probs = raw[:, classes.index(1)]
        else:
            attack_probs = [float(p == 1) for p in preds]

        created: list[dict] = []
        for (mac, window_id), pred, prob in zip(feature_df.index, preds, attack_probs):
            mac = str(mac).lower()
            if mac in self.blocklist:
                continue
            if pred == 1:
                alert = self._raise_alert(mac, window_id, float(prob), verbose)
                created.append(alert)
                self._strikes[mac] += 1
                if self._strikes[mac] >= self.block_after_alerts:
                    self._block(mac, verbose)
            else:
                # Consecutive means consecutive evaluated windows for this MAC.
                self._strikes[mac] = 0
        return created

    def _raise_alert(self, mac: str, window_id, prob: float, verbose: bool) -> dict:
        alert = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mac": mac,
            "window_id": int(window_id),
            "attack_probability": round(prob, 4),
            "strikes": self._strikes[mac] + 1,
            "action": "alert",
        }
        self._append_jsonl(self.alerts_path, alert)
        if verbose:
            print(f"ALERT {mac} window={window_id} p(attack)={prob:.3f} "
                  f"strike={alert['strikes']}")
        return alert

    def _block(self, mac: str, verbose: bool) -> None:
        if mac in self.blocklist:
            return
        self.blocklist.add(mac)
        tmp = self.blocklist_path.with_suffix(self.blocklist_path.suffix + ".tmp")
        tmp.write_text(json.dumps(sorted(self.blocklist), indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.blocklist_path)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mac": mac,
            "action": "blocklist",
            "enforcement": "not_configured",
            "dry_run": self.dry_run,
        }
        self._append_jsonl(self.alerts_path, entry)
        if verbose:
            print(f"BLOCKLISTED {mac} (no network enforcement performed)")

    @staticmethod
    def _append_jsonl(path: Path, obj: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, sort_keys=True) + "\n")
