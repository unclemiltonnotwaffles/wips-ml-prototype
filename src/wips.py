"""Offline detector: alerts and a proposed MAC blocklist, never network enforcement."""
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
    def __init__(self, model_path=config.MODEL_PATH, blocklist_path=config.BLOCKLIST_PATH,
                 alerts_path=config.ALERTS_PATH, block_after_alerts=config.BLOCK_AFTER_ALERTS,
                 dry_run=True, model=None):
        if block_after_alerts < 1:
            raise ValueError("block_after_alerts must be >= 1")
        self.model = model if model is not None else joblib.load(model_path)
        self.block_after_alerts = int(block_after_alerts)
        self.dry_run = bool(dry_run)  # compatibility only; no firewall command is executed
        self.blocklist_path, self.alerts_path = Path(blocklist_path), Path(alerts_path)
        self.blocklist_path.parent.mkdir(parents=True, exist_ok=True)
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        self.blocklist = self._load_blocklist()
        self._strikes = defaultdict(int)
        self._last_window = {}

    def _load_blocklist(self):
        if not self.blocklist_path.exists():
            return set()
        try:
            data = json.loads(self.blocklist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read blocklist {self.blocklist_path}: {exc}") from exc
        if not isinstance(data, list) or any(not isinstance(x, str) for x in data):
            raise ValueError("Blocklist must be a JSON array of MAC strings")
        invalid = [x for x in data if not _MAC_RE.fullmatch(x)]
        if invalid:
            raise ValueError(f"Invalid MAC address in blocklist: {invalid[:3]}")
        return {x.lower() for x in data}

    def process_windows(self, feature_df: pd.DataFrame, verbose=True):
        """Return only alerts created by this call; rows must be (MAC, window_id)."""
        if feature_df.empty:
            return []
        if not isinstance(feature_df.index, pd.MultiIndex) or feature_df.index.nlevels != 2:
            raise ValueError("Expected a two-level (source MAC, window_id) index")
        missing = sorted(set(FEATURE_COLUMNS) - set(feature_df.columns))
        if missing:
            raise ValueError(f"Missing model feature columns: {missing}")
        X = feature_df[FEATURE_COLUMNS]
        if X.isna().any().any():
            raise ValueError("Feature data contains NaN values")
        preds = self.model.predict(X)
        classes = list(getattr(self.model, "classes_", [0, 1]))
        if 1 not in classes:
            probs = [0.0] * len(feature_df)
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, classes.index(1)]
        else:
            probs = [float(p == 1) for p in preds]

        created = []
        for (raw_mac, raw_window), pred, probability in zip(feature_df.index, preds, probs):
            mac, window = str(raw_mac).lower(), int(raw_window)
            if mac in self.blocklist:
                continue
            previous = self._last_window.get(mac)
            if previous is not None and window != previous + 1:
                self._strikes[mac] = 0
            self._last_window[mac] = window
            if pred == 1:
                alert = self._raise_alert(mac, window, float(probability), verbose)
                created.append(alert)
                self._strikes[mac] += 1
                if self._strikes[mac] >= self.block_after_alerts:
                    self._block(mac, verbose)
            else:
                self._strikes[mac] = 0
        return created

    def _raise_alert(self, mac, window_id, probability, verbose):
        alert = {"ts": datetime.now(timezone.utc).isoformat(), "mac": mac,
                 "window_id": int(window_id), "attack_probability": round(probability, 4),
                 "strikes": self._strikes[mac] + 1, "action": "alert"}
        self._append_jsonl(self.alerts_path, alert)
        if verbose:
            print(f"ALERT {mac} window={window_id} p(attack)={probability:.3f} "
                  f"strike={alert['strikes']}")
        return alert

    def _block(self, mac, verbose):
        if mac in self.blocklist:
            return
        self.blocklist.add(mac)
        temporary = self.blocklist_path.with_suffix(self.blocklist_path.suffix + ".tmp")
        temporary.write_text(json.dumps(sorted(self.blocklist), indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(self.blocklist_path)
        event = {"ts": datetime.now(timezone.utc).isoformat(), "mac": mac,
                 "action": "blocklist", "enforcement": "not_configured",
                 "dry_run": self.dry_run}
        self._append_jsonl(self.alerts_path, event)
        if verbose:
            print(f"BLOCKLISTED {mac} (no network enforcement performed)")

    @staticmethod
    def _append_jsonl(path, obj):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, sort_keys=True) + "\n")
