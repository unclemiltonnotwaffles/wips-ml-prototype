"""
wips.py — the Prevention half of the prototype.

WIDS = detect only.  WIPS = detect + *respond*.
The response implemented here is MAC blocking:

    alert → strike counter → (≥ BLOCK_AFTER_ALERTS) → blocklist → optional iptables

WHY strikes and not "block on first alert"?
    Every ML model has false positives. Blocking on one alert means a single
    misclassification takes a legitimate device off the network. Requiring two
    consecutive attack windows from the same MAC drops the false-positive rate
    roughly quadratically while barely slowing true-positive response.

WHY dry_run=True by default?
    Because you will run this on a demo laptop. Accidentally inserting an
    iptables rule mid-presentation is a bad day. `_apply_block` is where you
    flip the switch, and the real command is clearly marked.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS

class WIPSDetector:
    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        blocklist_path: str = config.BLOCKLIST_PATH,
        alerts_path: str = config.ALERTS_PATH,
        block_after_alerts: int = config.BLOCK_AFTER_ALERTS,
        dry_run: bool = config.DRY_RUN,
    ):
        self.model = joblib.load(model_path)
        self.block_after_alerts = block_after_alerts
        self.dry_run = dry_run

        self.blocklist_path = Path(blocklist_path)
        self.alerts_path = Path(alerts_path)
        self.blocklist_path.parent.mkdir(parents=True, exist_ok=True)
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)

        if self.blocklist_path.exists():
            self.blocklist: set[str] = set(json.loads(self.blocklist_path.read_text()))
        else:
            self.blocklist = set()

        self._strikes: dict[str, int] = defaultdict(int)
        self.alerts: list[dict] = []

    def process_windows(self, feature_df: pd.DataFrame, verbose: bool = True) -> list[dict]:
        """Feed every row of a window-feature dataframe through the model."""
        if feature_df.empty:
            return []

        X = feature_df[FEATURE_COLUMNS].values
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]

        for (mac, window_id), pred, prob in zip(feature_df.index, preds, probs):
            if mac in self.blocklist:
                continue                       # already neutralised
            if pred != 1:
                continue

            self._raise_alert(mac, window_id, float(prob), verbose=verbose)
            self._strikes[mac] += 1
            if self._strikes[mac] >= self.block_after_alerts:
                self._block(mac, verbose=verbose)

        return self.alerts

    def _raise_alert(self, mac, window_id, prob, verbose):
        alert = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mac": mac,
            "window_id": int(window_id),
            "attack_probability": round(prob, 4),
            "strikes": self._strikes[mac] + 1,
            "action": "alert",
        }
        self.alerts.append(alert)
        self._append_jsonl(self.alerts_path, alert)
        if verbose:
            print(f"  ⚠️  ALERT  {mac}  window={window_id}  "
                  f"p(attack)={prob:.2f}  strike={alert['strikes']}")

    def _block(self, mac, verbose):
        if mac in self.blocklist:
            return
        self.blocklist.add(mac)
        self.blocklist_path.write_text(json.dumps(sorted(self.blocklist), indent=2))

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mac": mac,
            "action": "block",
            "dry_run": self.dry_run,
        }
        self._append_jsonl(self.alerts_path, entry)
        if verbose:
            tag = "DRY-RUN BLOCK" if self.dry_run else "BLOCKED"
            print(f"  🛡️  {tag}  {mac}  (added to blocklist)")

        self._apply_block(mac)

    def _apply_block(self, mac: str):
        """
        The single place where real enforcement happens.

        SAFETY: guarded by `self.dry_run`, which defaults to True in config.py.

        To actually drop frames from a MAC on an authorised lab host:

            subprocess.run(["iptables", "-I", "INPUT",
                            "-m", "mac", "--mac-source", mac,
                            "-j", "DROP"], check=True)

        Left commented out because a demo laptop is not a lab.
        """
        if self.dry_run:
            return
        # --- Uncomment ONLY on an authorised lab host ---
        # import subprocess
        # subprocess.run(["iptables", "-I", "INPUT", "-m", "mac",
        #                 "--mac-source", mac, "-j", "DROP"], check=True)
        pass

    @staticmethod
    def _append_jsonl(path: Path, obj: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")
