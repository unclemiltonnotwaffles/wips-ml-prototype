from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.awid3_loader import _normalise_one
from src.features import FEATURE_COLUMNS
from src.wips import WIPSDetector


class SequenceModel:
    classes_ = np.array([0, 1])

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)
        self.position = 0
        self.last_predictions = np.asarray([], dtype=int)

    def predict(self, frame):
        start = self.position
        stop = start + len(frame)
        self.last_predictions = self.predictions[start:stop]
        self.position = stop
        return self.last_predictions

    def predict_proba(self, frame):
        p = (self.last_predictions == 1).astype(float) * 0.9
        return np.column_stack([1 - p, p])


def feature_rows(mac, windows, frame_count=10):
    values = {name: [float(frame_count)] * len(windows) for name in FEATURE_COLUMNS}
    return pd.DataFrame(values, index=pd.MultiIndex.from_arrays(
        [[mac] * len(windows), windows], names=["src", "window_id"]))


def detector(tmp_path, predictions, threshold=2):
    return WIPSDetector(model=SequenceModel(predictions), block_after_alerts=threshold,
                        blocklist_path=tmp_path / "blocklist.json",
                        alerts_path=tmp_path / "alerts.jsonl")


def test_strike_counter_resets_after_normal_window(tmp_path):
    d = detector(tmp_path, [1, 0, 1])
    first = d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [1]), False)
    normal = d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [2]), False)
    last = d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [3]), False)
    assert len(first) == len(last) == 1
    assert normal == []
    assert not d.blocklist


def test_gap_breaks_consecutive_window_streak(tmp_path):
    d = detector(tmp_path, [1, 1])
    d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [10]), False)
    d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [12]), False)
    assert not d.blocklist


def test_two_consecutive_attack_windows_blocklist_once(tmp_path):
    d = detector(tmp_path, [1, 1])
    a = d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [10]), False)
    b = d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [11]), False)
    assert len(a) == len(b) == 1
    assert d.blocklist == {"aa:bb:cc:dd:ee:01"}
    assert json.loads((tmp_path / "blocklist.json").read_text()) == ["aa:bb:cc:dd:ee:01"]
    events = [json.loads(x) for x in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert [e["action"] for e in events] == ["alert", "alert", "blocklist"]
    assert events[-1]["enforcement"] == "not_configured"


def test_process_returns_only_new_alerts(tmp_path):
    d = detector(tmp_path, [1, 1, 1], threshold=9)
    assert len(d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [1]), False)) == 1
    assert len(d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [2]), False)) == 1
    assert len(d.process_windows(feature_rows("aa:bb:cc:dd:ee:01", [3]), False)) == 1


def test_bad_blocklist_rejected(tmp_path):
    path = tmp_path / "blocklist.json"
    path.write_text('["not-a-mac"]')
    with pytest.raises(ValueError):
        WIPSDetector(model=SequenceModel([0]), blocklist_path=path,
                     alerts_path=tmp_path / "alerts.jsonl")


def test_awid3_hex_frame_control_and_epoch():
    df = pd.DataFrame({
        "frame.time_epoch": [1700000000.25, 1700000001.25],
        "wlan.sa": ["AA:BB:CC:DD:EE:01"] * 2,
        "wlan.da": ["FF:FF:FF:FF:FF:FF"] * 2,
        "wlan.fc.type_subtype": ["0x000c", "0x0008"],
        "radiotap.dbm_antsignal": [-50, -55],
        "class": ["deauth", "normal"],
    })
    out = _normalise_one(df, "sample.csv")
    assert out["timestamp"].tolist() == [1700000000.25, 1700000001.25]
    assert out["fc_type"].tolist() == [0, 0]
    assert out["fc_subtype"].tolist() == [12, 8]
    assert out["label"].tolist() == [1, 0]


def test_awid3_human_timestamp():
    df = pd.DataFrame({
        "frame.time": ["2024-01-01 00:00:00.000000 UTC"],
        "wlan.sa": ["aa:bb:cc:dd:ee:01"],
        "wlan.da": ["ff:ff:ff:ff:ff:ff"],
        "wlan.fc.type": [0],
        "wlan.fc.subtype": [12],
    })
    out = _normalise_one(df, "human-time.csv")
    assert out["timestamp"].notna().all()
    assert out["timestamp"].iloc[0] == pytest.approx(1704067200.0)
