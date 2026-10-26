"""
test_features.py — lightweight sanity checks. No pytest needed.

Run:  python tests/test_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.features import (
    FEATURE_COLUMNS, prepare_packet_flags, build_window_features,
    train_test_split_time_ordered,
)
from src.synthetic import generate_traffic

def test_packet_flags():
    df = pd.DataFrame({
        "timestamp": [0.0, 0.5, 1.2],
        "src": ["aa"] * 3,
        "dst": ["ff"] * 3,
        "fc_type": [0, 0, 2],
        "fc_subtype": [12, 8, 0],
        "rssi": [-50.0, -60.0, -55.0],
        "length": [26, 250, 500],
        "label": [1, 0, 0],
    })
    f = prepare_packet_flags(df)
    assert f["is_deauth"].tolist() == [1, 0, 0], "deauth flag wrong"
    assert f["is_mgmt"].tolist() == [1, 1, 0], "mgmt flag wrong"
    assert f["is_data"].tolist() == [0, 0, 1], "data flag wrong"
    print("  ✓ packet flags")

def test_windowing():
    df = generate_traffic(duration_seconds=30, attack_episodes=1, seed=1)
    feats = build_window_features(df, window_seconds=1.0, deauth_flood_min=3)
    assert set(FEATURE_COLUMNS).issubset(feats.columns), "missing feature columns"
    assert "label" in feats.columns, "missing label"
    assert (feats["frame_count"] >= 1).all(), "empty windows leaked through"
    for col in ["deauth_ratio", "disassoc_ratio", "mgmt_ratio",
                "ctrl_ratio", "data_ratio"]:
        assert feats[col].between(0, 1).all(), f"{col} out of range"
    print(f"  ✓ windowing produced {len(feats):,} windows")

def test_deauth_flood_detected_in_features():
    df = generate_traffic(duration_seconds=30, attack_episodes=1, seed=2)
    feats = build_window_features(df, window_seconds=1.0, deauth_flood_min=3)
    attack_windows = feats[feats["label"] == 1]
    assert len(attack_windows) > 0, "no attack windows generated"
    assert attack_windows["deauth_count"].max() > 10, "flood not reflected in features"
    print(f"  ✓ {len(attack_windows)} attack windows; "
          f"max deauth/sec = {int(attack_windows['deauth_count'].max())}")

def test_time_split():
    df = generate_traffic(duration_seconds=60, attack_episodes=2, seed=3)
    feats = build_window_features(df)
    train, test = train_test_split_time_ordered(feats, 0.3)
    max_train_win = train.index.get_level_values("window_id").max()
    min_test_win  = test.index.get_level_values("window_id").min()
    assert max_train_win <= min_test_win, "time split is leaking future data"
    print(f"  ✓ time split clean: train ≤ {max_train_win}s, test ≥ {min_test_win}s")

if __name__ == "__main__":
    print("running tests ...")
    test_packet_flags()
    test_windowing()
    test_deauth_flood_detected_in_features()
    test_time_split()
    print("all tests passed ✔")
