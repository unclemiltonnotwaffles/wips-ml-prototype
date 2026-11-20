"""Packet-to-window feature extraction for deauthentication-flood research."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "frame_count", "deauth_count", "disassoc_count", "deauth_ratio",
    "disassoc_ratio", "mgmt_ratio", "ctrl_ratio", "data_ratio",
    "rssi_mean", "rssi_std", "unique_dst", "length_mean",
]
DEAUTH_SUBTYPE = 12
DISASSOC_SUBTYPE = 10


def prepare_packet_flags(df: pd.DataFrame) -> pd.DataFrame:
    required = {"fc_type", "fc_subtype"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing packet fields: {sorted(missing)}")
    out = df.copy()
    out["is_mgmt"] = (out["fc_type"] == 0).astype(int)
    out["is_ctrl"] = (out["fc_type"] == 1).astype(int)
    out["is_data"] = (out["fc_type"] == 2).astype(int)
    out["is_deauth"] = ((out["fc_type"] == 0) & (out["fc_subtype"] == DEAUTH_SUBTYPE)).astype(int)
    out["is_disassoc"] = ((out["fc_type"] == 0) & (out["fc_subtype"] == DISASSOC_SUBTYPE)).astype(int)
    return out


def build_window_features(df: pd.DataFrame, window_seconds: float = 1.0,
                           deauth_flood_min: int = 3) -> pd.DataFrame:
    if df.empty:
        raise ValueError("build_window_features received an empty dataframe")
    if window_seconds <= 0 or deauth_flood_min < 1:
        raise ValueError("window_seconds and deauth_flood_min must be positive")
    required = {"timestamp", "src", "dst", "fc_type", "fc_subtype", "rssi", "length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing packet fields: {sorted(missing)}")
    out = df.copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp", "src", "dst"])
    if out.empty:
        raise ValueError("No valid timestamp/source/destination rows")
    out = prepare_packet_flags(out)
    out["window_id"] = np.floor(out["timestamp"] / window_seconds).astype("int64")
    groups = out.groupby(["src", "window_id"], sort=True)
    agg = pd.DataFrame({
        "frame_count": groups.size(),
        "deauth_count": groups["is_deauth"].sum(),
        "disassoc_count": groups["is_disassoc"].sum(),
        "mgmt_count": groups["is_mgmt"].sum(),
        "ctrl_count": groups["is_ctrl"].sum(),
        "data_count": groups["is_data"].sum(),
        "rssi_mean": groups["rssi"].mean(),
        "rssi_std": groups["rssi"].std().fillna(0.0),
        "unique_dst": groups["dst"].nunique(),
        "length_mean": groups["length"].mean(),
    })
    for name, numerator in (
        ("deauth_ratio", "deauth_count"), ("disassoc_ratio", "disassoc_count"),
        ("mgmt_ratio", "mgmt_count"), ("ctrl_ratio", "ctrl_count"),
        ("data_ratio", "data_count"),
    ):
        agg[name] = agg[numerator] / agg["frame_count"]
    if "label" in out.columns and pd.to_numeric(out["label"], errors="coerce").fillna(0).max() > 0:
        labels = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
        agg["label"] = labels.groupby([out["src"], out["window_id"]]).max()
    else:
        agg["label"] = (groups["is_deauth"].sum() >= deauth_flood_min).astype(int)
    return agg[FEATURE_COLUMNS + ["label"]].copy()


def train_test_split_time_ordered(features: pd.DataFrame,
                                  test_fraction: float = 0.30):
    """Chronologically split global buckets without dividing labelled attack runs."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if features.empty or "window_id" not in (features.index.names or []):
        raise ValueError("Expected non-empty features indexed by (src, window_id)")
    if "label" not in features.columns:
        raise ValueError("Expected a label column for attack-episode-aware splitting")
    windows = np.array(sorted(features.index.get_level_values("window_id").unique()))
    if len(windows) < 2:
        raise ValueError("At least two distinct time windows are required for a temporal split")
    target_cut = int(np.floor(len(windows) * (1 - test_fraction)))
    target_cut = min(max(target_cut, 1), len(windows) - 1)

    # Avoid cutting a contiguous run of attack-labelled windows for the same
    # source across train/test. Pick the nearest safe chronological boundary.
    attack_rows = features[features["label"].astype(int) == 1]
    attack_runs = {}
    for source, group in attack_rows.groupby(level=0):
        attack_windows = np.array(sorted(group.index.get_level_values("window_id").unique()))
        attack_runs[source] = attack_windows

    safe_cuts = []
    for candidate in range(1, len(windows)):
        left, right = windows[candidate - 1], windows[candidate]
        crosses_episode = any(
            np.any((run[:-1] <= left) & (run[1:] >= right) & ((run[1:] - run[:-1]) == 1))
            for run in attack_runs.values() if len(run) > 1
        )
        if not crosses_episode:
            safe_cuts.append(candidate)
    if not safe_cuts:
        raise ValueError("No chronological split boundary keeps contiguous attack runs intact")
    cut = min(safe_cuts, key=lambda candidate: (abs(candidate - target_cut), candidate))
    train_windows, test_windows = set(windows[:cut]), set(windows[cut:])
    train = features[features.index.get_level_values("window_id").isin(train_windows)].copy()
    test = features[features.index.get_level_values("window_id").isin(test_windows)].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition")
    return train, test
