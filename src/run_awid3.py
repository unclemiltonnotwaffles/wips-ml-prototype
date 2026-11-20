"""Train on AWID3 CSV records and replay the chronological holdout tail."""
from __future__ import annotations
import argparse
from pathlib import Path
from . import config
from .awid3_loader import load_awid3_dir
from .features import build_window_features, train_test_split_time_ordered
from .train import train_model
from .wips import WIPSDetector


def main():
    parser = argparse.ArgumentParser(description="Train WIPS prototype on AWID3 CSVs")
    parser.add_argument("--data-dir", required=True, help="Folder containing AWID3 CSV files")
    parser.add_argument("--replay-fraction", type=float, default=0.25,
                        help="Fraction of final timeline to replay (0 < value < 1)")
    args = parser.parse_args()
    if not 0 < args.replay_fraction < 1:
        parser.error("--replay-fraction must be greater than 0 and less than 1")

    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.ALERTS_PATH).write_text("", encoding="utf-8")
    Path(config.BLOCKLIST_PATH).unlink(missing_ok=True)
    print("STEP 1: loading AWID3 records")
    packets = load_awid3_dir(args.data_dir)
    print("STEP 2: training and evaluating chronologically")
    metrics = train_model(packets)
    print("STEP 3: replaying the final chronological segment")
    features = build_window_features(packets, config.WINDOW_SECONDS, config.DEAUTH_FLOOD_MIN)
    _, chronological_test = train_test_split_time_ordered(features, config.TEST_FRACTION)
    # Replay a suffix of the held-out segment only, never training rows.
    unique_windows = sorted(chronological_test.index.get_level_values("window_id").unique())
    cut = max(0, int(len(unique_windows) * (1 - args.replay_fraction)))
    selected = set(unique_windows[cut:])
    replay = chronological_test[
        chronological_test.index.get_level_values("window_id").isin(selected)
    ].sort_index(level="window_id")
    detector = WIPSDetector(dry_run=True)
    alerts = detector.process_windows(replay, verbose=False)
    print(f"  replayed {len(replay):,} held-out windows")
    print(f"  alerts: {len(alerts):,}")
    print(f"  proposed blocklist entries: {len(detector.blocklist):,}")
    print("  network enforcement: not configured")
    print("SUMMARY")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"  {key}: {metrics[key] * 100:.2f}%")
    print(f"  artefacts: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
