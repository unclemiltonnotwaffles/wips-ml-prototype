"""
run_awid3.py — same demo, but trains on the real AWID3 dataset.

Usage:
    python -m src.run_awid3 --data-dir /path/to/AWID3/csv_folder
"""

from __future__ import annotations
import argparse
from pathlib import Path

from . import config
from .awid3_loader import load_awid3_dir
from .features import build_window_features
from .train import train_model
from .wips import WIPSDetector

def main():
    parser = argparse.ArgumentParser(description="Train WIPS on AWID3 CSVs")
    parser.add_argument("--data-dir", required=True,
                        help="Folder containing AWID3 *.csv files")
    parser.add_argument("--replay-fraction", type=float, default=0.25,
                        help="Last X of the timeline to replay after training (default 0.25)")
    args = parser.parse_args()

    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    Path(config.ALERTS_PATH).write_text("")

    print("=" * 70)
    print("STEP 1  loading AWID3")
    print("=" * 70)
    df = load_awid3_dir(args.data_dir)

    print("\n" + "=" * 70)
    print("STEP 2  training Random Forest")
    print("=" * 70)
    metrics = train_model(df)

    print("\n" + "=" * 70)
    print(f"STEP 3  replaying last {args.replay_fraction*100:.0f}% through WIPS")
    print("=" * 70)
    feats = build_window_features(df)
    n = len(feats)
    cut = int(n * (1 - args.replay_fraction))
    replay = feats.iloc[cut:].sort_index(level="window_id")

    detector = WIPSDetector(dry_run=config.DRY_RUN)
    alerts = detector.process_windows(replay, verbose=False)

    print(f"  replayed {len(replay):,} windows")
    print(f"  alerts: {len(alerts):,}")
    print(f"  unique MACs blocked: {len(detector.blocklist)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  accuracy  = {metrics['accuracy']*100:.2f} %")
    print(f"  precision = {metrics['precision']*100:.2f} %")
    print(f"  recall    = {metrics['recall']*100:.2f} %")
    print(f"  f1        = {metrics['f1']*100:.2f} %")
    print(f"  artefacts in {config.RESULTS_DIR}/")

if __name__ == "__main__":
    main()
