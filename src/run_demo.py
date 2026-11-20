"""One-command end-to-end demonstration using synthetic packet records."""
from __future__ import annotations
from pathlib import Path
from . import config
from .synthetic import generate_traffic
from .features import build_window_features
from .train import train_model
from .wips import WIPSDetector


def main() -> None:
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.ALERTS_PATH).write_text("", encoding="utf-8")
    Path(config.BLOCKLIST_PATH).unlink(missing_ok=True)
    print("STEP 1: generating synthetic training records")
    training = generate_traffic(duration_seconds=600, attack_episodes=10, seed=42)
    print(f"  {len(training):,} packets; {int(training['label'].sum()):,} attack-labelled")
    print("STEP 2: training and evaluating on a chronological holdout")
    metrics = train_model(training)
    print("STEP 3: generating a separate synthetic replay")
    replay_df = generate_traffic(duration_seconds=120, attack_episodes=1, seed=9999)
    windows = build_window_features(replay_df, config.WINDOW_SECONDS, config.DEAUTH_FLOOD_MIN)
    windows = windows.sort_index(level="window_id")
    print("STEP 4: offline detection; blocklist only, no network enforcement")
    detector = WIPSDetector(dry_run=True)
    alerts = detector.process_windows(windows, verbose=True)
    print("SUMMARY")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"  {key}: {metrics[key] * 100:.2f}%")
    print(f"  alerts created in this replay: {len(alerts)}")
    print(f"  MACs added to proposed blocklist: {len(detector.blocklist)}")
    print(f"  artefacts: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
