"""
run_demo.py — one-command end-to-end demonstration (synthetic traffic).

Run:  python -m src.run_demo
"""

from __future__ import annotations
from pathlib import Path

from . import config
from .synthetic import generate_traffic
from .features import build_window_features
from .train import train_model
from .wips import WIPSDetector

def main() -> None:
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    Path(config.ALERTS_PATH).write_text("")

    # ── 1. Training traffic ──────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1  generating 10 min of training traffic")
    print("=" * 70)
    train_df = generate_traffic(duration_seconds=600, attack_episodes=10, seed=42)
    print(f"        {len(train_df):,} packets, "
          f"{int(train_df['label'].sum()):,} attack packets")

    # ── 2. Train ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2  training Random Forest")
    print("=" * 70)
    metrics = train_model(train_df)

    # ── 3. Fresh capture to replay ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3  generating a FRESH capture (unseen attack) and replaying it")
    print("=" * 70)
    live_df = generate_traffic(duration_seconds=120, attack_episodes=1, seed=9999)
    live_feats = build_window_features(
        live_df,
        window_seconds=config.WINDOW_SECONDS,
        deauth_flood_min=config.DEAUTH_FLOOD_MIN,
    )
    print(f"        {len(live_feats):,} windows to replay")

    # ── 4. WIPS ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4  running WIPS detector (dry-run blocking)")
    print("=" * 70)

    detector = WIPSDetector(dry_run=config.DRY_RUN)
    live_feats = live_feats.sort_index(level="window_id")

    current_window = None
    alerts = []
    for (mac, win), _ in live_feats.iterrows():
        if current_window is None or win - current_window >= 15:
            print(f"\n  ── t ≈ {win}s ──")
            current_window = win
        sub = live_feats.loc[[(mac, win)]]
        alerts.extend(detector.process_windows(sub, verbose=True))

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  accuracy  = {metrics['accuracy']*100:.2f} %")
    print(f"  precision = {metrics['precision']*100:.2f} %")
    print(f"  recall    = {metrics['recall']*100:.2f} %")
    print(f"  f1        = {metrics['f1']*100:.2f} %")
    print(f"  alerts raised during replay: {len(alerts)}")
    print(f"  blocklist now contains: {sorted(detector.blocklist) or '(empty)'}")
    print(f"\n  artefacts in {config.RESULTS_DIR}/:")
    for f in sorted(Path(config.RESULTS_DIR).iterdir()):
        print(f"      {f.name}")

if __name__ == "__main__":
    main()
