"""
config.py — every tunable knob lives here.

WHY a separate config file?
    During the live demo the marker will ask "what happens if you change X?".
    If all numbers live in one file, you open it, change one, re-run, and
    answer the question live. It also *documents the reasoning* behind each
    default value — which is what a viva probes.
"""

# ─── Windowing ────────────────────────────────────────────────────────────────
# WHY 1 second?  A deauth flood at 100–500 frames/sec is obvious within one
# second, while normal traffic is sparse per device per second. Bigger windows
# delay detection; smaller windows make normal buckets too sparse for ratios.
WINDOW_SECONDS = 1.0

# A window is "attack" if it contains at least this many deauth frames.
# WHY 3?  A legitimate client sends ONE deauth when it disconnects. A flood
# sends dozens. 3 is a safe lower bound.
DEAUTH_FLOOD_MIN = 3

# ─── Model ────────────────────────────────────────────────────────────────────
N_ESTIMATORS = 100     # 100 trees: stable, fast.
MAX_DEPTH = 12         # cap depth to stop memorising individual MACs.
RANDOM_STATE = 42      # reproducibility — mandatory for a reportable number.
TEST_FRACTION = 0.30   # last 30 % of the timeline held out for testing.

# ─── WIPS response ────────────────────────────────────────────────────────────
# How many alerts from the SAME MAC before we block it?
# WHY 2?  One false positive shouldn't nuke a device; two consecutive attack
# windows from the same source is a very strong signal.
BLOCK_AFTER_ALERTS = 2

# dry_run=True → only LOG blocks, never touch the firewall.
# Set to False ONLY on an isolated lab host, and read src/wips.py first.
DRY_RUN = True

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
MODEL_PATH = "results/model.joblib"
CONFUSION_MATRIX_PATH = "results/confusion_matrix.png"
METRICS_PATH = "results/metrics.json"
ALERTS_PATH = "results/alerts.jsonl"
BLOCKLIST_PATH = "results/blocklist.json"
