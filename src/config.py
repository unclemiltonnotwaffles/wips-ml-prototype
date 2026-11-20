"""Central configuration for reproducible experiments."""
WINDOW_SECONDS = 1.0
DEAUTH_FLOOD_MIN = 3
N_ESTIMATORS = 100
MAX_DEPTH = 12
RANDOM_STATE = 42
TEST_FRACTION = 0.30

# Consecutive predicted attack windows required before adding a MAC to the
# proposed blocklist. This is not network enforcement.
BLOCK_AFTER_ALERTS = 2
# Retained only for backwards-compatible CLI/config consumers. The detector
# never executes host firewall commands; changing this flag does not enable
# enforcement.
DRY_RUN = True

RESULTS_DIR = "results"
MODEL_PATH = "results/model.joblib"
CONFUSION_MATRIX_PATH = "results/confusion_matrix.png"
METRICS_PATH = "results/metrics.json"
ALERTS_PATH = "results/alerts.jsonl"
BLOCKLIST_PATH = "results/blocklist.json"
