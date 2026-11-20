# Code Walkthrough

This document describes the current implementation. It distinguishes offline detection from real network prevention.

## Processing pipeline

1. `src/awid3_loader.py` normalises common AWID3 CSV columns, epoch or human-readable timestamps, and decimal/hexadecimal frame-control fields. Unparseable type/subtype values fail loudly rather than silently becoming zero. Address roles remain dataset/frame dependent and must be validated against the capture format.
2. `src/synthetic.py` generates reproducible packet-like records for demonstrations.
3. `src/features.py` creates per-source, per-global-second windows and extracts the fixed 12-column feature vector, including mean frame length. The temporal split uses unique global window IDs, so two MACs observed in the same second cannot be separated across train and test. It also selects the nearest boundary that does not divide a contiguous same-source attack-labelled run.
4. `src/train.py` checks that the training partition contains both classes, fits the Random Forest, evaluates the chronological holdout, and saves the exact class counts, confusion matrix, metrics, and feature importances.
5. `src/wips.py` scores feature windows, writes JSONL alerts, resets strike counts after a normal or missing time window, and persists a proposed MAC blocklist after the configured consecutive-attack threshold.

## Important detector behaviour

- `process_windows()` returns only alerts created by the current call, not the complete alert history.
- Consecutive means adjacent global window IDs for the same source. A normal window or a gap resets the strike count.
- The model score is not probability-calibrated; treat it as a ranking score, not a real-world probability. The detector validates the two-level `(source MAC, window_id)` index and required feature columns.
- The model's attack-score column is selected using `model.classes_`; it is not assumed that class 1 is always column index 1.
- A malformed persisted blocklist is rejected.

## Prevention boundary

The implementation does **not** run iptables, transmit deauthentication frames, capture live radio traffic, or communicate with an AP/controller. A MAC in `results/blocklist.json` is a proposed response, not an enforced network block. A host firewall rule would not stop a wireless client at the access point or protect other clients.

Production enforcement would require a separately designed, authorised AP/controller integration with identity checks, management-device allowlists, expiry, rollback, audit logs, rate limits, and safe failure handling. MAC addresses are spoofable and must not be treated as strong identity.

## Testing and evaluation

Run `python -m pytest`. Regression tests cover parsing, feature construction, temporal splitting, consecutive strikes, gaps, alert return semantics, and blocklist persistence. These tests do not establish model quality or validate live wireless enforcement.

Metrics are dataset-specific. Synthetic accuracy is a software-path check, not evidence of field performance. Preserve dataset provenance, exact capture selection, class counts, configuration, confusion matrix, and metrics for any reported result. Never copy old poster values into a new report without rerunning the experiment.
