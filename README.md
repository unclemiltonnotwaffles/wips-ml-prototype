# Wi-Fi Intrusion Detection and Response Prototype

A research prototype that extracts per-source, one-second 802.11 traffic features and uses a Random Forest classifier to identify deauthentication-flood-like activity. It supports generated synthetic records and compatible AWID3 CSV exports.

**Important scope boundary:** this repository processes offline records. It does not capture live radio traffic, manage an access point, disconnect wireless clients, or enforce network-wide MAC blocking. The response stage raises alerts and writes a proposed MAC blocklist only. The term WIPS describes the research goal, not a production-ready prevention capability.

## Quick start

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
python -m src.run_demo
```

Generated artifacts are written to `results/`: model, metrics JSON, confusion-matrix PNG, alert JSONL, and proposed blocklist JSON. These outputs are reproducibility artifacts, not proof of field performance.

## Processing pipeline

1. Read synthetic or AWID3 CSV packet records.
2. Validate and normalise timestamps, addresses, frame type/subtype, RSSI, and labels.
3. Aggregate by source address and global one-second time bucket.
4. Extract the fixed feature vector in `src/features.py`.
5. Train and evaluate a Random Forest using a chronological holdout split grouped by global time bucket.
6. Score replay windows, write alerts, and add a source MAC to a persistent *proposed* blocklist after consecutive attack predictions.

A normal prediction resets that source's strike counter. Each invocation of `process_windows` returns only the alerts created by that invocation.

## AWID3 input

```bash
python -m src.run_awid3 --data-dir /path/to/awid3/csv_folder
```

The loader supports common Wireshark field names and numeric epoch or human-readable timestamps. It accepts decimal and hexadecimal frame-control fields. Dataset exports vary: verify column semantics, address roles, frame-control encoding, label conventions, and timestamp timezone against the exact AWID3 release before interpreting results. Ambiguous or malformed frame-control values are rejected instead of silently converted to zero.

The adapter currently uses the selected transmitter/source aliases as the grouping key. 802.11 address roles vary by frame type and To-DS/From-DS bits; a rigorous study should use frame-aware address extraction rather than assume transmitter and end-host source are interchangeable.

## Evaluation discipline

Metrics are generated from the actual run and saved to `results/metrics.json`. The split is chronological by unique global window ID, so all sources in the same second stay on one side of the split. It also moves the boundary to keep contiguous attack-labelled windows for the same source in one partition. Both classes must occur in both the training and test partitions. Test metrics are meaningful only for the selected dataset, labels, and capture distribution; they do not establish generalisation to unseen networks, devices, attack tools, or environments.

Do not copy historical poster numbers as current results. Run the pipeline on a documented dataset version and retain its configuration, class counts, confusion matrix, and output metrics.

## Response and safety boundary

The detector writes alerts and a proposed MAC blocklist. It deliberately does not execute `iptables`, inject deauthentication frames, or call an AP/controller API. A host INPUT firewall rule would not block a wireless client at the access point or protect other clients.

Real enforcement requires a separately designed and authorised integration with the actual AP/controller/firewall, including device identity verification, allowlists for infrastructure/management MACs, expiry and rollback, audit logging, rate limits, fail-safe behaviour, and tests on an isolated lab network. Do not treat a spoofable MAC address as a strong identity.

## Tests

```bash
python -m pytest
```

Tests cover packet flags, feature windows, temporal partitioning, timestamp/frame-control parsing, strike reset, alert return semantics, and blocklist persistence. They do not validate live wireless capture or external network enforcement.

## Ethical use

Use only data you are authorised to analyse. Do not transmit disruptive wireless frames or block devices on networks you do not own or administer.
