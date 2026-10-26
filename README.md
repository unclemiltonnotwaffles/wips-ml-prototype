# Wi-Fi Intrusion Prevention System (WIPS) — ML Prototype

A working prototype that uses a **Random Forest** classifier to detect
**deauthentication flood attacks** on 802.11 traffic and **automatically blocks**
the offending MAC address.

Runs on **synthetic traffic** out of the box (no dataset needed) and on the
**AWID3** public dataset via a drop-in loader.

---

## Quick start (3 commands)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tests/test_features.py      # sanity checks
python -m src.run_demo             # full end-to-end demo
```

Outputs land in `results/`:

- `metrics.json` — accuracy / precision / recall / F1 + feature importances
- `confusion_matrix.png` — picture for the poster
- `alerts.jsonl` — every alert raised during the demo
- `blocklist.json` — MACs the WIPS blocked
- `model.joblib` — saved classifier

---

## What it does

1. **Reads** packet-level Wi-Fi records (synthetic generator or AWID3 CSVs).
2. **Aggregates** them into 1-second windows per source MAC.
3. **Extracts 11 simple features** (deauth count, frame-type ratios, RSSI mean/std, unique destinations, etc.).
4. **Classifies** each window as *normal* or *attack* using a Random Forest.
5. **Responds**: raises an alert; after 2 consecutive attacks from the same MAC, blocks it.

---

## Project layout

```
wips/
├── README.md
├── requirements.txt
├── poster/
│   └── poster.html            ← A1 poster (Chrome → Print → Save as PDF)
├── docs/
│   ├── CODE_WALKTHROUGH.md    ← "what and why" for every file
│   └── AI_USAGE.md            ← required AI-assistant statement
├── src/
│   ├── config.py              ← every tunable knob
│   ├── synthetic.py           ← demo traffic generator
│   ├── awid3_loader.py        ← AWID3 CSV loader
│   ├── features.py            ← windowing + feature extraction
│   ├── train.py               ← Random Forest training + evaluation
│   ├── wips.py                ← detector + alerting + blocking
│   ├── run_demo.py            ← end-to-end demo (synthetic)
│   └── run_awid3.py           ← end-to-end demo (AWID3)
└── tests/
    └── test_features.py       ← sanity checks (no pytest needed)
```

---

## Run against AWID3

```
python -m src.run_awid3 --data-dir /path/to/awid3/csv_folder
```

Get AWID3 from https://icsdweb.aegean.gr/awid/awid3 (free, email sign-up).
The loader auto-detects common Wireshark-style column names (`wlan.sa`,
`wlan.fc.type_subtype`, `radiotap.dbm_antsignal`, …) so no renaming is needed.

---

## How the ML works (short)

- **Windowing** — packets bucketed into `(src_mac, 1-second)` cells.
- **Features** — 11 numbers per cell (see `FEATURE_COLUMNS` in `src/features.py`).
- **Label** — a window is attack if it contains ≥ 3 deauth frames, or if any of
its packets were labelled attack in the source dataset.
- **Model** — `RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced")`.
- **Split** — *time-ordered* 70/30. This is stricter than a random split: it
prevents packets from the same attack episode leaking into both halves.

---

## The WIPS action

When `WIPSDetector` predicts *attack* for a MAC:

1. Write an alert to `results/alerts.jsonl`.
2. Increment that MAC's strike counter.
3. After 2 strikes, add the MAC to `results/blocklist.json` and log `[BLOCK]`.
4. If `DRY_RUN=False` **and** you are on a Linux lab host, it can run the real
`iptables -I INPUT -m mac --mac-source <mac> -j DROP` (commented out by default).

`DRY_RUN=True` is the default so you cannot accidentally firewall yourself.

---

## Tests

```
python tests/test_features.py
```

Checks: deauth flagging, window grouping, label correctness, time-split leakage.
No pytest required.

---

## Printing the poster

1. Open `poster/poster.html` in Chrome.
2. Edit `<Your Name>`, `<ID>`, `<Module Code>`, and the results numbers.
3. `Ctrl+P` → **Destination: Save as PDF** → **Paper size: A1** →
**Margins: None** → **Background graphics: ON** → Save.

---

## Ethical / safety note

All traffic is **synthetic** (generated in-process) or comes from the public
**AWID3** dataset. No live radio transmission is performed by this code.
If you adapt it to a real NIC in monitor mode, only do so on an **isolated lab
network you own or are explicitly authorised to test**.

</BDS:create_file>

<BDS:create_file fileName="requirements.txt">

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
joblib>=1.3

