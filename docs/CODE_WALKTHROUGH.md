# Code Walkthrough — "What is going on, and why"

File-by-file explanation written so you can defend every line during the demo.
Each section answers **what does it do** and **why was it built that way**.

---

## 1. `src/config.py` — one place for every number

**What:** All tunable parameters: window size, deauth threshold, tree count,
block policy, file paths.

**Why:** (a) You can answer "what if you change X?" live by editing one line.
(b) The comments document the reasoning behind the defaults, which is what the
viva probes. (c) Training and inference import the same constants, so
`WINDOW_SECONDS` can never silently drift between them — a classic cause of
"99 % train accuracy, terrible production accuracy" bugs.

---

## 2. `src/synthetic.py` — the data you can always demo

**What:** Generates a packet-level dataframe mimicking a home/lab Wi-Fi
capture: 15 legitimate devices, an AP, and N deauth-flood episodes. Schema is
identical to `awid3_loader.load_awid3_dir`.

**Why:**
- AWID3 needs an email sign-up; a demo cannot depend on network access.
- Replaying fresh synthetic traffic proves generalisation.
- Because the schema matches, every downstream module is dataset-agnostic.

**Key design choice:** ~1 % of legitimate clients emit **one** deauth when they
disconnect. Without that, `deauth_count > 0` would be a perfect rule and the ML
would be pointless. This forces the model to learn a *rate*, not just presence.

---

## 3. `src/awid3_loader.py` — the real-dataset adapter

**What:** Reads every `*.csv` under a folder, resolves Wireshark-style column
names via `COLUMN_ALIASES`, returns the 8-column schema.

**Why:**
- Column names differ between AWID3 releases; a central alias table handles all.
- Type/subtype may be packed (`type*16 + subtype`); loader unpacks it.
- Labels can be numeric or strings; `_normalise_label` maps all to `0/1`.
- Missing RSSI filled with column median (one missing radiotap header should
  not lose a frame).

---

## 4. `src/features.py` — the ML input, and the whole point

**What:** Aggregates packets into `(src_mac, 1-second)` windows and computes
`FEATURE_COLUMNS`.

**Why window per (source, second)?** A single deauth frame is indistinguishable
from a legitimate disconnect. A flood is a *rate* phenomenon — it only becomes
visible when you count frames per unit time. Windowing converts "is this a
flood?" into a supervised-learning question on aggregated rows.

**Why these 11 features?**

| Feature | What it captures | Why it matters |
|---|---|---|
| `frame_count` | total frames/sec | floods send far more than normal |
| `deauth_count` | deauth frames/sec | strongest deauth-flood signal |
| `disassoc_count` | disassoc frames/sec | adjacent attack family |
| `deauth_ratio` | deauth / total | makes the model shape-aware |
| `disassoc_ratio` | disassoc / total | ditto |
| `mgmt_ratio` | mgmt / total | normal traffic is mostly data |
| `ctrl_ratio` | ctrl / total | ACKs dominate normal traffic |
| `data_ratio` | data / total | ditto |
| `rssi_mean` | mean dBm | attackers are often physically close |
| `rssi_std` | spread of dBm | floods have almost no spread |
| `unique_dst` | distinct destinations | floods target broadcast |

**Why ratios as well as counts?** A busy legitimate AP can send hundreds of
frames/sec. Without ratios, "many frames" would look like an attack. Ratios
make the model distinguish *burst shape* from *burst volume*.

**Why time-ordered split?** A random split lets packets from the *same attack
episode* land in both halves. The test score then measures memorisation, not
generalisation.

---

## 5. `src/train.py` — the model

**What:** `RandomForestClassifier(n_estimators=100, max_depth=12,
class_weight="balanced")`, evaluated with accuracy / precision / recall / F1,
confusion matrix saved as PNG.

**Why Random Forest?**
- No feature scaling required.
- Robust to heavy-tailed count features.
- `class_weight="balanced"` handles the ~1:20 imbalance.
- `feature_importances_` gives you a bar chart for the viva.

**Why `max_depth=12`?** Uncapped forests memorise individual MACs and hit 100 %
train/test — a lookup table, not a detector. Capping depth forces generalisation.

**Why a confusion-matrix PNG?** It goes straight on the poster.

---

## 6. `src/wips.py` — the "P" in WIPS

**What:** Loads the model, scores every window, raises alerts, counts strikes,
blocks after `BLOCK_AFTER_ALERTS`.

**Why strikes, not block-on-first-alert?** Every ML model has false positives.
Requiring two consecutive attack windows from the same MAC reduces the
false-positive rate roughly quadratically while barely slowing response.

**Why `dry_run=True` by default?** A demo laptop is not a lab. `_apply_block` is
where the switch lives, and the real command is clearly commented out with a
safety note.

**Why JSONL for alerts?** Append-only, one-object-per-line, greppable,
streamable. A JSON array would require rewriting the whole file per alert.

---

## 7. `src/run_demo.py` — the demo script

**What:** Generates 10 min training traffic → trains → generates a *fresh*
2 min capture with one unseen attack → replays it through `WIPSDetector` with a
live timeline.

**Why fresh traffic for replay?** Because "the model classifies the data it
trained on" is not a demonstration of a working detector. Fresh seed, fresh
attacker MAC, fresh timing.

---

## 8. `tests/test_features.py` — proof the pipeline is correct

**What:** Four assert-based tests, no pytest required.

- deauth flags fire on the right subtypes
- windows have no empty rows and every ratio is in `[0, 1]`
- a synthetic flood actually shows up in the features
- the time split has no data leakage

If all four pass, the plumbing is structurally sound.

---

## What I would improve

1. Adversarial validation on non-deauth attacks (rogue AP, evil twin).
2. Rolling 5 s features to smooth isolated false positives.
3. Live monitor-mode capture via `scapy` → same detector.
4. Real WIPS action (iptables + active deauth injection) on an authorised lab.
5. `CalibratedClassifierCV` so `predict_proba` is a real probability.
6. Streaming learner (Hoeffding Tree) for concept drift.
