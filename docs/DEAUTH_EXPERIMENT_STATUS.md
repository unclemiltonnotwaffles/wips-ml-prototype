# Deauthentication WIDS/WIPS: Dataset Preparation and Baseline Experiment

**Status:** Packet-level baseline trained and evaluated; false-positive investigation completed to the pattern-analysis stage.  
**Last updated:** 2026-11-23

## 1. Current project objective

The current AWID3 experiment is scoped specifically to **binary Deauthentication detection**. It is not a 14-class classifier.

| Dataset label | Binary target |
|---|---:|
| Normal | 0 |
| Deauth | 1 |
| All other AWID3 attack labels | Excluded from this binary experiment |

The intended system is a Deauthentication-focused Wireless Intrusion Detection System (WIDS), with a later response/prevention workflow (WIPS). Packet-level classification is an initial component; operational attack detection also needs temporal aggregation and careful response controls.

## 2. Data preparation

### Cleaned source data

- Cleaned dataset: `processed_awid3/awid3_cleaned.csv`
- Records retained: **36,962,587**
- Source capture files: **745**
- Missing-label records excluded: **310**
- Records with insufficient selected features excluded: **2**
- Canonical AWID3 labels observed: **14**

The cleaning stage also normalised `wlan.fc.ds`, including hexadecimal values such as `0x00000002`.

### Capture-level train/test split

A grouped split was created using `source_file` as the group identifier. Candidate splits were searched to preserve label representation in both partitions.

| Measure | Training | Testing |
|---|---:|---:|
| Capture files | 596 | 149 |
| Records | 29,613,105 | 7,349,482 |

- Capture-file overlap: **0**
- Test record fraction: **19.88%**
- All 14 canonical labels were represented in both original partitions.

This avoids placing records from the same capture file in both partitions. It does not by itself guarantee broad generalisation across independent attack environments.

### Binary Deauth datasets

The Deauth-specific preparation stage produced:

| Measure | Training | Testing |
|---|---:|---:|
| Total records | 180,817 | 5,721,566 |
| Normal records | 150,000 | 5,713,441 |
| Deauth records | 30,817 | 8,125 |
| Normal capture files represented | 596 | 149 |
| Deauth capture files represented | 10 | 2 |

Training retained all Deauth records and used reservoir sampling to cap Normal records at 150,000. The test partition retained all Normal and Deauth records and was not rebalanced.

**Evaluation caveat:** only two independent Deauth capture files are represented in the test set. This makes attack generalisation estimates fragile. Results must be reported with this limitation.

## 3. Baseline features

The model uses 11 numeric features:

1. `frame.len`
2. `frame.time_delta`
3. `radiotap.channel.freq`
4. `wlan.fc.protected`
5. `wlan.fc.type`
6. `wlan.fc.retry`
7. `wlan.fc.subtype`
8. `wlan_radio.channel`
9. `wlan_radio.signal_dbm`
10. `to_ds`
11. `from_ds`

The original `wlan.fc.ds` field is decomposed into `to_ds` and `from_ds`. NumPy bitwise operations are used for this decomposition. `source_file` is retained for auditing and grouping only; it is not a model feature.

Protocol-dependent fields with substantial or class-dependent missingness (for example, datarate and IP/TCP/UDP fields) were excluded from this baseline rather than treating missing values as zero.

## 4. Random Forest baseline

The trained baseline is a scikit-learn Random Forest with:

- `n_estimators=200`
- `max_depth=20`
- `min_samples_split=4`
- `min_samples_leaf=2`
- `max_features="sqrt"`
- `class_weight="balanced_subsample"`
- `random_state=42`
- `n_jobs=-1`

The model was evaluated over the complete Deauth test dataset in chunks of 250,000 records at a probability threshold of **0.50**.

### Results

Confusion matrix (rows = actual [Normal, Deauth], columns = predicted [Normal, Deauth]):

```text
[[5709455    3986]
 [     77    8048]]
```

| Metric | Result |
|---|---:|
| True negatives | 5,709,455 |
| False positives | 3,986 |
| False negatives | 77 |
| True positives | 8,048 |
| Deauth precision | approximately 0.67 |
| Deauth recall | approximately 0.99 |
| Deauth F1-score | approximately 0.80 |
| Balanced accuracy | 0.994913 |
| Average precision | 0.964010 |
| ROC AUC | 0.999946 |
| False-positive rate | 0.00069765 (approximately 0.0698%) |

Accuracy alone is not informative for this highly imbalanced test set. The test set has approximately 703 Normal records per Deauth record. Precision, recall, F1, average precision, balanced accuracy, and the absolute false-positive count should be reported together.

### Feature importance

| Feature | Importance |
|---|---:|
| `wlan.fc.type` | 0.427644 |
| `wlan.fc.subtype` | 0.194503 |
| `wlan_radio.signal_dbm` | 0.134650 |
| `frame.len` | 0.119409 |
| `frame.time_delta` | 0.087639 |
| `wlan.fc.protected` | 0.021145 |
| `from_ds` | 0.008246 |
| `to_ds` | 0.004286 |
| `wlan.fc.retry` | 0.001289 |
| `wlan_radio.channel` | 0.000717 |
| `radiotap.channel.freq` | 0.000472 |

Feature importance describes how the fitted forest uses its inputs; it is not proof of causal importance or independent feature value.

## 5. False-positive investigation

The false-positive extraction identified all **3,986** Normal test records predicted as Deauth at threshold 0.50. A comparison sample of **23,000** correctly classified Normal records was also collected.

### Concentration by capture

| Capture | False positives | Normal records | False-positive rate |
|---|---:|---:|---:|
| `RogueAP_36.csv` | 2,609 | 49,931 | 5.2252% |
| `RogueAP_37.csv` | 1,165 | 49,895 | 2.3349% |
| `Kr00k_50.csv` | 135 | 43,478 | 0.3105% |
| Other captures combined | 77 | — | — |

The two RogueAP captures account for **3,774 / 3,986 (94.68%)** of all false positives.

### Frame-pattern finding

All 3,774 RogueAP false-positive records have:

- `wlan.fc.type = 0` (management frame)
- `wlan.fc.subtype = 12` (Deauthentication)
- `frame.len = 86`
- `wlan.fc.protected = 0`

Their mean received signal level is approximately **-40.56 dBm**.

Across all false positives, the subtype distribution is:

- Subtype 12: **99.523%**
- Subtype 10: **0.477%**

This indicates that the dominant error pattern is not random: Normal-labelled records in RogueAP captures contain Deauthentication management frames, and the packet-level classifier labels them as attacks.

**Interpretation must remain cautious.** A Deauthentication frame is not, by itself, proof of a malicious Deauthentication attack. The current results reveal a mismatch between packet-level frame characteristics and the binary dataset labels, or a need for contextual/temporal features. The original packet records and label semantics must be examined before changing labels, dropping features, or retraining.

## 6. Current limitations and evaluation discipline

1. Only two independent Deauth capture files are present in the test partition.
2. The highly imbalanced natural test distribution creates a meaningful false-alarm burden despite a low false-positive rate.
3. The current classifier predicts individual packet records; it does not yet establish whether a sequence constitutes an attack event.
4. The RogueAP false-positive pattern requires inspection of original source fields and dataset labelling semantics.
5. Threshold selection must use a separate validation set or capture-grouped validation. Do not tune the threshold against the final test set.
6. Do not claim live wireless detection, packet injection, client disconnection, or actual prevention from offline dataset results.

## 7. Work completed

- Cleaned and audited the AWID3 records.
- Audited feature quality, capture distribution, capture identity, and model inputs.
- Created a capture-grouped train/test partition with zero capture overlap.
- Prepared a binary Normal-versus-Deauth training and test dataset.
- Trained and saved a Random Forest baseline.
- Evaluated the full test partition in chunks.
- Extracted false-positive records and a Normal comparison sample.
- Analysed false-positive capture concentration and frame type/subtype patterns.

## 8. Next work

1. Inspect the original RogueAP capture columns and relevant packet records, including address fields, timestamps, sequence information, and any available Deauthentication reason-code fields.
2. Verify AWID3 label meaning for the affected RogueAP records against the dataset documentation/source.
3. Determine whether useful contextual fields are available and valid across captures.
4. Create a capture-grouped validation strategy for threshold selection.
5. Develop temporal/window-based detection so that attack decisions consider rates and repeated activity over time.
6. Evaluate the event-level detector on independent captures and report false alarms per time interval as well as packet-level metrics.
7. Keep any mitigation stage explicitly separated from detection and ensure responses are authorised, auditable, reversible, and tested safely.

## 9. Reproducibility artifacts

The working experiment directory currently contains the following named artifacts:

- `processed_awid3/awid3_cleaned.csv`
- `processed_awid3/awid3_train.csv`
- `processed_awid3/awid3_test.csv`
- `processed_awid3/train_capture_files.csv`
- `processed_awid3/test_capture_files.csv`
- `processed_awid3/grouped_split_report.csv`
- `processed_awid3/materialized_split_summary.csv`
- `processed_awid3/awid3_deauth_train.csv`
- `processed_awid3/awid3_deauth_test.csv`
- `models/deauth_random_forest.joblib`
- `results/deauth_baseline_metrics.csv`
- `results/deauth_classification_report.csv`
- `results/deauth_feature_importance.csv`
- `results/false_positive_records.csv`
- `results/false_positive_by_capture.csv`
- `results/false_positive_feature_summary.csv`
- `results/normal_comparison_sample.csv`
- `results/false_positive_capture_rates.csv`
- `results/false_positive_wlan_fc_type_distribution.csv`
- `results/false_positive_wlan_fc_subtype_distribution.csv`
- `results/false_positive_type_subtype_comparison.csv`

Large raw/processed datasets and trained binary model artifacts are intentionally not included in this documentation update. Add the reproducible scripts and compact reports to version control after checking dataset licensing, file sizes, and repository ignore rules.
