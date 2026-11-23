# AWID3 Deauthentication Detection Experiment

This experiment uses machine learning to identify Wi-Fi deauthentication frames in the AWID3 wireless dataset.

## Workflow

1. Clean the original AWID3 CSV files.
2. Split records into training and testing sets by capture file.
3. Prepare a binary dataset: Normal (0) and Deauth (1).
4. Train a Random Forest classifier.
5. Evaluate the model and investigate false positives.
6. Validate classification thresholds.

## Requirements

- Python 3.10 or newer
- AWID3 dataset in CSV format

Install dependencies with:

    python -m pip install -r requirements.txt

## Dataset configuration

Set the path to the original AWID3 CSV directory:

    $env:AWID3_RAW_DIR = "C:\path\to\AWID3\CSV"

## Run the experiment

Run these commands from the awid3_experiment directory:

    python prepare_awid3_dataset.py
    python create_grouped_split.py
    python materialize_grouped_datasets.py
    python prepare_deauth_datasets.py
    python train_deauth_model.py
    python investigate_false_positives.py
    python analyse_false_positive_patterns.py

Final validation:

    python run_final_deauth_checks.py --raw-dir "C:\path\to\AWID3\CSV"

## Model

The Random Forest uses 11 packet-level features and 200 trees, with a classification threshold of 0.50.

## Final test results

The test dataset contains 5,721,566 records.

| Metric | Result |
|---|---:|
| Deauth precision | 66.88% |
| Deauth recall | 99.05% |
| Deauth F1-score | 79.85% |
| Balanced accuracy | 99.49% |
| False-positive rate | 0.0698% |

See the results directory for aggregate evaluation reports.

## Limitations

This is an offline packet-classification experiment. It does not perform live wireless monitoring, packet injection, client disconnection, access-point control or network enforcement.

Some Normal-labelled AWID3 records contain deauthentication-subtype frames. Their presence does not, by itself, prove malicious activity.

## Data handling

Raw datasets, trained model files and packet-level diagnostic records are excluded from this repository. They must be supplied or generated locally.
