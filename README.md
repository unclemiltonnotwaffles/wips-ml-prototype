# Wi-Fi Deauthentication Detection Prototype

This project explores how machine learning can help identify suspicious Wi-Fi deauthentication activity.

A deauthentication frame is a Wi-Fi management frame that tells a device to disconnect from a wireless network. Attackers can send large numbers of these frames to disrupt wireless connections. The goal of this project is to detect patterns that may indicate this type of attack.

**This is a research prototype, not a live wireless protection system.** It analyses saved data and can generate alerts and a proposed blocklist. It does not monitor Wi-Fi traffic live, disconnect devices, or block attackers through an access point.

## How it works

At a high level, the project:

1. Reads Wi-Fi traffic records from a dataset or generated test data.
2. Extracts useful information from the records, such as frame type, frame subtype, signal strength, and timing.
3. Uses a Random Forest machine-learning model to classify activity.
4. Reports its predictions and evaluation results.
5. In the original prototype, can generate alerts and record MAC addresses in a proposed blocklist.

## Two parts of the project

The repository contains two related but separate experiments:

- **Original prototype:** groups Wi-Fi records into one-second time windows, extracts traffic features, and looks for deauthentication-flood-like activity. Its response stage produces alerts and a proposed blocklist.
- **AWID3 Deauthentication experiment:** analyses individual records from the AWID3 dataset and classifies them as either Normal (0) or Deauth (1). Other AWID3 attack categories are excluded from this binary experiment.

The AWID3 experiment is documented separately in [the experiment report](docs/DEAUTH_EXPERIMENT_STATUS.md).

## AWID3 experiment results

The Random Forest baseline was tested on 5,721,566 records that were kept separate from training. At the 0.50 classification threshold, it achieved:

| Measure | Result |
|---|---:|
| Deauth precision | 66.88% |
| Deauth recall | 99.05% |
| Deauth F1-score | 79.85% |
| Balanced accuracy | 99.49% |
| False positives | 3,986 |
| False negatives | 77 |

The model detected most Deauth records, but it also incorrectly flagged some Normal records. Most of these false alarms came from two RogueAP capture files. This is being investigated because a deauthentication frame is not automatically evidence of a malicious attack.

**Important limitation:** the test data contains only two independent Deauth capture files. More independent captures are needed before claiming that the model will work reliably in other networks and environments. These results come from offline dataset analysis, not live Wi-Fi testing.

## Run the original prototype

Set up the Python environment:

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Install the required packages and run the tests and demo:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
python -m src.run_demo
```

The demo writes its generated outputs to the `results/` directory.

To run the original prototype with AWID3 CSV files:

```bash
python -m src.run_awid3 --data-dir /path/to/awid3/csv_folder
```

Check the dataset's column names, labels, timestamps, and frame-control values before interpreting its results. AWID3 exports can differ.

## Safety and responsible use

Use only wireless data and networks you are authorised to analyse. The project does not perform real wireless blocking or transmit deauthentication frames. The proposed blocklist is only an output for review; it is not an enforced security control.

## Project documentation

- [Research poster](poster/poster.html)
- [Detailed AWID3 Deauthentication experiment report](docs/DEAUTH_EXPERIMENT_STATUS.md)
- [Statement of AI-assistant use](docs/AI_USAGE.md)
