# Statement of AI-Assistant Use

**Required by the coursework brief.** This statement records the AI tools used during development and the author's role in reviewing, running, testing, and interpreting the work. Copy the relevant summary onto the research poster if required by the coursework brief.

## AI tools used

Two AI coding assistants were used at different stages:

- **DeepSeek / BetterDeepSeek** — used during the initial prototype development, as described below.
- **ChatGPT (OpenAI)** — used during the later AWID3 Deauthentication-focused dataset preparation, model evaluation, error investigation, documentation, and poster update.

## What AI was used for

### Initial prototype development (DeepSeek / BetterDeepSeek)

1. **Project scaffolding** — suggestions for the file layout, `requirements.txt`, and module boundaries (`loader → features → train → wips`). The author reviewed and adjusted the structure.
2. **Synthetic traffic simulator** (`src/synthetic.py`) — assistance generating the simulator, including frame-type distribution weights and the simulated legitimate Deauthentication traffic behaviour.
3. **Initial hyperparameter suggestions** — suggestions for Random Forest settings. The author tested alternatives and selected the configuration used in the prototype.
4. **First-pass docstrings and comments** — drafted with AI assistance and reviewed for accuracy.

### AWID3 Deauthentication experiment (ChatGPT)

ChatGPT was used as a technical aid to:

1. **Plan and explain dataset preparation** — cleaning AWID3 CSV exports, normalising hexadecimal frame-control values, auditing labels and features, and preparing capture-grouped partitions.
2. **Develop and review analysis scripts** — assistance with scripts for feature-quality audits, capture identity checks, binary Normal-versus-Deauth dataset preparation, model training, false-positive extraction, and false-positive pattern analysis.
3. **Debug implementation issues** — including NumPy bitwise operations, dataset processing, and evaluation-code corrections.
4. **Interpret model evaluation** — explain confusion-matrix values, precision, recall, F1, balanced accuracy, average precision, ROC AUC, class imbalance, and false-positive rates.
5. **Design validation checks** — assist with a separate capture-grouped validation threshold sweep and a final evaluation script that evaluates the existing model on the untouched test partition and attempts to match false-positive feature signatures to original capture records.
6. **Prepare technical documentation** — assist with the AWID3 experiment status report, README updates, and the HTML research poster using the results produced by the author's local runs.

AI assistance did not execute the experiments on the author's computer or independently verify the raw dataset. The author ran the scripts locally, supplied the resulting terminal output for interpretation, and remains responsible for verifying the source data, code, reported results, and final submission.

## What the author did and verified

- Obtained and ran the AWID3 dataset locally and executed the cleaning, audit, split, preparation, training, investigation, and final validation scripts.
- Reviewed dataset labels, feature completeness, capture distributions, and capture overlap.
- Used source capture files as grouping units to keep capture files separate between the original training and test partitions.
- Confirmed the binary experiment scope: Normal = 0, Deauth = 1; other AWID3 attack labels were excluded.
- Ran the Random Forest training and evaluated the complete untouched test partition in chunks.
- Ran the false-positive investigations and the final validation script, then supplied the actual command output used to update the experiment report and poster.
- Reviewed the generated documentation and the limitations stated in it.

## Important interpretation and limitations

- The current AWID3 experiment is a **packet-level binary Deauthentication WIDS baseline**. It is separate from the repository's earlier window-based prototype.
- The held-out test partition contains only **two independent Deauth capture files**. This limits the strength of claims about generalisation to unseen Deauthentication attack environments.
- The final test metrics were measured at the baseline threshold of **0.50**. The threshold sweep was conducted on a separate validation partition; the final test results were not used to choose a threshold.
- The false-positive investigation found that 3,774 of 3,986 false positives came from two Normal-labelled RogueAP captures and shared Deauthentication management-frame characteristics. A Deauthentication frame alone does not establish malicious intent; source-label semantics and packet context still require investigation.
- The experiment uses offline CSV records. It does not demonstrate live monitor-mode capture, packet injection, client disconnection, or actual access-point enforcement.

## What the author should be able to explain

- The dataset cleaning, label mapping, feature selection, and capture-grouped split.
- Why `source_file` is used for grouping and auditing rather than as a model feature.
- The Random Forest configuration and the meaning of each reported evaluation metric.
- Why class imbalance makes precision and absolute false-positive counts important.
- Why the threshold sweep is separate from the untouched final test evaluation.
- Why the RogueAP false positives require label and packet-context investigation rather than automatic relabelling.
- The difference between the packet-level AWID3 experiment and the repository's window-based prototype.
- The distinction between an offline detection result, an alert, a proposed blocklist, and actual wireless prevention.

---

**Author:** Chiwo Newa  
**Date:** 23/11/26
