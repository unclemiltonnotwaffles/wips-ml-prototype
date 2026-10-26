# Statement of AI-Assistant Use

**Required by the coursework brief.** Copy the relevant box onto your poster.

---

## What AI was used for

An AI coding assistant (DeepSeek / BetterDeepSeek) was used to:

1. **Scaffold the project structure** — file layout, `requirements.txt`, module
   boundaries (`loader → features → train → wips`). Reviewed and adjusted by
   the author.
2. **Generate the synthetic traffic simulator** (`src/synthetic.py`) — the
   frame-type distribution weights and the "1 % legitimate deauth" trick were
   produced by the assistant after the author specified the behaviour.
3. **Draft initial hyperparameters** — the author then tested `max_depth` of 8,
   12, and unlimited, and kept 12 because unlimited depth overfit to individual
   MAC addresses.
4. **Write first-pass docstrings and comments** — reviewed by the author;
   several rewritten for accuracy.

## What the author did by hand

- Chose the 11 features in `FEATURE_COLUMNS` (assistant suggested ~20; author
  cut down to the ones that could be justified in one sentence each).
- Chose the **time-ordered train/test split** — assistant's first draft used
  `train_test_split(random_state=42)`, rejected as leaking attack episodes.
- Designed the **strike-count blocking policy** (assistant first drafted
  block-on-first-alert; author changed to `BLOCK_AFTER_ALERTS = 2`).
- Wrote the test assertions and ran them.
- Ran the demo, inspected the confusion matrix, adjusted `WINDOW_SECONDS` from
  5 s to 1 s after observing detection latency.

## What the author can explain line-by-line

- Every column of `FEATURE_COLUMNS` and why it was chosen.
- Why windowing is necessary for a rate-based attack.
- Random Forest vs a hand-written threshold, and when each is appropriate.
- Why the time-ordered split is stricter than a random split.
- The code path from packet to blocked MAC in `wips.py`.
- The safety rationale behind `DRY_RUN = True`.

---

*Signed: ____________________________  Date: ____________*
