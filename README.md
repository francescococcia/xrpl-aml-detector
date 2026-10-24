# XRPL AML detector (Decode & Detect, TeamFranco)

Scores every transaction in the synthetic XRPL test set with a 0–1 suspicion score for money-laundering patterns.

## Data layout (not in this repo)
Data lives next to this folder:
- `../decode-and-detect-training-data/train/` (labelled), `validation/` (unlabelled)
- `../decode-and-detect-test-data/` (40,101 transactions to score)

## Setup
```sh
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Reproduce predictions
TODO: exact command, runtime and hardware.

## What we did not write
See `ATTRIBUTION.md`.
