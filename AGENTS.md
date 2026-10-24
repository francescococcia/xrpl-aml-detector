# Agent guide: xrpl-aml-detector (team Follow the Drops)

This is the single source of rules and project knowledge for every AI assistant on this repo: Claude Code reads it
through `CLAUDE.md`, Cursor through `.cursor/rules/`, Codex-style agents read `AGENTS.md` directly, and chat assistants
without repo access (e.g. Grok) get it pasted or attached at the start of a conversation. Edit rules here, not in copies.

## 1. The task in one paragraph
Decode & Detect: The XRPL AI Hack (UK CBT, IDEALondon, Sat 24 Oct 2026, submissions close **16:30 London**).
Give every test payment (40,101) a 0-1 laundering suspicion score. Ranking metric: **100 × average precision** on
transaction-level scores, so only the order matters. One score per `TRANSACTION_HASH`, never per transfer leg.
Synthetic XRPL-shaped data (transactions, transfers, blocks, hourly market rates). Full facts: `knowledge/event.md`.

## 2. Hard rules (never break)
- **No data in git or the ZIP**: `*.parquet`, `labels.csv`, `market_rates.csv`, caches. Data lives outside the repo
  (`../decode-and-detect-training-data/train`, `../decode-and-detect-test-data`).
- **No secrets** in the repo (portal access code, API keys, tokens). The repo is public.
- **Label columns are never features**: `family`, `difficulty`, `scenario_id`, `cohort_id`. `scenario_id` is only the CV
  group key; `family` is only used for training weights (relay/ambiguous positives weight 0). Split membership is not a feature.
- **No external labels, data or services** without listing them in `ATTRIBUTION.md`.
- **Validate with `GroupKFold(5)` by `scenario_id`** (NaN → unique id). Any score-derived quantity (smoothing, stacking)
  must use same-fold out-of-fold scores only: the cross-fold stacking "+3.1 AP" was leakage.
- **The submitted CSV must be what the committed code produces** (organisers re-run it). Change a setting → re-run
  `predict.py` → upload that CSV.
- **Disclose AI help**: if you are a new AI tool on this repo, add yourself to the AI table in `ATTRIBUTION.md`.

## 3. Pipeline (current = v3 final)
| file | role |
|---|---|
| `starter/payments.py` | organiser helper `load_payments`; import it, do not edit |
| `src/features.py` | 175 per-payment graph, velocity, pair, multi-hop, leg/currency features |
| `src/flows.py` | 72 label-free flow features (shared-wallet components, 30/120/600 s undirected, 60/600/3600 s forward) + component ids |
| `src/common.py` | paths, CLI, model params, `train_weights`, `smooth_scores` (λ = **0.25**), feature cache `features_v5_*` |
| `src/train.py` | grouped CV report → `outputs/cv_results.md` |
| `src/predict.py` | LightGBM (seeds 42-46) + HGB + LR, rank blend 2:1:1, flow max-score smoothing, tie-breaker → `outputs/predictions.csv` |
| `src/explain.py` | top alerts with flow summary + LightGBM `pred_contrib` reasons → `outputs/example_alerts.md` |
| `make_zip.py` | submission ZIP (git HEAD + outputs + SUBMISSION.md with commit hash); checks event limits |
| `experiments/` | dropped ideas, documented; not part of the pipeline |

Results: CV AP v1 53.3 → v2 59.5 → v3 62.2 ± 4.8 (λ 0.5); public LB 57.4 → 63.5 → 64.4 → **64.5 (λ 0.25)**.
Weak spots: relays (~1 AP, indistinguishable from decoys), medium tier (9.8), hard tier (0.7), bipartite (58.5).

## 4. Commands
```bash
pip install -r requirements.txt                        # Python 3.13
python -m unittest discover -s tests                   # smoke tests, no data needed
python src/train.py   --cache-dir /tmp/xrpl-cache      # CV (~3 min)
python src/predict.py --cache-dir /tmp/xrpl-cache      # predictions + output checks (~1 min)
python src/explain.py --cache-dir /tmp/xrpl-cache      # example alerts
python make_zip.py                                     # only after the final commit
```

## 5. How to work
- Take **one card** from `knowledge/tasks.md`, do only that card, tick it, add a one-line note, commit with the card id,
  stop. A card bigger than 20 minutes gets split into new cards.
- Before claiming a gain: same folds, report mean ± std and per-fold AP, and say how many folds improved. Gains inside
  the noise (< ~0.5 AP or not most folds) are not adopted. Log tried ideas, adopted or not, in `outputs/cv_results.md`
  (below the hand-written-notes marker) and decisions in `knowledge/decisions.md`.
- Mark own measurements "(our data)"; every external claim gets a source link.
- Commit author is Francesco Coccia `<francesco_coccia@hotmail.com>`.
- Before pushing: smoke tests pass, no data or secrets staged (`git status`), docs match the code.

## 6. Submission checklist (by 16:30)
Repo public · `ATTRIBUTION.md` complete · re-run `train.py` + `predict.py` · commit · `python make_zip.py` ·
upload ZIP + CSV together · finish the supporting-material step · keep backups.

## 7. Where knowledge lives
| file | content |
|---|---|
| `knowledge/event.md` | event facts, rules, submission format |
| `knowledge/tracks.md` | kickoff idea list |
| `knowledge/data.md` | data profile |
| `knowledge/approach.md` | study guide: XRPL, AML typologies, features, phases 2-4 experiments, sources |
| `knowledge/decisions.md` | timestamped decisions + leaderboard log |
| `knowledge/tasks.md` | task cards |
| `knowledge/pitch-brief.md` | product/impact/security brief and 2-min pitch |
| `docs/method_report.md` | 4-page methodology report |
| `outputs/cv_results.md` | CV tables and every experiment |
