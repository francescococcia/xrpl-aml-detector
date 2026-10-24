# Tasks

## Setup
- [x] S0 Project folder + git
- [x] S1 venv + `pip install -r requirements.txt`
- [x] S2 Run starter notebook end to end, upload baseline CSV (first leaderboard score)
- [x] S3 Local validation: hold-out on train, same AP metric

## Feature cards (fill in after baseline)

## Submission (from 15:50)
- [x] P1 Make the GitHub repo public (required by 16:30)
- [x] P2 Methodology report (4 pages max) or slides (fewer than 10)
- [ ] P3 ZIP: code, README with commands, requirements + Python version, LICENSE, ATTRIBUTION, seeds, team list, commit hash, predictions CSV, validation results
- [ ] P4 Upload ZIP + CSV together, then finish the supporting-material step


## Plan (set 11:28)
- [x] 11:30 U1 Upload v1 predictions (CV 53.3) -> first leaderboard score
- [x] 12:15 M2 Relay hub scoring + stage-2 (tried, dropped; see cv_results.md phase 2)
- [x] 12:20 U2 Upload v2 if CV improved; compare leaderboard vs CV
- [x] 13:00 Lunch: read knowledge/training-pack.md + pitch-brief.md
- [ ] 13:30-15:00 M3 Tuning, seed ensemble, check test shift
- [x] 15:00 P0 Create public GitHub repo, push
- [x] 15:00-15:50 P2 Methodology report (4 pages max)
- [ ] 15:50 Feature freeze, final predictions, P3 build ZIP
- [ ] 16:10 P4 Upload ZIP + CSV, supporting materials (reminder set)

## Audit follow-ups (13:30)
- [x] A1 Split-stable deliv_asset_code, cache v5, n_jobs, ATTRIBUTION, make_zip.py, smoke tests
- [ ] A2 Re-run `src/train.py` + `src/predict.py` (cache v5), update numbers in README / method report if they move
- [ ] A3 Adversarial validation train vs test; drop or rank-normalise shifted features (hour/dow first)
- [ ] A4 Funder features (first incoming payment = account creator; shared funder, siblings in the same flow)
- [ ] A5 `python make_zip.py` after the final commit; upload ZIP + CSV; finish supporting-material step
