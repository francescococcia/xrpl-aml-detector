# Claude Code instructions

All project rules and knowledge are in `AGENTS.md` (shared with other AI assistants). Read it first:

@AGENTS.md

## Claude-specific notes
- Data is usually not in a cloud session container: say so instead of inventing CV or leaderboard numbers, and test
  code with `python -m unittest discover -s tests` or a small synthetic dataset outside the repo.
- Do not overwrite tracked files in `outputs/` (`cv_results.md`, `example_alerts.md`) with dummies when testing.
- Commit with `--author="Francesco Coccia <francesco_coccia@hotmail.com>"`.
