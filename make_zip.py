"""Build the Decode & Detect submission ZIP: the committed source (git HEAD), the test predictions, the CV results,
the example alerts and SUBMISSION.md (team, captain, commit hash, run command, hardware, seeds). Checks the event
limits: <= 8 MiB compressed, <= 64 MiB uncompressed, <= 2,000 files, no ZIP64, no data files."""
import argparse
import io
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
TEAM, CAPTAIN, MEMBERS = "Follow the Drops", "Francesco Coccia", ["Francesco Coccia"]
OUTPUTS = ("predictions.csv", "cv_results.md", "example_alerts.md", "final_settings.json")  # gitignored, so added explicitly
REQUIRED = ("outputs/predictions.csv", "outputs/cv_results.md")
FORBIDDEN = (".parquet", ".zip", "labels.csv", ".env")
MAX_ZIP, MAX_RAW, MAX_FILES = 8 * 2**20, 64 * 2**20, 2000


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def submission_md(commit, branch, remote):
    return f"""# Submission: {TEAM}

- Team: {TEAM}
- Captain: {CAPTAIN}
- Members: {", ".join(MEMBERS)}
- Commit: `{commit}` (branch `{branch}`)
- Repository: {remote}
- Predictions: `outputs/predictions.csv` (one row per test TRANSACTION_HASH, scores in [0, 1], no ties); the exact
  `predict.py` command and switches that produced it are in `outputs/final_settings.json`
- Validation results: `outputs/cv_results.md` (GroupKFold(5) by scenario_id)
- Model details and settings: `src/common.py` (LightGBM / HGB parameters, weights, smoothing), `src/predict.py` (blend);
  methodology in `docs/method_report.md` / `docs/method_report.pdf`
- Seeds: 42 everywhere; the final LightGBM averages seeds 42-46
- Environment: Python 3.13, `pip install -r requirements.txt`; this ZIP was built with Python {platform.python_version()}
  on {platform.system()} {platform.machine()}
- Run (about 4.2 min cold on 8 cores):

```
python src/train.py   --train-dir TRAIN --cache-dir /tmp/xrpl-cache
python src/predict.py --train-dir TRAIN --test-dir TEST --cache-dir /tmp/xrpl-cache
python src/explain.py --train-dir TRAIN --test-dir TEST --cache-dir /tmp/xrpl-cache
```
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(REPO.parent / "follow-the-drops-submission.zip"))
    p.add_argument("--allow-dirty", action="store_true", help="build even with uncommitted changes (not recommended)")
    args = p.parse_args()
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    if dirty and not args.allow_dirty:
        sys.exit(f"uncommitted changes; commit first so the ZIP matches its commit hash:\n{dirty}")
    missing = [f for f in REQUIRED if not (REPO / f).exists()]
    if missing:
        sys.exit(f"missing {missing}: run src/train.py and src/predict.py first")
    commit = git("rev-parse", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    try:
        remote = re.sub(r"//[^/@]+@", "//", git("remote", "get-url", "origin").strip())  # never write a token
    except subprocess.CalledProcessError:
        remote = "(no remote)"
    files = {}
    for name in git("ls-files", "-z").split("\0"):
        if name and (REPO / name).is_file():
            files[name] = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=REPO, check=True, capture_output=True).stdout
    for name in OUTPUTS:
        if (REPO / "outputs" / name).exists():
            files[f"outputs/{name}"] = (REPO / "outputs" / name).read_bytes()
    files["SUBMISSION.md"] = submission_md(commit, branch, remote).encode()
    bad = [f for f in files if f.endswith(FORBIDDEN)]
    assert not bad, f"data or secret files would be zipped: {bad}"
    raw = sum(len(b) for b in files.values())
    assert len(files) <= MAX_FILES, f"{len(files)} files > {MAX_FILES}"
    assert raw <= MAX_RAW, f"{raw / 2**20:.1f} MiB uncompressed > 64 MiB"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=False) as z:
        for name in sorted(files):
            z.writestr(f"{TEAM.lower().replace(' ', '-')}/{name}", files[name])
    data = buf.getvalue()
    assert len(data) <= MAX_ZIP, f"{len(data) / 2**20:.2f} MiB compressed > 8 MiB"
    Path(args.out).write_bytes(data)
    print(f"wrote {args.out}: {len(files)} files, {raw / 2**20:.2f} MiB raw, {len(data) / 2**20:.2f} MiB zipped, "
          f"commit {commit[:12]}{' (DIRTY)' if dirty else ''}")


if __name__ == "__main__":
    main()
