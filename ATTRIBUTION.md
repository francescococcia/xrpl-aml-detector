# Components we did not write

Everything not listed here was written by the team during the event (24 Oct 2026).

## Organiser material
| item | source | licence / terms |
|---|---|---|
| `starter/payments.py`, `starter/starter_notebook.ipynb` | Decode & Detect organisers (UK Centre for Blockchain Technologies) | supplied for the event; used unchanged, imported, not rewritten |
| training, validation and test data | Decode & Detect organisers | event terms (https://hackathon.ukcbt.org/terms); not included in this repo or the ZIP |

## Libraries (versions pinned in `requirements.txt`, Python 3.13)
| library | version | licence | source |
|---|---|---|---|
| pandas | 3.0.6 | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| numpy | 2.5.4 | BSD-3-Clause | https://github.com/numpy/numpy |
| pyarrow | 26.0.0 | Apache-2.0 | https://github.com/apache/arrow |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| LightGBM | 4.7.0 | MIT | https://github.com/microsoft/LightGBM |
| SciPy | 1.18.1 | BSD-3-Clause | https://github.com/scipy/scipy |
| networkx | 3.7 | BSD-3-Clause | https://github.com/networkx/networkx |
| tabulate | 0.10.0 | MIT | https://github.com/astanin/python-tabulate |

## External data and services
None. No external labels, datasets, pre-trained models or APIs are used by the pipeline.

## AI assistance (disclosure)
No LLM is called at run time: predictions and explanations come only from the code in `src/`.
AI coding assistants were used during development; every change was reviewed and measured by the team.
| tool / provider | used for | settings / reproducibility |
|---|---|---|
| Cursor (AI coding assistant) | drafting and editing code and docs, under `.cursor/rules/one-task.mdc` and `AGENTS.md` | interactive chat; outputs are not deterministic, so the committed code is the reproducible artefact |
| Grok (xAI) | first draft of the feature pipeline (`src/features.py`) | interactive chat; reviewed and rewritten by the team |
| Claude Code (Anthropic) | code audit; fix for the split-stable `deliv_asset_code`; `make_zip.py`; smoke tests; this file | interactive session; prompts asked to audit the solution and fix the findings |

Prompts were interactive task requests (one card from `knowledge/tasks.md` at a time), not part of the model.

## Sources consulted
Listed with links in `knowledge/approach.md` and `docs/method_report.md`.
