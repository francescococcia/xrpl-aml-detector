# Decode & Detect: The XRPL AI Hack

- Host: UK Centre for Blockchain Technologies (with Ripple, UCL CBT, XRPL Commons, Imperial DSS, KCL Blockchain, GirlsWhoML)
- Where: IDEALondon, 69 Wilson St, London EC2A 2BB
- When: Sat 24 Oct 2026 (London time). Arrival 09:30, check-in deadline 10:30.
- **Submissions close 16:30.** Final timetable (verification, awards) announced at the venue.
- Event page: https://luma.com/71nckopa
- Submission site: https://hackathon.ukcbt.org (team: TeamFranco). Terms: https://hackathon.ukcbt.org/terms

## Prizes
£10,000 total: £5,000 first, £3,500 second, £1,500 third. Ranked by test score, then eligibility + reproducibility checks. Leaderboard is provisional until verified.

## Task
Give every test transaction (40,101) a suspicion score 0–1 for planted money-laundering patterns. Synthetic data shaped like the XRPL (AWS/SonarX schema).

## Scoring
- Score = 100 × average precision (ties grouped at same threshold).
- Also shown: precision and recall at top 1% (ties broken by hash ascending).
- Best run per team counts. **50 runs per day per team**, reset 00:00 UTC (01:00 London).
- Sample preview starts one attempt; upload of the rest reuses it.

## Data facts
- Splits: train (labelled, labels.csv), validation (unlabelled, not scored by site), test. Each = 7 days. Wallets and hashes disjoint across splits.
- Tables: transactions, transfers (1–3 legs per tx, key TRANSACTION_HASH+TRANSFER_INDEX), blocks (join on LEDGER_INDEX + LEDGER_HASH), market_rates.csv (hourly USD price; PRICE_MICRO_USD / 1e6; ASSET = XRP or issuer:currency).
- XRP amounts are drops (string ints); IOUs are JSON {currency, issuer, value}. Group assets by TOKEN_ADDRESS **and** SYMBOL (USD from two issuers = different assets).
- SendMax is a ceiling, not the actual cost: read first executed leg.
- Only successful payments; no partial payments, failures, NFTs, batches. Fees always burned in XRP.
- Suspicious mix: 64% easy, 18% medium, 18% hard. Of 225 hard per split, 25 are pure noise (unlearnable).
- Hard cases: look at aggregate flows across longer histories and changing counterparties, not single amounts. **Test has batch variations absent from training**: avoid overfitting to exact train motifs.
- Legit look-alikes: commerce, peer payments, exchange/payroll hubs, multi-hop treasury flows.
- Laundering shapes from training: fan-in, fan-out, scatter-gather, gather-scatter, cycles, layer stacks, irregular networks.

## Submission format
- CSV: `TRANSACTION_HASH,score` (optional `reason`, not scored). Every test hash exactly once. ≤12 MiB, ≤100k rows.
- Code ZIP: ≤8 MiB compressed, ≤64 MiB uncompressed, ≤2,000 files, no ZIP64/encryption. Include source, README with exact command + hardware/time, dependency list + language version, LICENSE (MIT/BSD), list of things not written by us (libs, models, AI help), random seeds, team list, version control info, methodology doc (≤10 slides or ≤4 pages). No data, no venvs, no keys.
- Final entry also needs: team name, captain, member list.

## Rules
- Any language/tools/LLMs/pre-trained models allowed; disclose external data/services.
- No external labels, organiser files beyond what's supplied, generator seeds or test answers. Split membership is not a feature.
- Code must be built during the event and be reproducible.
- Teams 1–5, physically present only.
- Original work released under an approved open-source licence (commercial reuse allowed).

## Help on site
Mentors and tech staff for GPU environments, API credits and external-dataset questions.

## Added from slides + terms (10:58)
- Agenda: hacking 10:30-16:30, working lunch ~13:00. After 16:30: submission checks, winners, drinks.
- Sizes: 40,131 labelled training tx, 40,101 test tx. Top 1% = 402 tx (diagnostic only, with ROC-AUC and F1). Only AP decides ranking; ties are resolved by the organisers.
- Portal sign-in uses an organiser-issued access code. Upload ZIP + CSV together. **A CSV alone is not a complete final submission**: also complete the supporting-material step in the portal.
- ZIP must ALSO contain: the test predictions CSV, validation results, model details/settings (or model files), team name, captain, members, and the **commit hash** being submitted.
- Report/slides must cover: methodology, features, model or prompts, validation approach, results, interesting findings, limitations (incl. false positives, bias, uncertainty). Legible font.
- **Terms 5.3: the entry must be PUBLIC (open-source repo) by the 16:30 deadline**, with LICENSE + attribution notices.
- Terms 6.1: list every third-party lib/model/service with source + licence.
- If using an LLM: disclose model/provider, prompts, settings and reproducibility limits.
- Don't present results as evidence against real people; it's a research prototype, not a compliance tool.
- Problems at submission: contact Narmin Rahimli straight away. Keep own backups and upload early.
- Terms: https://hackathon.ukcbt.org/terms/2026-10-21
