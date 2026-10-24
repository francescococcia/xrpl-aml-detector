# Pitch brief: product, impact, security (Decode & Detect)
*Sources were opened today. "(our data)" = our own measurements; "(advice)" = framing suggestion.*

## 1. Problem
- **Scale.** An estimated 2–5% of global GDP, or $800bn–$2tn a year, is laundered. ([UNODC](https://www.unodc.org/unodc/en/money-laundering/overview.html))
- **Crypto is growing.** Illicit crypto addresses received at least $40.9bn in 2024 (possibly about $51bn). ([Chainalysis 2025](https://www.chainalysis.com/blog/2025-crypto-crime-report-introduction/)) On-chain laundering grew from $10bn (2020) to more than $82bn (2025). ([Chainalysis 2026](https://www.chainalysis.com/blog/2026-crypto-money-laundering/))
- **Transparent but hard to police.**
  - Every transaction is public and traceable, but accounts are pseudonymous. ([Chainalysis glossary](https://www.chainalysis.com/glossary/money-laundering/))
  - XRPL itself holds no personal identities. ([xrpl.org DEX](https://xrpl.org/docs/concepts/tokens/decentralized-exchange))
  - Ledgers close every 3–5 s ([close times](https://xrpl.org/docs/concepts/ledgers/ledger-close-times)), and some laundering services clear large transfers in about 1.6 min ([Chainalysis 2026](https://www.chainalysis.com/blog/2026-crypto-money-laundering/)).
  - Cross-currency path payments can hop across currencies inside one atomic transaction. ([Cross-currency](https://xrpl.org/docs/concepts/payment-types/cross-currency-payments))
- **Rules alone drown analysts.** Industry false-positive rates are "upwards of 90%". ([Weber et al. 2019](https://arxiv.org/abs/1908.02591))

## 2. User
- **Primary user:** a transaction-monitoring / compliance analyst at an exchange, custodian or other VASP that must spot and report suspicious activity. UK cryptoasset businesses register with the FCA under the MLRs and are supervised on a risk basis. ([FCA](https://www.fca.org.uk/firms/financial-crime/cryptoassets-aml-ctf-regime))
- **Secondary users:** stablecoin/IOU issuers (who can freeze trust lines, see [Trust lines](https://xrpl.org/docs/concepts/tokens/fungible-tokens/trust-line-tokens)) and FIU / law-enforcement analysts.

## 3. Product: "ranked alert queue"
- **Input.** XRPL-shaped tables (transactions, transfers, blocks, market_rates), the same schema as the free AWS/SonarX XRPL dataset. ([AWS Registry](https://registry.opendata.aws/aws-public-blockchain/))
- **Output.**
  - A 0–1 risk score per transaction, sorted into a queue.
  - For each alert: the top 3 reasons in plain English (from SHAP / feature contributions), e.g. "received from 7 new accounts in 20 min, forwarded 98% within 1 ledger window, crossed USD→EUR via path payment".
  - The likely pattern family (fan-in, chain, cycle, relay…).
  - A small subgraph view.
  - Links to the transaction hashes so the analyst can verify each claim on the ledger. (advice)
- **Why it fits XRPL.**
  - Features are built on payment-level flows: issuers and liquidity providers are treated as plumbing, not counterparties ([Rippling](https://xrpl.org/docs/concepts/tokens/fungible-tokens/rippling), [Paths](https://xrpl.org/docs/concepts/tokens/fungible-tokens/paths)).
  - Amounts use `delivered_amount` ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments)).
  - Values are normalised with hourly rates.
- **Model.** LightGBM on about 190 graph/flow features. Grouped-CV AP is 53.3, against 28.3 for the starter baseline (our data). This is the GBT + graph-features family that matched GNNs on IBM's AMLworld benchmark. ([Altman et al.](https://arxiv.org/abs/2306.16424))

## 4. Impact
- **Analyst time.** AP rewards putting true positives at the top of the queue ([sklearn AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)). Say: "at the top-N alerts our precision is X% vs Y% for the baseline" (fill in from out-of-fold results, our data).
- **Fewer false positives** means less wasted investigation and less harm to innocent customers. AML costs also drive financial exclusion. ([Weber et al.](https://arxiv.org/abs/1908.02591))
- **Regulatory context.**
  - The FATF Travel Rule requires VASPs to pass originator and beneficiary information with transfers. 99 jurisdictions have passed or are passing Travel Rule laws, and FATF says most on-chain illicit activity now involves stablecoins. ([FATF 2025](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/targeted-update-virtual-assets-vasps-2025.html))
  - Our USD/GBP/EUR IOUs are stablecoin-like issued tokens. ([Currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats))
- **Explainable and aligned with FATF red flags** (structuring, rapid pass-through, new accounts, needless currency conversion). ([FATF red flags](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/)) Analysts and regulators recognise these terms.

## 5. Security & responsible AI
| Risk | Mitigation |
|---|---|
| Privacy / pseudonymity: linking addresses to people | Score transactions, never identities. No off-chain enrichment in the prototype. Synthetic data only. (organiser rules) |
| False-positive harm (frozen funds, debanking) | Human-in-the-loop: the score triggers *review*, never automatic action. Show the reasons and the uncertainty. FATF itself treats a single red flag as context-dependent, not proof. ([FATF red flags](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/)) |
| Bias | Pattern shapes like fan-out also fit payroll and merchants ([Altman](https://arxiv.org/abs/2306.16424)). Report per-family precision and false-positive rate across legitimate look-alike decoys (our data). Don't use account IDs as features ([Altman](https://arxiv.org/abs/2306.16424)). |
| Adversarial adaptation | Launderers change behaviour when detection logic leaks; Altman released data, not generator code, for this reason ([Altman, Ethics](https://arxiv.org/abs/2306.16424)). Vendors migrate after enforcement ([Chainalysis 2026](https://www.chainalysis.com/blog/2026-crypto-money-laundering/)). Mitigate with diverse features, periodic retraining, and not publishing exact thresholds. |
| Model drift | Elliptic models failed after a dark-market shutdown ([Weber et al.](https://arxiv.org/abs/1908.02591)). Monitor feature shift (adversarial validation), retrain, and keep the human review loop. |
| Synthetic-data caveat | Performance may not transfer one-to-one to real data ([Altman, App. B](https://arxiv.org/abs/2306.16424)). Real labels are incomplete, while synthetic labels are complete ([Altman](https://arxiv.org/abs/2306.16424)). |
| Mis-reading amounts | Use `delivered_amount`, not `Amount` (the partial-payment exploit). ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments)) |
| No real allegations | Outputs are research prototypes, not compliance decisions. Disclose AI/LLM use. Open-source licence. (organiser rules) |

## 6. Limitations (state them honestly; judges reward it)
- Relays: about 200 suspicious relays hidden in about 10,000 legitimate ones are structurally identical, so recall there is limited (our data).
- 25 "ambiguous" labels are unlearnable by design (our data).
- Only 7 days and 676 senders, so CV variance is high. Report AP mean ± std across folds (our data; [sklearn CV](https://scikit-learn.org/stable/modules/cross_validation.html)).
- Time resolution is about 10 s ([close times](https://xrpl.org/docs/concepts/ledgers/ledger-close-times)).
- There is no off-chain KYC context, which real analysts would have.

## 7. Two-minute pitch script (~280 words)
> Every year, an estimated two to five percent of global GDP is laundered, up to two trillion dollars. Crypto makes this both easier and harder to hide. On the XRP Ledger every payment is public, but accounts are pseudonymous, ledgers close every few seconds, and a single path payment can hop from dollars to euros through a liquidity provider in one atomic step.
>
> Today, compliance analysts at exchanges face rules-based alert queues where most alerts are false positives. That wastes time and harms innocent customers.
>
> We built a ranked alert queue for XRPL. It reads ledger-shaped data (transactions, transfer legs, ledgers and hourly FX rates), rebuilds who really paid whom, and treats issuers and liquidity providers as plumbing, not suspects. From that flow graph we compute about 190 features, such as fan-in bursts, pass-through speed, how much of an inflow leaves again, cycles, chains and currency hops. A gradient-boosted model then scores every transaction.
>
> We validated it so that no sender appears in both training and validation. It reaches an average precision of 53, almost double the starter baseline. It is strongest on fan, chain and cycle schemes. Our hardest case is relays that look identical to thousands of legitimate ones, and we show exactly where it still struggles.
>
> Each alert comes with plain-English reasons tied to FATF red flags and links to the transaction hashes, so an analyst can verify it in seconds.
>
> It is a research prototype trained only on synthetic data. A human always decides, and we report false-positive risk, bias across look-alike patterns, and drift. The goal is fewer false alarms, faster investigations, and support for Travel Rule-era compliance on the XRP Ledger.

*(Swap in final AP numbers before speaking. Practise to land at about 1:55.)*

## 8. Suggested 4-page methodology report outline
1. **Problem & user (¼ p).** Scale ([UNODC](https://www.unodc.org/unodc/en/money-laundering/overview.html)), VASP analyst, ranked queue.
2. **Data & XRPL mechanics (½ p).** Tables, 1–3 legs per payment, rippling and path payments, drops → XRP, FX normalisation, `delivered_amount`; class balance 3.1% (our data).
3. **Features (¾ p).** A table of feature groups:
   - node degree/flow in rolling windows
   - pass-through ratio and dwell time
   - amount uniformity / structuring
   - account age / funding provenance
   - cycles, chains, scatter-gather motifs ([Altman](https://arxiv.org/abs/2306.16424))
   - currency-hop features
   - neighbourhood risk (out-of-fold)
   - mapping of each group to FATF red flags
4. **Model (¼ p).** LightGBM, class weighting, seeds/ensemble, hyperparameters.
5. **Validation (½ p).** StratifiedGroupKFold by sender/component ([sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)), leakage controls, adversarial-validation shift check, AP definition ([sklearn AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)).
6. **Results (¾ p).** AP vs baseline (mean ± std), precision-recall curve, precision@top-N, per-family AP table, feature importance / SHAP, 1–2 worked alert examples.
7. **Findings & limitations (½ p).** Relays, ambiguous labels, false-positive analysis on decoys, bias, uncertainty, drift, synthetic caveat.
8. **Responsible use, licence, AI disclosure (¼ p).** Human-in-the-loop, no real allegations, licence (e.g. MIT/Apache-2.0), LLM-use statement.
