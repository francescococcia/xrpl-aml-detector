# Pitch brief: product, impact, security (Decode & Detect) — team Follow the Drops
*Copied from the training pack and corrected against the internal audit (sections E and F) on 24 Oct 2026.
"(our data)" = our own measurements; "(advice)" = framing suggestion. Web sources were opened when written;
the Chainalysis pages were re-opened on 24 Oct 2026 and their figures checked.*

## 0. Numbers to quote (all checked 24 Oct 2026)
| | v1 | v2 | v3 (final) |
|---|---|---|---|
| CV AP ×100, GroupKFold(5) by scenario_id (our data) | 53.3 | 59.5 | **62.2 ± 4.8** |
| Public leaderboard | 57.4 | 63.5 | 64.4 |
| Features (measured from `feature_columns`) | 175 | 235 | **247** |

Starter baseline CV: 28.3. Top-1% review budget (v3, out-of-fold, our data): 401 alerts, about 350 true (precision 87.5%), about 28–30% of all laundering payments caught.

## 1. Problem
- **Scale.** An estimated 2–5% of global GDP, or $800bn–$2tn a year, is laundered. ([UNODC](https://www.unodc.org/unodc/en/money-laundering/overview.html), also quoted in the [Chainalysis glossary](https://www.chainalysis.com/glossary/money-laundering/))
- **Crypto is growing.**
  - Illicit crypto addresses received at least $40.9bn in 2024, and Chainalysis estimates the total may be closer to $51bn. ([Chainalysis 2025](https://www.chainalysis.com/blog/2025-crypto-crime-report-introduction/))
  - On-chain money laundering grew from $10bn in 2020 to more than $82bn in 2025, which Chainalysis calls a lower-bound estimate. ([Chainalysis 2026](https://www.chainalysis.com/blog/2026-crypto-money-laundering/))
- **Transparent but hard to police.**
  - Public blockchains record every transaction, but accounts are pseudonymous, not anonymous. ([Chainalysis glossary](https://www.chainalysis.com/glossary/money-laundering/))
  - XRPL itself holds no personal identities. ([xrpl.org DEX](https://xrpl.org/docs/concepts/tokens/decentralized-exchange))
  - Ledgers usually close every 3–5 s. ([close times](https://xrpl.org/docs/concepts/ledgers/ledger-close-times))
  - The fastest laundering services (Chainalysis's "Black U" type) cleared very large transfers in an average of 1.6 minutes in Q4 2025. ([Chainalysis 2026](https://www.chainalysis.com/blog/2026-crypto-money-laundering/))
  - Cross-currency path payments can hop across currencies inside one atomic transaction. ([Cross-currency](https://xrpl.org/docs/concepts/payment-types/cross-currency-payments))
- **Rules alone drown analysts.** As noted by Weber et al., industry-standard false-positive rates are "upwards of 90%". The paper states this without its own measurement. ([Weber et al. 2019](https://arxiv.org/abs/1908.02591))

## 2. User
- **Primary user:** a transaction-monitoring / compliance analyst at an exchange, custodian or other VASP that must spot and report suspicious activity. UK cryptoasset businesses register with the FCA under the MLRs and are supervised on a risk basis. ([FCA](https://www.fca.org.uk/firms/financial-crime/cryptoassets-aml-ctf-regime))
- **Secondary users:** stablecoin/IOU issuers (who can freeze trust lines, see [Trust lines](https://xrpl.org/docs/concepts/tokens/fungible-tokens/trust-line-tokens)) and FIU / law-enforcement analysts.

## 3. Product: "ranked alert queue" (batch / T+1h investigation mode)
- **Input.** XRPL-shaped tables (transactions, transfers, blocks, market_rates), the same schema as the free AWS/SonarX XRPL dataset. ([AWS Registry](https://registry.opendata.aws/aws-public-blockchain/))
- **Mode.** It runs in batch, not real time.
  - Features look both backwards and forwards in time within the data window, for example "did the receiver forward the money within 10 minutes?".
  - So a payment can be scored once about an hour of later activity is in (T+1h investigation mode).
  - A streaming variant with past-only windows is the next step. (our data: `src/features.py`, `src/flows.py`)
- **Output.**
  - A 0–1 risk score per transaction, sorted into a queue. Scores are unique, so there are no ties.
  - **Each alert's reconstructed flow:** n payments, wallets, duration and pass-through share. Example from `outputs/example_alerts.md`: "12 payments, 8 wallets in 228 s; 6 of 8 wallets both receive and send; they forward 96% of what they receive" (our data).
  - **Top 3 reasons in plain English**, from LightGBM's built-in per-feature contributions (`pred_contrib=True`; the last column is the expected value, see [LightGBM Booster.predict](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html)), implemented in `src/explain.py`. For example: "part of a 12-payment burst within 2 minutes", "1.5 payments per wallet in the 10-minute flow", "the only payment between this pair all week".
  - **Transaction hashes**, so the analyst can verify each claim on the ledger. (advice)
- **Why it fits XRPL.**
  - Features are built on payment-level flows. Issuers and liquidity providers are treated as plumbing, not counterparties. ([Rippling](https://xrpl.org/docs/concepts/tokens/fungible-tokens/rippling), [Paths](https://xrpl.org/docs/concepts/tokens/fungible-tokens/paths))
  - Amounts use `delivered_amount`. ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments))
  - Values are normalised with hourly rates.
- **Model.**
  - **Features:** 247 graph, flow-reconstruction and gap-regularity features.
  - **Blend:** LightGBM (×2 weight), HistGradientBoosting and logistic regression, plus flow-level smoothing.
  - **Result:** grouped-CV AP 62.2, against 28.3 for the starter baseline (our data).
  - This is the GBT + graph-features family that matched GNNs on IBM's AMLworld benchmark. ([Altman et al.](https://arxiv.org/abs/2306.16424))

## 4. Impact
- **Review-budget framing** (v3, out-of-fold, our data).
  - "If an analyst reviews 1% of traffic (401 alerts), about 7 in 8 are real laundering (87.5%), and that 1% holds 28–30% of all laundering payments."
  - AP rewards putting true positives at the top of the queue. ([sklearn AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html))
- **False alarms are almost all look-alikes, not ordinary users** (v3, out-of-fold, our data).
  - Every typology in this data has planted look-alike decoys.
  - Of the 250 highest-ranked false alarms, only 9 (3.6%) are background traffic. Over the top 1,250 false alarms, 155 (12%) are background.
  - The rest are decoys structurally shaped like laundering, mostly relay look-alikes.
  - So a false alert usually means reviewing a genuinely odd flow, not harassing a normal customer.
  - Fewer false positives also means less harm to innocent customers. AML costs also drive financial exclusion. ([Weber et al.](https://arxiv.org/abs/1908.02591))
  - *(The internal audit quoted 39/1,250 for v2. On the same v2 out-of-fold scores we measure 146/1,250, so we quote our own figures.)*
- **Explainable by construction.** The unit of an alert is a reconstructed flow, which an analyst can read in seconds, with the top-3 model reasons and the hashes.
- **Regulatory context.**
  - The FATF Travel Rule requires VASPs to pass originator and beneficiary information with transfers. 99 jurisdictions have passed or are passing Travel Rule laws, and FATF says most on-chain illicit activity now involves stablecoins. ([FATF 2025](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/targeted-update-virtual-assets-vasps-2025.html))
  - Our USD/GBP/EUR IOUs are stablecoin-like issued tokens. ([Currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats))
- **Aligned with FATF red flags** (structuring, rapid pass-through, new accounts, needless currency conversion). Analysts and regulators recognise these terms. ([FATF red flags](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/))

## 5. Security & responsible AI
| Risk | Mitigation |
|---|---|
| Privacy / pseudonymity: linking addresses to people | Only public ledger data is used. Pseudonymous addresses stay pseudonymous. We score transactions, never identities, with no off-chain enrichment or PII. Synthetic data only. (organiser rules) |
| False-positive harm (frozen funds, debanking) | Human-in-the-loop: the score triggers *review*, never automatic action. Show the flow, the reasons and the uncertainty. FATF treats a single red flag as context-dependent, not proof. ([FATF red flags](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/)) |
| Bias | Pattern shapes like fan-out also fit payroll and merchants ([Altman](https://arxiv.org/abs/2306.16424)). We report per-family AP and which decoys become false alarms (our data). Account IDs are not features ([Altman](https://arxiv.org/abs/2306.16424)). |
| Adversarial adaptation | Flow features are relative (ratios, spreads, burst density, gap regularity), not fixed thresholds. The data already shows what adaptation looks like: easy laundering scenarios span a median of 5 min, but medium ones are stretched to a median of 110 min, longer than the non-relay decoys (62 min), and medium AP falls to 9.8. Slowing down is the evasion to expect, so we plan longer-horizon features and periodic retraining (our data). Launderers change behaviour when detection logic leaks, which is why Altman released data, not generator code ([Altman, Ethics](https://arxiv.org/abs/2306.16424)). Vendors migrate to other channels after enforcement ([Chainalysis 2026](https://www.chainalysis.com/blog/2026-crypto-money-laundering/)). Don't publish exact thresholds. |
| Model drift | Elliptic models failed after a dark-market shutdown ([Weber et al.](https://arxiv.org/abs/1908.02591)). Monitor feature shift, retrain, keep the human review loop. |
| Synthetic-data caveat | Performance may not transfer one-to-one to real data ([Altman, App. B](https://arxiv.org/abs/2306.16424)). Real labels are incomplete, synthetic labels are complete ([Altman](https://arxiv.org/abs/2306.16424)). |
| Mis-reading amounts | Use `delivered_amount`, not `Amount` (the partial-payment exploit). ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments)) |
| No real allegations | Outputs are research prototypes, not compliance decisions. AI/LLM use is disclosed in `ATTRIBUTION.md`. MIT licence. (organiser rules) |

## 6. Limitations (state them honestly; judges reward it)
- **Relays are not separable.** There are about 200 suspicious relays among about 19,800 relay-family decoy transactions in train, and they are structurally identical. Within-family AP is about 1, i.e. chance (our data). Off-chain KYC / Travel Rule data would be needed there.
- **Unlearnable labels.** 25 "ambiguous" labels are unlearnable by design. They and the relay positives get zero training weight (our data).
- **Medium and hard tiers.** Medium AP is 9.8 and hard 0.7: slow flows that look like normal activity (our data).
- **Small sample.** Only 7 days and 676 senders, so CV variance is high. We report AP mean ± std across 5 scenario-grouped folds (our data; [sklearn CV](https://scikit-learn.org/stable/modules/cross_validation.html)).
- **Time resolution.** In this data, timestamps sit on a 2 s grid; the minimum gap between ledgers is 4 s and the median 12 s (our data). On mainnet, `close_time` is rounded to a 10 s resolution ([close times](https://xrpl.org/docs/concepts/ledgers/ledger-close-times)), so sub-10 s gaps would need care on real data.
- **Batch, not streaming.** Features use future context within the window (T+1h).
- **No off-chain KYC context**, which real analysts would have.

## 7. Two-minute pitch script (~290 words)
> Every year, an estimated two to five percent of global GDP is laundered, up to two trillion dollars. Crypto makes this both easier and harder to hide. On the XRP Ledger every payment is public, but accounts are pseudonymous, ledgers close every few seconds, and a single path payment can hop from dollars to euros in one atomic step.
>
> Compliance analysts face alert queues where, as Weber and colleagues note, most alerts are false positives. That wastes time and harms innocent customers.
>
> We built a ranked alert queue for XRPL. It rebuilds who really paid whom, treating issuers and liquidity providers as plumbing. Then it reconstructs money *flows*: bursts of payments chained through shared wallets. For each payment we measure its flow's size, speed, pass-through and how regular its timing is, 247 features in all. A blend of gradient-boosted models then scores every transaction.
>
> We validated it so that no laundering scenario appears in both training and validation. It reaches an average precision of 62, more than double the starter baseline. Give an analyst one percent of the traffic, about 400 alerts, and seven in eight are real laundering. When we are wrong, it is almost never an ordinary user. It is a look-alike flow designed to fool us.
>
> Each alert shows its flow, for example "12 payments through 8 wallets in under 4 minutes, 96% passed straight on", plus the model's top three reasons and the transaction hashes to verify.
>
> We are honest about limits. Relays hidden among twenty thousand identical ones can't be separated from ledger data alone. This is a batch investigation tool today, with streaming next. It is trained only on synthetic data, and a human always decides.

*(Update the leaderboard number for v3 before speaking. Practise to land at about 1:55.)*

## 8. Methodology report
See `docs/method_report.md` / `docs/method_report.pdf` (≤ 4 pages).
- **Validation is GroupKFold(5) by `scenario_id`** (NaN → unique id; shuffle, seed 42). ([sklearn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html))
- Grouping by sender (StratifiedGroupKFold) was only a robustness check. It gave 69.1 vs 53.3 by scenario for v1, a more optimistic estimate (our data).
