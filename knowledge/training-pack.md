# Decode & Detect: training pack (XRPL + AML)
*Fast study guide for Sat 24 Oct 2026. Every factual line links to a source opened today. "(our data)" = something we measured in the challenge dataset ourselves. "(advice)" = my recommendation, not a sourced fact.*

**Your 30-second story:** XRPL is a public ledger with ledgers closing every 3–5 s. Every payment is visible but pseudonymous. Launderers hide in graph shapes (fan-in, fan-out, chains, cycles, relays). We rank every transaction by risk using graph and flow features, and an analyst reviews the top of the queue.

---

## 1. Foundations of distributed ledgers & the XRP Ledger
**Key ideas**
- A consensus protocol lets participants agree on one ordered history without a central operator. It mainly solves the double-spend problem. ([xrpl.org: Consensus](https://xrpl.org/docs/concepts/consensus-protocol))
- XRPL does not use mining or proof-of-stake. Each server trusts a chosen list of validators (its UNL). Consensus keeps going while fewer than 20% of trusted validators are faulty, and confirming an invalid transaction would need more than 80% to collude. ([Consensus](https://xrpl.org/docs/concepts/consensus-protocol))
- Transactions are grouped into "ledger versions" (blocks). Each one holds the full current state, the set of transactions applied, and a header with an index, a hash and the parent. Once validated, a ledger is final. ([Consensus](https://xrpl.org/docs/concepts/consensus-protocol))
- New ledgers usually close every 3–5 s. On mainnet, `close_time` is rounded to the close-time resolution, currently 10 s, and strictly increases. ([Ledger close times](https://xrpl.org/docs/concepts/ledgers/ledger-close-times))
- **This competition's data is different:** every timestamp sits on a 2 s grid; the smallest gap between distinct timestamps is 4 s and the median is 12 s, in both train and test (our data, measured 24 Oct 2026).
- The XRP supply is fixed: XRP can be destroyed but never created. ([Currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats))

**What this means for our detector**
- In this dataset, gaps of a few seconds are real signal (2 s grid, our data). Our flow windows (30 s, 2 min…) and gap-regularity features rely on that. On real mainnet data, gaps under 10 s would be rounding noise, so bucket them there. (advice, based on the [close-time rules](https://xrpl.org/docs/concepts/ledgers/ledger-close-times))
- The `blocks` table equals ledgers. Ledger index is a clean ordering key for temporal features. (advice)

**Judge Qs**
- *Why is XRPL "final" faster than Bitcoin?* A validated ledger is final once trusted validators agree; there are no probabilistic confirmations. ([Consensus](https://xrpl.org/docs/concepts/consensus-protocol))
- *Is it anonymous?* No, it is pseudonymous: the ledger holds addresses, not personal identities. ([DEX limitations](https://xrpl.org/docs/concepts/tokens/decentralized-exchange); [Chainalysis glossary](https://www.chainalysis.com/glossary/money-laundering/))

## 2. XRPL architecture & transaction model
**Key ideas**
- **Accounts.** An account has an address, an XRP balance, a sequence number and a transaction history. No transaction exists just to create an account: a Payment that sends enough XRP to a new address "funds" (creates) it. ([Accounts](https://xrpl.org/docs/concepts/accounts))
- **Reserves.** On Mainnet the base reserve is 1 XRP plus 0.2 XRP for each owned object (trust lines, offers and so on). Reserve XRP can't be sent. Validators can change these values. ([Reserves](https://xrpl.org/docs/concepts/accounts/reserves))
- **Drops.** 1 drop = 0.000001 XRP. XRP amounts are strings of drops; tokens are `{currency, value, issuer}` objects with 15 significant digits. ([Currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats))
- **Issued currencies and trust lines.** USD, GBP and EUR tokens exist only on trust lines, which are two-way relationships with one shared balance. The side with the negative balance is the issuer. Each side sets a limit, which defaults to 0. ([Trust line tokens](https://xrpl.org/docs/concepts/tokens/fungible-tokens/trust-line-tokens))
- **Rippling.** When holder A pays holder B in the same token, the payment actually moves A → issuer → B. The issuer's net balance is unchanged. ([Rippling](https://xrpl.org/docs/concepts/tokens/fungible-tokens/rippling))
- **Paths and cross-currency.** Paths let value flow through rippling steps and through order-book or AMM steps. One Payment can use several paths. XRP-to-XRP payments are always direct. ([Paths](https://xrpl.org/docs/concepts/tokens/fungible-tokens/paths)) Cross-currency payments are atomic and can auto-bridge through XRP (for example USD → XRP → MXN) when that is cheaper. ([Cross-currency payments](https://xrpl.org/docs/concepts/payment-types/cross-currency-payments))
- **SendMax / delivered_amount / partial payments.** `SendMax` caps what the sender spends. With the `tfPartialPayment` flag, `Amount` is only a maximum. Always read `delivered_amount` from the metadata. Naive integrations that trust `Amount` have been exploited to steal from exchanges. ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments))
- **DEX / AMM.** XRPL has a native order-book DEX where an "Offer" is a limit order. Execution order within a ledger is deliberately unpredictable to discourage front-running. ([DEX](https://xrpl.org/docs/concepts/tokens/decentralized-exchange)) AMM pools hold two assets and use a constant-product formula, with a trading fee of 0–1% set by LP votes. Payments automatically use whichever of the order book or the AMM is cheaper. ([AMM](https://xrpl.org/docs/concepts/tokens/decentralized-exchange/automated-market-makers))
- **Fees are burned.** The reference transaction cost is 10 drops and rises with load. It is destroyed, not paid to anyone. ([Transaction cost](https://xrpl.org/docs/concepts/transactions/transaction-cost))
- **Result codes.** `tes` = success. `tec` = failed but the fee was still burned, final only in a validated ledger. `tef`/`tel`/`tem`/`ter` = failure, local error, malformed, retry. ([Transaction results](https://xrpl.org/docs/references/protocol/transactions/transaction-results))

**What this means for our detector**
- **Legs, not payments.** One payment can produce 1–3 transfer legs: a direct leg, a ripple through the issuer, or a path payment through a liquidity provider. (Mechanics: [Rippling](https://xrpl.org/docs/concepts/tokens/fungible-tokens/rippling), [Paths](https://xrpl.org/docs/concepts/tokens/fungible-tokens/paths).) Build the sender→receiver graph at payment level, and treat issuers and liquidity providers as infrastructure, not counterparties. Otherwise everyone appears to "fan in" to the two issuers. (advice / our data)
- **Normalise value.** Convert XRP drops to XRP, then convert everything to one unit with the hourly `market_rates`, so that amount-similarity features (such as "out ≈ in minus a small cut") work across currencies. (advice)
- **Cross-currency hops** (path_payment with SendMax in one currency and delivery in another) are a layering signal. FATF lists converting VAs into multiple types "without logical business explanation". ([FATF red flags PDF](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/))
- **Account age / funding.** A new account must be funded with XRP first, so the funder of a fresh account links the two. That makes "funded by X shortly before use" a strong graph edge. ([Accounts](https://xrpl.org/docs/concepts/accounts); FATF flags transfers "to a newly created or previously inactive account", [PDF](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/))

**Judge Qs**
- *Why use delivered_amount?* Partial payments can deliver far less than `Amount`. ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments))
- *What is rippling?* Same-currency value moving through the issuer's trust lines. ([Rippling](https://xrpl.org/docs/concepts/tokens/fungible-tokens/rippling))
- *Do failed transactions matter?* `tec` transactions still burn the fee and are recorded in validated ledgers. ([Transaction results](https://xrpl.org/docs/references/protocol/transactions/transaction-results))

## 3. Accessing & working with public on-chain data
**Key ideas**
- Both `xrpld` (formerly rippled) and Clio serve JSON-RPC and WebSocket APIs. The API has versions 1 and 2. xrpl-py ≥3 defaults to v2. ([HTTP/WebSocket APIs](https://xrpl.org/docs/references/http-websocket-apis))
- Clio is a read-optimised API server for validated history. It stores data in Cassandra/ScyllaDB using up to 4× less space than xrpld and forwards P2P-only calls (such as `submit`). ([Clio](https://xrpl.org/docs/concepts/networks-and-servers/the-clio-server))
- `delivered_amount` is available from `tx`, `account_tx` and `ledger` (expanded). ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments))
- **xrpl-py** is a pure-Python library with models for transactions and server API objects. Install it with `pip3 install xrpl-py`. ([xrpl-py docs](https://xrpl-py.readthedocs.io/en/stable/))
- **AWS Public Blockchain Data** offers free XRPL data maintained by SonarX at `s3://aws-public-blockchain/v1.1/sonarx/xrp/`, as date-partitioned Parquet updated daily. ([AWS Registry](https://registry.opendata.aws/aws-public-blockchain/)) SonarX indexes ledgers, transactions plus metadata, trust lines and offers. ([SonarX XRPL](https://www.sonarx.com/chains/xrp))

**What this means for our detector**
- The challenge tables (transactions, transfers, blocks, market_rates) follow this AWS/SonarX schema, so the same pipeline could in principle run on real public data. Our rules limit us to synthetic data and no real allegations. (organiser rules)
- Production story: stream new ledgers over WebSocket from Clio, update rolling graph features, re-score, and push alerts. (advice)

**Judge Qs**
- *How would this run live?* WebSocket subscription to a Clio/xrpld server, plus a nightly backfill from the AWS Parquet files. ([APIs](https://xrpl.org/docs/references/http-websocket-apis), [AWS](https://registry.opendata.aws/aws-public-blockchain/))

## 4. Domain knowledge: money-laundering patterns on public ledgers
**Three stages** ([UNODC](https://www.unodc.org/unodc/en/money-laundering/overview.html)): **placement** (moving funds away from the crime), **layering** (disguising the trail) and **integration** (spending funds that now look legitimate). Real cases may skip, merge or repeat stages.

| Pattern | Plain English | On-chain red flag | Source |
|---|---|---|---|
| Smurfing / structuring | Splitting a big sum into many small ones below thresholds | Many similar small amounts; amounts just under round thresholds | [FATF PDF](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/), [Chainalysis glossary](https://www.chainalysis.com/glossary/money-laundering/) |
| Money mules | People "renting out" accounts to receive and forward funds | Pass-through accounts that receive and quickly forward; new accounts | [Chainalysis 2026 CMLN](https://www.chainalysis.com/blog/2026-crypto-money-laundering/) |
| Fan-out | One account pays ≥2 distinct accounts | High out-degree burst | [Altman et al. 2023](https://arxiv.org/abs/2306.16424) |
| Fan-in | ≥2 accounts pay one account | High in-degree burst; "many unrelated wallets, small amounts, then onward transfer" | [Altman](https://arxiv.org/abs/2306.16424), [FATF PDF](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/) |
| Gather-scatter | Fan-in then fan-out at the **same** node | Hub with both high in- and out-degree in a short window | [Altman](https://arxiv.org/abs/2306.16424) |
| Scatter-gather | Source fans out to intermediates, which all fan in to one sink | Shared intermediate set between a source and a sink | [Altman](https://arxiv.org/abs/2306.16424) |
| Cycle | Funds return to the origin | Directed cycle; "strong indicator" | [Altman](https://arxiv.org/abs/2306.16424) |
| Random | Random walk among controlled accounts, no return | Short-lived chain of hops | [Altman](https://arxiv.org/abs/2306.16424) |
| Bipartite | A set of inputs pays a set of outputs | Dense input→output block | [Altman](https://arxiv.org/abs/2306.16424) |
| Stack | Bipartite plus another bipartite layer | Multi-layer dense blocks | [Altman](https://arxiv.org/abs/2306.16424) |
| Peel chain | Long chain where a small amount is peeled off at each hop and the rest moves on | Decreasing balance along a chain plus small side-outputs | [MITRE AADAPT](https://aadapt.mitre.org/techniques/ADT3028.005/), [Chainalysis glossary](https://www.chainalysis.com/glossary/money-laundering/) |
| Relays / rapid pass-through | Funds in, then (almost) all funds out quickly | Low dwell time, out ≈ in; FATF: deposit then "immediately" withdraw | [FATF PDF](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/), [Chainalysis glossary](https://www.chainalysis.com/glossary/money-laundering/) ("rapid movement of funds") |
| Aggregation / fragmentation services | Some services fragment funds, others consolidate them | Shift in amount-size mix and number of counterparties from inflow to outflow | [Chainalysis 2026 CMLN](https://www.chainalysis.com/blog/2026-crypto-money-laundering/) |
| DEX wash trading | Near-simultaneous buy and sell with no change in beneficial ownership, to fake volume | Matched buy/sell within minutes, <1% value difference, repeated; one controller funding many traders | [Chainalysis wash trading](https://www.chainalysis.com/blog/crypto-market-manipulation-wash-trading-pump-and-dump-2025/) |

Notes:
- Altman's 8 patterns are fan-out, fan-in, gather-scatter, scatter-gather, cycle, random, bipartite and stack. Our label families map onto them (fan_out_in ≈ scatter-gather or gather-scatter, multi_stage ≈ stack/chain combinations). That mapping is my interpretation. ([Altman](https://arxiv.org/abs/2306.16424); (our data))
- **Caveat on peel chains:** on Bitcoin, peel-chain-like shapes also arise naturally from change addresses, so the shape alone is not proof. ([Chainalysis blog](https://www.chainalysis.com/blog/common-blockchain-analysis-mistakes-cryptocurrency-investigations/)) This is the same "decoy" problem our labels contain.
- **Other FATF transaction red flags:** multiple high-value transfers in short succession (for example within 24h); transfers to a newly created or previously inactive account; frequent transfers to the same account from more than one person; converting at a loss; using multiple VAs or accounts with no logical business explanation. ([FATF PDF](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf/)) FATF groups indicators into technology, geography, transaction patterns, size, profiles and source of funds. ([FATF page](https://www.fatf-gafi.org/en/publications/Methodsandtrends/Virtual-assets-red-flag-indicators.html))
- **Speed:** Chainalysis reports that some laundering services clear very large transfers in about 1.6 minutes on average (Q4 2025). Dwell time is therefore a real signal. ([Chainalysis 2026](https://www.chainalysis.com/blog/2026-crypto-money-laundering/))

**What this means for our detector (relays, our weak spot)** (advice / our data)
- Our labels hide about 100 suspicious aggregate_relay and about 100 distributed_relay transactions among about 10,000 legitimate relays (our data). The shape is identical, so the signal must come from **context**:
  - funding provenance (who funded the relay account and when)
  - upstream and downstream risk propagated from neighbours already scored as suspicious (a 2-hop "guilt by association" score, built out-of-fold to avoid leakage)
  - amount retention (out/in ratio and the size of the cut)
  - dwell time
  - account age at first use
  - number of distinct upstream sources
  - currency switches along the route
  - whether the endpoints are themselves pattern members
- Score transaction-level **and** component-level features: relay pieces of one scheme share a connected component and time window.

**Judge Qs**
- *Isn't a fan-out just payroll?* Yes, and that is why the data has decoys. We use timing, amount uniformity, account age and downstream behaviour, not shape alone. ([Altman](https://arxiv.org/abs/2306.16424) notes laundering is also disguised as payroll and supplies)
- *Why is wash trading AML-relevant?* It fakes volume and can mask value transfer. Chainalysis estimated up to $2.57bn of suspected wash-trading volume on three chains in 2024 (an upper bound). ([Chainalysis](https://www.chainalysis.com/blog/crypto-market-manipulation-wash-trading-pump-and-dump-2025/))

## 5. Practical guidance & resources
- Docs: [xrpl.org concepts](https://xrpl.org/docs/concepts/accounts), [APIs](https://xrpl.org/docs/references/http-websocket-apis), [xrpl-py](https://xrpl-py.readthedocs.io/en/stable/), [AWS Public Blockchain Data](https://registry.opendata.aws/aws-public-blockchain/).
- Benchmarks to cite: **IBM AMLworld** synthetic AML data (Altman et al.). Its baselines are GBT + graph features, and GNNs (PNA). Ground truth is complete in synthetic data, unlike real data. ([arXiv 2306.16424](https://arxiv.org/abs/2306.16424)) **Elliptic** Bitcoin dataset: Random Forest on rich features beat GCN, and models failed after a dark-market shutdown (drift). ([arXiv 1908.02591](https://arxiv.org/abs/1908.02591))
- Practical takeaways from those papers:
  - Graph features plus gradient boosting is a strong, defensible baseline. ([Altman](https://arxiv.org/abs/2306.16424))
  - Account IDs were deliberately **not** used as features, to avoid memorising accounts. ([Altman](https://arxiv.org/abs/2306.16424))
  - In Altman et al., graph features were computed strictly from past data to prevent leakage. ([Altman, App. D](https://arxiv.org/abs/2306.16424))
  - **Our detector does not do this.** Several features look forwards within the data window, for example "did the receiver forward the money later?" and forward-linked flows up to 1 h. That is valid for a label-free batch investigation (T+1h), but not for real-time scoring. A streaming version would need past-only windows. (our data: `src/features.py`, `src/flows.py`)

## 6. Goals of the hackathon
- Build an AI detector that scores each test transaction 0–1. Ranking is 100 × average precision on the hidden test set. (organiser brief)
- Show you understand XRPL mechanics and AML typologies, and produce a reproducible, open-source research prototype, not a compliance tool. Disclose AI/LLM use. (organiser rules)

## 7. What is expected
- Submit predictions for all 40,101 test transactions before **16:30 London time (BST)**.
- Add a methodology report (≤4 pages or <10 slides) covering: methodology, features, model, validation, results, findings, limitations (false positives, bias, uncertainty), an open-source licence, and AI-use disclosure. (organiser brief; outline in `pitch-brief.md`)

## 8. Explaining AML
- **Definition:** money laundering is "the conversion or transfer of property, knowing that such property is derived from any offense(s), for the purpose of concealing or disguising the illicit origin…" (UN Vienna Convention, quoted by [UNODC](https://www.unodc.org/unodc/en/money-laundering/overview.html)).
- **Scale:** an estimated 2–5% of global GDP, or $800bn–$2tn a year, is laundered. ([UNODC](https://www.unodc.org/unodc/en/money-laundering/overview.html))
- **Crypto scale:**
  - Illicit addresses received $40.9bn in 2024. That is a lower bound, possibly about $51bn, and 0.14% of on-chain volume. Stablecoins carry 63% of illicit transaction volume. ([Chainalysis 2025](https://www.chainalysis.com/blog/2025-crypto-crime-report-introduction/))
  - On-chain laundering grew from $10bn (2020) to more than $82bn (2025). ([Chainalysis 2026](https://www.chainalysis.com/blog/2026-crypto-money-laundering/))
- **Regulation:**
  - FATF R.15 applies AML/CFT to VASPs. The Travel Rule (R.16 applied to VAs) requires originator and beneficiary information to travel with transfers. 99 jurisdictions have passed or are passing Travel Rule laws (June 2025). ([FATF 2025 update](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/targeted-update-virtual-assets-vasps-2025.html))
  - UK: cryptoasset firms must register with the FCA under the MLRs, and supervision is risk-based. A new UK crypto regime is expected from 25 Oct 2027. ([FCA](https://www.fca.org.uk/firms/financial-crime/cryptoassets-aml-ctf-regime))
- **Why ML?** Industry false-positive rates of "upwards of 90%" hamper rules-based monitoring. ([Weber et al. 2019](https://arxiv.org/abs/1908.02591))

---

## How to be most accurate (for the AP leaderboard)
1. **Know the metric.** AP = Σ(Rₙ − Rₙ₋₁)·Pₙ, which depends only on how you **order** transactions. ([sklearn AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html))
   - Calibration doesn't matter for the score; only ranking does. Any monotone transform (rank-averaging, logit) leaves AP unchanged.
   - Each positive pushed above a false positive at the top of the list is worth a lot.
2. **Grouped validation.** Our data has many transactions per sender (676 senders over 7 days, (our data)), so samples are not i.i.d. Use `GroupKFold`/`StratifiedGroupKFold` so that one group never appears in both train and validation. ([sklearn CV §3.1.2.4](https://scikit-learn.org/stable/modules/cross_validation.html))
   - **What we actually do:** `GroupKFold(5, shuffle=True, random_state=42)` grouped by the organiser's `scenario_id` (rows without a scenario get a unique id). That keeps all members of a laundering scheme or decoy in one fold. `scenario_id` is used only as the CV group key, never as a feature. (our data: `src/train.py`)
   - Grouping by sender (StratifiedGroupKFold) gave a more optimistic CV: 69.1 vs 53.3 by scenario for v1, because one scheme spans many senders. The wallet graph is a single connected component, so component grouping is impossible (our data: `experiments/group_robustness.py`).
   - Stratification can hide true fold variance with rare classes, so report the mean ± std of AP across folds. ([sklearn CV](https://scikit-learn.org/stable/modules/cross_validation.html))
3. **Avoid leakage.**
   - Fit any target-derived feature (neighbour label rates, out-of-fold risk propagation) inside the CV loop only. ([sklearn CV: transformations must be learnt on training folds](https://scikit-learn.org/stable/modules/cross_validation.html))
   - Don't use raw account IDs as features. ([Altman](https://arxiv.org/abs/2306.16424))
   - Don't tune on the test set. ([sklearn CV](https://scikit-learn.org/stable/modules/cross_validation.html))
4. **Test-time graph features.** Test transactions are a different set of accounts and time. Recompute graph features on the test graph itself, and on train plus test where legitimate, the same way you did for train. (advice)
5. **Check distribution shift.** (advice)
   - Compare feature distributions train vs test (KS test, or "adversarial validation": a classifier that predicts train-vs-test; AUC ≈ 0.5 means no shift).
   - Elliptic showed models collapsing after an unseen event, so shift is real. ([Weber et al.](https://arxiv.org/abs/1908.02591))
6. **Ensemble.**
   - Rank-average LightGBM over several seeds and feature subsets, plus one diverse model (for example XGBoost/CatBoost, or a GNN if time allows). Altman found GBT + graph features comparable to the best GNN (PNA). ([Altman](https://arxiv.org/abs/2306.16424))
   - Blend using out-of-fold predictions. ([sklearn: cross_val_predict for blending](https://scikit-learn.org/stable/modules/cross_validation.html))
7. **Per-family error analysis.** Compute AP per pattern family out-of-fold (our data) and spend the remaining time on the families that lose the most AP (relays). Don't chase the 25 "ambiguous" labels, which are unlearnable by design. (advice)
8. **Class imbalance.** Weighting the minority class (for example `scale_pos_weight`) is standard. ([Altman](https://arxiv.org/abs/2306.16424) tuned `scale_pos_weight` 1–10)

## Glossary
| Term | Meaning |
|---|---|
| AMM | Pool of two assets priced by a constant-product formula; LPs earn trading fees ([AMM](https://xrpl.org/docs/concepts/tokens/decentralized-exchange/automated-market-makers)) |
| AP | Average precision, a precision-recall summary that is rank-based ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)) |
| Auto-bridging | Token→XRP→token conversion when it is cheaper ([Cross-currency](https://xrpl.org/docs/concepts/payment-types/cross-currency-payments)) |
| Clio | Read-optimised XRPL API server ([Clio](https://xrpl.org/docs/concepts/networks-and-servers/the-clio-server)) |
| delivered_amount | Metadata field with the amount actually delivered ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments)) |
| Drop | 0.000001 XRP ([Currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats)) |
| Issuer | Account on the negative side of a trust line ([Trust lines](https://xrpl.org/docs/concepts/tokens/fungible-tokens/trust-line-tokens)) |
| Offer | Limit order on the XRPL DEX ([DEX](https://xrpl.org/docs/concepts/tokens/decentralized-exchange)) |
| Partial payment | Payment flag allowing delivery of less than `Amount` ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments)) |
| Path | Route through rippling and order-book/AMM steps ([Paths](https://xrpl.org/docs/concepts/tokens/fungible-tokens/paths)) |
| Reserve | Minimum XRP to hold an account: 1 XRP base + 0.2 XRP per owned object ([Reserves](https://xrpl.org/docs/concepts/accounts/reserves)) |
| Rippling | Same-currency transfer passing through the issuer ([Rippling](https://xrpl.org/docs/concepts/tokens/fungible-tokens/rippling)) |
| SendMax | Maximum the sender will spend ([Partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments)) |
| tec / tes | Fee-claimed failure / success result codes ([Results](https://xrpl.org/docs/references/protocol/transactions/transaction-results)) |
| Travel Rule | FATF R.16 applied to VA transfers ([FATF](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/targeted-update-virtual-assets-vasps-2025.html)) |
| UNL | Unique Node List of trusted validators ([Consensus](https://xrpl.org/docs/concepts/consensus-protocol)) |
| VASP | Virtual asset service provider, e.g. an exchange or custodian ([FATF](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/targeted-update-virtual-assets-vasps-2025.html); UK scope: [FCA](https://www.fca.org.uk/firms/financial-crime/cryptoassets-aml-ctf-regime)) |

---
*Source note: the FATF red-flag PDF (`...Virtual-Assets-Red-Flag-Indicators.pdf.coredownload.pdf`) is behind Cloudflare. A direct download returned a challenge page, so its quoted indicators were read from the search tool's extract of that PDF. The FATF HTML landing page and its six indicator categories were opened directly.*
