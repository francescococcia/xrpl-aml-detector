# Study guide: detecting laundering flows on the XRP Ledger

Skim the headings. **Bold** = what matters for the model. Every external claim has a link to a page opened while building this.
Claims marked *(our data)* come from our own analysis of the hackathon train set, not from a source.

---

## 1. XRPL in 5 minutes

| Concept | What it is | Source |
|---|---|---|
| Account | An address with an XRP balance, a sequence number and a transaction history. **Accounts are created by a Payment that funds them**, with a reserve of 1 XRP. | [XRPL: Accounts](https://xrpl.org/docs/concepts/accounts) |
| XRP & drops | XRP is native, has no issuer and is tracked in accounts. 1 drop = 0.000001 XRP. XRP amounts are strings of drops. | [XRPL: Currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats) |
| Tokens (IOUs) | Tokens always have an issuer. They are `{currency, value, issuer}` objects held in trust lines. | [XRPL: Currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats) |
| Trust line | A bidirectional RippleState entry with one shared balance. The negative side is the "issuer". Each side sets a limit, and there is one line per currency code per pair of accounts. Tokens can "ripple" through the issuer. | [XRPL: Trust line tokens](https://xrpl.org/docs/concepts/tokens/fungible-tokens/trust-line-tokens) |
| Issuer controls | Issuers can charge a **transfer fee**, which is deducted automatically on transfers, and they can freeze trust lines. | [XRPL: Trust line tokens](https://xrpl.org/docs/concepts/tokens/fungible-tokens/trust-line-tokens) |
| Fee | Each transaction destroys a small amount of XRP. The reference cost is 10 drops and rises with load. | [XRPL: Transaction cost](https://xrpl.org/docs/concepts/transactions/transaction-cost) |
| Payment | The only way to create accounts. `SendMax` is the most the source can spend. **Partial payments** can deliver less than `Amount`, and `delivered_amount` in the metadata is what actually arrived. | [XRPL: Payment](https://xrpl.org/docs/references/protocol/transactions/types/payment) |
| Paths | A path step either ripples through an account or trades on an order book or AMM. Intermediaries gain and lose roughly equal value. XRP→XRP payments use no paths. | [XRPL: Paths](https://xrpl.org/docs/concepts/tokens/fungible-tokens/paths) |
| Cross-currency payment | Atomic. Delivers a fixed amount at variable cost by consuming DEX offers, with auto-bridging through XRP. | [XRPL: Cross-currency payments](https://xrpl.org/docs/concepts/payment-types/cross-currency-payments) |
| DEX | An on-ledger order book (CLOB) since 2012. Offers are limit orders. **The same currency code from different issuers is a different asset.** The ledger holds no personal info. | [XRPL: Decentralized exchange](https://xrpl.org/docs/concepts/tokens/decentralized-exchange) |

**What this means for features:** one payment can be several transfer *legs*: a path, a conversion, or slippage between `SendMax` and the amount delivered. Our leg features (`n_legs`, `asset_change`, `conv_spread`, `sendmax_slack`) come from this. Values have to be normalised across assets before flows can be compared. We use the organiser's `market_rates` through `starter/payments.py`.

## 2. Money laundering vs fraud

- **Fraud** steals value from a victim. **Money laundering (ML)** disguises where criminal proceeds came from. Laundering runs in three stages: **placement → layering → integration** ([Altman et al. 2023](https://arxiv.org/abs/2306.16424)).
- Most laundering is never detected. The UN estimates it at 2–5% of global GDP ([Altman et al. 2023](https://arxiv.org/abs/2306.16424)).
- Crypto is a paradox: pseudonymity helps criminals, but public ledgers give investigators the full graph ([Weber et al. 2019, Elliptic](https://arxiv.org/abs/1908.02591)).
- FATF red flags for virtual assets: anonymity features, geography, unusual transaction patterns, size and frequency with "no logical business explanation", unusual sender or recipient profiles, and source of funds ([FATF 2020 overview](https://www.fatf-gafi.org/en/publications/Methodsandtrends/Virtual-assets-red-flag-indicators.html)).
- From the full FATF report (read through a search excerpt; the PDF itself is behind a bot wall):
  - structuring into small amounts below thresholds
  - multiple high-value transfers in short succession (within 24h), or to a newly created or inactive account
  - deposits that are moved on immediately

  Source: [FATF VA red flags PDF](https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf).

## 3. Laundering typologies (the hackathon families)

Definitions from [Altman et al. 2023 (AMLworld)](https://arxiv.org/abs/2306.16424). That paper's patterns match most of our label families.

| Typology | Shape | Feature signals we use |
|---|---|---|
| fan_out | 1 → k (k≥2) | `s_out_cnt_10m/1h`, `s_out_uniq_*`, amount similarity among siblings |
| fan_in | k → 1 | `r_in_cnt_*`, `r_w_in_uniq`, `r_burst_10m` |
| fan_out_in / scatter-gather | 1 → k → 1 | down/up 2-hop flows, `return_origin`, conservation |
| gather-scatter / multi_stage | k → 1 → k | `fwd_ratio_*`, `fund_ratio_*`, passthru, holding time |
| cycle | A → … → A | cycle BFS ≤3 hops in 24h/7d; cycles are a strong ML indicator per Altman et al. |
| bipartite | set A → set B | `pair_share_of_sender`, clustering, unique counterparties |
| stack | layered bipartite | multi-hop down/up flows, `chain_score` |
| chain | A→B→C… with fast forwarding | `s_hold_before`, `r_min_to_forward`, `down1_decay`, `chain_score` |
| random | random walk of transfers | pair novelty, LOO amount z-score |
| aggregate_relay / distributed_relay *(our data)* | hub(s) relay ~90–95% of inflow on a ~4.5h lag over days | episode features `ep_*`, `bal_*` (**still near random. See §6**) |

**Why graph features work:** a single transaction looks normal. The laundering *pattern* only shows across many transactions. Altman et al. show that gradient boosting on graph features is a strong baseline ([Altman et al. 2023](https://arxiv.org/abs/2306.16424)). On Elliptic, Random Forest with neighbourhood-aggregated features beat GCN, and the aggregated features beat local-only features for every model ([Weber et al. 2019](https://arxiv.org/abs/1908.02591)).

## 4. Our feature groups → which typology they catch

1. **Velocity windows** (10m/1h/6h/24h/7d): counts and sums in and out per wallet, plus forward/fund ratios. These pick up rapid in→out movement (fan, chain, the FATF "immediately moved" flag).
2. **Counterparty diversity**: unique senders and receivers per window. Catches fan and bipartite shapes.
3. **Holding time / forwarding lag**: how long value sits before moving on. Catches layering.
4. **Amount similarity / structuring**: near-identical sibling amounts and z-scores. Catches the structuring red flag.
5. **Pair features**: novelty of a pair and its share of the sender's flow. `pair_total` is our #1 feature.
6. **Multi-hop flow** (1–3 hops down and up), cycle return, chain score. Catches stack, chain and cycle shapes.
7. **Graph stats**: clustering, degree, components (networkx).
8. **Episodes / balances**: in vs out conservation per active episode, plus running-balance extremes. Aimed at relays.
9. **Transfer legs**: path or conversion complexity (§1).

Leakage checks *(our data)*: SOURCE_FEE, TRANSACTION_INDEX and SEQUENCE alone score AP ≈ random. LAST_LEDGER_SEQUENCE offset is a constant 20. None of these are shortcuts.

## 5. Metric and validation

- **Average precision**: AP = Σₙ (Rₙ − Rₙ₋₁)·Pₙ over score thresholds. It is not interpolated, and it measures **ranking**, not calibration ([scikit-learn AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)). Only the order of scores matters.
- The random baseline AP equals the positive rate, ≈3.1% in train *(our data)*.
- With class imbalance, the minority class is the one that matters. Altman et al. report minority-class F1 ([Altman et al.](https://arxiv.org/abs/2306.16424)), and Elliptic used weighted loss for the illicit class ([Weber et al.](https://arxiv.org/abs/1908.02591)).
- **GroupKFold** puts each group in exactly one test fold and never across train and test. `shuffle`/`random_state` are available since sklearn 1.6 ([scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)). We group by `scenario_id` (NaN → unique) so transactions from the same laundering scenario never leak across folds. Random K-fold would inflate CV.
- **LightGBM**: histogram-based, grows trees leaf-wise (best-first) so it can overfit on small data (limit with `max_depth`/leaves), and supports bagging, column subsampling and L1/L2 regularisation ([LightGBM features](https://lightgbm.readthedocs.io/en/latest/Features.html)).

## 6. Where we are weak and what to try *(our data)*

- Relay families (aggregate and distributed) are at about 1% within-family AP. They have only 6 positive scenarios each, and positives look like decoys at the transaction level.
- Each relay cohort has 1 positive scenario and 99 near-identical decoy scenarios (10–28 txs over ~5.5 days). Scenario-level statistics (out/in ratio, lag, rotation) put the positives at random percentiles among their decoys, so 6 positives per family cannot teach a model the difference *(our data)*.

## 7. Phase-2 experiments: what we tried, what we kept

All numbers are grouped 5-fold CV, 100×AP, same folds as phase 1 *(our data)*. Scripts are in `experiments/` and are **not** part of the submission pipeline.

| experiment | mean AP | std | verdict |
|---|---|---|---|
| stage-1 LightGBM (phase 1, kept) | **53.27** | 6.97 | kept |
| + 47 hub relay features (`experiments/hub_cv.py`) | 50.78 | 6.66 | dropped |
| stage-2 on neighbour aggregates only (`experiments/stacking_cv.py`) | 44.25 | 7.10 | dropped |
| stage-2 on base + neighbour aggregates | 51.95 | 7.60 | dropped |
| rank-blend 0.7·stage-1 + 0.3·stage-2 | 53.79 | 7.06 | dropped (+0.5 is far inside 1 std; doubles runtime) |

**Hub relay features (#1).** We computed label-free, per-wallet stats: daily out/in log-ratio (median, spread, number of consistent days), lag since last inflow (median, IQR, CV), time-of-day concentration, counterparty rotation and overlap. These are ratios and spreads, not fixed constants like "4.5 h" or "0.93", so they survive batch shifts. They lifted aggregate_relay only from 0.87 to 0.92 and hurt other families (fan_out 68.76→63.62, bipartite 63.25→60.51, multi_stage 82.17→79.70), so we dropped them *(our data)*.

**Stage-2 stacking (#2).** The idea comes from Elliptic: neighbourhood-aggregated features help every model ([Weber et al. 2019](https://arxiv.org/abs/1908.02591)). Here the thing we aggregate is the stage-1 *score*, over the sender/receiver neighbourhood, two hops, ±24 h, and high-score flow groups (connected components). To avoid leakage, the stage-1 scores used for training stage-2 are strictly out-of-fold (inner GroupKFold inside each outer fold). Stage-2 did not beat stage-1. The scenario-level grouping already keeps a scenario's txs in one fold, so neighbours' scores add little beyond the graph features stage-1 already sees *(our data)*.

**Steering ideas checked.**
- *Issuer / liquidity-provider collapsing.* `load_payments` already yields one effective sender→receiver edge per payment; the busiest wallet appears in only 256 of 40,131 train payments (0.6%). There is no issuer hub left to collapse *(our data)*.
- *Coarse time buckets.* XRPL ledger `close_time` is rounded by `close_time_resolution`, up to 120 s ([XRPL ledger header](https://xrpl.org/docs/references/protocol/ledger-data/ledger-header)). Our time features are already windows of 10 min or more, so we did not add buckets.
- *Grouping robustness.* `StratifiedGroupKFold` keeps groups non-overlapping while trying to preserve the class ratio ([scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)). Grouping by **sender** gives 69.12 ± 3.68, which is much higher than 53.27 by scenario. Scenario grouping is therefore the stricter, less optimistic estimate, and we keep it. The wallet graph is one connected component, so component grouping is impossible (`experiments/group_robustness.py`) *(our data)*.

**Lesson for the pitch.** Negative results are honest results. On relays, the data generator makes positives and decoys statistically identical at every level we could measure. More features there mostly add noise for the families we already detect well.

## Sources opened
- https://xrpl.org/docs/concepts/accounts
- https://xrpl.org/docs/references/protocol/data-types/currency-formats
- https://xrpl.org/docs/concepts/tokens/fungible-tokens/trust-line-tokens
- https://xrpl.org/docs/concepts/tokens/fungible-tokens/paths
- https://xrpl.org/docs/concepts/payment-types/cross-currency-payments
- https://xrpl.org/docs/concepts/tokens/decentralized-exchange
- https://xrpl.org/docs/references/protocol/transactions/types/payment
- https://xrpl.org/docs/concepts/transactions/transaction-cost
- https://arxiv.org/abs/2306.16424 — Altman et al., *Realistic Synthetic Financial Transactions for AML Models* (AMLworld)
- https://arxiv.org/abs/1908.02591 — Weber et al., *AML in Bitcoin: Experimenting with GCNs* (Elliptic)
- https://www.fatf-gafi.org/en/publications/Methodsandtrends/Virtual-assets-red-flag-indicators.html
- https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Virtual-Assets-Red-Flag-Indicators.pdf (content via search excerpt)
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- https://lightgbm.readthedocs.io/en/latest/Features.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- https://xrpl.org/docs/references/protocol/ledger-data/ledger-header
