# Data profile (train, 11:05)

- 40,131 tx, 60,557 transfer legs (1 leg: 23,681 tx, 2 legs: 12,474, 3 legs: 3,976). path_payment 36,876 legs, direct 23,681.
- Only 676 distinct senders over ~168 h (7 days).
- Suspicious: 1,250 (3.1%). Random AP is about 3.
- labels.csv columns: TRANSACTION_HASH, is_suspicious, family, difficulty, scenario_id (1,497), cohort_id (12). family/scenario/cohort are NOT available at test time: use only for analysis and grouped CV.
- Families (suspicious = easy/medium/hard; decoys = "normal" rows with the same shape):
  fan_in, fan_out, fan_out_in, bipartite, chain, cycle, stack, multi_stage, random: about 100-300 each, roughly half suspicious, half look-alike decoys.
  aggregate_relay (102 hard vs 10,098 normal) and distributed_relay (98 hard vs 9,702 normal): the hard cases hide inside huge legit relay populations.
  ambiguous: 25 hard, unlearnable noise. background: 18,000 normal.
- No sender is purely suspicious: all 443 senders with a suspicious tx also send legit ones, so wallet ID alone is not enough; score the tx in its flow context.
- MEMOS, SOURCE_TAG, FLAGS are empty or constant: useless.
- Validation tip: hold out by scenario_id or wallet (GroupKFold); test wallets are disjoint from train.
