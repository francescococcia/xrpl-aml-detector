# Agent rules

Take exactly one card from knowledge/tasks.md. Do only that card. Tick it, write a one-line note for the next agent, commit with the card id, and stop. If the card is bigger than 20 minutes, split it into new cards instead of continuing. Never commit or zip data files (*.parquet, labels.csv). Never write secrets to the repo.

Context: read knowledge/event.md first. Metric is average precision on transaction-level scores; one score per TRANSACTION_HASH, never per transfer.
