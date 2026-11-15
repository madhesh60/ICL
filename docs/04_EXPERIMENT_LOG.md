# 04 — Experiment Log

One row per run. One change per row. Raw outputs live in `solution/runs/<id>/`.
Splits: `dev300` = 300 held-out labeled examples per task (seed 42); `final200` = untouched second holdout; `test` = hidden-label submission set.

| id | date | task | split | config (one-line diff vs previous row for this task) | ctx tokens | nulls | acc | notes |
|---|---|---|---|---|---|---|---|---|
| E000 | | 1 | dev300 | official baseline prompt (`method.py`), 8K ctx, raw completion | 8K | | | reproduction of baseline |

## Decisions made from this log
(none yet)
