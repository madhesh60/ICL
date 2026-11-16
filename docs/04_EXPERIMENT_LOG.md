# 04 — Experiment Log

One row per run. One change per row. Raw outputs live in `runs/<id>/` (git-ignored); the
run report (`<split>.report.json`) records the settings, git commit and token use.
Splits: `dev` = 300 held-out labelled examples per task (hash split, seed 42); `holdout` =
untouched second set; `test` = hidden-label submission set.

## Offline measurements (no language model; regenerate with `cairn bench`)

| id | date | task | split | what was measured | result | decision |
|---|---|---|---|---|---|---|
| O001 | 2026-11-16 | all | test (baseline zip) | official baseline answers lost to null/invalid output | 32% overall; task 4 59%; task 8 100% | zero-null decoding is the first priority |
| O002 | 2026-11-16 | 1–7 | dev | majority label vs nearest-examples (kNN-15) vote | t2 28.3/49.0, t5 61.4/68.3, t6 64.3/58.0, t7 0/2.7 | these are the floors any model run must beat |
| O003 | 2026-11-16 | 2,5,6,7 | dev | example coverage: file order vs random vs Cairn coverage | t2 0.754/0.759/0.960, t6 0.587/0.590/0.728, t7 0.782/0.785/0.857 | use query-aware coverage selection |
| O004 | 2026-11-16 | 6 | pool | longest same-label streak in the shared block | file order 18, Cairn 7 | interleave labels |
| O005 | 2026-11-16 | 7 | dev (200) | GPU prompt tokens to read, 4 clusters, simulated prefix cache | grouped 0.20M vs arrival 1.62M of 6.21M | always group + prime |
| O006 | 2026-11-16 | 6 | dev | sentence-pair overlap signal, stacked with kNN vote | 94.7% (threshold alone 93%) | stacked calibration is the default; signal kept on the task card |
| O007 | 2026-11-16 | 8 | examples | static checker on the 184 reference kernels | 183 pass; 1 real typo (`tl.cidv`) | checker trusted |
| O008 | 2026-11-16 | 8 | test | signatures parsed | 166/166 | pin the signature in every prompt |

## Model runs (Qwen3-4B; to be filled on a GPU machine)

| id | date | task | split | config (one-line diff vs previous row for this task) | ctx tokens | acc (95% range) | share per ladder step | notes |
|---|---|---|---|---|---|---|---|---|
| E000 | | 1–7 | dev | `configs/default.yaml` | ≥30K | | | first real run |
| E001 | | 1–7 | dev | `configs/pure_icl.yaml` | ≥30K | | | what ICL alone gives |

## Decisions made from this log
- O001 → answers are read by one-token scoring or constrained generation; every ladder ends in a model-free answer.
- O003/O004 → `selection: coverage`, `coverage_target: queries`, `bulk_order: interleave`.
- O005 → the scheduler groups by prompt beginning and sends one primer first.
- O006 → `calibration: stack` in `configs/default.yaml`; `configs/pure_icl.yaml` keeps the strict alternative.
