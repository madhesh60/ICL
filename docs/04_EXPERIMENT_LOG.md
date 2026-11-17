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
| O009 | 2026-11-17 | 1,3,4 | all examples | do the labels follow the exact rule? (true rule run over every example) | 5500/5500, 3997/3997, 3993/3993 match | label noise is NOT why program induction failed in E000 |

## Model runs (Qwen3-4B)

Hardware for E000: Kaggle, 2 x Tesla T4 16 GB, vLLM 0.12.0, tensor parallel 2, fp16, eager,
`max-model-len 40960`, prefix caching on, YaRN x4. Full log: `docs/results/kaggle_run_2026-11-16_raw.md`.
Analysis: `docs/06_FIRST_REAL_RUN.md`.

| id | date | task | split | config (one-line diff vs previous row for this task) | ctx tokens | acc (95% range) | share per ladder step @ accuracy | seconds | notes |
|---|---|---|---|---|---|---|---|---|---|
| E000 | 2026-11-16 | 1 | dev (20) | `configs/default.yaml` @ 4599680 | ≥30K | 70% (48..85) | think_vote 85% @ 82%, generate 15% @ 0% | 1609 | **no `program` source**: induction never produced an accepted program |
| E000 | 2026-11-16 | 2 | dev (20) | same | ≥30K | 70% (48..85) | score 35% @ 29%, think_vote 65% @ 92% | 719 | cheap rung "sure" and wrong; expensive rung excellent |
| E000 | 2026-11-16 | 3 | dev (20) | same | ≥30K | 100% (84..100) | think_vote 100% @ 100% | 1422 | no `program` source; thinking carried it |
| E000 | 2026-11-16 | 4 | dev (20) | same | ≥30K | 70% (48..85) | think_vote 100% @ 70% | 1302 | no `program` source; character slips on long lists |
| E000 | 2026-11-16 | 5 | dev (20) | same | ≥30K | 70% (48..85) | score 75% @ 87%, think_vote 25% @ 20% | 322 | escalation hurt (n=5, thin) |
| E000 | 2026-11-16 | 6 | dev (20) | same | ≥30K | 95% (76..99) | score 90% @ 100%, think_vote 10% @ 50% | 302 | overlap signal carries it |
| E000 | 2026-11-16 | 7 | dev (20) | same | ≥30K | 20% (8..42) | generate 55% @ 18%, think_vote 45% @ 22% | 599 | knowledge-bound; ECE 0.54 (over-confident) |
| E000 | 2026-11-16 | 8 | dev (20) | same, `CAIRN_TRITON_INTERPRET=0` | ≥16K | consistency 0.000 | static OK 95%, runs 0%, torch match 0% | 1006 | executor ran nothing; no reason printed (fixed: outcome histogram) |
| E001 | planned | all | dev (60) | commit after E000 fixes: short induction block + 4096 tokens + no-think retry; singleton-accuracy gate; t2 `use_scoring: false`; t5 `escalate: false`; t1/3/4 backup `think_max_tokens: 2048`; Triton-friendly test shapes; trivia vote pooling | ≥30K | predictions in `06_FIRST_REAL_RUN.md` §6 | | | run with `scripts/kaggle_run.sh` (LIMIT=60) |
| E002 | planned | 5 | dev (100) | `escalate: true` vs `false` via `cairn tune --task 5 --limit 100` | ≥30K | | | | decides E000's provisional task-5 choice properly |
| E003 | planned | 2 | dev (100) | `vote_n` 3 vs 5; `think_max_tokens` 768 vs 1024 | ≥30K | | | | cost/accuracy trade for the most expensive task |

## Decisions made from this log
- O001 → answers are read by one-token scoring or constrained generation; every ladder ends in a model-free answer.
- O003/O004 → `selection: coverage`, `coverage_target: queries`, `bulk_order: interleave`.
- O005 → the scheduler groups by prompt beginning and sends one primer first.
- O006 → `calibration: stack` in `configs/default.yaml`; `configs/pure_icl.yaml` keeps the strict alternative.
- O009 + E000 (tasks 1/3/4) → program induction reads a 48-example slice with a 4,096-token budget and retries without thinking when no code comes back; attempts are printed in the log.
- E000 (task 2) → a conformal singleton is only trusted if, on held-out examples, singletons were ≥85% accurate (`min_singleton_accuracy`); task 2 skips scoring altogether.
- E000 (task 5) → `escalate: false`, provisional until E002.
- E000 (task 8) → execution outcomes are reported as a histogram; the driver guesses shapes that are multiples of 16 first.
- E000 (all) → 20 inputs per task cannot separate 65% from 80%; the next dev pass uses 60, and `cairn tune` uses 100+ for any single decision.
