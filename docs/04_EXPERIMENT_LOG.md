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
| O010 | 2026-11-17 | 8 | examples vs test | inputs with a parseable wrapper signature | demonstrations 0/184; test 166/166 | the dev split cannot measure execution; measure task 8 on test inputs (metric needs no gold) |
| O011 | 2026-11-17 | 8 | test | test functions with a PyTorch function of the same name | 62/166 (div, tanh, matmul, conv2d, ...) | correctness is checkable for ~37% of test kernels; the rest only for "runs and is finite" |
| O013 | 2026-11-18 | 2 | all examples | least-squares weight of each word in the label (verbs 2,755 / nouns 2,685 sentences) | verbs: is +0.02, are −0.02, has +0.57; nouns: living +0.81, it −0.44, front +0.30 | helper verbs not counted; compound words counted separately; pronouns not nouns (rules in `tasks/counting.py`, pinned by a data test) |
| O012 | 2026-11-17 | 7 | trivia answers | longest gold answer; gold answers containing ':' or starting with "wait" | 11 words; 1 (`the 3:10 to yuma`); 0 | replies over 12 words or starting with "wait"/"clue:" are talk, not answers |

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
| E001 | 2026-11-17 | 1 | dev (60) | @ e34413c: 48-example induction prompt, 4096 tokens, no-think retry | ≥30K | **100%** (94..100) | program 100% @ 100% | 282 | accepted in round 0 (one of two candidates ran out of budget) |
| E001 | 2026-11-17 | 2 | dev (60) | `use_scoring: false` | ≥30K | 45% (33..58) | generate 97% @ 43%, think_vote 3% @ 100% | 143 | **prediction missed**: cheap constrained samples answered first, unanimous and wrong |
| E001 | 2026-11-17 | 3 | dev (60) | as task 1 | ≥30K | **100%** (94..100) | program 100% @ 100% | 398 | first candidate used `eval` (rejected by the sandbox); round 1 accepted |
| E001 | 2026-11-17 | 4 | dev (60) | as task 1 | ≥30K | 80% (68..88) | think_vote 100% @ 80% | 3831 | **prediction missed**: `''.join(text)` 8 times (string read as list); 7 read timeouts (40 sequences vs 16) |
| E001 | 2026-11-17 | 5 | dev (60) | `escalate: false` | ≥30K | 85% (74..92) | score 100% @ 85% | 128 | stacked calibration 78.4 → 80.0% held-out; singletons 88% right on held-out |
| E001 | 2026-11-17 | 6 | dev (60) | unchanged | ≥30K | 93.3% (84..97) | score 93% @ 96%, think_vote 7% @ 50% | 408 | journal counterfactual: the 4 escalated had 3 right before thinking, 2 after |
| E001 | 2026-11-17 | 7 | dev (60) | vote pooling by meaning | ≥30K | 20% (12..32) | generate 68% @ 20%, think_vote 32% @ 21% | 941 | 40/48 errors wrong knowledge, 6 near misses, 2 the model talking |
| E001 | 2026-11-17 | 8 | dev (24) | Triton-friendly shapes, outcome histogram | ≥16K | consistency 0.000 | static OK 96%, `not_run` 24/24 | 1315 | **prediction missed**: dev inputs have no signature (O010), execution never attempted |
| E002 | 2026-11-17 | 1 | dev (60) | @ f13d95a | ≥30K | **100%** | program 100% | 253 | round 0 thinking ran out; no-think round 1 accepted |
| E002 | 2026-11-17 | 3 | dev (60) | same | ≥30K | **100%** | program 100% | 252 | as task 1 |
| E002 | 2026-11-17 | 4 | dev (60) | `repr()` assert induction | ≥30K | **100%** | program 100% | 134 | `''.join(ast.literal_eval(text))` accepted in round 0 |
| E002 | 2026-11-17 | 2 | dev (60) | probe | ≥30K | 43.3% (32..56) | generate 100% @ 43% | 1925 | probe: cheap 50% vs thinking 60% → cheap_only; **124/200 thinking samples truncated at 768 tokens** |
| E002 | 2026-11-17 | 5 | dev (60) | probe | ≥30K | 85.0% (74..92) | score 100% | 1362 | probe: cheap 75% vs thinking 80% → cheap_only |
| E002 | 2026-11-17 | 6 | dev (60) | probe | ≥30K | **96.7%** (89..99) | score 100% | 2146 | probe: cheap 95% vs thinking 78% (8 vs 1, p = 0.039) → cheap_only |
| E002 | 2026-11-17 | 7 | dev (60) | probe | ≥30K | 16.7% (9..28) | generate 98% | 2054 | probe: cheap 20% vs **thinking 38%** (9 vs 2 overall, p = 0.065); split groups lost the signal → cheap_only |
| E002 | 2026-11-17 | 8 | dev (24) | old notebook cell ran dev | ≥16K | 0.000 | `not_run` 24/24 | 1289 | still not measured; CLI now redirects to test inputs |
| E003 | planned | all | dev (60); task 8 on test inputs | counting rules + listing rung (task 2); probe tests overall difference first; task 2 thinking 1536 tokens; task 8 auto-redirect | ≥30K | predictions in `08_THIRD_RUN.md` §8 | | | any notebook cell works; `scripts/kaggle_run.sh` preferred |
| E004 | planned | 2 | dev (100) | `vote_n` 3 vs 5 (paired), only if the probe picks thinking | ≥30K | | | | the cost lever for the most expensive task |
| E005 | planned | 8 | test (24) | differential testing: model-written PyTorch reference for functions without a twin | ≥16K | | | | only after E002 shows how many kernels run at all |

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
- E001 (tasks 1/3) → the repaired induction works: verified programs, 60/60, no model call per input.
- E001 (task 2) → a setting is not a behaviour: per-task "should this task think?" is now measured by the escalation probe instead of set by hand; the hand-set task-5 `escalate: false` is removed for the same reason.
- E001 (task 4) → induction examples are `repr()` asserts; feedback reports the return type and "returned its input unchanged"; repeated code is detected and called out.
- E001 (task 8) + O010 → task 8 is measured on test inputs; kernels proven wrong are replaced by a loud stub (S_exec²/S_call algebra).
- E001 (task 7) → no new algorithm (knowledge-bound); talking replies are rejected by the normaliser (O012).
- E001 (systems) → workers × vote_n ≤ server `--max-num-seqs`; tasks 1–4 use 3 workers; timeout 900 s.
- E002 (tasks 1/3/4) → all three exact tasks solved by verified programs.
- E002 (task 2) + O013 → counting rules in the system prompt; listing rung (model names words, code counts); thinking budget 1536.
- E002 (task 7) → selection bias in E000/E001 ("thinking ≈ cheap") corrected by the paired probe; the probe now tests the overall difference first, groups only choose where to think.
- E002 (task 8) → the CLI measures code tasks on test inputs whenever dev inputs cannot be executed.
