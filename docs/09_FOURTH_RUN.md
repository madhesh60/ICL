# Fourth Qwen3-4B run and kernel replay

**Run date:** November 18, 2026 · **Analysis and replay:** November 24, 2026 · **Generation commit:** `7d1dddd`

Cairn answered **340 of 420 development questions correctly (80.95%)**. Task 2 contributed all 15 additional correct answers over the previous run. The original kernel run recorded zero callable predictions. Repairing the executor made six of the same 24 predictions callable in the CPU interpreter, including four that matched PyTorch. The two measurements use different execution environments. An official score requires a full test submission. [Original run](../RESULTS_NOW.md) · [Saved predictions and reports](../dev_results_now/) · [CPU replay](results/e003_cpu_replay.json)

## How the project reached this run

| Date | Evidence and decision |
|---|---|
| Nov 14–15 | The baseline review documented output parsing failures and the mismatch between the baseline prompt budget and the required context. The plan introduced separate demonstration, calibration, development and holdout roles. [Original review](01_CODEBASE_EXPLAINED.md) |
| Nov 16, E000 | The first Qwen run scored 99/140 (70.7%). Verified programs supplied no final answers. Subsequent changes shortened the induction prompt and increased its completion budget. An earlier README gave 67.9% for this mean. The seven counts in the original log sum to 99. [Raw results](results/kaggle_run_2026-11-16_raw.md) · [First run](06_FIRST_REAL_RUN.md) |
| Nov 17, E001 | The development sample grew to 60 per task. Programs answered every sampled question correctly on tasks 1 and 3. Task 4 exposed an ambiguity between a string and the list it represented. Executable examples made the input type explicit. [Second run](07_SECOND_RUN.md) |
| Nov 17, E002 | Programs reached 60/60 on all three exact tasks. The probe compared cheap and costly inference on the same questions. It exposed selection bias in earlier routing comparisons and frequent truncation on task 2. The next run lengthened the reasoning budget and tried listing counted words. [Third run](08_THIRD_RUN.md) |
| Nov 18, E003 | The measured policy selected reasoning for counting and direct generation for trivia. Task 8 finally used test signatures. Reading the child exceptions exposed a common import failure. [Run log](../RESULTS_NOW.md) |
| Nov 24 | The saved journals were rescored by ID. A CPU replay established the first numerical kernel matches. The new retrieval, counting and trivia paths are available for the next FlagScale experiment. [Implementation](../cairn/) |

## Results by task

Tasks 1–7 each use the first 60 IDs in the seeded development split. Their sample sizes are equal, so the pooled accuracy and mean per-task accuracy coincide. The intervals below are per-task Wilson 95% intervals. Task 8 uses the first 24 of 166 unlabelled test inputs.

| Task | Correct | Accuracy (95% interval) | E002 correct | E003 source or decision |
|---|---:|---:|---:|---|
| 1 · closest integers | 60/60 | 100% (94.0–100) | 60 | Verified program |
| 2 · count nouns and verbs | 41/60 | 68.3% (55.8–78.7) | 26 | Reasoning first |
| 3 · Collatz | 60/60 | 100% (94.0–100) | 60 | Verified program |
| 4 · concatenation | 60/60 | 100% (94.0–100) | 60 | Verified program |
| 5 · sadness | 51/60 | 85.0% (73.9–91.9) | 51 | Calibrated label score |
| 6 · same genre | 58/60 | 96.7% (88.6–99.1) | 58 | Calibrated label score |
| 7 · trivia | 10/60 | 16.7% (9.3–28.0) | 10 | 59 direct answers, one nearest-example fallback |
| 8 · Triton | 0/24 callable on Kaggle | Local consistency 0.000 | Dev signatures unsuitable | 22 import failures, two static rejections |

### Counting

The probe scored the cheap policy at 13/40 and the reasoning-first policy at 28/40. Seventeen disagreements favoured reasoning and two favoured the cheap policy (exact paired p = 0.0007). On development, reasoning supplied 36 correct answers out of 48. Twelve inputs produced zero usable reasoning votes and fell back to direct generation, which answered five correctly. None of the 19 missed inputs contained the exact gold answer among the recorded reasoning votes. Changing the tie-breaker alone would recover zero here.

The task took 7,719 seconds. Probe and production together used 514,585 completion tokens on the reasoning rung. Missing votes have several possible causes. New runs record finish reasons and valid vote counts so truncation can be distinguished from parsing failures. [Task 2 report](../dev_results_now/task2/dev.report.json)

Two changes address different failure modes:

1. **Select examples by grammatical structure.** A noun-count query retrieves noun-count demonstrations, and a verb-count query retrieves verb-count demonstrations. BM25 ranks short sequences of function words, auxiliaries and word endings. This lets captions about different objects illustrate the same construction. With twelve neighbors and similarity-weighted voting alone, lexical retrieval scored **150/300** and structural retrieval **158/300** on dev. Lexical alone won 42 paired cases, structural alone won 50 (p = 0.466). The eight-answer difference motivates a Qwen ablation. Its size is uncertain. [Feature construction](../cairn/tasks/counting.py) · [Retrieval](../cairn/context/builder.py)
2. **Bind each counting decision to a word occurrence.** Qwen receives numbered positions and emits one bit per position. The decoder requires the exact length, sums the marked positions and votes over counts. Two occurrences of the same word occupy separate slots. The prompt uses the mask format in its system instruction and final reminder. A missing or malformed sample supplies zero confidence support. If every reasoning vote is unreadable, the mask provides another way to answer before the final direct call. The earlier list format remains available for comparison. [Decision logic](../cairn/pipeline/annotator.py)

### Trivia

The cheap policy won 8/40 calibration questions and reasoning first won 14/40. Nine disagreements favoured reasoning and three favoured the cheap policy (p = 0.146). The configured threshold of 0.10 selected the cheaper policy. The prior forecast of 25–42% missed the observed 16.7%.

Only three of the 50 dev misses had an exact gold answer elsewhere in the recorded votes. Perfect selection among those recorded responses could therefore reach at most **13/60** on this batch. The new costly path asks Qwen for a short description of the answer type and relationship in the clue. Those terms retrieve official demonstrations for a second pass. The model answers the original clue, and the journal counts how many examples the new search introduced. The next probe uses a fixed 80 calibration questions, alongside a plain-reasoning control for the ablation. [Task 7 report](../dev_results_now/task7/dev.report.json) · [Strategy settings](../configs/default.yaml)

### Kernel generation

The original console said static checking passed 0%. It counted final candidate tiers, and a runtime error moved a statically valid candidate below the reporting cutoff. All 22 import failures carried the same exception, `ValueError: @jit functions should be defined in a Python file`. The matching task-8 journal was recovered from the local run directory. Its accompanying prediction file and the supplied export have identical SHA-256 digest `8adc505734f703b804be8b209034fe33d5332bb297bfb6df685c05bdba32e75a`. [Original metrics and grouped journal errors](results/e003_kernel_diagnosis.json)

The parent now writes each candidate to a temporary module and keeps it alive while the child loads and calls the wrapper. Static checks, execution and reference comparison have separate outcomes. A reference match requires two successful numerical trials. The repair loop skips duplicate source within a query.

| CPU replay outcome | Count | What the test established |
|---|---:|---|
| Matched PyTorch | 4 | `sqrt`, `sub`, `add` and `relu` matched on both trials |
| Callable without reference | 2 | `relu_sqrt` and `sigmoid_argmax` returned finite tensors |
| Call failed on a reference-accepted input | 6 | These wrappers raised on an input accepted by PyTorch |
| Unconfirmed error | 9 | Guessed inputs raised and no reference validated those inputs |
| No valid inputs | 1 | The input generator found no arguments accepted by the reference for `grid_sample` |
| Static rejection | 2 | The checker flagged unknown Triton names before execution |

The local consistency calculation is `(4/24)² / (6/24) = 0.1111`. The replay used Python 3.10.11, CPU PyTorch 2.9.1 and triton-windows 3.5.1.post24 in interpreter mode. GPU compilation, device-specific behavior, optional parameters and the organiser's input suite still require validation. The raw replies and the saved prediction file were preserved. [Per-kernel outcomes](results/e003_cpu_replay.json) · [Original predictions](../dev_results_now/task8/openseek-8.jsonl) · [Triton Windows compatibility](https://github.com/triton-lang/triton-windows)

## Reproduction

From the repository root after installing Cairn:

```bash
cairn audit-dev --run-dir dev_results_now
cairn --config configs/default.yaml baseline --task 2 -k 12 --neighbors lexical
cairn --config configs/default.yaml baseline --task 2 -k 12 --neighbors structural
cairn replay-kernels dev_results_now/task8/openseek-8.jsonl
# With torch and Triton installed in the executor environment:
CAIRN_TRITON_INTERPRET=1 cairn replay-kernels dev_results_now/task8/openseek-8.jsonl --execute
```

The audit reads the split settings recorded with each run, joins predictions to official labels by ID, and checks the recomputed counts and accuracy against the reports. Static-only replay leaves numerical metrics empty. Execution runs model-generated Python in child processes. Use an isolated evaluation environment for this workload.

## Next measured comparison

On November 25 the first full-finals launcher exceeded the Kaggle session duration. The interrupted run supplied no new accuracy figure in the reported output. The revised launcher checks completed splits, restores saved finals output, and stops on a session budget. [Session recovery and finals procedure](KAGGLE_FINALS.md)

Start the competition FlagScale endpoint using the README. Use separate run directories for each settings file so the journals remain attributable. Begin with the E003 settings at commit `7d1dddd` on the same development IDs.

| Experiment | Change independently | Record |
|---|---|---|
| Task 2 retrieval | Switch `context.structural_neighbors` with a fixed tail size and `tail_per_label: 0` in both arms | Paired exact-match wins, noun and verb results, prompt tokens |
| Task 2 format | Switch `strategy.use_word_mask` with fixed retrieval and a fixed reasoning budget | Accuracy, usable samples, zero-vote recoveries, completion tokens |
| Task 7 retrieval | Toggle `strategy.use_clue_expansion` with the same 80 probe IDs and reasoning settings | Paired accuracy, new examples retrieved, generated tokens and elapsed time |
| Task 8 | Replay the saved predictions on GPU, then regenerate on the same 24 signatures | Static pass, callable and correct counts, all failure statuses, consistency |

```bash
cairn --config configs/default.yaml run --task 2,7 --split dev --limit 60 --run-dir runs/e004-dev
CAIRN_TRITON_INTERPRET=0 cairn --config configs/default.yaml run --task 8 --split test --limit 24 --run-dir runs/e004-kernels
```

The larger study uses the development split for method selection, the reserved holdout for a final text check and all eight complete test files for competition packaging.

### Paired Kaggle run

Import the generated one-cell notebook `cairn-e004.ipynb` into a fresh Kaggle GPU T4 x2 session with Internet enabled. Its embedded snapshot includes the current Cairn source, the three official task files, the same Qwen vocabulary used in E003, and the original predictions. To rebuild the notebook from this checkout, run `python scripts/package_kaggle_e004.py`. The same command writes a ZIP for use as a notebook input.

The launcher verifies the snapshot and checks all 144 IDs against E003 before loading Qwen. It pins the official Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c` and launches the vLLM 0.12.0 engine through the single-node FlagScale v1.0.0 serving entry point. All three model shard hashes and the tokenizer hash in the E003 download log match this revision. Task 8 generation uses one worker to leave GPU memory for repair subprocesses. After generation, the launcher releases the model server and replays the old and new kernels on the same device using the repaired executor. [Kaggle launcher](../scripts/kaggle_e004.sh) · [Paired evaluation](../scripts/compare_e003.py)

The run saves `/kaggle/working/cairn-e004/comparison.json`, both per-kernel replay reports, per-task journals, the resolved serving configuration and package versions. The text results include the IDs gained and lost and an exact paired test. The kernel comparison measures the contribution of regenerated predictions under a common local evaluator. These subset measurements do not establish a leaderboard score or isolate individual prompt and retrieval changes.

On November 24 the Windows suite passed 175 tests including real Triton interpreter checks. Two tests for POSIX resource limits were skipped. The offline CLI also produced eight run reports from two inputs per task. The distribution wheel contained the CLI, kernel driver and Triton API table. The local machine had CPU PyTorch, a 2 GB GPU and no service at the configured Qwen endpoint. Model-side ablations and the full FlagScale run require a serving host.
