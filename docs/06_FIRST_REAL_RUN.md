# 06 — The first real run: is it true, what went wrong, and how to think about it

> Companion to `docs/LEARNING_GUIDE.md` (the ideas) and `docs/04_EXPERIMENT_LOG.md` (the
> numbers). This file is about **one event**: the first time Qwen3-4B actually ran the
> pipeline (Kaggle, 2 x Tesla T4, 2026-11-16, 20 dev inputs per task). It answers three
> questions, in order: *Can the numbers be trusted? What do they say? How do you reason from
> a result like this to the next change, every time?*
>
> Raw log: `docs/results/kaggle_run_2026-11-16_raw.md`. Everything below is derived from
> that log plus the code and data in this repository. Where I could not verify something, it
> says so.

---

## 0. Verdict in five lines

1. **The results are real.** They came from a live vLLM server running Qwen3-4B, not from the
   stand-in model. The evidence is in §1.
2. **The measurement is honest but weak.** No answer could have leaked from the prompt into the
   test, but 20 inputs per task means every accuracy carries a range of roughly ±20 points.
3. **Three things were broken, none of them the model.** The program path never produced a
   program (tasks 1, 3, 4), the cheap scoring gate was trusted when it should not have been
   (task 2), and the kernel executor could not run anything (task 8). All three are fixed in
   code and the fixes are tested.
4. **One real weakness is the model.** Task 7 (trivia) is limited by what a 4B model knows. The
   pipeline can stop losing answers to formatting; it cannot add knowledge.
5. **The next run should be dramatically better on 1, 3, 4 and 2,** and it should take minutes,
   not half an hour, per exact task. Predictions, with ranges, are written down in §6 so we
   can check my judgement afterwards.

---

## 1. Is it real? The validation I actually did

A number is only as good as the thing that produced it. Before reading a single accuracy, ask
*what produced it*, and check each link.

| question | where I looked | what I found | verdict |
|---|---|---|---|
| Which backend answered? | `configs/default.yaml` -> `backend.kind: openai`, `base_url: 127.0.0.1:2026` | The real server. The stand-in model only runs with `configs/offline.yaml` (`kind: demo-reader`). | real |
| Was a model really there? | the `cairn doctor` block in the log | 1,323 characters of reasoning for "17 + 26", 5 logprob candidates, 4,016 cached prompt tokens on the second call (1.09 s -> 0.14 s). A fake cannot produce these. | real |
| Could the test answers be in the prompt? | `cairn/data.py` `split_examples`; `TaskRunner.__init__` builds the prompt from `splits.pool` only | Split by SHA-256 of `(seed, id)`; the four roles never overlap; a test asserts no dev id appears in any prompt. | no leak |
| Is accuracy computed the way the organisers compute it? | `cairn/eval/metrics.py` `is_correct` | Exact match after stripping outer spaces. Same as the official description. | honest |
| Does each task's time make sense for a real GPU? | `Time` column in the summary | 300 to 1,600 s per task for 20 inputs plus preparation. A mock answers in milliseconds. | real |
| Could the numbers be luck? | Wilson interval, printed in the log | 14/20 -> 48%..85%. Yes, largely. | weak |

**The one thing I could not verify:** the per-input journal and the report JSON files stayed on
Kaggle (`/kaggle/working/runs/dev/task*/dev.report.json`). They hold the preparation notes
(program attempts, calibration numbers) and the token usage per stage. Everything in §3 about
*why* things failed is therefore reasoning from the code and the visible symptoms, marked as
such. The next run prints these notes directly in the log so this gap closes.

**Remember:** validate the *producer* of a number before the number. Config, server evidence,
split, metric, timing, error bar: six checks, ten minutes, and you know what you are holding.

---

## 2. What the numbers say, task by task

The accuracy line is the least informative line in the log. The breakdown by *source* (which
rung of the decision ladder answered) is where the information is. Read it like a doctor
reads a blood panel: each line is a separate organ.

| task | accuracy | breakdown (share of inputs @ accuracy of that rung) | what it tells us |
|---|---|---|---|
| 1 smallest gap | 70% | `think_vote` 85% @ 82%, `generate` 15% @ 0% | **No `program` line at all.** The verified-program path never fired. 15% of inputs got no valid answer from 5 thinking votes and fell to a plain guess, which was always wrong. |
| 2 count nouns/verbs | 70% | `score` 35% @ 29%, `think_vote` 65% @ 92% | The cheap rung was "sure" on a third of inputs and wrong on most of them. The expensive rung is excellent. The gate is the problem, not the model. |
| 3 collatz step | 100% | `think_vote` 100% @ 100% | Again no `program` line. Thinking carried it, at about 70 s per input. |
| 4 concatenate | 70% | `think_vote` 100% @ 70% | No `program` line. Thinking makes character-level slips on 13-item lists. |
| 5 tweet sadness | 70% | `score` 75% @ 87%, `think_vote` 25% @ 20% | The cheap rung is good; escalation *hurt* (1 right out of 5). Thin evidence, structural reason: see §3D. |
| 6 same genre | 95% | `score` 90% @ 100%, `think_vote` 10% @ 50% | The sentence-overlap signal found during the offline analysis carries the task. |
| 7 trivia | 20% | `generate` 55% @ 18%, `think_vote` 45% @ 22% | Both rungs poor and *equally* poor. When more thinking does not help, the limit is knowledge, not reasoning. |
| 8 kernels | 0.000 | static check OK 95%, runs 0%, matches 0% | 19 of 20 kernels passed the static check and **not one ran**. All-or-nothing results point at the environment or the harness, not at the model. |

Two reading habits made this table possible:

* **Absence is evidence.** The missing `program` source on tasks 1, 3 and 4 is the single most
  important fact in the log, and it is a line that is *not there*. Always check that every path
  you expected to run has left a trace.
* **Uniform failure is a harness smell.** 0 of 20 kernels running, when 19 of 20 are
  syntactically sound Triton with the right signature, is not how a model fails. Models fail
  unevenly.

---

## 3. The root causes, with the reasoning chain

### 3A. Program induction never produced a program (tasks 1, 3, 4)

**Symptom.** No prediction came from the `program` source.

**What the code did.** `TaskRunner._prepare_program` built the induction prompt from the whole
30K-token example block (`prefix_messages(0)`), asked for a `solve()` function with
**thinking on and a 2,048-token reply budget**, up to 4 rounds x 2 candidates, and accepted
only a program that reproduced 150 verify and 150 confirm examples exactly.

**Hypotheses, ranked by how cheap they are to test:**

| # | hypothesis | cost to test | result |
|---|---|---|---|
| H1 | The labels are noisy, so a *correct* program can never score 100% and is always rejected | 1 minute: run the true rule over every example | **Rejected.** 5,500 / 3,997 / 3,993 labels match the rule exactly (0 mismatches). |
| H2 | The reply budget ran out while thinking, so no code block ever came back ("no code block found" on every attempt) | needs the report notes from Kaggle | **Most likely.** Qwen3 thinking over a 31K-token prompt easily exceeds 2,048 tokens; with the vLLM reasoning parser the visible answer is then empty, and `extract_code("")` is `None`. |
| H3 | The sandbox failed to run the candidate (so every candidate scored 0%) | the sandbox tests pass on Linux; the same child-process pattern worked for the kernel driver | unlikely |
| H4 | The model wrote wrong programs 8 times in a row for "join these strings" | contradicts what a 4B model can do on a 5-line task when it actually gets to write code | unlikely |

**Fix (all in code, all tested):**

1. The induction prompt now shows a **short, evenly spaced slice of 48 examples**, not 30K
   tokens. Writing a five-line rule needs a few dozen cases and a clear head; the program is
   then proven on 300 held-out examples anyway (`induction_examples`).
2. Reply budget **4,096 tokens** (`induction_max_tokens`), room for thinking *and* code.
3. If a whole round yields no code, the next round runs **without thinking**. A plain reply
   cannot be swallowed by a think block (`ProgramInducer`, test
   `test_no_code_with_thinking_on_retries_without_thinking`).
4. Every attempt is logged with its **finish reason** (`length` = budget ran out), its verify
   accuracy and its first failing example, and the run prints "program: ACCEPTED" or "program:
   not accepted, best verify 93%" so the log alone tells the story next time.
5. The backup path (thinking + vote) for these tasks gets 2,048 thinking tokens instead of
   1,024: listing 45 pairwise differences does not fit in 1,024.

**Predicted effect.** Tasks 1, 3, 4 at or near 100%, and each run in minutes instead of
20-plus, because an accepted program answers every input with **zero model calls**.

**Why this matters beyond the score.** The program path is the clearest example of the
project's central idea: left-over labelled examples are a *unit-test suite* for what the
model proposes. When the suite never gets a candidate to test, the whole idea is switched off
silently. The fix is as much about *visibility* (print the attempts) as about the budget.

### 3B. A coverage promise is not an accuracy promise (task 2)

**Symptom.** The `score` rung answered 35% of inputs and was right on 29% of them.

**Why it is not a paradox.** Conformal prediction promises that the true label is inside the
predicted *set* at least 90% of the time, **averaged over all inputs**. It says nothing about
the accuracy of the inputs where the set happens to contain one label. With 7 labels, the
promise can be kept by making most sets large while the few one-label sets are mostly wrong.
I had treated "one label in the set" as "sure", which the mathematics never said.

**Why counting is special.** A single next-token guess at "how many verbs" cannot list the
words; it pattern-matches sentence length. The expensive rung, which lists the words, was right
92% of the time. The structural reason, not just the number, is why the fix below is safe to
apply from 20 inputs.

**Fix.**

1. **A second gate, measured on held-out examples.** The runner now computes, on the second
   calibration half, how often the cheap rung's one-label answers are *right*. Below 85%
   (`min_singleton_accuracy`, judged only with at least 10 such answers), the cheap rung may
   not end the ladder: every input thinks and votes, and the vote is over all labels, not the
   distrusted set. Test: `test_runner_distrusts_a_cheap_rung_that_is_confidently_wrong`.
2. **Task 2 skips scoring entirely** (`use_scoring: false`): the structural argument says the
   cheap rung cannot work for counting, so do not pay for it.

**Predicted effect.** Task 2 close to the 92% the expensive rung showed, at the cost of five
768-token thinking votes per input.

### 3C. The kernel executor ran nothing (task 8)

**Symptom.** 19/20 static OK, 0/20 callable, consistency 0.000.

**Reasoning.** `code_quality` counts a kernel as callable only if its execution status is one
of `exec_ok`, `wrong_result`, `called_no_reference`. "Static OK" means the tier was at least
2, which (reading `Annotator._judge`) means the status was one of `unavailable`,
`no_valid_inputs`, `timeout`, `not_run` or `unconfirmed_error`. The log did not say which.
That gap is itself a defect: a 0% must always come with a reason.

Two explanations fit, and both are probably true:

* **`unconfirmed_error` for most inputs.** The test tasks are fused operations
  (`fused_bmm_rmsnorm_gelu_dropout_sub`, ...) with no PyTorch twin, so the driver can only try
  guessed inputs. Its guesses were tiny odd shapes such as `(6, 10)`. Triton rejects those even
  for a *correct* kernel: `tl.dot` needs every side to be at least 16, `tl.arange` needs a
  power-of-two length, and generated kernels often derive a block size from the input size. A
  correct kernel fed `(6, 10)` raises, and we counted it as broken.
* **Compile time.** The first Triton compile in a fresh child process on a T4 can take tens of
  seconds; with a 180 s cap and four workers sharing two GPUs that vLLM already occupies at 85%,
  some runs may have hit `timeout` or failed to initialise CUDA.

**Fix.**

1. The driver's shape guesses are now **multiples of 16 and powers of two first**
   (`(32, 64)`, `(64, 16)`, ...), with the odd shapes kept last for PyTorch functions that only
   accept them.
2. `code_quality` reports a **histogram of execution outcomes**, and the run prints it, so the
   next log says "unconfirmed_error: 14, timeout: 3, ..." instead of "runs 0%".

**Predicted effect.** Some kernels become callable; the score becomes non-zero. Task 8 will
still be the weakest task: a 4B model writing fused Triton kernels is hard, and the official
metric (`S_exec^2 / S_call`) punishes kernels that run but compute the wrong thing.

### 3D. Escalation hurt on tweet sadness (task 5)

**Symptom.** The cheap rung was right 87% of the time on the inputs it was sure about; the
expensive rung was right on 1 of the 5 unsure inputs.

**Evidence quality.** Five inputs. This is not a measurement, it is an anecdote. So why act on
it at all? Because there is a *structural* reason to expect it: binary sentiment is a surface
judgement that small models make better fast than slow. Long deliberation over a tweet tends
to talk the model out of the obvious reading. The 20-input run did not create this belief; it
agreed with it.

**Decision.** `escalate: false` for task 5: unsure inputs take the calibrated best guess. This
is explicitly provisional and the config file says so. The next run with 100 inputs decides it
properly; the experiment log holds the prediction.

### 3E. Trivia is knowledge-bound (task 7)

Both rungs sit near 20%. When "think harder" and "answer fast" score the same, the model does
not know the answers, and no prompt engineering fixes that.

What *can* still be lost is a known answer that is written two ways and splits the vote:
"the beatles" 2 votes, "beatles" 2 votes, "elvis" 3 votes -> wrong. Votes are now pooled by
meaning (articles, quotes and punctuation dropped) and the winning *spelling* is the one the
model used most, so the output still looks like the examples (`trivia_key`,
`majority_vote(group_key=...)`). Expected gain: a few points at most. The honest expectation
for task 7 stays low; the report should say so rather than hide it.

### 3F. Two defects the run exposed in the tooling, not the pipeline

* The sandbox child process was started with an empty environment and used POSIX-only alarm
  signals. On Linux (Kaggle) this works; on Windows (where you develop) the child could not
  start at all and four tests failed. The child now gets the one variable Windows needs and
  skips the alarm where it does not exist; the two tests that are genuinely about POSIX
  resource limits are skipped on Windows with a reason.
* The repository's `tests/` folder was not a package, so a stray `tests` package in
  site-packages shadowed it. One `__init__.py` fixes that permanently.

---

## 4. Time is a resource: the budget for the next runs

The log's `Time` column, per task, for 20 inputs plus preparation (2 x T4, TP=2, fp16, eager):

| task | seconds | what the time went to (reasoned, not measured per stage) |
|---|---|---|
| 1 | 1,609 | 8 failed induction attempts over a 31K prompt (~800 s) + 20 x 5 thinking votes |
| 2 | 719 | 150 one-token calibration calls + 13 x 5 thinking votes |
| 3 | 1,422 | failed induction + thinking votes |
| 4 | 1,302 | same |
| 5 | 322 | mostly one-token scoring; 5 escalations |
| 6 | 302 | same |
| 7 | 599 | direct + 3 samples for all, thinking for 9 |
| 8 | 1,006 | 20 x (up to 3 rounds x 2 candidates x up to 3,072 tokens) + execution attempts |

Rules of thumb that fall out of it, for this hardware:

* A **thinking vote (5 x 1,024 tokens) costs about 30 to 40 s per input.** 500 test inputs of
  pure thinking is 4 to 6 hours. That is a whole Kaggle session for one task.
* **One-token scoring costs about 1 to 2 s per input** once the shared prefix is cached.
* An **accepted program costs nothing per input.** Tasks 1, 3, 4 collapse from hours to minutes.
  That is why `scripts/kaggle_run.sh` runs them first.
* **Task 8 is the hog**: 166 test inputs x up to 6 generations of 3K tokens. Budget a session
  for it alone, or lower `codegen_rounds` to 2 if time is short.

Kaggle gives 12 hours per session and 30 GPU hours per week. The journal makes every run
resumable, so a session boundary costs nothing *if* the notebook version is saved so that
`/kaggle/working` survives. Plan the sessions around the table, not around hope.

---

## 5. How to think like this, in every case

The user asked for "how to think like you in all cases". There is no trick; there is a short
procedure, applied stubbornly. Each step below names the moment in this run where it mattered.

1. **Ask what produced the number before you read the number.** Config, backend, split,
   metric, timing, error bar (§1). *Moment:* confirming the run was the real model took ten
   minutes and made every later conclusion worth having.

2. **Read the breakdown, not the average.** An average hides which part is broken. *Moment:*
   task 2 at 70% looked "fine"; the breakdown showed one rung at 29% and the other at 92%.

3. **Treat absence as evidence.** List what *should* have left a trace and check each.
   *Moment:* the missing `program` source was the biggest finding and the easiest to miss.

4. **Distrust uniform failure.** Models fail unevenly; harnesses fail uniformly. *Moment:*
   0 of 20 kernels running with 19 of 20 statically sound.

5. **Write hypotheses down and rank them by the cost of the test, then run the cheapest.**
   *Moment:* "labels are noisy" cost one minute to kill (§3A, H1). Had it been true, every
   other fix would have been wasted.

6. **Separate the guarantee you have from the guarantee you need.** Read the theorem's
   statement, not its reputation. *Moment:* conformal coverage is marginal over all inputs; it
   never promised that one-label sets are accurate (§3B).

7. **Decide by structure when the evidence is thin, and schedule the real test.** Five inputs
   prove nothing, but "a single token cannot count words" is a reason. Act on the reason,
   write the prediction, re-test with enough data. *Moment:* tasks 2 and 5.

8. **Make every failure say why.** A "0%" without a reason costs a whole GPU session to
   diagnose. *Moment:* task 8 now prints its outcome histogram; induction prints every attempt.

9. **Price the next run before starting it.** Seconds per input per rung, times inputs, times
   tasks, against the session limit. *Moment:* §4. Without it, a test run would have died at
   hour 12 with three tasks done.

10. **Predict before you measure.** Write the expected range for each task before the run (§6).
    Afterwards, the gap between prediction and result is the most honest measure of whether
    you understand the system.

11. **Fix the tooling that let the defect hide, not only the defect.** *Moment:* the Windows
    sandbox and the `tests` package. Neither changed a score; both would have hidden the next
    bug.

12. **Keep the path to the honest answer open.** A fix that cannot be turned off, or a number
    that cannot be reproduced, is a liability. Every change here is a config key with a
    comment that names the evidence; every number has a command that regenerates it.

**The picture to keep:** you are a detective, not a cheerleader. The result is a witness
statement. Your job is to find where it is lying, where it is incomplete, and what the
cheapest question is that would settle it.

---

## 6. Predictions for the next run (60 dev inputs per task, same hardware)

Written *before* the run, so they can be checked. The 95% range for 60 inputs is about
±12 points; a prediction "lands" if the measured interval overlaps the predicted one.

| task | predicted accuracy | basis | if it misses low, look first at |
|---|---|---|---|
| 1 | 97..100% | accepted program; data is 100% rule-consistent | the printed program attempts: finish reasons, best verify accuracy |
| 2 | 82..95% | the expensive rung's 92% on 13 inputs; now applied to all | thinking budget 768 too small for long sentences; votes split 2-2-1 |
| 3 | 97..100% | accepted program | same as task 1 |
| 4 | 97..100% | accepted program | same as task 1; also check that `solve` returns a string, not a list |
| 5 | 72..82% | 87% on the sure 75%, plus the calibrated guess on the rest | the calibration note: held-out accuracy before and after; singleton accuracy |
| 6 | 88..97% | overlap signal; 95% seen | nothing yet; this one is working |
| 7 | 18..32% | knowledge-bound; pooling adds a little | the vote counts in the journal: are known answers split by spelling? |
| 8 (consistency) | 0.05..0.25 | some kernels become callable; most fused ops still fail | the outcome histogram; if `unconfirmed_error` dominates, the input guesses are still wrong |

Expected wall time with programs accepted: tasks 1, 3, 4 under 10 minutes each; task 2 about
40 minutes; 5 and 6 about 5 minutes each; 7 about 20 minutes; 8 about 60 minutes. Roughly 2.5
hours for the whole dev pass at 60 inputs.

---

## 7. After the next run: what to look at, in order

1. `program:` lines for tasks 1, 3, 4. ACCEPTED or not. If not, the attempt log says whether
   the budget ran out (`finish=length`), whether the code was close (verify 95%+: read the first
   failure), or whether nothing came back at all.
2. `calibration:` lines for 5 and 6. Held-out accuracy before and after; the share of sure
   answers and how often they were right; whether singletons were trusted.
3. The source breakdown per task, as in §2. Any rung below 70% with a large share is the next
   target.
4. Task 8's outcome histogram. `unconfirmed_error` dominant -> input guessing; `timeout`
   dominant -> compile time, raise the cap or lower workers; `unavailable` -> the child cannot
   import torch, environment problem.
5. The `Time` column against §6. A task far over budget is a config problem (votes, tokens,
   rounds), not a model problem.
6. Paste the eight `dev.report.json` files (or at least their `notes` and `usage` sections)
   into `docs/results/`. They are the raw material for the technical report's ablation tables.

---

## 8. Small glossary for this file

* **source / rung**: which stage of the decision ladder produced an answer (`program`,
  `score`, `think_vote`, `generate`, `codegen`, `fallback_*`). Recorded on every prediction.
* **singleton**: a conformal prediction set that contains exactly one label; the cheap rung's
  way of saying "sure".
* **coverage**: the share of inputs whose true label is inside the predicted set. This is what
  conformal prediction guarantees; it is an average over *all* inputs.
* **finish reason `length`**: the server stopped the reply because the token budget was used
  up. With thinking on, this usually means the visible answer is empty.
* **verify / confirm**: the two held-out example sets a model-written program must reproduce
  exactly; failures on *verify* are shown back to the model, *confirm* is never shown.
* **callable (task 8)**: the generated wrapper ran on guessed inputs and returned tensors,
  right or wrong. The official score needs callable *and* correct.
* **Wilson interval**: the 95% range printed next to every accuracy; for 14/20 it is
  48%..85%, which is why 20 inputs can only find large effects.
