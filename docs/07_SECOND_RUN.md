# 07 — The second real run: scoring my own predictions, and deciding what to build next

> Companion to `06_FIRST_REAL_RUN.md`. That file diagnosed the first run (E000) and wrote
> down predictions for the next one. This file is about what happened when those
> predictions met reality (E001), and about the decision the run forces: **fix the code, or
> add new algorithms?** It is written so that the reasoning can be reused, not only read.
>
> Raw log: `docs/results/kaggle_run_2026-11-17_raw.md`. Per-task reports (settings, usage,
> preparation notes): `docs/results/e001_reports/`. Kaggle, 2 x Tesla T4, vLLM 0.12.0,
> commit `e34413c`, 60 development inputs per task (24 for task 8), 2026-11-17.

---

## 0. Verdict in six lines

1. **Five of eight predictions landed.** Tasks 1 and 3 reached 100% through verified
   programs; tasks 5, 6 and 7 landed inside their predicted ranges.
2. **The three misses were all mine, not the model's**, and each one has a different lesson:
   a setting I never traced through the code (task 2), a contract the model could misread
   (task 4), and a development split that cannot measure what it is supposed to (task 8).
3. **The prompt layout works in practice.** The server reused **96% to 100%** of prompt
   tokens from its cache on every task. The 30K-token prefix is read once, not 500 times.
4. **The answer to "fix or add?" is: fix three things, add one algorithm.** The new
   algorithm is an *escalation probe* that measures, per task, whether thinking is worth its
   cost, instead of me guessing it from twenty inputs.
5. **Task 7 gets no new algorithm**, on purpose: its errors are missing knowledge, and no
   prompt or vote adds knowledge to a 4B model.
6. **Predictions for the next run are in §8**, together with a time budget that says the full
   test run needs two Kaggle sessions.

---

## 1. Scoring the predictions

`06_FIRST_REAL_RUN.md` §6 wrote down ranges *before* this run. A prediction "lands" if the
measured 95% interval overlaps it.

| task | predicted | measured (95% interval) | landed? | the belief it tested |
|---|---|---|---|---|
| 1 | 97–100% | **100%** (94–100) | yes | a short induction prompt + bigger budget lets the model write the rule |
| 2 | 82–95% | **45%** (33–58) | **no** | `use_scoring: false` sends every input to thinking |
| 3 | 97–100% | **100%** (94–100) | yes | same as task 1 |
| 4 | 97–100% | **80%** (68–88) | **no** | same as task 1 |
| 5 | 72–82% | **85%** (74–92) | yes | calibrated one-token scoring is the right rung |
| 6 | 88–97% | **93.3%** (84–97) | yes | the overlap signal carries the task |
| 7 | 18–32% | **20%** (12–32) | yes | trivia is knowledge-bound |
| 8 | 0.05–0.25 | **0.000** | **no** | Triton-friendly shapes let kernels run |

**Why bother predicting at all?** Because a result without a prediction cannot surprise you,
and surprise is the only signal that one of your beliefs is wrong. Without the table, "task 2
at 45%" reads as "task 2 is hard". With it, it reads as "I believed something about my own
code that is false". The second reading leads to the fix.

---

## 2. The three misses, one by one

Each miss is written the same way: what I believed, what was actually true, how I found it,
and the rule I keep.

### 2A. Task 2: a setting is not a behaviour

**Belief.** After E000, one-token scoring was right on only 29% of the inputs it was sure
about, while thinking and voting was right on 92%. I set `use_scoring: false` for task 2 and
predicted 82–95%, expecting every input to think and vote.

**What was true.** Without scoring, a classification task falls through to the free-form
path. That path first asks for one constrained answer plus three quick samples, and stops if
they agree. They always agreed. The breakdown says it plainly:

```
task 2 [dev] accuracy 45.0% (27/60)
    generate         97% of inputs, accuracy 43.1%
    think_vote       3% of inputs, accuracy 100.0%
```

**How I found it.** The journal records the votes of every input. They read `{'1': 4.0}`,
`{'4': 4.0}`, `{'4': 4.0}` ... four samples, unanimous, and wrong more than half the time.
The errors are spread symmetrically around the right count (−2: 3, −1: 14, +1: 13, +2: 3),
which is what guessing looks like.

**Two lessons, both general:**

* **Trace a setting to the line of code it changes, and test the behaviour you expect.**
  I changed a switch and assumed its effect. A five-line test ("with these settings, the first
  rung is thinking") would have failed before the GPU session began. That test now exists
  (`test_reason_first_routes_a_classification_task_straight_to_thinking`), and it asserts the
  E001 behaviour too, so the mistake is documented in code.
* **Agreement measures confidence, not correctness.** Self-consistency is a good confidence
  signal *when the model's mistakes are random*. When the model is confidently wrong, every
  sample makes the same mistake and agreement is perfect. This is the same lesson as E000's
  conformal sets, in a new costume: **a gate must be checked against held-out answers, never
  trusted because it looks principled.**

### 2B. Task 4: the model misread the contract

**Belief.** E000's induction failed because thinking ate the token budget. With a shorter
prompt, a 4,096-token budget and a no-thinking retry, the model would write the obvious
`''.join(...)`.

**What was true.** It did write `''.join(...)`, eight times. Every reply, read straight from
the run's response cache (`runs/dev/task4/responses.sqlite`), was this:

```python
def solve(text: str) -> str:
    return ''.join(text)
```

`text` is the *string* `"['cutting', 'y', 'T']"`. Joining the characters of a string returns
the same string. The function returned its input unchanged, and every check showed:

```
Expected: cuttingyTthePtheZbNonlyLlossD
Got: ['cutting', 'y', 'T', 'the', ...
```

The thinking replies show the confusion directly: 18,000 characters of "the input is a list,
but the function is given a string, so perhaps the problem statement is wrong..." until the
budget ran out.

**Why the repair loop could not fix it.** The feedback showed the same failure every round,
and the model sent back the same code every round. A loop that adds no new information
converges on its first answer. Four rounds were four copies of round one.

**How it is fixed, without writing the program for the model:**

1. **The examples are shown as executable asserts with `repr()`**:
   `assert solve("['p', 'that.', 'o']") == 'pthat.o'`. The quotes around the list now make
   it impossible to read the input as anything but a string. The instruction says so in plain
   words and names `ast.literal_eval` as the safe way to parse it.
2. **Feedback names the kind of mistake.** It shows the call with `repr()`, and adds a note
   when the function *returned its input unchanged* or *returned a list, not a str*. The
   sandbox now reports the return type to make the second note possible.
3. **Repeats are detected.** Code is compared after removing comments and blank lines; a
   repeated attempt is not re-run, and the feedback says "you wrote this same function
   before, and it failed the same way; change the approach".

**Lesson:** when a capable model keeps making a simple mistake, suspect the *question*. Make
the contract executable, and make every error message say what kind of error it is.

### 2C. Task 8: a development split that cannot measure the target

**Belief.** E000 found no kernel running, so I changed the test shapes to multiples of 16 and
added an outcome histogram. I predicted a consistency score of 0.05–0.25.

**What was true.** The histogram did its job on the first try:

```
execution outcomes: {'not_run': 24}
```

`not_run` means execution was never attempted, which happens only when the task input has
no wrapper signature to call. A check over the data took one minute:

| set | inputs with a parseable wrapper signature |
|---|---|
| the 184 demonstration kernels (pool, calib, dev) | **0** |
| the 166 test inputs | **166** |

The demonstrations are "describe this existing code" tasks; the test inputs are "here is a
specification and the exact signature, write it" tasks. **The development split for task 8
does not resemble the test set**, so it can never measure executability. E000's zero almost
certainly had the same cause.

**Fix.** Task 8's metric needs no gold answer: running the kernel and comparing it with
PyTorch is reference-free. So task 8 is now measured on **test inputs** (`--split test
--limit 24`), and the Kaggle script does that automatically during a development pass.
62 of the 166 test functions have a PyTorch twin (`div`, `tanh`, `matmul`, `conv2d`, ...), so
about 37% of test kernels can be proven right or wrong; the rest can only be shown to run.

**Lesson:** before trusting a validation split, check that it looks like the test set *in the
way your metric cares about*. Here a one-minute count would have saved a GPU session.

---

## 3. Why the hits landed

A hit deserves the same scrutiny as a miss: was it right for the right reason?

* **Tasks 1 and 3: certainty, not luck.** The accepted programs are short and correct
  (sort, then the smallest neighbour difference; halve or triple-plus-one). Each passed 150
  verification and 150 confirmation examples it was never shown. With labels 100%
  rule-consistent (O009), 60/60 is the expected outcome, and each task took under seven
  minutes with **zero model calls per input**.
* **Task 3 also shows the sandbox earning its keep.** The first accepted-looking program used
  `eval`; the sandbox rejected it before running, the feedback said why, and the next round
  used `ast.literal_eval`.
* **Task 5: calibration on held-out examples.** The stacked calibrator moved held-out accuracy
  from 78.4% to 80.0%; one-label answers were right 88% of the time on held-out inputs and
  85% on dev. The two numbers agree, which is what a sound calibration looks like.
* **Task 6: the data detective work paid off.** Held-out accuracy rose from 74.0% with the
  model's scores alone to 94.7% with the sentence-overlap signal stacked in. On dev: 93.3%.
* **Task 7: the prediction was right because the ceiling was understood.** Both rungs sat
  near 20% again.

---

## 4. Free experiments: what the journal already knew

Every prediction is stored with its metadata. That makes some experiments free: no GPU, just
a script over the journal.

**Task 6: did escalation help?** The scoring rung's opinion is stored even when the input
was escalated. For the four escalated inputs:

| gold | score rung alone | after thinking + vote |
|---|---|---|
| N | N | N |
| Y | Y | Y |
| Y | Y (0.506) | **N** |
| Y | Y (0.813) | **N** |

Thinking turned 3 right into 2 right. With E000's one escalated input, the pooled record is
3 of 6 for thinking. That is no evidence that thinking helps, and some that it hurts.

**Task 7: is anything recoverable?** Of 48 errors, 6 are near misses (`the llama` vs
`llama`, `allen ginsberg` vs `ginsberg`, `lake victoria` vs `victoria`), 2 are the model
talking instead of answering (`wait, the correct answer should be...`, `clue: he was a
losing super bowl quarterback`), and **40 are simply wrong answers** (`thumbelina` for
`tom thumb`, `iron` for `bronze`). The near misses go in both directions (an article added,
an article dropped), so no single formatting rule fixes them. The talking replies are now
rejected by the normaliser so they cannot win a vote.

**Lesson:** store enough per prediction to answer the next question without re-running. The
journal turned "should task 6 escalate?" from a GPU experiment into a ten-line script.

---

## 5. Systems findings

**The tree-shaped prompt works.** Share of prompt tokens served from the server's cache:

| task | 2 | 4 | 5 | 6 | 7 | 8 |
|---|--:|--:|--:|--:|--:|--:|
| generate / score / think calls | 0.96–1.00 | 1.00 | 0.98 | 0.97 | 0.93–1.00 | 0.82–0.98 |

For task 5, the 249 calibration calls read 7.7M prompt tokens, of which 98% came from the
cache. This is the measured version of the design claim in the README: one shared prefix per
task, grouped sending, one primer request.

**Timeouts came from over-subscription.** Task 4's backup path logged seven 600-second read
timeouts. Eight workers each asked for five thinking samples: 40 sequences against a server
configured for 16 at once (`--max-num-seqs 16`). Requests queued past the timeout and were
retried, wasting work. The rule now written in the settings: **workers × votes ≤ the server's
sequence limit.** Tasks where every input may think use 3 workers (15 sequences).

**Thinking is the cost driver.** A five-vote thinking call produced about 3,400 completion
tokens (tasks 2, 6, 7) and up to 4,250 with a 2,048-token budget (task 4). Task 4's backup
path generated 254,845 tokens in about 3,560 seconds: roughly **70 tokens per second** for
the whole server under that load. An accepted program generated 7,758 tokens *once* and then
answered all 60 inputs for free.

---

## 6. The decision: fix the code, or add algorithms?

The tempting answer to "make it better" is "add more techniques". The disciplined answer
starts from where the points are, and asks what *mechanism* is missing at each place.

| task | E001 | what the errors are | missing mechanism? | action | expected gain |
|---|--:|---|---|---|--:|
| 1, 3 | 100% | none | no | none | 0 |
| 4 | 80% | a contract misread; a repair loop that repeats | no, a defect | **fix** the prompt and feedback | up to +20 points |
| 2 | 45% | thinking never reached | yes: nothing measures whether a rung is worth its cost | **add** the escalation probe | +40 points or more |
| 5 | 85% | mixed | no | let the probe decide escalation | 0 to +3 |
| 6 | 93.3% | escalation flipped right answers | same mechanism as task 2 | let the probe decide | +1 to +2 |
| 7 | 20% | missing knowledge (40 of 48) | not one Cairn can supply | **no** new algorithm; reject talking replies | about +1, and saves time |
| 8 | 0.000 | not measured at all | measurement, then abstention | **fix** the measurement; add metric-aware abstention | unknown until measured |

So the answer is: **three fixes and one new algorithm**, because only one row shows a missing
mechanism. Building more around task 7 (bigger tails, more votes, answer rewriting) would be
activity without a reason: the error analysis says the answers are not in the model.

### 6A. The new algorithm: the escalation probe

**The problem it solves.** In two runs I set "should this task think?" by hand, from 5 to 20
inputs, and was wrong in both directions: task 2 needed thinking and did not get it; tasks 5
and 6 got thinking that hurt. A decision that has to be made per task, and that can be
measured, should be measured.

**What it does** (`TaskRunner._prepare_policy`). Before labelling, take 40 held-out examples
from the second calibration half (never in a prompt, not used to fit the calibrator). Answer
each one twice: by the cheap rung alone, and by thinking + voting alone. Split them by
whether the cheap rung was *sure* (it would have stopped the ladder). Then:

```
thinking wins a paired test on the inputs the cheap rung was SURE about   -> reason_first
thinking wins a paired test on the inputs the cheap rung was UNSURE about -> ladder
otherwise                                                                  -> cheap_only
```

**Why each design choice:**

* **Paired, on the same inputs.** Comparing two accuracies measured on different inputs
  mixes the method difference with the input difference. On the same inputs, only the
  disagreements carry information, and an exact McNemar test says whether their split is more
  lopsided than chance.
* **Split by sure / unsure.** These are two different questions. Task 2 needs thinking even
  where the cheap rung is sure (it is confidently wrong). A healthy ladder needs thinking only
  where the cheap rung is unsure.
* **Ties go to the cheaper policy.** Spending more compute requires evidence (p ≤ 0.10);
  saving compute does not. With 40 examples, a small real benefit can be missed, and the cost
  of that miss is small; a large benefit, like task 2's, is detected easily.
* **It reuses what is already cached.** The cheap answers for classification tasks are the
  same one-token calls the calibrator already made, so they come from the disk cache.
* **Every number is recorded.** The report holds the probe's accuracies, the disagreement
  counts and p-values, and the run prints them, so the technical report can show *why* each
  task uses the policy it uses.

**A flaw caught by writing the third test.** The first version answered the probe's cheap
pass with a single direct answer. A single answer always "agrees with itself", so every
probe input looked sure, the unsure group was always empty, and the probe could never choose
the ladder. Two tests passed anyway, because they only exercised `reason_first` and
`cheap_only`. The third test, built for the case where thinking helps *only* on unsure
inputs, failed. The fix is a `cheap_rung_only` mode: the full cheap rung, quick samples
included, and never thinking. The probe measures with it, and the `cheap_only` policy answers
with it, so production behaves exactly as the probe measured. **Lesson:** write one test per
branch of a decision rule; the branch you did not test is the one that is broken.

**What it costs.** Forty thinking calls per task, about 130,000 completion tokens, roughly 20
to 30 minutes on two T4s. It replaces a hand-set guess with a measurement every run.

### 6B. Metric-aware abstention for task 8

The task-8 score is `S_exec² / S_call`. For a fixed number of correct kernels, every kernel
that runs but computes the wrong result *lowers* the score:

| out of 166 | kernels that run | of which correct | score |
|---|--:|--:|--:|
| hand in everything | 30 | 10 | 0.020 |
| replace the 20 proven wrong by a stub that fails loudly | 10 | 10 | 0.060 |

So when execution **proves** the best candidate wrong after every repair round (it disagrees
with PyTorch's own function, or returns NaN or infinity), Cairn hands in a stub that raises,
and keeps the rejected code in the journal for analysis. Kernels that run but cannot be
checked are kept: they might be right. This follows the metric's stated intent (code that
runs should be right) and is reported openly, not hidden.

### 6C. Algorithms considered and rejected, with reasons

| idea | why not now |
|---|---|
| Retrieval from an external knowledge source for trivia | forbidden: only the official data may be used |
| More votes or a longer tail for trivia | both rungs are equally wrong; the answers are not in the model |
| An answer-formatting rule for trivia (articles) | near misses go both ways (article added and dropped); no single rule fixes them |
| Differential testing for task 8 (the model writes a PyTorch reference, the Triton kernel is compared with it) | the strongest next idea for the 104 functions without a PyTorch twin, but it cannot be tested without Triton; first measure what runs at all |
| Guideline induction for task 2 | worth trying only if thinking alone stays below 90%; the probe result decides |
| Fewer votes for task 2 (3 instead of 5) | a cost lever; needs its own paired comparison (E003) |

---

## 7. Everything that changed, in one table

| change | where | evidence | test |
|---|---|---|---|
| escalation probe with paired test; `reason_first` and `cheap_rung_only` paths | `pipeline/runner.py`, `pipeline/annotator.py` | E000 + E001, tasks 2, 5, 6, 7 | `test_probe_picks_reason_first_...`, `..._cheap_only_...`, `..._the_ladder_...` |
| `use_scoring: false` alone documented as NOT reaching thinking | test only | E001 task 2 | `test_reason_first_routes_a_classification_task_straight_to_thinking` |
| induction examples as `repr()` asserts; contract stated in words | `pipeline/runner.py`, `induction/program.py` | E001 task 4 cached replies | `test_program_induction_sees_a_short_example_block` |
| feedback names the mistake; return type reported; repeats detected | `induction/program.py`, `induction/sandbox.py` | E001 task 4 | `test_the_e001_type_confusion_is_named_...`, `test_a_non_string_return_...` |
| task 8 measured on test inputs | `cli.py`, `scripts/kaggle_run.sh` | 0/184 vs 166/166 signatures | existing code-quality tests |
| abstain on proven-wrong kernels | `pipeline/annotator.py` | the metric's algebra | `test_a_kernel_proven_wrong_is_replaced_by_a_loud_stub` |
| reject "talking" trivia replies | `decoding/parse.py` | 2 of 48 task-7 errors | `test_norm_trivia_rejects_the_model_talking_...` |
| workers × votes ≤ 16; timeout 900 s | `configs/default.yaml` | 7 read timeouts on task 4 | config load test |
| hand-set `escalate: false` for task 5 removed | `configs/default.yaml` | replaced by the probe | — |

---

## 8. Predictions for the next run (E002) and a time budget

### Predictions (60 dev inputs; task 8 on 24 test inputs)

| task | predicted | basis | if it misses low, look first at |
|---|---|---|---|
| 1, 3 | 100% | accepted programs, unchanged path | the `program:` lines |
| 4 | 97–100% | the contract is now unambiguous | the attempt log: is the code still `''.join(text)`? |
| 2 | 82–95% | probe picks `reason_first`; thinking was 14 of 15 so far | the `policy:` line; if `cheap_only`, the probe failed to detect the gap |
| 5 | 80–90% | probe likely picks `cheap_only`; scoring was 85–87% | the probe's unsure row |
| 6 | 90–97% | probe likely picks `cheap_only`; escalation removed 1 correct answer of 4 | the probe's unsure row |
| 7 | 17–30% | knowledge-bound; probe likely `cheap_only` | the vote counts in the journal |
| 8 | a first real measurement: callable 10–50%, consistency 0.00–0.20 | executions now happen | the outcome histogram (see below) |

Predicted mean over tasks 1–7: **about 85–89%**, against 74.8% in E001.

How to read task 8's histogram: mostly `unconfirmed_error` → the argument guesses are wrong
for these signatures; mostly `timeout` → compile time, raise the limit or lower workers;
mostly `wrong_result` → the kernels are wrong (and are now abstained); mostly
`called_no_reference` → they run but cannot be checked, which is where differential testing
would help next.

### Time budget for the full test run (500 inputs per task, 166 for task 8)

| task | dominant cost | estimate |
|---|---|---|
| 1, 3, 4 | induction once; programs answer every input | 5–15 min each |
| 5, 6 | calibration (250–300 one-token calls) + 500 scoring calls + probe | 25–40 min each |
| 7 | 500 × (1 + 3 cheap samples) + probe | 30–50 min |
| 2 | probe + 500 thinking calls × ~3,400 tokens at 70–150 tokens/s | **3–7 hours** |
| 8 | 166 × up to 6 generations of 3K tokens + execution | **2–3 hours** |

That is more than one 12-hour Kaggle session can safely hold. **Run the test split in two
sessions**: session A with tasks `1 3 4 5 6 7`, session B with tasks `2 8`, both into the
same run folder (the journal resumes; save the notebook version so `/kaggle/working`
survives). If task 2's time is a problem, E003 tests `vote_n: 3` against 5 with a paired
comparison before cutting it.

---

## 9. Rules to keep (added to the list in `06_FIRST_REAL_RUN.md` §5)

13. **A setting is not a behaviour.** Trace it to the code it changes, and write the test
    that proves the behaviour before spending GPU time on it.
14. **Agreement is confidence, not correctness.** A gate is trustworthy only after it has
    been checked against held-out answers.
15. **When a capable model repeats a simple mistake, rewrite the question.** Make the
    contract executable; make every error message name the kind of error.
16. **A loop that adds no new information converges on its first answer.** Detect repeats and
    say so.
17. **Check that the validation split looks like the test set, in the way the metric cares
    about.** If the metric needs no labels, measure on the real test inputs.
18. **Measure decisions that can be measured.** Where a choice can be tested on held-out data
    before labelling, test it every run instead of setting it by hand.
19. **Add an algorithm only where the error analysis shows a missing mechanism.** Missing
    knowledge is not a missing mechanism.
20. **Size client concurrency to the server.** Workers × parallel samples must fit the
    server's sequence limit, or requests queue into timeouts.
21. **Write one test per branch of a decision rule.** The untested branch is the broken one.
22. **Score your predictions.** Five of eight is a measure of how well the system is
    understood; the three misses are the most valuable lines in this file.
