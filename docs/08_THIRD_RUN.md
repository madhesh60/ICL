# 08 — The third real run: the probe corrects me, and task 2 gets a counting method

The next run took place on November 18. Its analysis compares the forecasts below with the measured outcomes. [Fourth run](09_FOURTH_RUN.md)

> Companion to `07_SECOND_RUN.md`, whose §8 predicted this run. Raw log:
> `docs/results/kaggle_run_2026-11-17b_raw.md`. Per-task reports: `docs/results/e002_reports/`.
> Kaggle, 2 x Tesla T4, commit `f13d95a`, 60 dev inputs per task, 2026-11-17.

---

## 0. Verdict in six lines

1. **Tasks 1, 3 and 4 are solved:** verified programs, 60/60 each, two to four minutes per task.
   Task 6 rose to **96.7%**. The mean over tasks 1–7 is **77.4%** (E001: 74.8%).
2. **The escalation probe worked, and it contradicted two of my beliefs.** On task 2 thinking
   was only 60% on held-out inputs, not the 93% I had claimed. On task 7 thinking was **38% vs
   20%** for the cheap rung, which I had called "knowledge-bound, thinking adds nothing".
3. **Task 2's thinking was starved, not stupid.** 124 of 200 thinking samples hit the 768-token
   limit before giving an answer.
4. **Task 2's labels follow a convention the task never states**, measured from the data:
   each word of a compound noun counts, helper verbs ("is", "are") do not.
5. **Task 8 was run on dev inputs again** (the notebook cell was the older script), so nothing
   could execute. The CLI now redirects to test inputs by itself, whatever script calls it.
6. **What changed:** counting rules in the prompt, a listing rung where code does the counting,
   a probe that tests the overall difference first. Predictions for the next run in §8.

---

## 1. Scoring the predictions from `07_SECOND_RUN.md` §8

| task | predicted | measured (95% interval) | landed? |
|---|---|---|---|
| 1, 3 | 100% | **100%**, **100%** | yes |
| 4 | 97–100% | **100%** (94–100), program accepted in round 0 | yes |
| 2 | 82–95% | **43.3%** (32–56) | **no** |
| 5 | 80–90% | **85.0%** (74–92) | yes |
| 6 | 90–97% | **96.7%** (89–99) | yes |
| 7 | 17–30% | 16.7% (9–28) | yes (interval overlaps) |
| 8 | callable 10–50% | not measured (dev inputs again) | **no** |

Probe decisions predicted: `reason_first` for task 2 (**wrong**: it chose `cheap_only`), and
`cheap_only` for 5, 6 and 7 (right as predicted, but for task 7 the prediction itself rested
on a mistaken belief; see §2B).

Six of eight landed. The task 4 fix (assert-style examples) worked on its first attempt: the
model wrote `''.join(ast.literal_eval(text))` in round 0.

---

## 2. What the probe taught me

The probe answers 40 held-out inputs twice, cheap and with thinking, on the **same** inputs.
That is exactly what earlier runs lacked, and it showed two of my beliefs were wrong.

### 2A. Task 2: "thinking is right 14 of 15 times" was a small, lucky sample

| | cheap rung | thinking |
|---|--:|--:|
| probe, 40 held-out inputs | 50% | **60%** |
| finished before the token limit | — | **76 of 200 samples** |

**Why my earlier number was wrong.** The 14-of-15 came from the few inputs that reached the
thinking rung in E000 and E001, and from thinking replies that happened to finish. Most do not.
Reading the cached replies shows why: the model debates whether "par" is a typo for "pair",
re-checks earlier examples, and runs out of its 768 tokens mid-sentence. **A truncated reply
is silent:** it simply contributes no vote. Nothing in the accuracy line said "62% of your
votes were empty". The finish reasons did.

### 2B. Task 7: "both rungs are near 20%" compared different inputs

In E000 and E001 thinking only saw the inputs the cheap rung was unsure about, which are the
hard ones. Comparing "cheap on easy-and-hard inputs" with "thinking on hard inputs" and
finding them equal says nothing about thinking. That is **selection bias**, and I drew a
conclusion from it ("knowledge-bound, no new algorithm"). The probe, on the same inputs:

| | n | only cheap right | only thinking right | p |
|---|--:|--:|--:|--:|
| all probe inputs | 40 | 2 | **9** | 0.065 |
| cheap rung sure | 29 | 2 | 7 | 0.18 |
| cheap rung unsure | 11 | 0 | 2 | 0.50 |

Thinking nearly doubles trivia accuracy (20% → 38%). Task 7 is still the hardest task, but
"knowledge-bound" was too strong: some answers *are* in the model, and reasoning retrieves them.

### 2C. My own probe rule threw that signal away

The first rule tested only the sure group and the unsure group, separately. Splitting 40 inputs
into 29 and 11 cut the evidence in pieces: 9-to-2 overall (p = 0.065, below the 0.10 bar)
became 7-to-2 and 2-to-0, neither significant, so the rule chose `cheap_only`.

**The fix:** the overall paired test decides *whether* thinking pays; the groups decide only
*where* to spend it (`reason_first` if it also helps where the cheap rung is sure, otherwise
`ladder`). A test now builds the exact situation: each group 5-to-0 (p = 0.0625 each, above a
0.05 bar), overall 10-to-0 (p = 0.002), and checks the probe does not choose `cheap_only`.

**Lesson:** every split of a sample costs statistical power. Test the question you actually
need answered on all the data you have, and split only for secondary questions.

---

## 3. Task 2: finding the counting convention in the data

The task definition says only "count the number of nouns/verbs". The labels were evidently
produced by a consistent procedure (likely a part-of-speech tagger), and the model, reasoning
like a person, disagrees with it in systematic ways. The convention has to be found.

### 3A. Step one: let the data say which words count

If a word is counted as a verb, a sentence containing it should, on average, have a label one
higher. So fit `label ≈ Σ weight(word) × count(word) + constant` by least squares over all
2,755 verb questions (and separately the 2,685 noun questions), for the 170 most common words.

| word | weight in verb labels | reading |
|---|--:|---|
| is | **+0.02** | never counted |
| are | **−0.02** | never counted |
| has | +0.57 | counted when it is the main verb |
| sitting | +0.53 | usually counted |
| covered | +0.87 | counted |

| word | weight in noun labels | reading |
|---|--:|---|
| living | **+0.81** | counted as a noun in "living room" |
| it | −0.44 | pronouns are not nouns |
| front, top | +0.30, +0.29 | counted in "in front of", "on top of" |

The fit is rough (words are correlated, so single weights wobble), but frequent words with
weights near 0 or near 1 are reliable. This took one minute and no model.

### 3B. Step two: read labelled sentences that test each hypothesis

| sentence | label | count under the convention |
|---|--:|---|
| A tennis player holding a racket on the tennis court | 5 nouns | tennis, player, racket, tennis, court |
| a living room with some brick walls and a fireplace | 5 nouns | living, room, brick, walls, fireplace |
| A bench sitting in front of a wall and next to a door | 4 nouns | bench, front, wall, door (not "next") |
| A diner with large pepsi signs on the front of it | 4 nouns | diner, pepsi, signs, front (not "it") |
| A man is using a cell phone to photograph a barn | 2 verbs | using, photograph (not "is") |
| A elephant that is standing on a floor at a bowling alley | 1 verb | standing (not "is", not "bowling") |

A person would say "tennis court" is one thing and "is using" contains a verb. The labels
disagree, consistently. These sentences are now a test (`test_the_counting_rules_agree_with_
the_labels`): if a rule is ever edited into something the data contradicts, the test fails.

### 3C. Step three: state the rules where every rung reads them

The rules (`cairn/tasks/counting.py`, `COUNTING_RULES`) go into the system prompt through the
task card's new `guidelines` field, so the listing rung and the thinking rung both see them.
This is instruction design grounded in measurement, not guesswork.

### 3D. Step four: let the model name the words and code count them

A count is the length of a list. The new **listing rung** (`Annotator._decompose`) asks, with
thinking off and the format enforced by the server:

```
Nouns: tennis, player, racket, tennis, court
```

Code computes `len()` (counting words, so "tennis court" written as one item still counts 2).
Five sampled lists vote on the count. Why this shape:

* **It plays to each side's strength.** Naming the nouns in a sentence is a reading task a 4B
  model does well; holding a running count while deliberating is where it slips.
* **It cannot ramble.** Twenty-odd tokens per sample, no thinking, so no truncation.
* **It is readable by construction.** The server forces the `Nouns: a, b, c` shape. If a server
  rejects the format constraint, the same request is retried once without it; a broken
  constraint costs a retry, never the input.
* **It is cheap.** About 5 × 25 tokens per input, against 5 × 768 for thinking.

The probe then compares this rung with thinking (now given 1,536 tokens, so it can finish)
on 40 held-out inputs, and picks the better one. That comparison is the real test; §8 gives
the prediction.

**Lesson:** when a model must produce a number derived from items, ask for the items and
compute the number. *Decompose, then let code do the arithmetic.*

---

## 4. Task 8: fix it in the tool, not in the script

`scripts/kaggle_run.sh` in the repository already ran task 8 on test inputs. The notebook cell
used for this run was an older copy, so task 8 ran on dev inputs a third time and executed
nothing (`not_run`: 24 of 24).

A fix that depends on everyone using the right script is fragile. Now `cairn run --task 8
--split dev` itself notices that no dev input carries a wrapper signature, prints

```
task 8: dev inputs have no wrapper signature to execute; measuring on the first 60 TEST inputs instead
```

and measures on test inputs. The report records `measured_on: test`. Whatever cell runs next,
task 8 will finally be measured.

**Still unknown:** whether generated kernels run on the Kaggle GPU at all. Three runs, zero
executions. The next run gives the first real number; §8 says how to read it.

---

## 5. Cost of this run

| task | wall time | what the time went to |
|---|--:|---|
| 1, 3, 4 | 253 s, 252 s, 134 s | one induction call each, then nothing |
| 2 | 1,925 s | probe: 40 thinking calls, 130,276 tokens |
| 5 | 1,362 s | probe thinking 95,817 tokens; scoring is cheap |
| 6 | 2,146 s | probe thinking 148,875 tokens |
| 7 | 2,054 s | probe thinking 137,047 tokens |
| 8 | 1,289 s | 24 code generations on dev inputs |

The probe costs 20–35 minutes per task. It paid for itself twice this run: it kept tasks 5 and
6 from thinking where thinking hurts (task 6: cheap 95% vs thinking 78%, 8-to-1, p = 0.039),
and it exposed the task 7 mistake.

Prompt cache reuse stayed at 94–100% for every text task.

---

## 6. Everything that changed

| change | where | evidence | test |
|---|---|---|---|
| counting rules in the system prompt (task card `guidelines`) | `tasks/counting.py`, `tasks/base.py`, `context/render.py` | word regression + labelled sentences | `test_the_counting_rules_agree_with_the_labels` |
| listing rung: model names, code counts, samples vote | `pipeline/annotator.py` (`_decompose`) | 124/200 thinking samples truncated | `test_the_listing_rung_counts_in_code` |
| retry without the format constraint if the server rejects it | same | a rejection would otherwise fall to the model-free answer | `test_a_rejected_format_constraint_costs_a_retry_...` |
| unreadable lists hand over to the normal ladder | same | defensive | `test_unreadable_lists_hand_over_...` |
| probe: overall paired test first, groups only choose where | `pipeline/runner.py` | task 7: 9-to-2 overall, split lost it | `test_probe_acts_on_an_overall_difference_...` |
| task 2 thinking budget 768 → 1,536 | `configs/default.yaml` | truncation | config load |
| task 8 measured on test inputs whenever dev cannot execute | `cli.py` | three runs of `not_run` | `test_code_task_on_dev_is_measured_on_test_inputs` |

---

## 7. Rules to keep (continuing `07_SECOND_RUN.md` §9)

23. **Compare methods on the same inputs.** A method that only sees the hard cases will look
    worse than it is (task 7: selection bias).
24. **Check finish reasons.** A truncated reply is silent; it just casts no vote. Count how many
    replies ended on the token limit before trusting an accuracy.
25. **Find the label convention in the data.** Regress labels on input features, then read
    labelled examples that test each hypothesis, then pin the findings in a test.
26. **Decompose: let the model name things and let code do arithmetic.**
27. **Every split costs power.** Answer the main question on all the data; split only for the
    secondary one.
28. **Fix it in the tool, not in the instructions for using the tool.** If a mistake can be
    made by calling the tool the wrong way, make the tool refuse or correct it.
29. **A small lucky sample is still small.** 14 of 15 on inputs that reached a rung by a filter
    is not the rung's accuracy.

---

## 8. Predictions for the next run (E003)

| task | predicted | basis | if it misses low, look first at |
|---|---|---|---|
| 1, 3, 4 | 100% | programs | `program:` lines |
| 2 | **65–85%** | listing + rules vs thinking at 1,536 tokens; the probe keeps the better | the `policy:` lines; the `lists` in the journal: which words were listed wrongly |
| 5 | 80–90% | unchanged | — |
| 6 | 93–99% | unchanged | — |
| 7 | **25–42%** | probe now acts on 9-to-2-like results → `reason_first` or `ladder` | the policy line; truncation of thinking (30% at 1,024 tokens) |
| 8 | callable 10–50%, consistency 0.00–0.20 | first real measurement on test inputs | the outcome histogram (`07_SECOND_RUN.md` §8 explains each status) |

Predicted mean over tasks 1–7: **about 80–87%**.

The task 2 range is wide on purpose: the listing rung has never run on the real model. If the
lists are mostly right but the count still misses, compare the listed words with the rules; if
the probe picks thinking instead, the listing rung lost a fair comparison, which is also an
answer.

**Time.** If the probe picks thinking for task 7, its full test run costs about 500 × 3,000
tokens, roughly 3–6 hours on two T4s. Task 2's listing rung is cheap (minutes) unless the probe
prefers thinking. Plan the test run in two sessions, as in `07_SECOND_RUN.md` §8.
