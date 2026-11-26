# Learning Guide: how Cairn works, and how to think when you build things like it

## November 26 · Choosing the next experiment from a checkpoint

The saved run finished development tasks 1, 3, 4, 5 and 6. On those same 300 questions E003 had 289 correct and the new run had 284. Task 5 accounted for the whole difference. Task 7 had nine saved answers from the 60-question batch. The scheduler groups questions by prompt prefix, so those nine were scattered across the batch. Counting the first nine rows of an older journal would compare different clues. Join on the official question ID.

| Task | Evidence so far | Algorithm and next decision |
|---|---|---|
| 1, 3, 4 · exact transformations | Verified programs scored 60/60 on each development batch in both runs. | Keep these policies fixed during the targeted comparison. Reuse validated development results when resuming finals. The complete test files are still required. |
| 2 · noun and verb counts | E003 scored 41/60. None of its 19 misses had the correct count among the recorded reasoning votes. | Retrieve demonstrations with similar grammatical structure. Ask Qwen to mark individual word positions and sum the valid marks in code. Inspect wrong counts, missing votes and fallback use alongside paired accuracy. |
| 5 · tweet sadness | A 40-question calibration probe favored reasoning, 36 correct to 30. On development, switching all 60 tweets to reasoning gained one answer and lost six. | Use the calibrated label scorer directly. Replay scored 51/60 using November calibration replies and E003 label-score replies, compared with 46/60 from November reasoning. The same development batch guided this decision. A fresh run and an independent check are still needed. |
| 6 · same genre | 58/60 in both runs, with the same two misses. | Keep the current policy. A change here would need evidence on additional examples. |
| 7 · trivia | E003 scored 10/60. Only three of its 50 misses contained the gold answer elsewhere in the votes. November scored 4/9 on the nine finished clues, up from 3/9 on those IDs. | The reasoning path generates search terms for the answer type and relationship, retrieves official demonstrations, then answers the original clue. Check all 60 clues. The nine-question overlap is an additional paired check. |
| 8 · Triton kernels | The old executor failed while importing 22 candidates. The saved CPU replay later found four numerical matches. | Write generated kernels to importable Python files. Replay the original and regenerated predictions on the same GPU using the same 24 signatures and reference trials. Report callable count, PyTorch matches and local consistency together. |

Two mistakes shaped the run design. The original finals launcher put a full 500-question test batch between development batches and exceeded the Kaggle session limit. Separately, comparing new GPU results directly to the old broken executor would credit an evaluator repair to the model. The targeted run finishes the two 60-question text comparisons and the paired 24-kernel replay separately from finals.

Read `comparison.json` by task. For text, inspect IDs that moved from wrong to right and from right to wrong. For kernels, inspect which IDs gained numerical matches and why others failed. Multiple implementation changes entered this comparison together. To attribute a gain to one algorithm, hold the other settings fixed and rerun that component on the same IDs. Keep the reserved holdout for evaluating the selected text policies. The eight-task finals archive requires all 3,666 official test predictions. [Run history](09_FOURTH_RUN.md) · [Checkpoint results](KAGGLE_FINALS.md)

## An interview walkthrough from the November 24 investigation

1. **Describe the constraint.** Qwen3-4B must label eight tasks from official demonstrations. The prompt budget, held-out examples and finite inference time each affect the design.
2. **Quantify the problem.** Task 2 improved by 15 answers on 60 dev inputs. On trivia, guessing a better tie-breaker could recover at most three of fifty misses. These two tasks call for different experiments.
3. **Walk through a failure end to end.** The task-8 report printed zero static passes. The journal showed 22 JIT import errors. Static replay of the submitted bytes passed 22 kernels. File-backed import allowed six to run, and reference checks validated four. Explain what evidence each stage contributes.
4. **State the remaining uncertainty.** A 24-kernel CPU replay tests the harness and a subset of generated code. The competition uses its own GPU inputs. For the new counting and trivia methods, compare fixed IDs, cost and paired outcomes before reporting a gain.

[Commands, evidence and original predictions](09_FOURTH_RUN.md)

This guide is for you, not for a reader of the code. The goal is that after reading it you
can rebuild the ideas yourself, explain them in an interview, and know what to do when
something breaks — without needing an AI to tell you.

How each idea is explained:

1. **One line** — the whole idea in a sentence.
2. **Picture** — the same idea in everyday life.
3. **See it** — a tiny real example from this project (real numbers, not made up).
4. **Step by step** — what happens inside, in order.
5. **Remember** — one line to keep.

How to read it: keep the code open next to it. Every section names the file. Run the
commands. Do the exercises at the end. Then write your own notes in
`docs/03_RESEARCH_NOTEBOOK.md` — writing it in your own words is what makes it stick.

---

## Contents

**Part A — the ideas**
1. In-context learning
2. Tokens: the ruler for everything
3. Why a long prompt can hurt a small model
4. Splitting the examples so you cannot fool yourself
5. Finding similar examples (BM25)
6. Choosing which examples fill the prompt (diminishing returns)
7. Ordering the prompt, and the server's memory (KV cache)
8. The data detective story: how the task-6 signal was found
9. Reading the answer: one-token scoring
10. Calibration: correcting a model that leans
11. Conformal sets: knowing when you don't know
12. Self-consistency: asking five times
13. Program induction: let the model write the rule, then test it
14. Checking generated GPU code without a GPU
15. Session memory without breaking the cache

**Part B — how to think like an engineer here**
16. The order I worked in, and why
17. Error handling: the way of thinking
18. Testing: what to test and how
19. Measuring honestly: error bars and fair comparisons
20. Design habits that kept the code clean
21. Bugs found while building this, and what each one teaches

**Part C — practice**
22. Exercises
23. Reading order for the code
24. Interview questions with short answers
25. Glossary

**Part D — the real runs**
26. Reading a real result (pointer to `06_FIRST_REAL_RUN.md`)
27. Scoring your own predictions (pointer to `07_SECOND_RUN.md`)
28. The escalation probe: measuring whether thinking pays
29. Finding a label convention in the data
30. Decomposition: the model names, code computes

---

# Part A — the ideas

## 1. In-context learning

**One line:** you teach a model a new job by showing it solved examples inside the
question, without changing the model at all.

**Picture:** a new card game. Nobody explains the rules; you watch five rounds, then you
play. Your brain did not change shape, you just used what you saw.

**See it:** this is (shortened) what Cairn sends for task 3:

```
Input: [72, 29, 49]
Output: [36, 88, 148]

Input: [27, 99, 188, 149]
Output: [82, 298, 94, 448]

Input: [91, 63, 148, 8, 6]
Output:
```

The correct continuation is `[274, 190, 74, 4, 3]`, and the model's job is to find it from the pattern alone.

**Step by step:**
1. The model reads all the text from left to right.
2. While reading, it notices the pattern (odd → ×3+1, even → ÷2).
3. When it reaches the last `Output:`, the most likely next words are the ones that fit
   the pattern.
4. Nothing is saved afterwards. The next question starts from zero again.

**Why this project is hard:** the rules force 30,000 tokens of examples into every
prompt, and the model is small (4 billion numbers inside it, which is small for this kind
of model). Part of this guide is about making those 30,000 tokens help instead of hurt.

**Remember:** in-context learning = examples in the prompt, no training.

---

## 2. Tokens: the ruler for everything

**One line:** a model does not read letters or words, it reads *tokens* — small pieces of
text — and every limit is counted in tokens.

**Picture:** like Lego bricks. A sentence is built from bricks; common words are one big
brick, rare words need several small ones.

**See it** (real output of Qwen's tokenizer, `cairn/tokens.py`):

| text | tokens | pieces |
|---|---|---|
| `Sad` | 1 | `Sad` |
| `Not sad` | 2 | `Not`, ` sad` |
| `tiramisu` | 3 | `t`, `iram`, `isu` |
| `12345` | 5 | `1`,`2`,`3`,`4`,`5` — **every digit is its own token** |
| `😭` | 1 | the emoji |
| one task-1 example | 24 | `Input`, `:`, ` [`, `5`, `9`, `,` … |

**Step by step (how Cairn uses this):**
1. The rule says "at least 30,000 tokens of examples". So we need a ruler that counts
   exactly like Qwen counts.
2. The real ruler is Qwen's vocabulary file. It ships in this repository
   (`OpenSeek/hf_openseek/tokenizer/qwen.tiktoken`), so no download is needed.
3. Each example is written so it ends with a blank line. Qwen never glues tokens across a
   line break, so the count of 900 examples together = the sum of their separate counts.
   A test checks this on real data (`test_token_counts_add_up_across_examples`).
4. If only a rough ruler is available, Cairn adds a safety margin (15%) so the prompt
   can be a bit too long, but never too short.

**Two consequences you should notice:**
* Numbers are expensive. Task 1 examples are short to read but cost 24 tokens each.
* `Sad` and `Not sad` start with different tokens (`Sad` vs `Not`). That is what makes
  one-token scoring possible later (section 9).

**Remember:** measure in the model's units, and when your ruler is rough, err on the safe side.

---

## 3. Why a long prompt can hurt a small model

**One line:** small models remember the start and the end of a long prompt, blur the
middle, lean toward the most common answer, and copy whatever they read last.

**Picture:** reading a 60-page manual in one go, then answering one question. You
remember the first page, the last page, and the general feel ("most answers were N").

**See it:** in the file order of task 6, there is a run of **18 "N" answers in a row**
(measured, `cairn bench`). A small model that has just read 18 "N"s is pushed toward "N".

**Step by step (what each effect means for design):**
1. *Lost in the middle* → put what matters most at the very end, next to the question.
2. *Majority drift* → don't let one label dominate; spread labels evenly; correct the lean
   afterwards (calibration).
3. *Format copying* → the last examples set the answer format, so make them real
   question/answer turns in exactly the format you want.
4. *Middle is weakest* → the bulk of the 30K is "carried safely", the tail does the work.

**Remember:** design mantra — make the long part harmless, the end decisive, and the
answer impossible to lose.

---

## 4. Splitting the examples so you cannot fool yourself

**One line:** split the solved examples into groups that never mix — some to teach, some
to check — before doing anything else.

**Picture:** a teacher who uses some solved homework on the board and keeps the rest in a
sealed envelope for a quiz. If quiz questions were also on the board, a perfect quiz score
would mean nothing.

**See it** (`cairn inspect`, task 6):

| role | count | used for |
|---|---|---|
| pool | 4,700 | the only examples allowed inside a prompt |
| calib | 300 | fitting corrections, testing programs |
| dev | 300 | comparing designs |
| holdout | 200 | one last honest number at the end |

**Step by step (`cairn/data.py`):**
1. Each example id is turned into a fingerprint with a hash function (SHA-256 of
   `"42:<id>"`). A hash scrambles text into a number that looks random but is always the
   same for the same text.
2. Examples are sorted by that number and sliced into the four groups.
3. Because it depends only on the id, the split is identical on every computer, and adding
   new examples never moves old ones between groups.
4. A test asserts no example is in two groups. Another asserts no dev example ever appears
   in a prompt.

**Why not just shuffle with `random`?** Random shuffles change with the seed, the Python
version and the file order. Hashing the id does not.

**Remember:** decide how you will check yourself *before* you build anything.

---

## 5. Finding similar examples (BM25)

**One line:** BM25 scores how well an example matches a question by counting shared words,
giving rare words much more weight than common ones.

**Picture:** searching your photos for "tiramisu". Photos tagged "the" are useless;
photos tagged "tiramisu" are gold. Rare words tell things apart.

**See it** (test `test_bm25_prefers_rare_matching_words`): documents "the cat sat",
"the cat ran", "the tiramisu dessert". Question: "a tiramisu for the cat". The winner is
the tiramisu document, because "tiramisu" appears in 1 document, "cat" in 2, "the" in all.

**Step by step (`cairn/retrieval/bm25.py`):**
1. *Rarity* (called IDF): a word in few documents gets a big weight, a word in all of them
   gets almost zero.
2. *Repeats*: a word appearing 3 times counts more than once, but not 3 times more.
3. *Length*: long documents don't win just by having more words.
4. Score = sum over the question's words of rarity × (repeat part).
5. All of this is precomputed once, so one search is a few array additions (~1 ms).

**Why no neural "embedding" model?** It would be a second model (against the rules), use
GPU memory, and give different results on different machines. BM25 is exact, fast,
explainable, and for short texts it is a strong baseline.

**Remember:** rare shared words are the strongest evidence of similarity.

---

## 6. Choosing which examples fill the prompt (diminishing returns)

**One line:** pick examples one at a time, always the one that adds the most *new*
information per token, until the token minimum is reached.

**Picture:** packing a suitcase for a specific trip. The first umbrella is very useful,
the second adds little, the third adds nothing. You pack for *this* trip, not for every
possible trip.

**See it** (tiny version of what `cairn/selection/submodular.py` does). The value of a set
is the sum over words of √(how many chosen examples contain that word). Square root =
"each extra copy helps less".

* A = {funeral, cry}, B = {funeral, rain}, C = {party}
* Pick A first. Value = √1 + √1 = 2.
* Gain of B: "funeral" goes from 1 to 2 copies → √2 − √1 = 0.41; "rain" is new → +1.
  Total **+1.41**.
* Gain of C: "party" is new → **+1.00**.
* B wins, but its "funeral" only counted 0.41 — the second copy is worth less.

**Step by step:**
1. Each example gets a cost (its tokens) and a set of features (words and word pairs).
2. Each feature gets a weight: rarity × how often *the inputs we must label* contain it.
   That is the "query-aware" part: we pack for the actual trip.
3. Greedy loop: take the example with the best gain ÷ cost, update counts, repeat until
   ≥ 30,000 tokens.
4. Label quotas stop one label from taking over. If quotas would leave us short, the token
   minimum wins (a rule beats a preference).
5. *Lazy trick:* gains can only shrink as we add examples, so an old gain is an upper bound.
   Keep examples in a priority queue, re-check only the top one. Most examples are never
   re-scored. A test proves the lazy version picks exactly what the slow version picks.

**Measured result** (`cairn bench`, share of each dev input's informative words that appear
in the prompt):

| task | file order | random | Cairn |
|---|---|---|---|
| 2 | 0.754 | 0.759 | **0.960** |
| 6 | 0.587 | 0.590 | **0.728** |
| 7 | 0.782 | 0.785 | **0.857** |

**Remember:** "diminishing returns + greedy" is a pattern you will reuse everywhere
(summaries, search results, test selection).

---

## 7. Ordering the prompt, and the server's memory (KV cache)

**One line:** put everything that repeats at the start and everything specific at the end,
because the server can reuse its work for a repeated start.

**Picture:** a group of people all asking questions about the same 60-page manual. One
person reads it and leaves notes; everyone else uses the notes and only reads their own
last page. If people bring the manual in a different order every time, nobody can use the
notes.

**See it:** the prompt Cairn builds (`cairn/context/builder.py`):

```
[system: task + output rule]            same for every question of the task
[30K tokens of chosen examples]          same for every question of the task
[session examples]                       grows only at its end
[12 nearest examples, as chat turns]     different for each question
[the question]
```

Measured with a simulator of the server's cache (200 prompts, task 7, 4 topic groups):

| send order | prompt tokens the GPU must read |
|---|---|
| as they come | 1.62 million |
| grouped by beginning (Cairn) | **0.20 million** |

Same prompts, same answers, 8× less work, only from the order of sending.

**Step by step:**
1. While reading a prompt, the model writes down internal notes for every token (the
   "KV cache": keys and values for attention).
2. vLLM stores these notes in blocks of 16 tokens, labelled by the text up to that block.
3. When a new prompt starts with exactly the same text, those blocks are reused instead of
   recomputed.
4. So: identical things first, variable things last, and send same-beginning prompts
   together.
5. *Primer:* the first prompt of a group is sent alone; the others wait until it is done, so
   they all find its notes (otherwise 16 workers compute the same 30K tokens at once).

**Remember:** in serving, the order of requests is a performance feature.

---

## 8. The data detective story: how the task-6 signal was found

**One line:** before building anything clever, ask where the data came from; the answer
can be worth more than any model.

**Picture:** a detective who checks the receipts before interviewing suspects.

**See it:** task 6 says "decide if sentence 2 is from the same genre as sentence 1". The
numbers that started the investigation:

| method | dev accuracy |
|---|---|
| always answer "N" | 64.3% |
| nearest examples vote | **58.0%** (worse than guessing "N"!) |

When "find similar examples" is *worse* than always guessing the common answer, word
similarity is the wrong idea of "similar". Something else decides the label.

**Step by step (how I reasoned):**
1. The examples mention genres like "telephone", "slate", "travel". Those are the genres of
   a known dataset: MNLI.
2. In MNLI, every sentence 2 was *written by a person who read sentence 1*.
3. So a "Y" pair (same genre) is most likely the original pair, and an "N" pair had its
   sentence 2 swapped in from somewhere else.
4. Hypothesis: "Y" pairs share many words between the two sentences; "N" pairs don't.
5. Test it with one number: the share of sentence 2's content words found in sentence 1.
   Average for "Y": **0.49**. For "N": **0.12**.
6. A simple threshold on that number: **93%** on dev. Combined with the nearest-examples
   vote in the stacked calibrator: **94.7%**. No model at all.

**What Cairn does with it:** the number is a *signal* on the task card
(`cairn/tasks/signals.py`). It is never used alone; the stacked calibrator (section 10)
learns on held-out examples how much to trust it next to the language model.

**Remember:** read the data's history. The cheapest win in this whole project came from
thinking about where the examples came from, not from the model.

---

## 9. Reading the answer: one-token scoring

**One line:** instead of letting the model write an answer and then searching the text for
it, read the model's probability for each allowed answer directly.

**Picture:** instead of asking someone to write an essay and then guessing their opinion
from it, you ask "how sure are you of A? of B?".

**See it:** the official baseline asked the model to wrap answers in `<label>` tags and
then searched the text. Measured from its own output file:

| task | answers lost (empty or impossible) |
|---|---|
| 4 | 59% |
| 5 | 24% |
| 8 | **100%** (its parser drops answers longer than 100 characters; every kernel is longer) |

With one-token scoring, for task 5 the server returns something like
`{"Sad": -0.4, "Not": -1.2, "The": -6.0}` (log-probabilities). "Sad" and "Not" are the
first tokens of the two labels; "The" fits neither and is ignored. There is always an answer.

**Step by step (`cairn/decoding/score.py`):**
1. Ask for exactly 1 token, with the top 20 candidates and their log-probabilities.
2. For each label, add up the probability of the tokens that are the start of that label.
3. A label that did not show up gets a small floor ("unlikely", never "impossible").
4. Normalise so they sum to 1. That is the answer *and* the confidence.

**When it does not work:** if two labels start with the same token (e.g. "no" and "not
sure"). `distinct_starts()` checks this; Cairn then falls back to constrained generation.

**Constrained generation** (`cairn/decoding/constraints.py`): for answers that must be
generated, the server is told the shape ("one of these words", or a pattern like
`[1, 2, 3]`), and it cannot produce anything else. Cairn checks again on its side anyway.

**Remember:** never let a correct answer get lost in parsing; ask for probabilities or
force the shape.

---

## 10. Calibration: correcting a model that leans

**One line:** measure how the model is biased on examples where you know the answer, then
subtract that bias from every new answer.

**Picture:** a friend who says "yes" 70% of the time no matter what. When they say "yes"
it means less; when they say "no" it means a lot. You learn to correct for them.

**See it:** in a test (`test_vector_scaling_removes_a_label_lean`), a fake model has a
built-in lean toward label 0. After fitting on 300 known examples, accuracy on 300 new ones
rises by more than 10 points.

**Step by step (`cairn/decoding/calibrate.py`):**
1. Score 150 *calib* examples with the model (we know their answers).
2. Find numbers *a* (one sharpness) and *b* (one shift per label) so that
   `a × model_score + b` matches the truth best. This is a tiny logistic regression.
3. The loss is bowl-shaped (mathematicians call it convex), so simple downhill steps reach
   the single best answer every time. No luck involved.
4. A small pull toward "change nothing" (L2 penalty) stops over-reacting.
5. Apply `a × score + b` to every new input.

**Stacked calibration** (the default): same idea, but the inputs are the model's scores
*plus* cheap signals (nearest-examples vote, task signals like the task-6 overlap). It
learns how much to trust each. If a signal is useless its weight goes to about zero
(tested: a strong signal gets more than 3× the weight of noise).

**Three calibrators and when to use them:**
* *Batch*: needs no labels at all; subtract each label's average score over many inputs.
* *Contextual*: score empty inputs like "N/A"; whatever the model says there is pure lean.
* *Vector / stacked*: needs labelled examples — we have hundreds left over, so this is the default.

**Remember:** a model's raw confidence is a guess; calibrate it against known answers.

---

## 11. Conformal sets: knowing when you don't know

**One line:** use held-out examples to pick a threshold so that "the set of labels above the
threshold" contains the right answer at least 90% of the time — guaranteed.

**Picture:** a doctor who, after a quick check, either says "it's a cold" (sure) or "it's a
cold or the flu, let's run a test" (unsure). The quick check is calibrated so the true
illness is in what they say at least 90% of the time.

**See it** (real computation, `cairn/decoding/conformal.py`). Ten held-out examples; the
model's probability for the *true* label was:

`0.95, 0.90, 0.88, 0.80, 0.75, 0.70, 0.60, 0.55, 0.40, 0.30`

With α = 0.2 (allow 20% misses):
1. "Surprise" score = 1 − probability: `0.05, 0.10, 0.12, 0.20, 0.25, 0.30, 0.40, 0.45, 0.60, 0.70`.
2. Position = ⌈(10 + 1) × 0.8⌉ = ⌈8.8⌉ = 9. The 9th smallest surprise is **0.60**.
3. Rule for new inputs: keep every label with surprise ≤ 0.60, i.e. probability ≥ 0.40.

| new input | probabilities (Sad, Not sad) | kept | action |
|---|---|---|---|
| A | 0.85, 0.15 | {Sad} | answer "Sad", done (1 cheap call) |
| B | 0.55, 0.45 | {Sad, Not sad} | unsure → think and vote between the two |
| C | 0.30, 0.70 | {Not sad} | answer "Not sad", done |

**Step by step (how Cairn uses it):**
1. Calibrate on half of *calib*, fit the conformal threshold on the *other* half (using the
   same examples for both would break the guarantee).
2. One label in the set → answer immediately.
3. Two or more → escalate: thinking mode + 5 votes, allowed answers limited to the set.

**Why this matters:** the expensive step costs ~50× more than the cheap one. A guarantee-
based gate spends it only where the model is really unsure. In an offline run on task 6,
95% of inputs stopped at the cheap step.

**Remember:** turn "the model seems sure" into "the right answer is in this set 90% of the
time", then spend compute only on sets bigger than one.

---

## 12. Self-consistency: asking five times

**One line:** sample several reasoned answers and take the most common one.

**Picture:** asking five friends the way to the station. If four say "left", go left. A
wrong answer can be wrong in many ways; a right answer is right in one way, so right
answers pile up.

**See it** (`test_majority_vote_normalizes_before_counting`): replies `"Paris"`,
`"paris."`, `" PARIS "`, `"Rome"` → cleaned to `paris` ×3, `rome` ×1 → answer `paris`,
agreement 75%. Without cleaning, "Paris" would have lost to nothing because each spelling
would count as a different answer.

**Step by step (`cairn/decoding/vote.py`):** clean each reply with the task's normaliser →
drop replies that are not valid answers → count → winner, and its share as confidence.

**Remember:** clean before you count, and use agreement as a confidence number.

---

## 13. Program induction: let the model write the rule, then test it

**One line:** for jobs with an exact rule, ask the model to write the rule as code, and
accept the code only if it reproduces every answer on examples it has never seen.

**Picture:** a student explains their method for long division. You don't trust the
explanation; you check it on the homework, and then on a surprise quiz.

**See it** (`test_program_is_repaired_from_feedback_then_accepted`):
* Round 1: the model writes `abs(xs[0] - xs[1])` (wrong: it only looks at the first two
  numbers). It gets 11% of the test cases right.
* Cairn sends back the failures: `Input: [98, -15, 35, -86, 27]  Expected: 8  Got: 113`.
* Round 2: the model writes "sort, then smallest neighbour difference". 100% on the
  first check, 100% on the hidden second check → accepted.
* The accepted program then labels all 5,500 task-1 examples correctly in 0.2 seconds.

**Step by step (`cairn/induction/program.py`):**
1. Show the examples, ask for `def solve(text: str) -> str`.
2. Run the code on set *verify* (held-out). Not 100%? Show up to 5 failures, ask for a fix.
3. 100% on *verify*? Run it on set *confirm*, which the repair loop never sees.
4. Why the second set? A model under pressure can write `if text == "...": return "8"` for
   every shown failure — a lookup table, not a rule. It passes *verify* and fails
   *confirm*. A test checks exactly this case.
5. Not accepted after 4 rounds? The job falls back to normal in-context labelling.

**Safety** (`cairn/induction/sandbox.py`): the code is read before running (only safe
standard-library imports, no `open`, no `eval`, no `__dunder__` tricks), runs in a separate
process with `-I -S` (no user packages), with limits on CPU (10 s), memory (512 MB), file
size, and time per call. Tests prove an endless loop, a 10 GB string and `import os` are all
stopped.

**Remember:** a model's code is a suggestion; held-out examples turn it into a proof.

---

## 14. Checking generated GPU code without a GPU

**One line:** first read the code for typical mistakes, then actually run it on the CPU and
compare its output with PyTorch's.

**Picture:** a driving test with two parts: a written test (rules), then a road test.

**See it:** the static check on the dataset's own 184 reference kernels: 183 pass, and the
one it flags really is broken (it calls `tl.cidv`, a typo of `cdiv` that doesn't exist).
Then real runs on CPU (tests in `tests/test_kernel.py`):

| kernel | result |
|---|---|
| division kernel using `x / y` | `exec_ok` — matches `torch.div` |
| same kernel using `x * y` | `wrong_result` — largest error 2372 |
| kernel with an undefined name | `import_error` |
| wrapper with an endless loop | `timeout` after 8 s |

**Step by step (`cairn/tasks/kernel.py`, `kernel_exec.py`, `_kernel_driver.py`):**
1. Read the exact function header from the task (`div(input, other, *, rounding_mode=None, out=None)`)
   and pin it in the prompt. 166/166 headers are parsed.
2. Static check: imports triton, has a `@triton.jit` kernel, every `tl.xxx` name exists
   (the list was read from the installed Triton package, not typed from memory), the kernel
   is launched, no PyTorch inside the kernel body, the wrapper accepts the calls the header allows.
3. Run in a child process with `TRITON_INTERPRET=1` (Triton simulates the GPU with numpy).
4. Trick for valid inputs: if PyTorch has a function with the same name, try guesses on
   PyTorch first. The first guess PyTorch accepts is a valid input, and PyTorch's output is
   the right answer. Then call the generated code with the same inputs and compare.
5. Each failure becomes one sentence fed back to the model for repair.

**The metric lesson.** The official score is `S_exec × (1 − (S_call − S_exec)/S_call)`,
which simplifies to **S_exec² ÷ S_call**. Work it out with 100 kernels, 40 correct:

| | runs | correct | score |
|---|---|---|---|
| 20 extra kernels run but compute the wrong thing | 60 | 40 | 0.267 |
| those 20 fail loudly instead | 45 | 40 | **0.356** |

Same number of correct kernels, higher score when wrong code does *not* run. That is why
Cairn's last resort is a stub that raises an error, never code that quietly lies.

**Remember:** always do the algebra on the metric; it tells you which mistakes cost the most.

---

## 15. Session memory without breaking the cache

**One line:** in long runs, add sure answers to the prompt so later answers stay
consistent, but only ever append, and only between waves of work.

**Picture:** a shared notebook in a team. You may only add lines at the bottom, and only
at the end of each hour, so everybody's page numbers stay valid.

**Step by step (`cairn/pipeline/memory.py`):**
1. Identical input seen before → return the same answer (no model call).
2. An answer joins the session block only if it is very sure (≥ 0.95), came from a trusted
   step, and its label isn't already over its share (a wrong answer copied forward would
   spread errors).
3. New entries wait until the current wave of 64 inputs is finished, then are appended.
4. Because entries only appear at the end, the cached beginning of the prompt stays valid.
   A test checks that every later session block *starts with* the earlier one.

**Remember:** any feature that changes the prompt during a run must respect the cache and
must not let mistakes feed on themselves.

---

# Part B — how to think like an engineer here

## 16. The order I worked in, and why

1. **Read the rules and write down the hard parts.** Only one model; 30K minimum; 8 very
   different jobs; scoring formula for job 8. The hard parts decide the design.
2. **Read the data with code, not with eyes.** Counts, label shares, input patterns. Every
   regular expression in `tasks/registry.py` was checked against every input (0 misses)
   before it was used.
3. **Measure how the existing solution fails.** The baseline loses 32% of answers to
   format. That is the first, cheapest thing to fix, and it gives a clear story.
4. **Set up honest measurement before building.** The splits (section 4) and the model-free
   floors (majority, nearest examples) came before any clever idea.
5. **Build the seam first.** One small interface for "ask the model" (`llm/base.py`), so
   everything else could be built and tested with fake models on a laptop.
6. **Build from the bottom up, testing each layer as it is finished:** tokens → search →
   selection → prompt → answer reading → decision ladder → runner → command line.
7. **Run end to end early**, even with a fake model. The first full run exposed problems
   that unit tests didn't (e.g. resuming with changed settings, section 21).
8. **Write down every number with how to reproduce it** (`cairn bench`).

The habit behind all of it: *what is the cheapest experiment that tells me whether this
idea is worth building?*

---

## 17. Error handling: the way of thinking

**One line:** sort every failure into "my fault", "temporary", or "the other side is
down", and give each kind exactly one response.

**Picture:** a restaurant order going wrong. Wrong dish written on the ticket (your fault)
→ fix the ticket, don't resend it. Kitchen busy (temporary) → wait a bit and ask again.
Kitchen on fire (down) → stop taking orders, keep the list of what was served.

**See it** (`cairn/llm/openai_compat.py`):

| what happened | category | response |
|---|---|---|
| HTTP 400/422 (prompt too long, bad field) | our request is wrong | `BadRequestError`; **never retried**; that one input gets the model-free answer |
| HTTP 429/5xx, timeout, dropped connection | temporary | retry up to 5 times, waiting 1 s, 2 s, 4 s… plus a random part |
| still failing, or 8 failures in a row | the server is down | `ServerUnavailableError`; circuit breaker opens; **the run stops** |

**Step by step (the reasoning behind each choice):**
1. *Why not retry a 400?* The same request will fail the same way. Retrying only wastes
   time and hides the real problem.
2. *Why random waiting (jitter)?* If 16 workers all wait exactly 2 s, they all hit the
   server again at the same instant. Randomness spreads them out.
3. *Why a circuit breaker?* Hammering a server that is down keeps it down. After 8 failures
   in a row Cairn stops calling for 60 s, then lets one test call through.
4. *Why stop the run instead of using fallbacks when the server is down?* Fallback answers
   are weak. If the server is dead and we quietly fill 500 answers with fallbacks, the run
   *looks* finished but is ruined. Stopping is honest, and the journal makes restarting free.
5. *Why fallbacks for single bad inputs?* One odd input should not stop 4,000 others. It
   gets a model-free answer marked with confidence 0, so reports show how often it happened.

**The four rules I use everywhere:**
* **Never fail silently.** Every fallback is labelled (`source`, `confidence`, `reason`).
* **Fail fast at start-up.** Settings are fully checked before the first model call; a typo
  stops the program immediately (`config.py`). `cairn doctor` checks the server before a long run.
* **Make work resumable.** Every finished answer is written to disk at once
  (`pipeline/journal.py`): append-only, flushed, and a half-written last line is ignored.
* **Make repeats safe** (the word is *idempotent*: doing it twice = doing it once). The disk
  cache means asking the same question twice costs nothing and gives the same answer.

**Remember:** classify the failure first; the right response follows from the category.

---

## 18. Testing: what to test and how

**One line:** test the promises your code makes, using fakes for slow or external things,
and use real data wherever it is cheap.

**What was tested here, and the trick behind each:**

| what | trick |
|---|---|
| retries, bad requests, circuit breaker | a tiny fake HTTP server (`tests/fake_server.py`) that can be told to fail N times |
| the whole pipeline | fake models: `ScriptedBackend` (prepared replies), `FunctionBackend` (any rule), `DemoReaderBackend` (copies nearest examples) |
| a speed-up is still correct | compare lazy greedy with the slow obvious greedy, item by item |
| a guarantee holds | conformal coverage averaged over 30 random trials ≥ 88% |
| crash recovery | a fake model that dies on call 7; rerun; check only the unfinished inputs are re-asked |
| the checker doesn't cry wolf | run it on the dataset's 184 known-good kernels |
| reproducibility | hash the same text in two separate Python processes, expect identical vectors |
| security limits | an endless loop, a 10 GB string, `import os`: all stopped |

**Step by step (how to decide what to test):**
1. Write down what the function *promises* ("never returns an empty answer", "the minimum
   is always met", "the split never overlaps").
2. Test the promise, not the implementation details. Then you can rewrite the inside freely.
3. For each external thing (network, GPU, time), ask: what is the smallest fake that still
   behaves like the real thing *in the way I care about*?
4. Add one test per bug you fix, so it can never come back (section 21).

**Remember:** a test suite is a list of promises that a machine checks for you every time.

---

## 19. Measuring honestly: error bars and fair comparisons

**One line:** a single accuracy number is a guess; give it a range, and compare methods on
the same inputs.

**Picture:** flipping a coin 10 times and getting 7 heads doesn't prove the coin is unfair.

**See it** (`cairn/eval/stats.py`, real numbers):
* 71 right out of 100 → 95% range **61.5% to 79.0%**.
* 73 right out of 100 → **63.6% to 80.7%**. The ranges overlap almost completely, so "73 is
  better than 71" is not shown.
* With 300 inputs, 213 right → 65.6% to 75.8%: three times the data, much narrower range.

**Paired comparison (McNemar test):** two methods on the same 100 inputs. Method A got 64,
B got 72. Looks like +8 points. But only the inputs where they *disagree* matter: A alone
right on 4, B alone right on 12. If they were equally good, each disagreement would be a
coin flip. The chance of a 4-vs-12 split or more lopsided from fair coins is **p = 0.077**.
So even +8 points on 100 inputs is not convincing evidence. Get more data before deciding.

**Step by step (the rules I follow):**
1. Only the dev split decides; the test split is never used for choices.
2. Change one thing at a time; write it in `docs/04_EXPERIMENT_LOG.md`.
3. Report a range, not just a number.
4. Adopt a change only if it is at least not worse within the noise; break ties toward the
   cheaper method.
5. Look at *where* errors come from (each answer records which ladder step produced it).

**Remember:** if you cannot say how sure you are about a number, you don't know it yet.

---

## 20. Design habits that kept the code clean

1. **One seam for the outside world.** Everything talks to the model through
   `LLMBackend.generate(request) -> response`. Swapping vLLM for a fake is one line.
2. **Differences as data, not as `if`s.** Each task is a small card (`tasks/registry.py`):
   kind, view, normaliser, constraint, signals. The engine never says `if task_id == 5`.
   Adding a ninth task = writing one card.
3. **Settings as typed data.** Every setting lives in a frozen dataclass with a default and a
   comment; YAML only overrides. Unknown keys are errors.
4. **Small pure functions where possible.** Parsing, scoring, calibration, voting take inputs
   and return outputs, no hidden state. They are trivial to test.
5. **Determinism on purpose.** Python's built-in `hash()` gives different values in every
   run (it is salted for security). Cairn uses `zlib.crc32` and SHA-256 instead, so clusters
   and splits are identical everywhere.
6. **Explain *why* in comments, not *what*.** The code says what; the comment says why this
   way and not another.
7. **Records with provenance.** Every prediction carries `source`, `confidence` and `meta`
   (cluster, tokens, votes, failures). Debugging starts from data, not from guesses.

**Remember:** good structure is what lets you change your mind cheaply.

---

## 21. Bugs found while building this, and what each one teaches

| bug | how it showed up | lesson |
|---|---|---|
| A search test failed: "tiramisu" did not beat "cat" | both words appeared in exactly one document, so it was a tie, not a bug in BM25 | when a test fails, first ask whether the test's assumption is true |
| The checker flagged a correct dataset example (two of its kernels) as "uses torch inside the kernel" | `torch.float16` appeared in a *decorator*, which runs as normal Python | validate a checker on known-good inputs before trusting it on new ones |
| Changed settings had no effect in a rerun | the journal resumed answers made with the old settings | any cache or resume needs a fingerprint of what produced it; Cairn now refuses to mix |
| Exact-match accuracy for code was always 0% | code has no single right string | choose the metric to fit the output; task 8 now reports static OK / runs / matches PyTorch |
| The baseline loses 100% of code answers | its parser drops answers longer than 100 characters | read the failure *distribution*, not only the average |
| A formatter spread a word list over 50 lines | a style tool's automatic fix | read automated changes like a teammate's changes |
| The sandbox set its limits with `preexec_fn` | found in review: Python's docs warn it can deadlock in programs that use threads | read the warnings section of the docs for every system call you use; the fix was to let the child set its own limits |
| The CPU-limit test reported "crashed: no output" | soft and hard CPU limits were equal, so the system sent SIGKILL instead of the CPU-limit signal | an error message is a feature: test that each failure says what really happened |
| The tuner never printed its "is the winner really better?" test | by the last round the runner-up had been dropped, so there was no second entry to compare | test the output a user will read, not just the internal values |

**Remember:** keep a list like this; it becomes your personal rulebook.

---

# Part C — practice

## 22. Exercises

Do them in order. Each one says what you will learn.

1. **Run and read** (30 min). `make test`, `cairn inspect`, `cairn context --task 6 --show`.
   Find, in the printed prompt, the shared block, the nearest examples, and the rule
   reminder. *Learn: the prompt layout.*
2. **Token intuition** (20 min). In Python, count tokens for 10 inputs of task 1 and 10 of
   task 7 with `cairn.tokens.get_counter()`. Why does task 1 cost more per character?
   *Learn: tokens are not words.*
3. **Write BM25 yourself** (1–2 h). Without looking, implement it for 5 short documents.
   Compare with `BM25Index` scores. *Learn: rarity weighting.*
4. **Change one setting, predict first** (1 h). Before running, write down what you expect
   `cairn context --task 6 --selection random` to do to coverage and streaks. Run it. Were
   you right? *Learn: forming and testing predictions.*
5. **Find a signal for task 2** (2–3 h). Look at verb questions with label 0 vs 2. Is there
   a cheap measurement that separates them (word endings like "-ing", "is/are")? Add it to
   `tasks/signals.py` and a card, then rerun `cairn bench` section 5. *Learn: data detective
   work, and that most ideas fail — record yours either way.*
6. **Break something on purpose** (1 h). Make `parse.norm_integer` return `None` for
   negative numbers. Which tests fail? Would you have caught it without tests? Undo it.
   *Learn: what tests protect.*
7. **Conformal by hand** (30 min). Redo section 11's table with α = 0.1. What changes, and
   why? Check with `SplitConformal`. *Learn: the coverage-vs-cost trade-off.*
8. **Metric algebra** (30 min). For task 8, if you could turn 10 "wrong but runs" kernels into
   "correct", or 20 into "fails loudly", which is better? *Learn: optimise what is scored.*
9. **First real run** (GPU needed, half a day). `bash scripts/gpu_run.sh` with `STAGE=dev`.
   Fill `docs/04_EXPERIMENT_LOG.md`. Which ladder step is right least often?
   *Learn: the real numbers, and where to work next.*
10. **Design your own improvement** (open). Pick the weakest task from exercise 9, write a
    hypothesis, the cheapest experiment to test it, and the decision rule *before* running it.

## 23. Reading order for the code

One day, in this order (each file is short and starts with a "why" explanation):

1. `cairn/types.py`, `cairn/data.py` — the records and the splits
2. `cairn/tokens.py` — the ruler
3. `cairn/retrieval/bm25.py`, `cairn/selection/submodular.py` — search and choice
4. `cairn/context/render.py`, `cairn/context/builder.py` — the prompt
5. `cairn/llm/base.py`, `cairn/llm/openai_compat.py`, `cairn/llm/cache.py` — talking to the model
6. `cairn/decoding/*` — reading answers (score, calibrate, conformal, vote, parse)
7. `cairn/pipeline/annotator.py` — the decision ladder (the heart of it)
8. `cairn/pipeline/runner.py`, `scheduler.py`, `journal.py`, `memory.py` — running at scale
9. `cairn/induction/*`, `cairn/tasks/kernel*.py` — programs and code checking
10. `cairn/eval/*`, `cairn/cli.py` — measuring and the command line
11. `tests/` — read the tests of each module after the module: they show its promises

## 24. Interview questions with short answers

* **Why not just use the most similar examples?** The rule requires 30K tokens anyway, and
  similarity alone can mislead (task 6: nearest examples scored *below* always-"N"). Cairn
  uses coverage for the bulk and similarity for the last few examples.
* **How do you stay within budget and still meet the minimum?** Exact token counts from the
  model's own vocabulary, additive by construction (tested), with a safety margin when the
  counter is rough.
* **How do you know when to spend more compute?** Conformal prediction on held-out
  examples gives a set with ≥ 90% coverage; escalate only when the set has more than one label.
* **Why is your system fast with 30K-token prompts?** Shared beginnings + grouped, primed
  sending let the server reuse its KV cache: 0.20M instead of 1.62M tokens read in the
  measured case.
* **How do you trust model-written code?** Static checks, a sandbox, and two held-out sets —
  one for repair feedback, one hidden — so memorising programs are caught.
* **What happens if the server dies at hour 3?** The circuit breaker stops the run; the
  journal has every finished answer; the same command resumes; settings fingerprints stop
  you from mixing runs.
* **What would you do with a GPU tomorrow?** Run dev on all tasks, read the per-step error
  breakdown, race settings per task with successive halving, keep changes that win a paired test.
* **What was the biggest single insight?** Task 6's label is really "was sentence 2 written
  from sentence 1": one overlap number gives 93–95%. Found by reading the data's origin.

## 25. Glossary

| word | plain meaning |
|---|---|
| token | a small piece of text the model reads; limits are counted in these |
| in-context learning (ICL) | teaching by examples inside the prompt, no training |
| prompt | everything you send to the model in one go |
| demonstration / example | a solved input→output pair shown in the prompt |
| KV cache / prefix cache | the model's notes for text it already read; reusable if the text repeats at the start |
| BM25 | classic search score: shared words, rare words count more |
| IDF | "inverse document frequency": how rare a word is |
| submodular | "each extra copy helps less" (diminishing returns) |
| greedy algorithm | a set of steps that always takes the best-looking next choice |
| log-probability | the logarithm of a probability; easier to add up; closer to 0 = more likely |
| calibration | correcting a model's confidence so it matches how often it is right |
| logistic regression | a simple formula that turns several numbers into probabilities, fitted on examples |
| convex | bowl-shaped; walking downhill always reaches the one lowest point |
| L2 penalty | a small pull toward zero that stops a fit from over-reacting |
| conformal prediction | a method that returns a set of answers with a guaranteed hit rate |
| self-consistency | sample several answers, take the most common |
| program induction | the model writes the rule as code |
| sandbox | a locked room for running untrusted code |
| static check | reading code for mistakes without running it |
| Triton | a language for writing GPU programs in Python style |
| interpreter mode | Triton running its GPU code slowly on the CPU, for checking |
| circuit breaker | stop calling a failing service for a while |
| backoff + jitter | wait longer after each failure, plus a random bit |
| idempotent | doing it twice has the same effect as doing it once |
| journal | an append-only file of finished work, for resuming |
| fingerprint (hash) | a short code computed from data; same data, same code |
| successive halving | race many options on a little data, keep the better half, repeat |
| McNemar test | a fair way to compare two methods on the same inputs |
| Wilson interval | a reliable error range for an accuracy number |
| YaRN | a setting that stretches a model's position system to read longer prompts |
| vLLM | a fast server for running language models |

---

**The whole guide in one line:** look at the data first, decide how you'll check yourself
before you build, put the right examples where the model reads best, never lose an answer,
spend effort only where a measured check says "unsure", and make every failure loud,
labelled and recoverable.

---

# Part D — the real runs

## 26. Reading a real result

Everything above was written before Qwen3-4B had run once. On 2026-11-16 it ran on Kaggle,
and the result was the best lesson in this project so far: three of the eight numbers were
produced by broken machinery rather than by the model, and the log *looked* fine.

The full account is in [`06_FIRST_REAL_RUN.md`](06_FIRST_REAL_RUN.md). Read it after
sections 11, 13 and 19 of this guide. It covers:

* **How to check that a result is real** before reading it (config, server evidence, split,
  metric, timing, error bar).
* **Reading the breakdown instead of the average**, and treating a *missing* line as the
  loudest evidence (the program path left no trace on tasks 1, 3, 4).
* **The difference between the guarantee you have and the one you need**: conformal
  coverage versus the accuracy of one-label sets (task 2).
* **Ranking hypotheses by the cost of testing them**: "the labels are noisy" took one minute
  to rule out and would have invalidated every other fix.
* **A twelve-step way of thinking** (its section 5) that applies to any result, not only this one.
* **Predictions for the next run**, written before it happens, so judgement can be scored too.

**Remember:** the first real run is not where the project is finished; it is where you find
out which of your beliefs were about the model and which were about your own tooling.

---

## 27. Scoring your own predictions

Before the second run, `06_FIRST_REAL_RUN.md` wrote down a predicted range for every task.
After it, [`07_SECOND_RUN.md`](07_SECOND_RUN.md) scored them: **five of eight landed.** The
three misses were the most useful part of the run, because each one exposed a belief about
Cairn itself that was false:

| miss | the false belief | the rule it left behind |
|---|---|---|
| task 2: 45%, predicted 82–95% | "switching scoring off sends every input to thinking" | a setting is not a behaviour: trace it to the code, test the behaviour |
| task 4: 80%, predicted 97–100% | "a bigger budget is all induction needs" | when a capable model repeats a simple mistake, rewrite the question |
| task 8: 0.000, predicted 0.05–0.25 | "the dev split measures executability" | check that the validation split looks like the test set, in the way the metric cares about |

**Remember:** a prediction you never wrote down cannot prove you wrong, so it cannot teach you.

---

## 28. The escalation probe: measuring whether thinking pays
29. Finding a label convention in the data
30. Decomposition: the model names, code computes

**One line:** before labelling, answer some held-out examples both the cheap way and the
expensive way, and let a paired test decide how much thinking this task deserves.

**Picture:** a shop deciding whether to pay for express delivery. It does not argue about it;
it sends forty parcels both ways and counts which ones arrived on time. If express is not
clearly better, it keeps the cheap service.

**See it** (real numbers from the two runs):

| task | cheap rung | thinking | what a hand-set rule did |
|---|---|---|---|
| 2 count nouns/verbs | 43% (four samples, unanimous) | 14 of 15 | skipped thinking: 45% |
| 6 same genre | 3 of 4 on the escalated inputs | 2 of 4 on the same inputs | thought, and lost one answer |
| 7 trivia | 20% | 21% | thought on a third of inputs for no gain |

One rule cannot be right for all three. A measurement can.

**Step by step** (`TaskRunner._prepare_policy`):
1. Take 40 examples from the second half of the calibration split. They were never in a
   prompt, and the calibrator was fitted on the *first* half, so they are fair judges.
2. Answer each one twice: cheap rung only, and thinking + 5 votes only.
3. Mark each example *sure* or *unsure*, by whether the cheap rung would have stopped there.
4. On the sure examples, count the disagreements: "only cheap right" (b) and "only thinking
   right" (c). An exact McNemar test gives the chance of a split this lopsided if the two were
   equally good. Do the same on the unsure examples.
5. Decide: thinking wins on the **sure** ones → think for every input (`reason_first`); it wins
   only on the **unsure** ones → the normal ladder; otherwise → never think (`cheap_only`).
   "Wins" means c > b and p ≤ 0.10.

**Why "ties go to cheap":** spending compute needs evidence, saving it does not. With 40
examples a small benefit may be missed, and missing it costs a little accuracy. A large
benefit, like task 2's 43% versus 93%, gives p below 0.0001 and cannot be missed.

**Why paired:** two accuracies measured on different inputs mix "which method" with "which
inputs". On the same inputs, every input where both are right or both wrong cancels out, and
only the disagreements are evidence.

**Try it yourself:** run `pytest tests/test_pipeline.py -k probe`. Two fake models stand in for
the two cases: one whose quick answers are confidently wrong, one where thinking adds nothing.
Read the test, predict the chosen policy, then run it.

**Remember:** if a decision can be measured before you commit to it, measure it every time.

---

## 29. Finding a label convention in the data

**One line:** when labels come from some procedure you cannot see, recover the procedure's
rules from the labels themselves before asking a model to imitate them.

**Picture:** a new job where the old clerk filed everything by rules nobody wrote down. You
do not guess; you look at a few hundred filed folders, notice what always goes together,
then check your guesses on folders you have not looked at.

**See it** (task 2, `docs/08_THIRD_RUN.md` §3): a least-squares fit of each verb label on the
words of its sentence gave "is" a weight of +0.02 and "are" −0.02. Those words appear in
hundreds of sentences and never raise the count, so helper verbs are not counted. On the
noun side "living" scored +0.81: in "living room", both words count.

**Step by step:**
1. Fit `label ≈ Σ weight(word) × count(word) + constant` with `numpy.linalg.lstsq`.
2. Read the weights of frequent words: near 1 means "counted", near 0 means "not counted".
3. Read labelled sentences that test each hypothesis directly ("A man is using a cell phone to
   photograph a barn" has 2 verbs: using, photograph).
4. Write the rules down, and put the test sentences in a test, so the rules cannot drift.

**Try it:** run the regression yourself for the noun questions and find two more words whose
weight surprises you. Then find a labelled sentence that confirms or refutes each.

**Remember:** a model cannot follow a rule nobody stated. Find it, state it, test it.

---

## 30. Decomposition: the model names, code computes

**One line:** when the answer is a number derived from items, ask the model for the items and
let code compute the number.

**Picture:** at a till, the cashier scans each item and the machine adds up the bill. Nobody
asks the cashier to look at the trolley and guess the total.

**See it:** asked "how many nouns?", Qwen3-4B's four quick samples agreed and were right 43% of
the time; asked to think, it ran out of tokens on 62% of samples. Asked instead for
`Nouns: tennis, player, racket, tennis, court`, it does a reading task, and `len()` does the
counting (`Annotator._decompose`, `cairn/tasks/counting.py`).

**Step by step:**
1. After the input, ask for the evidence in a fixed format, with thinking off.
2. Let the server enforce the format (structured output), so every reply can be read.
3. Compute the answer in code from each reply; vote over several sampled replies.
4. If the server rejects the format, retry once without it; if nothing is readable, fall back
   to the normal ladder.

**Remember:** a small model reads well and counts badly. Give each job to whoever does it best.
