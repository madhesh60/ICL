# 03 — Research Notebook (thinking space)

> This is the scratchpad. Write hypotheses, half-ideas, rejected ideas, and results here.
> Nothing in this file needs to be polished. Promote decisions to `02_PLAN.md`; promote numbers to `04_EXPERIMENT_LOG.md`.
> Convention: `H#` = hypothesis, `I#` = idea, `Q#` = open question, `R#` = result, `X#` = rejected.

---

## A. The core tension, written out once

```
   forced:  ≥30K tokens of demonstrations in the prompt
   model:   4B params, effective attention degrades with length
   observed (literature):
     - "lost in the middle": recall is U-shaped over position
     - many-shot helps big models; small models plateau early, then drift
     - majority-label bias and recency bias grow with #shots
     - format copying is strong: the model imitates the *nearest* examples
   therefore:
     - the 30K is mostly "dead weight" we must carry safely
     - the LAST ~1–2K tokens before the query are where accuracy is decided
     - the FIRST ~1K tokens set the frame (task + induced rules)
     - decoding must be constrained so no answer is lost to formatting
```

Design mantra: **make the 30K harmless (or mildly useful), make the tail decisive, make decoding lossless.**

---

## B. Hypotheses to test (each becomes an ablation row)

- **H1** Zero-null decoding (constrained) alone lifts every task by the baseline's null rate (5–61 points of raw yield; accuracy gain = null rate × accuracy on those).
- **H2** kNN tail (BM25 top-8 right before the query) beats random tail by ≥5 points on tasks 2, 5, 6, 7.
- **H3** Bulk demos ordered label-interleaved beat file order on 5/6 (removes label streaks → less majority bias).
- **H4** Accuracy vs context length is flat or *declining* from 30K → 60K for tasks 5/6/7 with a 4B model; if so we stay at the minimum (30K) and say why in the report. (This is a nice report finding either way.)
- **H5** Contextual calibration adds 2–6 points on 5 and 6 (class prior 59/41 and 67/33 → model will over-predict the majority).
- **H6** Thinking mode helps 1/2/3/4/7, hurts latency, and needs a fallback; it does not help 5/6 when scoring by logprob.
- **H7** Program induction reaches ~100% on 1/3/4 and the induced program passes 100% of held-out demos on the first or second try.
- **H8** Task 2 errors are dominated by convention mismatches (compound nouns, gerunds, auxiliaries), not by counting mistakes → induced guidelines fix more than CoT does.
- **H9** Task 8: forcing the exact wrapper signature from the test input ("Wrapper Entry Information") removes the "inconsistent naming" failure class the Season-1 winner described.
- **H10** Task 7: same-category demos teach answer *shape* (single word, "the X", person's full name) and add 3–5 points over topical kNN.

---

## C. Idea bank (not yet decided)

### Context construction
- **I1 Two-zone prompt.** Zone A (cached bulk, identical per task), Zone B (per-query kNN + restated rules + query). Keeps prefix caching and puts relevance where attention is strongest.
- **I2 Relevance-ascending tail.** Order kNN demos so similarity increases toward the query.
- **I3 Hard-negative tail for classification.** For tasks 5/6, include the nearest demo of *each* label in the tail so the decision boundary is visible locally.
- **I4 Compressed bulk.** For the bulk zone, use the shortest representative demos (cheap tokens) so the 30K budget covers more label variety; keep long/representative ones for the tail.
- **I5 Chunk-and-summarize bulk.** Ask the model (once) to write "what I learned from these 500 examples" and put that summary in the header; the raw 500 stay as cached bulk. Cheap and reportable.
- **I6 Position-aware duplication.** Repeat the task definition every ~10K tokens inside the bulk (anchors). Test whether anchors reduce drift.
- **I7 Query-first echo.** Put the query *also* at the top ("You will be asked about: …") so the model reads the demos with the question in mind. Known to help long-context QA; cheap to test.

### Decoding
- **I8 Logprob scoring for closed label sets** (2, 5, 6): compute log P(label | prompt) for each label via `echo`+`logprobs` or by asking for `max_tokens=1` with `logprobs=20`. No parsing, no nulls, gives confidence.
- **I9 Guided decoding** (`guided_choice`, `guided_regex`) when generating: always valid output.
- **I10 Self-consistency** with normalized voting for free-form (7) and reasoning (2).
- **I11 Confidence-gated escalation.** Cheap non-thinking pass first; if margin between top-2 labels < τ, run the expensive thinking + vote pass. Saves 70%+ of compute on easy samples.

### Induction (the "the model teaches itself from demos" family)
- **I12 Guideline induction** (tasks 2, 5, 6): "Here are 300 labeled examples. Write the annotation guideline a human annotator must have followed, with edge cases." Insert into header. One call per task, stored as text.
- **I13 Program induction** (tasks 1, 3, 4): "Here are 300 input→output pairs. Write a Python function f(input_str)->str that reproduces every output." Verify on 200 held-out demos. Regenerate with failing cases shown if <100%.
- **I14 Rule induction for task 7 formatting:** derive rules like "answers are lowercase; keep leading 'the'; no trailing period; for 'X-LETTER WORDS' answer is one word of X letters".

### Task 8 specific
- **I15 Signature lock.** Parse `Wrapper Entry Information: name(args) -> Tensor` and pre-fill `def name(args):` in the prompt's answer prefix (completion API, model continues the body).
- **I16 Family routing.** Classify the test description into {elementwise unary, elementwise binary w/ broadcast, reduction, matmul/bmm, normalization, fused chain, linear algebra, conv} and retrieve 3–6 example kernels of that family with BM25 over descriptions.
- **I17 Generate → check → repair.** Static (ast, imports, `@triton.jit`, signature) → compile on GPU with random tensors → compare with a torch reference the model also writes (or we map from the description, e.g. `torch.div`) → feed the traceback back for ≤2 repairs.
- **I18 Best-of-n with a verifier.** n=3 candidates, keep the one that passes the most checks.
- **I19 Minimal-kernel prior.** For ops the example set lacks (LU solve, conv2d), allow a "torch-fallback wrapper + trivial Triton kernel" so at least the wrapper is callable — only if scoring rewards callability separately (S_Call). Check rules.

### Rejected or parked
- **X1** Sentence-transformer embeddings (MiniLM) → second model, rule risk. Use BM25/TF-IDF or Qwen hidden states.
- **X2** Fine-tuning / LoRA → not ICL, forbidden in spirit.
- **X3** Web retrieval for task 7 → forbidden ("could not access external retrieval tools").
- **X4** Hand-written solvers for 1/3/4 submitted as the method → not an LLM method; only acceptable if the *model* writes the program (I13).

---

## D. Per-task thinking

### Task 1 — closest integers
- Pure ICL failure mode: the model eyeballs; lists of 8–10 numbers with negatives are error-prone. Thinking helps: "sort → adjacent differences → min".
- Program induction should be trivial for the model (one line). Verification = run on 200 held-out demos.
- Pure-ICL path: `guided_regex` `^\d{1,3}$`, thinking on, n=5 vote.
- Q1: does the organizer normalize "33" vs "33.0"? Output exactly the integer string.

### Task 2 — count nouns / verbs
- What convention? Sentences are COCO captions; the source is likely a Natural-Instructions task built on a POS tagger. Offline analysis idea: run NLTK/spaCy locally **only to understand** the convention (which tag set counts "boiling" as a verb? does it count NN, NNS, NNP? are auxiliaries ("is", "are") verbs?). Then write the guideline — or better, let the model induce it (I12) and compare with what we learned.
- Known traps: compound nouns ("train tracks"), gerunds as nouns vs verbs, "being", "is" + participle (1 or 2 verbs?), proper nouns.
- Decoding: thinking, then `guided_choice` over "0".."6". Or logprob scoring over the 7 labels (compare both).
- The two sub-questions (nouns vs verbs) have very different priors; retrieve kNN demos **of the same question type** only.

### Task 3 — collatz step
- Even/odd then arithmetic per element; errors compound with list length (up to 9). Program induction → 100%.
- Pure path: thinking + `guided_regex` for `^\[\d+(, \d+)*\]$`. Output must match exactly the Python list format with ", " separators.

### Task 4 — concatenate strings
- Tokenization makes char-level copying error-prone (spaces removed, punctuation kept, case kept). Program induction → 100%.
- Pure path: thinking "write each element, then join"; constrained output is hard (any string) → use strict post-parse.

### Task 5 — tweet sadness
- SemEval-2018 E-c "sadness" label. Human agreement is not perfect; ceiling maybe ~85%.
- Approach: logprob scoring Sad vs Not sad; balanced demos; contextual calibration; kNN tail with hard negatives (I3). Emojis matter: keep raw text, UTF-8 safe.
- Watch for sarcasm / "crying from laughter" (😭 used positively, see example 4 → "Not sad").

### Task 6 — same genre (MNLI-derived)
- Y = S2 is the original hypothesis → strong topical overlap with S1. N = S2 is random from another genre → usually off-topic and stylistically different (telephone transcripts have disfluencies; government is formal; fiction is narrative past tense; travel is descriptive; slate is opinionated).
- So the real features: topical relatedness(S1,S2) and style(S2)==genre. The induced guideline (I12) should say exactly this.
- Prior 67% N → calibration essential.
- kNN: retrieve by S2 text (style) and by genre (stratify demos so all 10 genres appear).

### Task 7 — Jeopardy
- Knowledge-bound: a 4B model knows maybe 35–45% of these. Our leverage is **not losing correct knowledge to formatting**: lowercase, keep articles as in demos, no trailing punctuation, "what is" removed, surnames vs full names (demos show full names: "robin williams", "erich segal").
- Category-kNN (same or similar category) teaches shape; thinking lets the model reason about clue wordplay ("BEFORE & AFTER" categories).
- Self-consistency n=8 with normalized voting; tie-break by non-thinking greedy answer.
- Metric uncertainty Q2: exact match only? If so, answer normalization is worth several points.

### Task 8 — Triton kernels
- Example/test format mismatch → the model has to generalize from "describe existing kernel → code" to "spec → code". Put 1–2 *synthetic* test-style examples (we can rewrite a few example descriptions into the test format **using Qwen3-4B**) at the end of the context.
- Signature lock (I15) + family routing (I16) + repair loop (I17).
- Need GPU with Triton to run checks; on Kaggle T4 Triton works (compute capability 7.5 is fine for most kernels; `tl.dot` needs fp16/bf16 and ≥16 block sizes).
- Scoring unknown exactly (S_Call / S_Execution per the repo README). Maximize "callable with the given signature" first, correctness second.

---

## E. Experiment log template (copy rows into `04_EXPERIMENT_LOG.md`)

| id | date | task | split | config (one-line diff vs previous) | ctx tokens | nulls | acc | notes |
|---|---|---|---|---|---|---|---|---|
| E000 | | 1 | dev300 | official baseline prompt, 8K ctx, raw completion | 8K | | | reproduction |
| E001 | | 1 | dev300 | + chat template, no_think, guided_regex | 30K | | | H1 |
| E002 | | 1 | dev300 | + kNN tail (BM25 k=8) | 30K | | | H2 |
| ... | | | | | | | | |

Rules for the log: one change per row; record the seed; keep the raw outputs under `solution/runs/<id>/`.

---

## F. Report skeleton (start filling as results land)

1. Problem and constraints (30K min context, Qwen3-4B only, 8 heterogeneous tasks).
2. Diagnosis of the baseline (null rates, format leakage, 8K context) — table from `01_CODEBASE_EXPLAINED.md` §4.
3. Method overview: two-zone context, induction, constrained/scored decoding, verification loops.
4. Per-task designs (one subsection each) with the ablation that justified each choice.
5. Context-length study (30K/45K/60K) and ordering study — the "long-context science" the judges want.
6. Results table: baseline vs ours per task; nulls; compute cost (tokens, time, cache hit rate).
7. Reproducibility: environment, seeds, single command, runtime on the organizers' hardware.
8. Limitations and what we'd do with a bigger model.

---

## G. Running questions

- Q1 Exact-match normalization used by the grader (case? whitespace? "33" vs "33.0")?
- Q2 Task 7 grading: exact match or fuzzy?
- Q3 Task 8 formula and whether "callable but wrong" earns partial credit.
- Q4 Is model-generated code execution allowed for 1/3/4?
- Q5 Is any auxiliary non-LLM model (BM25 is not a model; TF-IDF is not; a POS tagger would be) allowed at inference time? Assume no.
- Q6 Must the 30K be *demonstrations only*, or does the instruction text count toward it? Assume demonstrations only; be safe.
- Q7 Deadline / round.

---

## H. Results so far (offline, 2026-11-16)

- **R1 (H1 partly confirmed, offline):** the baseline loses 32% of answers to null or
  impossible output; task 8 loses 100% (its parser drops answers ≥ 100 characters).
- **R2 (new):** task 6's label is "was sentence 2 written from sentence 1" (MNLI origin).
  Content-word overlap: mean 0.49 for Y vs 0.12 for N; threshold alone 93% on dev; stacked
  with the kNN vote 94.7%. kNN vote alone is *below* majority (58.0% vs 64.3%), which is what
  pointed to the wrong notion of similarity.
- **R3 (H3 support):** the file-order shared block of task 6 has an 18-long "N" streak;
  interleaving brings it to 7.
- **R4:** query-aware coverage lifts example coverage strongly on task 2 (0.75 → 0.96),
  task 6 (0.59 → 0.73), task 7 (0.78 → 0.86). Proxy only; needs a model run to price it.
- **R5:** with clusters, send order matters more than anything else for prefix reuse
  (0.20M vs 1.62M tokens read). With one shared beginning it does not matter.
- **R6:** one example kernel in the dataset calls `tl.cidv` (typo); the dataset is not
  perfectly clean, so never treat examples as ground truth for API names.

## I. Next ideas (not built)

- **I20** Task 2 signal hunt: are there cheap surface cues (word endings, auxiliaries) that
  separate verb counts 0/1/2? Measure before building.
- **I21** Context-length sweep with the real model: 30K vs 45K vs 60K on tasks 5/6/7 (H4).
- **I22** Use the conformal set size as a per-input routing signal for *where* the
  per-input tail should come from (same-label neighbours vs diverse neighbours).
