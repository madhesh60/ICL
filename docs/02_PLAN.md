# 02 — Plan: Winning FlagOS Track 3 with Qwen3-4B

> Companion to `01_CODEBASE_EXPLAINED.md` (what exists) and `03_RESEARCH_NOTEBOOK.md` (thinking space).
> This file is the *execution* plan. Update the checkboxes as we go.

## Status (November 2026)

The planned `solution/` package was built as **`cairn/`** (see the root `README.md`).
Everything that does not need a GPU is done and tested (137 tests); the measured numbers
are in `docs/results/offline_benchmarks.md`. What remains needs a Linux GPU box:
`bash scripts/gpu_run.sh` (or `STAGE=dev` to stop after the dev score).

| plan item | status | where |
|---|---|---|
| data loader, seeded leak-free splits, token counting | done | `cairn/data.py`, `cairn/tokens.py` |
| BM25 retrieval (no extra model) | done | `cairn/retrieval/` |
| context builder ≥ 30K, shared prefix + kNN tail | done, plus query-aware coverage selection, label interleaving, optional clusters | `cairn/context/builder.py`, `cairn/selection/` |
| chat template + thinking control + constrained decoding + strict parser (0 nulls) | done | `cairn/llm/openai_compat.py`, `cairn/decoding/` |
| logprob scoring + calibration (tasks 2, 5, 6) | done, plus stacked calibration and conformal escalation | `cairn/decoding/` |
| program induction + verification (tasks 1, 3, 4) | done, with a hidden confirm set and a sandbox | `cairn/induction/` |
| guideline induction | done, kept only if a paired test says it does not hurt | `cairn/induction/guideline.py` |
| self-consistency voting (2, 7, 8) | done | `cairn/decoding/vote.py` |
| task 8: signature lock, static check, compile/run, repair loop | done; runs on CPU via Triton interpreter mode | `cairn/tasks/kernel*.py` |
| everything logged, resumable runs | done: journal, response cache, run reports, settings fingerprint | `cairn/pipeline/`, `cairn/cli.py` |
| submission zip in the official format | done, with validation | `cairn/submit.py` |
| **real Qwen3-4B runs, ablations, context-length sweep** | **open — needs a GPU** | `cairn run`, `cairn tune` |

New since the plan: the task-6 data finding (sentence overlap alone gives 93–95% on dev,
see the learning guide §8), and the measured 32% format loss of the official baseline.

---

## 1. Reframing the problem correctly

You described it as "we have a small context window and must show examples". The rule is actually the reverse and harder:

* We **must** put **≥30K tokens of demonstrations** in every prompt (≥16K for task 8). That is roughly 500–1900 examples per prompt depending on the task.
* The model is **small (4B)**. Research (LongICLBench, "lost in the middle") shows small models get *worse*, not better, as you add more shots beyond a point: they attend to the beginning and end, forget the middle, drift toward the majority label, and copy formats from nearby examples.
* So the competition is really: **"Given that you are forced to use a 30K-token context with a 4B model, how do you select, order, and frame the demonstrations so that the model still answers accurately, and how do you decode so that nothing is lost to formatting?"**

Everything below is organized around that.

---

## 2. What "winning" requires (two scores)

| Component | Weight | How we attack it |
|---|---|---|
| Prediction accuracy on 8 tasks | 60% | Per-task algorithms (Section 5), zero-null decoding, verification loops |
| Technical report quality | 40% | Ablations with numbers, clear method taxonomy, reproducible code, honest analysis of failures. **Plan the report from day 1**, not at the end. |

Weights are from the repo README and are **[UNVERIFIED]**; even if the split differs, both matter.

---

## 3. Infrastructure plan (blocker #1: no GPU locally)

Your laptop GPU (2 GB) cannot run Qwen3-4B. Choose one:

| Option | Cost | Notes |
|---|---|---|
| **Kaggle Notebook (free)** 2× T4 16 GB, 30 h/week | 0 | Qwen3-4B fp16 fits with TP=2; KV cache for 32K context is tight but workable with `max_model_len=40000`, `gpu_memory_utilization=0.95`. Triton works (Linux). Sessions die after 12 h → checkpoint outputs to jsonl continuously. Good for development. |
| **Google Colab Pro** (L4 24 GB / A100 40 GB) | ~$10–50/month | Easiest for long contexts (A100). |
| **RunPod / Vast.ai** (A10G/4090/A100) | ~$0.3–1.5/hour | Full control, persistent volume, vLLM Docker image. Best for final full runs. |
| **AWS g5.xlarge** (A10G 24 GB) | ~$1/hour | Your AWS MCP is configured (it timed out this session); usable if you already have credits. |

Decision for now: **develop on Kaggle/Colab, do final runs on a rented 24–80 GB GPU.** Keep everything behind an OpenAI-compatible endpoint so the code is identical everywhere (and matches the organizers' FlagScale deployment).

Serving config we want (differences from `llm_config.yaml`):
* `enable_prefix_caching: true` (remove `no_enable_prefix_caching`). The 30K shared demo block becomes a cached prefix → prompt processing goes from 500× to ~1× per task.
* `max_model_len: 65536` (or 131072 with YaRN if we test >60K contexts).
* `enable_chunked_prefill: true`, `max_num_batched_tokens: 16384+`.
* Guided decoding backend available (`xgrammar` default in vLLM ≥0.8).

---

## 4. Code layout we will build (separate from the baseline)

```
ICL/
├── docs/                      these documents + experiment log
└── solution/                  <- our package (clean, reproducible, submittable)
    ├── README.md              how to run everything in one command
    ├── requirements.txt
    ├── configs/
    │   ├── serve_vllm.yaml    FlagScale/vLLM config with prefix caching + YaRN
    │   └── tasks.yaml         per-task strategy switches (k_nn, budget, mode, n_votes ...)
    ├── data/
    │   ├── load.py            load task JSON; normalize; dev-split (seeded) that never leaks into the ICL pool
    │   └── tokens.py          Qwen tokenizer length cache (so we meet 30K exactly)
    ├── retrieval/
    │   ├── bm25.py            BM25 / TF-IDF retrievers (no extra model → rule-safe)
    │   └── qwen_embed.py      optional: embeddings from Qwen3-4B hidden states via the server (rule-safe)
    ├── context/
    │   ├── builder.py         assembles [header][bulk demos][kNN demos][footer][query] under a token budget
    │   ├── ordering.py        ordering strategies (random, relevance-ascending, label-interleaved ...)
    │   └── induction.py       asks the model to induce guidelines / programs from demos
    ├── llm/
    │   ├── client.py          OpenAI-compatible client: completions, chat, logprobs, guided decoding, n-sampling, retries
    │   └── parse.py           robust answer extraction + normalization
    ├── tasks/
    │   ├── base.py            Task interface: build_prompt(sample) → decode(sample) → postprocess
    │   ├── t1_closest.py  t2_count.py  t3_collatz.py  t4_concat.py
    │   ├── t5_sadness.py  t6_genre.py  t7_jeopardy.py t8_kernel.py
    ├── eval/
    │   ├── score.py           exact-match accuracy on the dev split; per-task error analysis
    │   └── kernel_check.py    task 8: ast parse, signature check, compile, run, compare to torch reference
    ├── run.py                 `python run.py --task 1..8 --split dev|test --config ...`
    └── submit.py              builds the zip with the 8 jsonl files in the official format
```

---

## 5. The algorithm plan, per task

The notebook (`03_RESEARCH_NOTEBOOK.md`) has the reasoning; this is the decision table. "Tier" = order to implement. Expected accuracy = my estimate for Qwen3-4B, to be replaced by dev-split numbers.

| Task | Primary technique | Backup / ensemble | Expected (baseline → ours) |
|---|---|---|---|
| 1 closest_integers | **Program induction from demos** (model writes the rule as Python after seeing the demos; we verify on 200 held-out demos; execute) | Thinking mode + `guided_regex` integer + self-consistency n=5 | ~45% → ~100% (program) / ~85% (ICL) |
| 2 count_nouns/verbs | **Convention induction** (model derives an explicit tagging guideline from demos) + kNN demos (BM25) at the end + thinking "list each word → POS → count" + **majority vote n=5** + `guided_choice` over 0–6 | logprob scoring over "0".."6" | ~40% → 60–75% |
| 3 collatz | Program induction (as task 1) | Thinking + `guided_regex` for `[ints]` | ~35% → ~100% / ~85% |
| 4 concat | Program induction (as task 1) | Thinking + character-level "spell it out" CoT | ~55% → ~100% / ~90% |
| 5 sadness | **Logprob scoring** of "Sad" vs "Not sad" + label-balanced demos + **contextual calibration** (subtract bias measured on content-free inputs) + kNN tail | Thinking mode vote | ~65% → 78–85% |
| 6 same_genre | Logprob scoring Y/N + calibration (67% N prior!) + genre-stratified demos + kNN on both sentences | Thinking with "is S2 topically tied to S1 and in style X?" | ~55% → 70–80% |
| 7 jeopardy | **Category-kNN demos** (learn answer *shape* from same-category examples) + thinking + **self-consistency n=5–8** with normalized voting (lowercase, strip trailing punct) | Non-thinking greedy fallback when thinking truncates | ~30% → 35–45% (knowledge-bound) |
| 8 kernels | **Signature-locked generate → static check → compile/run on GPU → repair loop** (≤3 rounds) with BM25-retrieved example kernels (elementwise/reduction/matmul families) filling 16K | Torch-reference self-test generated by the model; pick the candidate that passes | 39% non-null → compile ≥80%, correct 40–60% |

### The "program induction" decision (tasks 1, 3, 4) — read this

These three tasks are deterministic. Asking Qwen3-4B to *infer the rule from the 30K of demonstrations, write it as a Python function, and verify it against held-out demonstrations* is legitimate ICL (instruction induction / program-aided LMs are published techniques) and will give ~100%. **Risk:** a judge may view it as sidestepping "annotation by the LLM". Mitigation:
1. The program is produced **by Qwen3-4B from the in-context demos**, never hand-written by us.
2. We keep a **pure-ICL path** (thinking + constrained decoding) and report both numbers.
3. We ask the organizers (if there is a channel) whether model-generated code execution counts; the Season-1 winner publicly used code execution for task 8, which suggests tooling is accepted.

If the rule turns out to forbid execution, we still have the pure path, and the thinking-mode numbers will remain strong.

---

## 6. Cross-cutting techniques (apply to all tasks)

1. **Zero-null decoding.** Chat template with `enable_thinking` controlled explicitly; `guided_choice` for closed label sets; `guided_regex` for ints/lists; strict post-parse with fallbacks. Target: 0 nulls on all 8 tasks (baseline loses 5–61%).
2. **Budget-exact 30K context.** Count tokens with the Qwen tokenizer; fill to ≥30K (we will test 30K, 45K, 60K to find the sweet spot and report it).
3. **Shared-prefix layout for prefix caching.** `[header][BULK demos — identical for all 500 queries][kNN demos — per query][footer with restated rules][query]`. The bulk is cached once; only the tail is recomputed.
4. **Ordering.** Bulk demos: label-interleaved, random seed fixed. kNN tail: relevance ascending so the most similar demo is adjacent to the query (recency effect).
5. **Instruction placement.** Rules at top and restated right before the query (lost-in-the-middle mitigation). No example labels in the instruction text (the baseline's "Good Review" leak).
6. **Induced guidelines.** One extra call per task: show the model a few hundred demos and ask it to write the annotation guideline (edge cases, conventions). Insert that into the header. Store the induced text in the repo for reproducibility.
7. **Self-consistency** where answers are generated (2, 7, 8): n samples at T=0.6–0.7, normalized majority vote.
8. **Calibration** where answers are scored (5, 6): contextual calibration (Zhao et al. 2021) using content-free inputs ("N/A", "", "[MASK]").
9. **Retrieval without extra models.** BM25 over inputs (rank_bm25) or TF-IDF (scikit-learn). Optionally Qwen3-4B hidden-state embeddings through the server. Never a second neural model.
10. **Everything logged.** Every run writes prompt hash, config, per-sample raw output, parsed answer, latency → feeds the report's ablation tables.

---

## 7. Evaluation protocol (so we never waste a submission)

* **Dev split:** for each task, hold out 300 labeled examples (seeded). The ICL pool = remaining examples. Dev accuracy ≈ test accuracy because test inputs come from the same distribution (verified: same length/label distributions).
* **Task 8 dev:** hold out 30 of the 184 examples; measure (a) `ast.parse` OK, (b) `@triton.jit` present, (c) wrapper name/signature matches, (d) compiles and runs on random inputs, (e) matches a torch reference where the description maps to a torch op.
* **Metric:** exact match after the organizer-style normalization (strip; we assume case-sensitive unless told otherwise — test both).
* **Ablation discipline:** change one thing at a time; record in `docs/04_EXPERIMENT_LOG.md` (template in the notebook).
* **Baseline reproduction first:** run the official `method.py` prompt on the dev split to get the true starting point.

---

## 8. Phased schedule

### Phase 0 — Setup (Day 1–2)
- [ ] Decide GPU option; get a Linux GPU session running.
- [ ] Download Qwen3-4B; apply YaRN patch; serve with vLLM (prefix caching ON); run `api_test.py`-style smoke test incl. `logprobs`, `guided_choice`, `n`.
- [ ] Create `solution/` skeleton; data loader; seeded dev split; token-length cache.
- [ ] Reproduce the official baseline on dev (fix its path bugs locally, do not edit the upstream repo).

### Phase 1 — Quick wins (Day 2–4)
- [ ] Chat template + explicit thinking control; `guided_choice`/`guided_regex`; strict parser → **0 nulls**.
- [ ] Context builder that meets ≥30K with shared prefix + kNN tail.
- [ ] Run all 8 tasks on dev. This alone should beat the baseline by a wide margin.

### Phase 2 — Task-specific algorithms (Day 4–10)
- [ ] Tasks 1/3/4: program induction + verification; pure-ICL thinking path as backup.
- [ ] Tasks 5/6: logprob scoring + calibration + balanced demos.
- [ ] Task 2: convention induction + kNN + vote.
- [ ] Task 7: category-kNN + self-consistency + answer normalization.
- [ ] Task 8: signature lock + compile/run/repair loop + torch-reference self-test.

### Phase 3 — Ablations and context-length study (Day 10–14)
- [ ] Sweep: context 30K / 45K / 60K; ordering strategies; k for kNN; n for voting; thinking vs not.
- [ ] Error analysis per task (confusion tables, failure categories).
- [ ] Freeze configs in `tasks.yaml`.

### Phase 4 — Deliverables (Day 14–18)
- [ ] Full test runs → `submit.py` → zip in the exact format.
- [ ] Technical report (≥5 pages): problem, method taxonomy, per-task design, ablation tables, context-length findings, reproducibility, limitations.
- [ ] Reproducibility package: Dockerfile or `create_env.sh`, `run_all.sh`, pinned requirements, seeds.
- [ ] Submit; iterate with remaining daily submissions.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Rules forbid code execution for tasks 1/3/4 | Keep pure-ICL path; report both; ask organizers early |
| Rules forbid any auxiliary model (even an embedder) | BM25/TF-IDF only; or Qwen3-4B's own embeddings |
| Organizers' deployment lacks guided decoding | Make constrained decoding optional; parser must be robust without it (regex + retry + majority) |
| Thinking mode runs away (infinite loops) | `max_tokens` cap, T=0.6, non-thinking fallback, timeout per call |
| Prefix caching unavailable on their side | Still correct, just slower; keep bulk prefix identical anyway |
| KV memory at 30K+ on a 24 GB GPU | fp16/bf16 model = 8 GB; 30K KV for 4B ≈ 4–5 GB; fine for batch 1–4. Use `max_model_len` 48K unless sweeping |
| Dev split optimistic vs hidden test | Same distribution (verified); still hold a second untouched 200-example split for the final check |
| Deadline unknown (Season 1 ended June 2026) | Confirm with FlagScale contact before heavy spend |

---

## 10. Open questions for you (answer when you can; nothing here blocks Phase 0)

1. **Which round / deadline?** Season 1 ended June 2026. Is this Season 2, an internal FlagScale assignment, or coursework? Is there a submission portal you can access?
2. **Do you have an official rules PDF** with the exact scoring formula (especially task 8) and the normalization used for exact match? If yes, drop it in `docs/` and I will update the plan.
3. **GPU access:** Kaggle/Colab account, or budget for RunPod/AWS? This decides how fast Phase 0 goes.
4. **Is code execution by the model acceptable** (for tasks 1/3/4 and 8)? If no official answer, we run both paths and decide at submission time.
