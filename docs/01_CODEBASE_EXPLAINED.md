# 01 — Codebase Explained: OpenSeek / FlagOS Track 3 (Long-Context ICL Annotation)

> Written 2026-11-14. Everything here was verified by reading the repo and the data files.
> Things that could NOT be verified are marked **[UNVERIFIED]**.
>
> **Update 2026-11-16:** this describes the repository *before* Cairn was added. The new
> solution lives in `cairn/` (see the root `README.md`); `AGENTS.md` now holds the challenge
> text; the baseline's format losses were re-measured in `docs/results/offline_benchmarks.md`.

---

## 0. TL;DR

* The repo you were given is **OpenSeek** (BAAI + FlagScale). 95% of it is about *pre-training* a 1.4B DeepSeek-style model. **You do not need that part.**
* The part that matters is one folder: `OpenSeek/openseek/competition/LongContext-ICL-Annotation/`.
* It is the baseline for **FlagOS Open Computing Global Challenge – Track 3: "LLM Automatic Data Annotation in Long-Context Scenarios"**.
* The job: using **only Qwen3-4B**, label 8 test sets by **In-Context Learning (ICL)** where the prompt must contain **at least 30K tokens of demonstrations** (16K for task 8). No fine-tuning. The model is small, the context is huge, and the model degrades with long context. That is the whole game.
* Score = **prediction accuracy (60%) + technical report (40%)** [from repo README, UNVERIFIED on the official page].
* The provided baseline is weak: it violates the 30K rule, has broken paths, and 5%–61% of its predictions are `null`.

---

## 1. What the "report" (the repo READMEs) says, in plain words

### 1.1 OpenSeek (the outer project)
OpenSeek is BAAI's open effort to reproduce/beat DeepSeek: data (CCI4.0 dataset), algorithms (MTP, hyper-params), and systems (FlagScale, their Megatron-based training framework). The outer `README.md`, `docs/`, `configs/`, `evaluation/`, `hf_openseek/`, `tools/`, `benchmarks/` all serve that pre-training goal. Their results (e.g. `docs/algorithm_exp_results.md`: MTP lifts the 100B-token baseline from 0.3655 to 0.3897 average) are **irrelevant to our task** except as background.

### 1.2 The competition (what we actually do)
From `openseek/competition/LongContext-ICL-Annotation/README.md`, `data/README.md`, `src/README.md` and web sources:

| Item | Value |
|---|---|
| Organizers | FlagOS community, BAAI, CCF ODTC |
| Platform | Kaggle page + FlagOS platform (flagos.io) |
| Model | **Qwen3-4B only.** "The use of any model other than Qwen3-4B is strictly prohibited." |
| Serving | FlagScale → vLLM, OpenAI-compatible API on port 2026 |
| Long context | Patch `config.json` with YaRN `factor: 4.0` → 131K window |
| Tasks | 8 datasets, 500 test samples each (166 for task 8) |
| Context rule | **Minimum ICL context 30K tokens** (tasks 1–7), **16K** (task 8) |
| Submission | zip of 8 jsonl files, each line `{"test_sample_id": ..., "prediction": ...}` |
| Scoring | 60% prediction performance + 40% technical solution **[UNVERIFIED]** |
| Reproducibility | Organizers re-run your code on their own Qwen3-4B deployment |
| Report | Technical report, min 5 pages, plus code zip + open-source PR |
| Prizes | 30K / 20K / 10K RMB (Season 1) |
| Season 1 timeline | Registration Jan 2026, results early June 2026 (**already concluded**) |

**Important timeline note:** Season 1 is over (winners: OpenSeek team 1st, TrialBlazers 2nd, StreamCrossers 3rd). You are working on this in Nov 2026, so either this is Season 2, an internal assignment from FlagScale, or a course project. **Confirm the deadline and ruleset with whoever gave you the task** — see open questions in `02_PLAN.md`.

What the Season 1 winner reported (PR Newswire): hardest parts were trivia QA (task 7, no retrieval allowed) and natural-language-to-code (task 8). They used multi-round invocation + content review and got **code compile success above 80%**.

---

## 2. Repository map (only what matters)

```
ICL/                                   <- your project root (git repo, 1 commit)
├── AGENTS.md                          (empty)
├── docs/                              <- NEW: these documents
└── OpenSeek/                          <- nested git repo (your fork: madhesh60/OpenSeek)
    ├── README.md, docs/, configs/ ... <- pre-training stuff: IGNORE
    ├── evaluation/                    <- lm-eval / lighteval harnesses for pretraining: IGNORE
    └── openseek/
        ├── algorithm/, baseline/, data/, system/   <- pre-training: IGNORE
        └── competition/
            ├── pz/README.md                        (empty)
            └── LongContext-ICL-Annotation/         <<< THE TASK >>>
                ├── README.md / READMD_cn.md        quick start (env, model, YaRN, serve, run)
                ├── data/
                │   ├── README.md                   dataset rules (30K min context, format)
                │   └── openseek-{1..8}_*.json      the 8 tasks (examples + hidden-label tests)
                ├── outputs/
                │   ├── README.md                   submission format
                │   └── result.zip                  organizers' baseline predictions (v1)
                └── src/
                    ├── main.py        baseline runner (official)
                    ├── method.py      baseline prompt / example selection / API call (official)
                    ├── llm_config.yaml FlagScale+vLLM serve config
                    ├── api_test.py    smoke test for the API
                    ├── create_env_nvidia.sh  env setup (Linux, CUDA)
                    ├── annotate.py    ALTERNATIVE pipeline (written by "MiniMax Agent", not official)
                    ├── retrieval.py   FAISS + MiniLM retriever   (MiniMax pipeline)
                    ├── prompt.py      prompt templates            (MiniMax pipeline)
                    ├── dataset8.py    code-exec self-correction   (MiniMax pipeline)
                    ├── evaluate.py    local eval                  (MiniMax pipeline)
                    ├── submit.py      zip packager                (MiniMax pipeline)
                    ├── requirements.txt
                    └── README.md      "7-phase winning strategy" (MiniMax pipeline)
```

Two pipelines live side by side in `src/`:
1. **Official baseline** = `main.py` + `method.py` (+ `llm_config.yaml`, `api_test.py`).
2. **"MiniMax Agent" pipeline** = `annotate.py`, `retrieval.py`, `prompt.py`, `dataset8.py`, `evaluate.py`, `submit.py`. Someone generated it with an AI agent and committed it. It has never been run end-to-end (see bugs below).

---

## 3. The data (verified by script)

Every task file has the same JSON shape:
```json
{ "task_id": "openseek-1", "task_name": "...", "Definition": ["<task description>"],
  "examples":     [{"id": "...", "input": "...", "output": ["<label>"]}, ...],
  "test_samples": [{"id": "...", "input": "..."}, ...],
  "License": [...] }
```

| Task | Name | #examples | #test | avg tokens/example | labels | ~examples that fit in 30K tok |
|---|---|---|---|---|---|---|
| 1 | closest_integers | 5500 | 500 | ~8 | integer 0–~99 (81 distinct) | ~1900 |
| 2 | count_nouns_verbs | 5440 | 500 | ~27 | "0".."6" (7 classes) | ~850 |
| 3 | collatz_conjecture | 3997 | 500 | ~12 | a list (unique per sample) | ~1500 |
| 4 | conala_concat_strings | 3993 | 500 | ~19 | a string (unique per sample) | ~1100 |
| 5 | tweet_sadness | 1899 | 500 | ~25 | Sad 59% / Not sad 41% | ~900 |
| 6 | mnli_same_genre | 5500 | 500 | ~53 | N 67% / Y 33% | ~490 |
| 7 | jeopardy_answer | 5499 | 500 | ~31 | free text, lowercase (4768 distinct) | ~770 |
| 8 | kernel_generation | 184 | 166 | ~430 in + ~1370 out | Triton Python code | ~8 full examples in 16K |

Facts that shape the algorithm design:

* **Tasks 1, 3, 4 are deterministic.** I verified every one of the 5500 / 3997 / 3993 labels is reproduced by a 1-line program (min adjacent diff of the sorted list; `x/2 if even else 3x+1`; join the strings). Task 1 lists have 3–10 ints; task 3 lists have 2–9 ints in 1..200; task 4 lists have 2–19 strings.
* **Task 2** sentences look like MS-COCO captions. Nouns questions and verbs questions are ~50/50. Label distributions differ sharply: nouns → mostly 3–4; verbs → mostly 0–1. The labels follow *some specific POS tagger's convention*, which the model must infer from demos (e.g. is "train tracks" 1 or 2 nouns? is a gerund a verb?).
* **Task 5** is SemEval-2018 Task 1 (emotion "sadness" presence). Binary, with emojis/hashtags.
* **Task 6** is built from MNLI: `Sentence 1 ... Sentence 2 ... Genre: <genre of S1>.` → **Y** if S2 is from the same genre (it is then the original MNLI hypothesis, usually topically related to S1), **N** if S2 was swapped in from another genre. 67% N. Genres seen: telephone, government, fiction, travel, slate.
* **Task 7** is closed-book trivia. Answers are lowercase, may contain punctuation ("mckinley, taft &/or clinton"), and keep articles ("the nutty professor"). Format matters as much as knowledge; categories like "5-LETTER WORDS" dictate the answer shape.
* **Task 8**: the 184 examples are *descriptions of existing kernels* → code, but the 166 tests use a **different format** ("Functional Description / Wrapper Entry Information / Math / other", ending with "verify if the Triton wrapper aligns with the provided func_inputs"). The wrapper **function name and signature are given in the test input** and must be reproduced exactly. Example outputs are long (median 4K chars, max 17K).
* No test input appears verbatim in any example set (no leakage shortcut). No duplicates anywhere.

---

## 4. The official baseline, line by line

### `main.py`
* Loads task JSON, takes `examples[:100]` only, iterates `test_samples`.
* Calls `build_prompt(task_description, text)` then `select_examples(...)` **once** (cached in `examples_str`) and string-replaces `[[EXAMPLES]]`.
* Calls `annotate(input_prompt)` → writes `{"test_sample_id", "prediction"}` per line to `../outputs/openseek-{id}-v{n}.jsonl`.
* `--max_input_length` is parsed but the length check is commented out.
* **Bug:** `TASK_FILES` uses `./data/...` for tasks 1–7 but `../data/...` for task 8. Run from `src/`, tasks 1–7 paths do not exist.
* **Bug:** `--tokenizer_path` default is an absolute path on the author's server.

### `method.py`
* `build_prompt`: a long "Role / Core Task / Guidelines / Examples / Text to Annotate / Final Requirement" template asking the model to reason **and** wrap the answer in `<label>…</label>`. It contains illustrative labels like "Good Review", "Neutral Review" — which **leak into predictions** (task 4 output has 24× "Neutral Review", task 8 has 11× "Good Review").
* `select_examples`: greedy fill of examples **in file order** until **8192 tokens** → directly violates the 30K minimum. Reloads the tokenizer from a hard-coded path.
* `count_answer`: regex all `<label>` tags, majority-vote them, return `None` if none found or if the answer is ≥100 chars. This is where the nulls come from.
* `annotate_nvidia`: raw `POST /v1/completions` with `max_tokens=10000`, default sampling, **no chat template, no thinking control**. `annotate_ascend`: chat API at T=0.7.

### `llm_config.yaml`
vLLM via FlagScale: `../Qwen3-4B`, port 2026, `gpu_memory_utilization: 0.9`, **`no_enable_prefix_caching: true`** (prefix caching OFF — we will want it ON; see plan).

### What the organizers' baseline actually produced (`outputs/result.zip`, v1)

| Task | null predictions | obviously garbage strings |
|---|---|---|
| 1 | 47 / 500 (9%) | 26 |
| 2 | 81 / 500 (16%) | 15 |
| 3 | 113 / 500 (23%) | 108 ("...", "RESULT", template echoes) |
| 4 | 137 / 500 (27%) | 88 ("Neutral Review", "Annotation Result") |
| 5 | 88 / 500 (18%) | 18 |
| 6 | 105 / 500 (21%) | 3 (plus some genre names instead of Y/N) |
| 7 | 24 / 500 (5%) | 11 |
| 8 | 102 / 166 (61%) | 9 |

So before any "intelligence", the baseline throws away roughly a fifth of its answers to formatting failures. Fixing output control alone is a big jump.

---

## 5. The "MiniMax Agent" pipeline (`annotate.py` & friends): what it tries and why it is not usable as-is

Ideas in it are reasonable (retrieval ICL, instructions top+bottom, sliding history, self-correction for code, local eval, packaging). The implementation has blocking problems:

1. **Rule risk:** `retrieval.py` embeds with `all-MiniLM-L6-v2` (a second model). The rules say only Qwen3-4B may be used. Use BM25/TF-IDF (no neural model) or Qwen3-4B's own hidden states instead.
2. **Violates 30K:** uses 8 demos per prompt (k_similar=4 + k_diverse=4).
3. **Wrong submission key:** writes `"id"` instead of `"test_sample_id"`; `submit.py` and `evaluate.py` also expect `"id"`. The zip it builds would fail the format in `outputs/README.md`.
4. **Thinking-mode trap:** `run_model` uses chat completions with `max_tokens=50` and never disables Qwen3 thinking → the budget is spent inside `<think>` and the visible answer is empty.
5. **Task 8 self-check is broken:** runs code with local `python -c` (Triton needs Linux + CUDA), and treats "no stdout" as failure although kernels print nothing.
6. `get_diverse_demos` takes the *first* example of each label in file order, not a balanced or random sample.
7. `evaluate.py` has no ground truth for test samples and no dev split, so it cannot measure anything.
8. Wrong API port (8000 vs 2026 in `llm_config.yaml`) and model name.

Conclusion: treat `src/` as **reference material**, not as the starting point. We build our own clean solution package (see `02_PLAN.md`).

---

## 6. Environment reality check (your machine)

* Windows 11, Python 3.10.11. Installed: torch 2.9.1 (**CPU build**), transformers 4.56.1, sentence-transformers, faiss-cpu, openai, requests, tqdm, scikit-learn.
* GPU: NVIDIA GeForce with **2 GB VRAM**, driver 511.69 / CUDA 11.6. **Cannot host Qwen3-4B** (8 GB in bf16 + tens of GB of KV cache at 30K+ context).
* Triton (needed to *execute* task 8 kernels) does not run on Windows.
* `FlagScale/` and `Qwen3-4B/` are not cloned/downloaded anywhere.

Therefore: all model inference must run on a **Linux box with a ≥24 GB GPU** (A10G / L4 / A100 / 4090), rented or cloud. The local machine is fine for data analysis, retrieval index building, prompt construction, evaluation, and report writing. Options are listed in `02_PLAN.md`.

---

## 7. Qwen3-4B facts we will exploit

* Dense 4B model, native context 32K; **131K with YaRN factor 4** (the README's patch).
* **Hybrid thinking:** chat template supports `enable_thinking=True/False`; soft switches `/think` and `/no_think` in the user turn. Thinking output is inside `<think>…</think>`.
* Recommended sampling (Qwen team): thinking → T=0.6, top_p=0.95, top_k=20; non-thinking → T=0.7, top_p=0.8, top_k=20. **Greedy decoding in thinking mode can loop forever**; always cap `max_tokens` and keep a non-thinking fallback.
* Served by vLLM, so the OpenAI-compatible endpoint also supports: `logprobs` / `echo` (score candidate labels by likelihood), `guided_choice` / `guided_regex` / `guided_json` (constrained decoding → zero null predictions), `n` (multiple samples for self-consistency), and **automatic prefix caching** (shared 30K demo prefix computed once per task instead of 500 times).

These four server features (logprob scoring, constrained decoding, n-sampling, prefix caching) are the systems backbone of the plan.
