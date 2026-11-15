# FlagOS Long-Context Data Annotation Competition

## Competition Track 3: Automatic Data Annotation for Large Models in Long-Context Scenarios

This repository implements a complete solution for the **FlagOS Ultra-Long Context Annotation** competition using **Qwen3-4B** with the FlagScale framework.

---

## Competition Overview

**Objective**: Develop long-context in-context learning (ICL) frameworks with effective example selection, context organization, and multi-step reasoning capabilities to improve annotation accuracy toward human-level performance.

**Key Challenges**:
- Ultra-long contexts (tens of thousands to hundreds of thousands of tokens)
- Memory capacity requirements
- Reasoning ability requirements
- Key information extraction

**Evaluation**:
- Prediction Performance Score: 60%
- Technical Solution Score: 40%

---

## The Winning Strategy: 7-Phase Approach

Based on the competition analysis, here's the proven strategy to achieve high scores:

### Phase 1: Context Engineering
- **Treat context like CPU cache**: Place most important instructions at TOP and BOTTOM of prompt
- **Avoid "lost-in-the-middle" syndrome**: Less critical content goes in the center

### Phase 2: Prompt & Instruction Design
- Use chain-of-thought: "Think step by step before giving your final label"
- Enforce output format: "Respond ONLY with the label. Do not explain."
- Test both thinking and non-thinking modes

### Phase 3: Retrieval-Based ICL (THE WINNING STRATEGY)
- Instead of random example selection, use **FAISS semantic retrieval**
- Find the **most similar examples** to each test input
- Combine with **diversity sampling** (cover all label types)

### Phase 4: Multi-Turn Continuous Annotation
- Maintain sliding window of recent annotations
- Use history for consistency across 1000+ samples

### Phase 5: Dataset 8 Special Strategy
- Dataset 8 scores on **code execution**, not just label accuracy
- Implement **self-correction loop**: generate → run → fix if crash
- Repeat up to 3 times

### Phase 6: Local Evaluation
- Build local evaluation script (don't waste daily submissions)
- Test ideas one by one (ablation studies)

### Phase 7: Strategic Submission
- Submit up to 5 times/day
- Keep improving based on leaderboard feedback

---

## Repository Structure

```
LongContext-ICL-Annotation/
├── data/                           # Competition datasets (8 tasks)
│   ├── openseek-1_closest_integers.json
│   ├── openseek-2_count_nouns_verbs.json
│   ├── openseek-3_collatz_conjecture.json
│   ├── openseek-4_conala_concat_strings.json
│   ├── openseek-5_semeval_2018_task1_tweet_sadness_detection.json
│   ├── openseek-6_mnli_same_genre_classification.json
│   ├── openseek-7_jeopardy_answer_generation_all.json
│   └── openseek-8_kernel_generation.json
├── src/                            # Implementation modules
│   ├── retrieval.py               # FAISS semantic retrieval
│   ├── prompt.py                  # Prompt template builder
│   ├── dataset8.py                # Dataset 8 (code gen) handler
│   ├── annotate.py                # Main annotation pipeline
│   ├── evaluate.py                # Local evaluation
│   ├── submit.py                  # Submission packaging
│   ├── method.py                  # Original baseline methods
│   ├── main.py                    # Original main script
│   ├── requirements.txt           # Python dependencies
│   └── ...
├── outputs/                        # Prediction outputs
│   └── README.md
└── README.md                       # This file
```

---

## Module Descriptions

### retrieval.py - Smart Example Picker
Implements semantic retrieval using FAISS for finding the most similar examples to each test input.

**Key Functions**:
- `build_index()`: Build FAISS index from demo examples
- `get_top_k_similar()`: Find k most similar examples
- `get_diverse_demos()`: Get examples covering all label types (WINNING TRICK)

### prompt.py - Prompt Template Builder
Constructs prompts with optimized structure to avoid "lost-in-the-middle" problem.

**Key Functions**:
- `build_annotation_prompt()`: Build full prompt with top/bottom instructions
- `build_code_generation_prompt()`: Special prompt for Dataset 8
- `extract_label()`: Extract label from LLM response

### dataset8.py - Code Generation Handler
Handles Dataset 8 with self-correction loop for executable code.

**Key Functions**:
- `run_code_safely()`: Execute generated Python code
- `annotate_with_correction()`: Self-correction loop
- `validate_triton_code()`: Validate Triton kernel code

### annotate.py - Main Pipeline
The complete annotation pipeline that orchestrates all components.

**Usage**:
```bash
python annotate.py --all              # Run all 8 tasks
python annotate.py --task_id 1        # Run specific task
```

### evaluate.py - Local Evaluation
Test accuracy locally before wasting daily submissions.

**Usage**:
```bash
python evaluate.py --all              # Evaluate all predictions
python evaluate.py --task_id 1        # Evaluate specific task
```

### submit.py - Submission Packaging
Package predictions into ZIP file for FlagOS upload.

**Usage**:
```bash
python submit.py                     # Create submission.zip
```

---

## Installation

### 1. Install Dependencies

```bash
cd openseek/competition/LongContext-ICL-Annotation/src
pip install -r requirements.txt
```

### 2. Download Model Weights

```bash
# Using HuggingFace
hf download Qwen/Qwen3-4B --local-dir Qwen3-4B

# Or using ModelScope
modelscope download --model Qwen/Qwen3-4B
```

### 3. Configure Long-Context

In `Qwen3-4B/config.json`, update the rope scaling configuration:

```json
"rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}
```

### 4. Set Up FlagScale

Clone and set up FlagScale:

```bash
# Clone FlagScale repository
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale

# Install dependencies
./install/install-requirements.sh --env train

# Or use Docker (recommended)
docker pull openseek2025/openseek:flagscale-20250527
```

---

## Running the Pipeline

### Step 1: Start FlagScale Server

```bash
# In a separate terminal, start FlagScale
cd FlagScale
python run.py --config-path .. --config-name llm_config action=run
```

### Step 2: Test API Connection

```bash
python api_test.py
```

### Step 3: Run Annotation

```bash
# Navigate to source directory
cd src

# Run all 8 datasets
python annotate.py --all

# Or run specific dataset
python annotate.py --task_id 1

# With custom settings
python annotate.py --all --k_similar 4 --k_diverse 4 --max_examples 100
```

### Step 4: Local Evaluation

```bash
# Evaluate all predictions
python evaluate.py --all

# Evaluate specific task
python evaluate.py --task_id 1
```

### Step 5: Package for Submission

```bash
# Create submission package
python submit.py

# Or with custom directory
python submit.py --predictions_dir ../outputs
```

---

## Competition Datasets

| ID | Dataset | Task Type |
|----|---------|-----------|
| 1 | closest_integers | Find minimum absolute difference |
| 2 | count_nouns_verbs | POS tagging |
| 3 | collatz_conjecture | Mathematical reasoning |
| 4 | conala_concat_strings | Code generation |
| 5 | semeval_tweet_sadness | Tweet sentiment classification |
| 6 | mnli_same_genre | Natural language inference |
| 7 | jeopardy_answer | Jeopardy answer generation |
| 8 | kernel_generation | **Triton kernel code generation** |

**Note**: Dataset 8 requires special handling as it's evaluated on code execution, not just accuracy.

---

## Technical Highlights

### Retrieval-Based ICL
```python
from retrieval import SemanticRetriever

retriever = SemanticRetriever()
retriever.build_index(demo_examples)

# Get diverse examples combining similarity + label coverage
demos = retriever.get_diverse_demos(test_input, k_similar=4, k_diverse=4)
```

### Optimized Prompt Structure
```
[SYSTEM INSTRUCTION]  <- TOP: Most important instructions
[TASK DEFINITION]
[ANNOTATION EXAMPLES] <- MIDDLE: Examples
[TARGET SAMPLE]       <- BOTTOM: Repeat key instructions
[REMINDER]
```

### Multi-Turn Consistency
```python
annotation_history = []  # Sliding window

# Recent annotations improve consistency
recent = annotation_history[-3:]
prompt = build_prompt(task_description, demos, test_input, recent_history=recent)
```

### Self-Correction Loop for Dataset 8
```python
for attempt in range(max_retries):
    code = generate_code(prompt)
    success, output = run_code_safely(code)
    if success:
        return code, output
    # Fix and retry
    prompt = f"Fix this error: {error}"
```

---

## Submission Instructions

1. **Run the pipeline** for all 8 datasets
2. **Package predictions** using `submit.py`
3. **Upload to FlagOS**: https://flagos.io/RaceDetail?id=296fmsd8
4. **Submit technical report** (min 5 pages) by deadline

### Submission Checklist
- [ ] Prediction results for ALL 8 datasets in JSONL format
- [ ] Zipped into single file
- [ ] Technical report (PDF, min 5 pages)
- [ ] ZIP with requirements.txt, inference scripts, README
- [ ] Open-source release to OpenSeek repository

---

## Key Takeaways

1. **Context construction wins**: 40% of technical score
2. **Retrieval-based ICL with diversity sampling** is the winning strategy
3. **Self-correction loop** significantly boosts Dataset 8 score
4. **Iterate relentlessly** with local evaluation
5. **Submit strategically**: Up to 5 times/day

---

## References

- Competition: [FlagOS Open Computing Global Challenge](https://flagos.io/RaceDetail?id=296fmsd8)
- OpenSeek: [FlagAI-Open/OpenSeek](https://github.com/FlagAI-Open/OpenSeek)
- FlagScale: [FlagOpen/FlagScale](https://github.com/FlagOpen/FlagScale)
- Qwen3-4B: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

---

## Author

MiniMax Agent - Built based on FlagOS competition strategy guides
