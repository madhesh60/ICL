THE CODE:
%%bash
#!/usr/bin/env bash

# ============================================================
# CAIRN + QWEN3-4B | KAGGLE EXPERIMENT RUNNER
# ============================================================

# -------------------- EXPERIMENT SETTINGS -------------------

TASKS="${TASKS:-1 3 4 2 5 6 7 8}"
LIMIT="${LIMIT:-60}"
LIMIT8="${LIMIT8:-24}"

SPLIT="${SPLIT:-dev}"

BRANCH="${BRANCH:-claude/ai-project-enhancement-z6ct2n}"
REPO_URL="${REPO_URL:-https://github.com/madhesh60/ICL.git}"

# -------------------- PATH CONFIGURATION --------------------

W="/kaggle/working"
REPO="$W/cairn"
MODEL_DIR="${MODEL_DIR:-/tmp/Qwen3-4B}"
RUNS="$W/runs/$SPLIT"
PORT=2026

# -------------------- ENVIRONMENT ---------------------------

set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CAIRN_TRITON_INTERPRET=0

mkdir -p "$W"

# Save all main script output.
exec > >(tee -a "$W/cairn_run.log") 2>&1

say() {
    printf '\n==================================================\n'
    printf ' %s\n' "$*"
    printf '==================================================\n'
}

up() {
    curl -sf --connect-timeout 3 \
        "http://127.0.0.1:$PORT/v1/models" >/dev/null
}

say "CAIRN EXPERIMENT STARTED"

date

echo "Tasks       : $TASKS"
echo "Dev limit   : $LIMIT"
echo "Task 8 limit: $LIMIT8"
echo "Split       : $SPLIT"
echo "Branch      : $BRANCH"
echo "Model       : $MODEL_DIR"

# ============================================================
# 1. GPU CHECK
# ============================================================

say "1/7 GPU CHECK"

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: NVIDIA GPU unavailable."
    echo "Enable GPU in Kaggle notebook settings."
    exit 1
}

nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv

NGPU=$(nvidia-smi -L | wc -l)

if [ "$NGPU" -lt 1 ]; then
    echo "ERROR: No NVIDIA GPU detected."
    exit 1
fi

echo "Detected GPUs: $NGPU"

# ============================================================
# 2. CUDA DRIVER LIBRARY CHECK
# ============================================================

say "2/7 CUDA DRIVER LIBRARY CHECK"

CUDA_DRIVER=$(
    ldconfig -p 2>/dev/null |
    awk '/libcuda\.so\.1 / {print $NF; exit}' || true
)

if [ -z "${CUDA_DRIVER:-}" ] || [ ! -f "$CUDA_DRIVER" ]; then

    for candidate in \
        /usr/local/nvidia/lib64/libcuda.so.1 \
        /usr/local/nvidia/lib64/libcuda.so \
        /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
        /lib/x86_64-linux-gnu/libcuda.so.1
    do
        if [ -f "$candidate" ]; then
            CUDA_DRIVER="$candidate"
            break
        fi
    done

fi

if [ -z "${CUDA_DRIVER:-}" ] || [ ! -f "$CUDA_DRIVER" ]; then
    echo "ERROR: NVIDIA driver library not found."
    echo "Restart Kaggle session and check GPU settings."
    exit 1
fi

CUDA_LINK_DIR="$W/cuda-link"

mkdir -p "$CUDA_LINK_DIR"

ln -sf "$CUDA_DRIVER" "$CUDA_LINK_DIR/libcuda.so"

export LIBRARY_PATH="$CUDA_LINK_DIR:/usr/local/nvidia/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_LINK_DIR:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_LIBCUDA_PATH="$CUDA_LINK_DIR"

echo "CUDA driver: $CUDA_DRIVER"

ls -l "$CUDA_LINK_DIR/libcuda.so"

if ! gcc -print-file-name=libcuda.so | grep -q '^/'; then
    echo "ERROR: Linker cannot find libcuda.so."
    exit 1
fi

echo "CUDA linker check passed."

# ============================================================
# 3. DOWNLOAD PROJECT CODE
# ============================================================

say "3/7 GET PROJECT CODE"

rm -rf "$REPO"

git clone --depth 1 \
    -b "$BRANCH" \
    "$REPO_URL" \
    "$REPO"

cd "$REPO"

echo "Repository cloned successfully."

git log --oneline -1

# ============================================================
# 4. INSTALL DEPENDENCIES
# ============================================================

say "4/7 INSTALL DEPENDENCIES"

if ! python -c "import vllm, cairn" >/dev/null 2>&1; then

    echo "Installing CAIRN and vLLM..."

    pip install -q -e ".[dev]" "vllm==0.12.0"

else

    echo "Dependencies already installed."

fi

python - <<'PY'
import vllm
import cairn

print("vLLM version:", vllm.__version__)
print("CAIRN import: OK")
PY

# ============================================================
# 5. MODEL DOWNLOAD AND CONFIGURATION
# ============================================================

say "5/7 MODEL SETUP"

if [ ! -f "$MODEL_DIR/config.json" ]; then

    mkdir -p "$MODEL_DIR"

    echo "Downloading Qwen3-4B..."

    hf download Qwen/Qwen3-4B \
        --local-dir "$MODEL_DIR"

else

    echo "Model already downloaded."

fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: Model config.json missing."
    exit 1
fi

# Apply YaRN long-context configuration.

python - "$MODEL_DIR/config.json" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path) as f:
    cfg = json.load(f)

want = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}

if cfg.get("rope_scaling") != want:
    cfg["rope_scaling"] = want

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

print("RoPE configuration:", cfg["rope_scaling"])
print("Model configuration ready.")
PY

# ============================================================
# 6. START vLLM SERVER
# ============================================================

say "6/7 START MODEL SERVER"

start_server() {

    local name="$1"
    shift

    local LOG="$W/vllm_${name}.log"

    echo "Starting attempt: $name"
    echo "Arguments: $*"

    if up; then
        echo "Server already running."
        return 0
    fi

    nohup vllm serve "$MODEL_DIR" \
        --served-model-name Qwen3-4B \
        --host 127.0.0.1 \
        --port "$PORT" \
        --dtype half \
        --max-model-len 40960 \
        --enforce-eager \
        --enable-prefix-caching \
        --reasoning-parser qwen3 \
        --enable-prompt-tokens-details \
        "$@" > "$LOG" 2>&1 &

    local pid=$!

    echo "Server PID: $pid"
    echo "Server log: $LOG"

    for i in $(seq 1 90); do

        if up; then
            echo "vLLM server ready."
            return 0
        fi

        if ! kill -0 "$pid" 2>/dev/null; then

            echo "Server crashed: $name"

            grep -iE \
                "error|exception|memory|cuda|failed|nccl|unsupported" \
                "$LOG" | tail -40 || true

            echo "Full log: $LOG"

            return 1
        fi

        sleep 10

        if [ $((i % 6)) -eq 0 ]; then
            echo "Server loading: $((i / 6)) minutes"
        fi

    done

    echo "Server startup timed out."

    kill "$pid" 2>/dev/null || true

    return 1
}

if ! up; then

    start_server A \
        --tensor-parallel-size "$NGPU" \
        --disable-custom-all-reduce \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 16 \
    || start_server B \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 \
        --max-num-seqs 8 \
    || {
        echo "Both server attempts failed."
        echo "Check vllm_A.log and vllm_B.log."
        exit 1
    }

else

    echo "Existing model server detected."

fi

if ! up; then
    echo "ERROR: Model server not responding."
    exit 1
fi

echo "Running CAIRN environment diagnostics..."

cairn --config configs/default.yaml doctor

# ============================================================
# 7. RUN EXPERIMENTS
# ============================================================

say "7/7 RUN CAIRN EXPERIMENTS"

mkdir -p "$RUNS"

echo "Evaluation directory: $RUNS"

for t in $TASKS; do

    echo
    echo "=================================================="
    echo "STARTING TASK $t"
    echo "=================================================="

    if [ "$SPLIT" = "test" ]; then

        cairn --config configs/default.yaml -v run \
            --task "$t" \
            --split test \
            --run-dir "$RUNS" \
        || echo "TASK $t FAILED. Continuing."

    elif [ "$t" = "8" ]; then

        echo "Task 8 uses TEST inputs."
        echo "Task 8 input limit: $LIMIT8"

        cairn --config configs/default.yaml -v run \
            --task 8 \
            --split test \
            --limit "$LIMIT8" \
            --run-dir "$RUNS" \
        || echo "TASK 8 FAILED. Continuing."

    else

        echo "Task $t uses $SPLIT split."
        echo "Input limit: $LIMIT"

        cairn --config configs/default.yaml -v run \
            --task "$t" \
            --split "$SPLIT" \
            --limit "$LIMIT" \
            --run-dir "$RUNS" \
        || echo "TASK $t FAILED. Continuing."

    fi

done

# ============================================================
# FINAL RESULTS SUMMARY
# ============================================================

say "FINAL EXPERIMENT RESULTS"

python - "$RUNS" "$SPLIT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
split = sys.argv[2]

paths = sorted(
    root.glob(f"task*/{split}.report.json")
)

# Task 8 is evaluated using TEST inputs during DEV experiments.
if split != "test":
    paths += sorted(
        root.glob("task8/test.report.json")
    )

if not paths:
    print("No evaluation reports found.")
    sys.exit(0)

for path in paths:

    try:

        r = json.loads(path.read_text())

        t = r.get("task", path.parent.name)
        secs = r.get("seconds", 0)

        if "eval" in r:

            e = r["eval"]

            print(
                f"\nTask {t}: "
                f"Accuracy {e['accuracy']:.1%} "
                f"(n={e['n']}, "
                f"95% CI {e['ci95'][0]:.0%}..{e['ci95'][1]:.0%}) "
                f"| {secs:.0f}s "
                f"| Sources {r.get('sources')}"
            )

        elif "code_quality" in r:

            q = r["code_quality"]

            print(
                f"\nTask {t}: "
                f"Consistency {q['consistency']:.3f} "
                f"| Callable {q['callable']:.0%} "
                f"| PyTorch Match {q['matches_torch']:.0%} "
                f"| Outcomes {q.get('statuses')} "
                f"| {secs:.0f}s"
            )

        else:

            print(
                f"\nTask {t}: "
                f"Predictions {r.get('n')} "
                f"| Sources {r.get('sources')} "
                f"| {secs:.0f}s"
            )

        notes = r.get("notes", {})

        # Program verification

        prog = notes.get("program")

        if isinstance(prog, dict):

            print(
                "Program:",
                f"accepted={prog.get('accepted')}",
                f"attempts={prog.get('attempts')}",
                f"best_verify={prog.get('best_verify_accuracy')}"
            )

        # Policy selection

        pol = notes.get("policy")

        if isinstance(pol, dict):

            print(
                "Policy:",
                f"chosen={pol.get('chosen')}",
                f"cheap={pol.get('cheap_accuracy')}",
                f"thinking={pol.get('costly_accuracy')}",
                f"probe_examples={pol.get('probe_examples')}"
            )

        # Calibration

        cal = notes.get("calibration")

        if isinstance(cal, dict):

            c = cal.get("conformal")

            if isinstance(c, dict):

                print(
                    "Calibration:",
                    f"singleton_rate={c.get('singleton_rate')}",
                    f"singleton_accuracy={c.get('singleton_accuracy')}",
                    f"trusted={c.get('singletons_trusted')}"
                )

    except Exception as error:

        print(f"Could not read {path}: {error}")

print("\nReports directory:", root)
print("Summary completed.")
PY

# ============================================================
# SUBMISSION GENERATION
# ============================================================

if [ "$SPLIT" = "test" ]; then

    say "GENERATING TEST SUBMISSION"

    cairn submit \
        --run-dir "$RUNS" \
        --out "$W/submission.zip"

    echo "Submission saved: $W/submission.zip"

fi

# ============================================================
# COMPLETION
# ============================================================

say "EXPERIMENT COMPLETED"

echo "Main log : $W/cairn_run.log"
echo "vLLM A   : $W/vllm_A.log"
echo "vLLM B   : $W/vllm_B.log"
echo "Reports  : $RUNS"

date

say "DONE"


THE RESULTS:
==================================================
 CAIRN EXPERIMENT STARTED
==================================================
Wed Nov 18 05:45:48 AM UTC 2026
Tasks       : 1 3 4 2 5 6 7 8
Dev limit   : 60
Task 8 limit: 24
Split       : dev
Branch      : claude/ai-project-enhancement-z6ct2n
Model       : /tmp/Qwen3-4B

==================================================
 1/7 GPU CHECK
==================================================
index, name, memory.total [MiB], memory.used [MiB]
0, Tesla T4, 15360 MiB, 0 MiB
1, Tesla T4, 15360 MiB, 0 MiB
Detected GPUs: 2

==================================================
 2/7 CUDA DRIVER LIBRARY CHECK
==================================================
CUDA driver: /usr/local/nvidia/lib64/libcuda.so.1
lrwxrwxrwx 1 root root 36 Nov 18 05:45 /kaggle/working/cuda-link/libcuda.so -> /usr/local/nvidia/lib64/libcuda.so.1
CUDA linker check passed.

==================================================
 3/7 GET PROJECT CODE
==================================================
Cloning into '/kaggle/working/cairn'...
Repository cloned successfully.
7d1dddd Act on the third real run: task 2 counting method, probe power fix, task 8 measured by the CLI

==================================================
 4/7 INSTALL DEPENDENCIES
==================================================
Installing CAIRN and vLLM...
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gradio-client 2.7.2 requires huggingface-hub<3.0,>=1.16.0, but you have huggingface-hub 0.36.2 which is incompatible.
gradio 6.30.0 requires huggingface-hub<3.0,>=1.16.0, but you have huggingface-hub 0.36.2 which is incompatible.
gradio 6.30.0 requires pillow<13.0,>=8.0, but you have pillow 13.0.0 which is incompatible.
diffusers 0.42.0 requires huggingface-hub<3.0,>=1.32.0, but you have huggingface-hub 0.36.2 which is incompatible.
vLLM version: 0.12.0
CAIRN import: OK

==================================================
 5/7 MODEL SETUP
==================================================
Downloading Qwen3-4B...
Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]Downloading 'merges.txt' to '/tmp/Qwen3-4B/.cache/huggingface/download/PtHk0z_I45atnj23IIRhTExwT3w=.31349551d90c7606f325fe0f11bbb8bd5fa0d7c7.incomplete'
Downloading 'model-00001-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/Y6g195DXbsCiN_Ka1NwH6jyLkvc=.328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223.incomplete'
Downloading '.gitattributes' to '/tmp/Qwen3-4B/.cache/huggingface/download/wPaCkH-WbT7GsmxMKKrNZTV4nSM=.52373fe24473b1aa44333d318f578ae6bf04b49b.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/.gitattributes
Downloading 'generation_config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/3EVKVggOldJcKSsGjSdoUCN1AyQ=.20a8a9156fc8c3f25295ca067f61fdf120d517c5.incomplete'
Downloading 'model-00002-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/D5-QTaCbl9VUFv9Wr0C28eFewRc=.6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5.incomplete'
Downloading 'LICENSE' to '/tmp/Qwen3-4B/.cache/huggingface/download/DhCjcNQuMpl4FL346qr3tvNUCgY=.6634c8cc3133b3848ec74b9f275acaaa1ea618ab.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/merges.txt
Download complete. Moving file to /tmp/Qwen3-4B/generation_config.json
Fetching 13 files:   8%|▊         | 1/13 [00:00<00:03,  3.09it/s]Downloading 'README.md' to '/tmp/Qwen3-4B/.cache/huggingface/download/Xn7B-BWUGOee2Y6hCZtEhtFu4BE=.2de5ee7eee214bb55ea33ec7505c5838a7adf7f6.incomplete'
Downloading 'config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/8_PA_wEVGiVa2goH2H4KQOQpvVY=.e49eccdc32f36da9c09cfa0e737084f9e0105e5e.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/LICENSE
Download complete. Moving file to /tmp/Qwen3-4B/README.md
Download complete. Moving file to /tmp/Qwen3-4B/config.json
Downloading 'model-00003-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/NmVQgThqhsRkG2FlXLh-M51MilA=.e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1.incomplete'
Downloading 'tokenizer.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/HgM_lKo9sdSCfRtVg7MMFS7EKqo=.aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4.incomplete'
Downloading 'model.safetensors.index.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/yVzAsSxRSINSz-tQbpx-TLpfkLU=.95c0a0059df040d75dc6c396b174382cf61d2f91.incomplete'
Downloading 'tokenizer_config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/vzaExXFZNBay89bvlQv-ZcI6BTg=.417d038a63fa3de29cfde265caedae14d1a58d92.incomplete'
Downloading 'vocab.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/j3m-Hy6QvBddw8RXA1uSWl1AJ0c=.4783fe10ac3adce15ac8f358ef5462739852c569.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/model.safetensors.index.json
Download complete. Moving file to /tmp/Qwen3-4B/tokenizer_config.json
Download complete. Moving file to /tmp/Qwen3-4B/vocab.json
Download complete. Moving file to /tmp/Qwen3-4B/model-00003-of-00003.safetensors
Download complete. Moving file to /tmp/Qwen3-4B/tokenizer.json
Download complete. Moving file to /tmp/Qwen3-4B/model-00001-of-00003.safetensors
Fetching 13 files:  54%|█████▍    | 7/13 [00:25<00:22,  3.77s/it]Download complete. Moving file to /tmp/Qwen3-4B/model-00002-of-00003.safetensors
Fetching 13 files: 100%|██████████| 13/13 [00:25<00:00,  1.96s/it]
/tmp/Qwen3-4B
RoPE configuration: {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
Model configuration ready.

==================================================
 6/7 START MODEL SERVER
==================================================
Starting attempt: A
Arguments: --tensor-parallel-size 2 --disable-custom-all-reduce --gpu-memory-utilization 0.85 --max-num-seqs 16
Server PID: 214
Server log: /kaggle/working/vllm_A.log
Server loading: 1 minutes
Server loading: 2 minutes
Server loading: 3 minutes
Server loading: 4 minutes
vLLM server ready.
Running CAIRN environment diagnostics...
checking http://127.0.0.1:2026/v1 (model Qwen3-4B)

[PASS] server reachable: models: Qwen3-4B
[PASS] plain answer (thinking off): got 'OK'
[PASS] token probabilities (logprobs): 5 candidates returned
[PASS] constrained answers (structured outputs): works with structured_api: structured_outputs
[PASS] thinking mode: 1323 characters of reasoning, answer '17 + 26 = 43\n\n**Step'
[PASS] prefix cache: second call: 4016 cached tokens, 1.01s -> 0.16s

==================================================
 7/7 RUN CAIRN EXPERIMENTS
==================================================
Evaluation directory: /kaggle/working/runs/dev

==================================================
STARTING TASK 1
==================================================
Task 1 uses dev split.
Input limit: 60
2026-11-18 05:57:41,767 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-18 06:02:05,736 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-18 06:02:05,736 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-18 06:02:05,736 INFO cairn.induction.program: program round 0 gave no code; retrying without thinking
2026-11-18 06:02:11,371 INFO cairn.induction.program: program round 1: 100.0% right on 150 verify examples
2026-11-18 06:02:11,437 INFO cairn.induction.program: program accepted in round 1: 150/150 verify, 150/150 confirm
2026-11-18 06:02:11,437 INFO cairn.pipeline.runner: task 1: program accepted after 3 attempt(s)
task 1 [dev] accuracy 100.0% (60/60, 95% range 94.0%..100.0%), ECE 0.000
    program          100% of inputs, accuracy 100.0%
    program: ACCEPTED after 3 attempt(s); verified on 150 + 150 held-out examples

run folder: /kaggle/working/runs/dev

==================================================
STARTING TASK 3
==================================================
Task 3 uses dev split.
Input limit: 60
2026-11-18 06:02:13,027 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-18 06:06:35,298 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-18 06:06:35,298 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-18 06:06:35,298 INFO cairn.induction.program: program round 0 gave no code; retrying without thinking
2026-11-18 06:06:39,937 INFO cairn.induction.program: program round 1: 100.0% right on 150 verify examples
2026-11-18 06:06:40,002 INFO cairn.induction.program: program accepted in round 1: 150/150 verify, 150/150 confirm
2026-11-18 06:06:40,002 INFO cairn.pipeline.runner: task 3: program accepted after 3 attempt(s)
task 3 [dev] accuracy 100.0% (60/60, 95% range 94.0%..100.0%), ECE 0.000
    program          100% of inputs, accuracy 100.0%
    program: ACCEPTED after 3 attempt(s); verified on 150 + 150 held-out examples

run folder: /kaggle/working/runs/dev

==================================================
STARTING TASK 4
==================================================
Task 4 uses dev split.
Input limit: 60
2026-11-18 06:06:41,552 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-18 06:09:01,077 INFO cairn.induction.program: program round 0: 100.0% right on 150 verify examples
2026-11-18 06:09:01,148 INFO cairn.induction.program: program accepted in round 0: 150/150 verify, 150/150 confirm
2026-11-18 06:09:01,149 INFO cairn.pipeline.runner: task 4: program accepted after 1 attempt(s)
task 4 [dev] accuracy 100.0% (60/60, 95% range 94.0%..100.0%), ECE 0.000
    program          100% of inputs, accuracy 100.0%
    program: ACCEPTED after 1 attempt(s); verified on 150 + 150 held-out examples

run folder: /kaggle/working/runs/dev

==================================================
STARTING TASK 2
==================================================
Task 2 uses dev split.
Input limit: 60
2026-11-18 06:09:02,741 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-18 07:02:49,494 INFO cairn.pipeline.runner: task 2: escalation probe on 40 held-out inputs: cheap 32%, thinking 70% -> reason_first
task 2 [dev] accuracy 68.3% (41/60, 95% range 55.8%..78.7%), ECE 0.202
    generate         20% of inputs, accuracy 41.7%
    think_vote       80% of inputs, accuracy 75.0%
    policy: REASON_FIRST from a probe on 40 held-out inputs: cheap 32%, thinking 70%
        all    n= 40 cheap 32% vs thinking 70% (only-cheap 2, only-thinking 17, p=0.001)
        sure   n= 35 cheap 34% vs thinking 66% (only-cheap 2, only-thinking 13, p=0.007)
        unsure n=  5 cheap 20% vs thinking 100% (only-cheap 0, only-thinking 4, p=0.125)

run folder: /kaggle/working/runs/dev

==================================================
STARTING TASK 5
==================================================
Task 5 uses dev split.
Input limit: 60
2026-11-18 08:17:43,280 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-18 08:39:36,233 INFO cairn.pipeline.runner: task 5: escalation probe on 40 held-out inputs: cheap 75%, thinking 82% -> cheap_only
task 5 [dev] accuracy 85.0% (51/60, 95% range 73.9%..91.9%), ECE 0.100
    score            100% of inputs, accuracy 85.0%
    policy: CHEAP_ONLY from a probe on 40 held-out inputs: cheap 75%, thinking 82%
        all    n= 40 cheap 75% vs thinking 82% (only-cheap 2, only-thinking 5, p=0.453)
        sure   n= 33 cheap 79% vs thinking 82% (only-cheap 2, only-thinking 3, p=1.000)
        unsure n=  7 cheap 57% vs thinking 86% (only-cheap 0, only-thinking 2, p=0.500)
    calibration: stack on 124 examples, held-out accuracy 78.4% -> 80.0%; sure on 76% of held-out inputs, right on 88% of those

run folder: /kaggle/working/runs/dev

==================================================
STARTING TASK 6
==================================================
Task 6 uses dev split.
Input limit: 60
2026-11-18 08:40:00,783 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-18 09:16:03,701 INFO cairn.pipeline.runner: task 6: escalation probe on 40 held-out inputs: cheap 95%, thinking 75% -> cheap_only
task 6 [dev] accuracy 96.7% (58/60, 95% range 88.6%..99.1%), ECE 0.026
    score            100% of inputs, accuracy 96.7%
    policy: CHEAP_ONLY from a probe on 40 held-out inputs: cheap 95%, thinking 75%
        all    n= 40 cheap 95% vs thinking 75% (only-cheap 9, only-thinking 1, p=0.021)
        sure   n= 38 cheap 95% vs thinking 74% (only-cheap 9, only-thinking 1, p=0.021)
        unsure n=  2 cheap 100% vs thinking 100% (only-cheap 0, only-thinking 0, p=1.000)
    calibration: stack on 150 examples, held-out accuracy 74.0% -> 94.7%; sure on 95% of held-out inputs, right on 95% of those

run folder: /kaggle/working/runs/dev

==================================================
STARTING TASK 7
==================================================
Task 7 uses dev split.
Input limit: 60
2026-11-18 09:16:43,399 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-18 09:50:39,415 INFO cairn.pipeline.runner: task 7: escalation probe on 40 held-out inputs: cheap 20%, thinking 35% -> cheap_only
task 7 [dev] accuracy 16.7% (10/60, 95% range 9.3%..28.0%), ECE 0.629
    fallback_nearest 2% of inputs, accuracy 0.0%
    generate         98% of inputs, accuracy 17.0%
    policy: CHEAP_ONLY from a probe on 40 held-out inputs: cheap 20%, thinking 35%
        all    n= 40 cheap 20% vs thinking 35% (only-cheap 3, only-thinking 9, p=0.146)
        sure   n= 29 cheap 28% vs thinking 41% (only-cheap 3, only-thinking 7, p=0.344)
        unsure n= 11 cheap 0% vs thinking 18% (only-cheap 0, only-thinking 2, p=0.500)

run folder: /kaggle/working/runs/dev

==================================================
STARTING TASK 8
==================================================
Task 8 uses TEST inputs.
Task 8 input limit: 24
2026-11-18 09:52:07,815 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
task 8 [test] static check OK 0%, runs 0%, matches PyTorch 0%, consistency score 0.000 (n=24)
    execution outcomes: {'import_error': 22, 'not_run': 2}
task 8 [test] wrote 24 predictions, sources: {'codegen': 24}

run folder: /kaggle/working/runs/dev

==================================================
 FINAL EXPERIMENT RESULTS
==================================================

Task 1: Accuracy 100.0% (n=60, 95% CI 94%..100%) | 270s | Sources {'program': 60}
Program: accepted=True attempts=3 best_verify=1.0

Task 2: Accuracy 68.3% (n=60, 95% CI 56%..79%) | 7719s | Sources {'think_vote': 48, 'generate': 12}
Policy: chosen=reason_first cheap=0.325 thinking=0.7 probe_examples=40

Task 3: Accuracy 100.0% (n=60, 95% CI 94%..100%) | 267s | Sources {'program': 60}
Program: accepted=True attempts=3 best_verify=1.0

Task 4: Accuracy 100.0% (n=60, 95% CI 94%..100%) | 140s | Sources {'program': 60}
Program: accepted=True attempts=1 best_verify=1.0

Task 5: Accuracy 85.0% (n=60, 95% CI 74%..92%) | 1336s | Sources {'score': 60}
Policy: chosen=cheap_only cheap=0.75 thinking=0.825 probe_examples=40
Calibration: singleton_rate=0.76 singleton_accuracy=0.8842 trusted=True

Task 6: Accuracy 96.7% (n=60, 95% CI 89%..99%) | 2202s | Sources {'score': 60}
Policy: chosen=cheap_only cheap=0.95 thinking=0.75 probe_examples=40
Calibration: singleton_rate=0.9533 singleton_accuracy=0.951 trusted=True

Task 7: Accuracy 16.7% (n=60, 95% CI 9%..28%) | 2123s | Sources {'generate': 59, 'fallback_nearest': 1}
Policy: chosen=cheap_only cheap=0.2 thinking=0.35 probe_examples=40

Task 8: Consistency 0.000 | Callable 0% | PyTorch Match 0% | Outcomes {'import_error': 22, 'not_run': 2} | 1949s

Reports directory: /kaggle/working/runs/dev
Summary completed.

==================================================
 EXPERIMENT COMPLETED
==================================================
Main log : /kaggle/working/cairn_run.log
vLLM A   : /kaggle/working/vllm_A.log
vLLM B   : /kaggle/working/vllm_B.log
Reports  : /kaggle/working/runs/dev
Wed Nov 18 10:24:37 AM UTC 2026

==================================================
 DONE
==================================================