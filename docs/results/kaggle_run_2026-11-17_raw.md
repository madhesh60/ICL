this is thr second kaggle run:
code:
%%bash
#!/usr/bin/env bash

# ============================================================
# CAIRN + Qwen3-4B | KAGGLE EVALUATION
# ============================================================

# -------------------- SETTINGS -------------------------------

TASKS="${TASKS:-1 3 4 2 5 6 7 8}"
LIMIT="${LIMIT:-60}"
SPLIT="${SPLIT:-dev}"

BRANCH="${BRANCH:-claude/ai-project-enhancement-z6ct2n}"
REPO_URL="${REPO_URL:-https://github.com/madhesh60/ICL.git}"

W="/kaggle/working"
REPO="$W/cairn"
MODEL_DIR="${MODEL_DIR:-/tmp/Qwen3-4B}"
RUNS="$W/runs/$SPLIT"
PORT=2026

# -------------------- ENVIRONMENT -----------------------------

set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CAIRN_TRITON_INTERPRET=0

mkdir -p "$W"

# Save main execution logs while displaying them.
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

say "CAIRN EVALUATION STARTED"

date
echo "Tasks: $TASKS"
echo "Limit: $LIMIT"
echo "Split: $SPLIT"

# ============================================================
# 1. GPU CHECK
# ============================================================

say "1/7 GPU CHECK"

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: NVIDIA GPU unavailable."
    echo "Enable GPU in Kaggle settings."
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
    exit 1
fi

CUDA_LINK_DIR="$W/cuda-link"

mkdir -p "$CUDA_LINK_DIR"

ln -sf "$CUDA_DRIVER" "$CUDA_LINK_DIR/libcuda.so"

export LIBRARY_PATH="$CUDA_LINK_DIR:/usr/local/nvidia/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_LINK_DIR:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_LIBCUDA_PATH="$CUDA_LINK_DIR"

echo "CUDA driver: $CUDA_DRIVER"
echo "CUDA linker directory: $CUDA_LINK_DIR"

ls -l "$CUDA_LINK_DIR/libcuda.so"

if ! gcc -print-file-name=libcuda.so | grep -q '^/'; then
    echo "ERROR: Linker cannot locate libcuda.so."
    exit 1
fi

echo "CUDA linker check passed."

# ============================================================
# 3. DOWNLOAD PROJECT
# ============================================================

say "3/7 GET PROJECT CODE"

rm -rf "$REPO"

git clone --depth 1 \
    -b "$BRANCH" \
    "$REPO_URL" \
    "$REPO"

cd "$REPO"

echo "Repository downloaded successfully."

git log --oneline -1

# ============================================================
# 4. INSTALL DEPENDENCIES
# ============================================================

say "4/7 INSTALL DEPENDENCIES"

if ! python -c "import vllm, cairn" >/dev/null 2>&1; then

    echo "Installing CAIRN and vLLM..."

    pip install -q -e ".[dev]" "vllm==0.12.0"

fi

python - <<'PY'
import vllm
import cairn

print("vLLM version:", vllm.__version__)
print("CAIRN import: successful")
PY

# ============================================================
# 5. MODEL DOWNLOAD AND CONFIGURATION
# ============================================================

say "5/7 MODEL SETUP"

if [ ! -f "$MODEL_DIR/config.json" ]; then

    mkdir -p "$MODEL_DIR"

    hf download Qwen/Qwen3-4B \
        --local-dir "$MODEL_DIR"

else
    echo "Model already downloaded."
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: Model config missing."
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
            echo "vLLM server is ready."
            return 0
        fi

        if ! kill -0 "$pid" 2>/dev/null; then

            echo "Server crashed: $name"

            echo "Last error messages:"

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
        echo "Inspect vllm_A.log and vllm_B.log."
        exit 1
    }

else
    echo "Existing vLLM server detected."
fi

if ! up; then
    echo "ERROR: Model server is not responding."
    exit 1
fi

echo "Testing CAIRN environment..."

cairn --config configs/default.yaml doctor

# ============================================================
# 7. RUN CAIRN EVALUATION
# ============================================================

say "7/7 RUN EVALUATION"

mkdir -p "$RUNS"

echo "Split: $SPLIT"
echo "Tasks: $TASKS"
echo "Questions per task: $LIMIT"

for t in $TASKS; do

    echo
    echo "------------------------------------------"
    echo "STARTING TASK $t"
    echo "------------------------------------------"

    if [ "$SPLIT" = "test" ]; then

        if cairn --config configs/default.yaml -v run \
            --task "$t" \
            --split test \
            --run-dir "$RUNS"; then

            echo "Task $t completed."

        else
            echo "Task $t FAILED. Continuing."
        fi

    else

        if cairn --config configs/default.yaml -v run \
            --task "$t" \
            --split "$SPLIT" \
            --limit "$LIMIT" \
            --run-dir "$RUNS"; then

            echo "Task $t completed."

        else
            echo "Task $t FAILED. Continuing."
        fi

    fi

done

# ============================================================
# FINAL RESULTS
# ============================================================

say "FINAL EVALUATION SUMMARY"

python - "$RUNS" "$SPLIT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
split = sys.argv[2]

reports = sorted(
    root.glob(f"task*/{split}.report.json")
)

if not reports:
    print("No evaluation reports found.")
    sys.exit(0)

for path in reports:

    try:

        r = json.loads(path.read_text())

        task = r.get("task", path.parent.name)
        secs = r.get("seconds", 0)

        if "eval" in r:

            e = r["eval"]

            print(
                f"\nTask {task}: "
                f"Accuracy {e['accuracy']:.1%} "
                f"(n={e['n']}, "
                f"95% CI {e['ci95'][0]:.0%}..{e['ci95'][1]:.0%}) "
                f"| {secs:.0f}s"
            )

        elif "code_quality" in r:

            q = r["code_quality"]

            print(
                f"\nTask {task}: "
                f"Consistency {q['consistency']:.3f} "
                f"| Callable {q['callable']:.0%} "
                f"| PyTorch Match {q['matches_torch']:.0%} "
                f"| {secs:.0f}s"
            )

            print("Outcomes:", q.get("statuses"))

        else:

            print(
                f"\nTask {task}: "
                f"Predictions {r.get('n')} "
                f"| Time {secs:.0f}s"
            )

        # Program verification information

        notes = r.get("notes", {})

        prog = notes.get("program")

        if isinstance(prog, dict):

            print(
                "Program:",
                f"accepted={prog.get('accepted')}",
                f"attempts={prog.get('attempts')}",
                f"best_verify={prog.get('best_verify_accuracy')}"
            )

        # Calibration information

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
# TEST SPLIT SUBMISSION
# ============================================================

if [ "$SPLIT" = "test" ]; then

    say "GENERATING SUBMISSION"

    cairn submit \
        --run-dir "$RUNS" \
        --out "$W/submission.zip"

    echo "Submission: $W/submission.zip"

fi

# ============================================================
# COMPLETION
# ============================================================

say "EVALUATION FINISHED"

echo "Main logs: $W/cairn_run.log"
echo "vLLM attempt A logs: $W/vllm_A.log"
echo "vLLM attempt B logs: $W/vllm_B.log"
echo "Reports: $RUNS"

date


RESULTS:
==================================================
 CAIRN EVALUATION STARTED
==================================================
Tue Nov 17 07:25:41 AM UTC 2026
Tasks: 1 3 4 2 5 6 7 8
Limit: 60
Split: dev

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
CUDA linker directory: /kaggle/working/cuda-link
lrwxrwxrwx 1 root root 36 Nov 17 07:25 /kaggle/working/cuda-link/libcuda.so -> /usr/local/nvidia/lib64/libcuda.so.1
CUDA linker check passed.

==================================================
 3/7 GET PROJECT CODE
==================================================
Cloning into '/kaggle/working/cairn'...
Repository downloaded successfully.
e34413c Fix what the first real Qwen3-4B run exposed; document the run and how to reason from it

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
CAIRN import: successful

==================================================
 5/7 MODEL SETUP
==================================================
Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]Downloading 'model-00001-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/Y6g195DXbsCiN_Ka1NwH6jyLkvc=.328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223.incomplete'
Downloading 'generation_config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/3EVKVggOldJcKSsGjSdoUCN1AyQ=.20a8a9156fc8c3f25295ca067f61fdf120d517c5.incomplete'
Downloading 'model-00002-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/D5-QTaCbl9VUFv9Wr0C28eFewRc=.6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/generation_config.json
Downloading 'config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/8_PA_wEVGiVa2goH2H4KQOQpvVY=.e49eccdc32f36da9c09cfa0e737084f9e0105e5e.incomplete'
Downloading 'merges.txt' to '/tmp/Qwen3-4B/.cache/huggingface/download/PtHk0z_I45atnj23IIRhTExwT3w=.31349551d90c7606f325fe0f11bbb8bd5fa0d7c7.incomplete'
Downloading '.gitattributes' to '/tmp/Qwen3-4B/.cache/huggingface/download/wPaCkH-WbT7GsmxMKKrNZTV4nSM=.52373fe24473b1aa44333d318f578ae6bf04b49b.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/config.json
Download complete. Moving file to /tmp/Qwen3-4B/.gitattributes
Fetching 13 files:   8%|▊         | 1/13 [00:00<00:03,  3.14it/s]Downloading 'README.md' to '/tmp/Qwen3-4B/.cache/huggingface/download/Xn7B-BWUGOee2Y6hCZtEhtFu4BE=.2de5ee7eee214bb55ea33ec7505c5838a7adf7f6.incomplete'
Downloading 'LICENSE' to '/tmp/Qwen3-4B/.cache/huggingface/download/DhCjcNQuMpl4FL346qr3tvNUCgY=.6634c8cc3133b3848ec74b9f275acaaa1ea618ab.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/README.md
Download complete. Moving file to /tmp/Qwen3-4B/LICENSE
Download complete. Moving file to /tmp/Qwen3-4B/merges.txt
Downloading 'model-00003-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/NmVQgThqhsRkG2FlXLh-M51MilA=.e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1.incomplete'
Downloading 'tokenizer.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/HgM_lKo9sdSCfRtVg7MMFS7EKqo=.aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4.incomplete'
Downloading 'model.safetensors.index.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/yVzAsSxRSINSz-tQbpx-TLpfkLU=.95c0a0059df040d75dc6c396b174382cf61d2f91.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/model.safetensors.index.json
Downloading 'vocab.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/j3m-Hy6QvBddw8RXA1uSWl1AJ0c=.4783fe10ac3adce15ac8f358ef5462739852c569.incomplete'
Downloading 'tokenizer_config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/vzaExXFZNBay89bvlQv-ZcI6BTg=.417d038a63fa3de29cfde265caedae14d1a58d92.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/tokenizer_config.json
Download complete. Moving file to /tmp/Qwen3-4B/vocab.json
Download complete. Moving file to /tmp/Qwen3-4B/tokenizer.json
Download complete. Moving file to /tmp/Qwen3-4B/model-00003-of-00003.safetensors
Download complete. Moving file to /tmp/Qwen3-4B/model-00001-of-00003.safetensors
Fetching 13 files:  54%|█████▍    | 7/13 [00:28<00:25,  4.31s/it]Download complete. Moving file to /tmp/Qwen3-4B/model-00002-of-00003.safetensors
Fetching 13 files: 100%|██████████| 13/13 [00:29<00:00,  2.25s/it]
/tmp/Qwen3-4B
RoPE configuration: {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
Model configuration ready.

==================================================
 6/7 START MODEL SERVER
==================================================
Starting attempt: A
Arguments: --tensor-parallel-size 2 --disable-custom-all-reduce --gpu-memory-utilization 0.85 --max-num-seqs 16
Server PID: 207
Server log: /kaggle/working/vllm_A.log
Server loading: 1 minutes
Server loading: 2 minutes
Server loading: 3 minutes
Server loading: 4 minutes
vLLM server is ready.
Testing CAIRN environment...
checking http://127.0.0.1:2026/v1 (model Qwen3-4B)

[PASS] server reachable: models: Qwen3-4B
[PASS] plain answer (thinking off): got 'OK'
[PASS] token probabilities (logprobs): 5 candidates returned
[PASS] constrained answers (structured outputs): works with structured_api: structured_outputs
[PASS] thinking mode: 1323 characters of reasoning, answer '17 + 26 = 43\n\n**Step'
[PASS] prefix cache: second call: 4016 cached tokens, 1.08s -> 0.15s

==================================================
 7/7 RUN EVALUATION
==================================================
Split: dev
Tasks: 1 3 4 2 5 6 7 8
Questions per task: 60

------------------------------------------
STARTING TASK 1
------------------------------------------
2026-11-17 07:36:07,751 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 07:40:49,050 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-17 07:40:49,110 INFO cairn.induction.program: program round 0: 100.0% right on 150 verify examples
2026-11-17 07:40:49,166 INFO cairn.induction.program: program accepted in round 0: 150/150 verify, 150/150 confirm
2026-11-17 07:40:49,166 INFO cairn.pipeline.runner: task 1: program accepted after 2 attempt(s)
task 1 [dev] accuracy 100.0% (60/60, 95% range 94.0%..100.0%), ECE 0.000
    program          100% of inputs, accuracy 100.0%
    program: ACCEPTED after 2 attempt(s); verified on 150 + 150 held-out examples

run folder: /kaggle/working/runs/dev
Task 1 completed.

------------------------------------------
STARTING TASK 3
------------------------------------------
2026-11-17 07:40:50,742 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 07:45:22,435 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-17 07:45:22,436 INFO cairn.induction.program: program round 0: 0.0% right on 150 verify examples; first failure: "Input: [183, 82, 79, 4, 13, 62]\nExpected: [550, 41, 238, 2, 40, 31]\nGot: ERROR rejected: use of 'eval' is not allowed"
2026-11-17 07:47:28,247 INFO cairn.induction.program: program round 1: 100.0% right on 150 verify examples
2026-11-17 07:47:28,324 INFO cairn.induction.program: program accepted in round 1: 150/150 verify, 150/150 confirm
2026-11-17 07:47:28,324 INFO cairn.pipeline.runner: task 3: program accepted after 3 attempt(s)
task 3 [dev] accuracy 100.0% (60/60, 95% range 94.0%..100.0%), ECE 0.000
    program          100% of inputs, accuracy 100.0%
    program: ACCEPTED after 3 attempt(s); verified on 150 + 150 held-out examples

run folder: /kaggle/working/runs/dev
Task 3 completed.

------------------------------------------
STARTING TASK 4
------------------------------------------
2026-11-17 07:47:30,052 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 07:51:55,807 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-17 07:51:55,808 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-17 07:51:55,808 INFO cairn.induction.program: program round 0 gave no code; retrying without thinking
2026-11-17 07:51:57,197 INFO cairn.induction.program: program round 1: 0.0% right on 150 verify examples; first failure: "Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']\nExpected: cuttingyTthePtheZbNonlyLlossD\nGot: ['cutting', 'y', 'T', 'the"
2026-11-17 07:51:57,246 INFO cairn.induction.program: program round 1: 0.0% right on 150 verify examples; first failure: "Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']\nExpected: cuttingyTthePtheZbNonlyLlossD\nGot: ['cutting', 'y', 'T', 'the"
2026-11-17 07:51:58,454 INFO cairn.induction.program: program round 2: 0.0% right on 150 verify examples; first failure: "Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']\nExpected: cuttingyTthePtheZbNonlyLlossD\nGot: ['cutting', 'y', 'T', 'the"
2026-11-17 07:51:58,502 INFO cairn.induction.program: program round 2: 0.0% right on 150 verify examples; first failure: "Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']\nExpected: cuttingyTthePtheZbNonlyLlossD\nGot: ['cutting', 'y', 'T', 'the"
2026-11-17 07:51:59,652 INFO cairn.induction.program: program round 3: 0.0% right on 150 verify examples; first failure: "Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']\nExpected: cuttingyTthePtheZbNonlyLlossD\nGot: ['cutting', 'y', 'T', 'the"
2026-11-17 07:51:59,706 INFO cairn.induction.program: program round 3: 0.0% right on 150 verify examples; first failure: "Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']\nExpected: cuttingyTthePtheZbNonlyLlossD\nGot: ['cutting', 'y', 'T', 'the"
2026-11-17 07:51:59,706 WARNING cairn.pipeline.runner: task 4: no program accepted after 8 attempt(s) (best verify accuracy 0.0%); falling back to in-context labelling
2026-11-17 08:03:30,735 WARNING cairn.llm.openai_compat: call failed (ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=2026): Read timed out. (read timeout=600)); retry 1 in 0.8s
2026-11-17 08:04:49,403 WARNING cairn.llm.openai_compat: call failed (ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=2026): Read timed out. (read timeout=600)); retry 1 in 0.8s
2026-11-17 08:18:34,930 WARNING cairn.llm.openai_compat: call failed (ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=2026): Read timed out. (read timeout=600)); retry 1 in 0.4s
2026-11-17 08:19:05,968 WARNING cairn.llm.openai_compat: call failed (ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=2026): Read timed out. (read timeout=600)); retry 1 in 0.3s
2026-11-17 08:24:27,521 WARNING cairn.llm.openai_compat: call failed (ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=2026): Read timed out. (read timeout=600)); retry 1 in 0.5s
2026-11-17 08:25:51,313 WARNING cairn.llm.openai_compat: call failed (ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=2026): Read timed out. (read timeout=600)); retry 1 in 0.4s
2026-11-17 08:32:20,530 WARNING cairn.llm.openai_compat: call failed (ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=2026): Read timed out. (read timeout=600)); retry 1 in 0.8s
task 4 [dev] accuracy 80.0% (48/60, 95% range 68.2%..88.2%), ECE 0.068
    think_vote       100% of inputs, accuracy 80.0%
    program: not accepted after 8 attempt(s), best verify accuracy 0.0%; answers came from the model
        round 0 thinking=True finish=length verify=0.0% token budget ran out before any code
        round 0 thinking=True finish=length verify=0.0% token budget ran out before any code
        round 1 thinking=False finish=stop verify=0.0% Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']
E
        round 1 thinking=False finish=stop verify=0.0% Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']
E
        round 2 thinking=False finish=stop verify=0.0% Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']
E
        round 2 thinking=False finish=stop verify=0.0% Input: ['cutting', 'y', 'T', 'the', 'P', 'the', 'Z', 'b', 'N', 'only', 'L', 'loss', 'D']
E

run folder: /kaggle/working/runs/dev
Task 4 completed.

------------------------------------------
STARTING TASK 2
------------------------------------------
2026-11-17 08:51:22,165 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
task 2 [dev] accuracy 45.0% (27/60, 95% range 33.1%..57.5%), ECE 0.550
    generate         97% of inputs, accuracy 43.1%
    think_vote       3% of inputs, accuracy 100.0%

run folder: /kaggle/working/runs/dev
Task 2 completed.

------------------------------------------
STARTING TASK 5
------------------------------------------
2026-11-17 08:53:46,594 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
task 5 [dev] accuracy 85.0% (51/60, 95% range 73.9%..91.9%), ECE 0.100
    score            100% of inputs, accuracy 85.0%
    calibration: stack on 124 examples, held-out accuracy 78.4% -> 80.0%; sure on 76% of held-out inputs, right on 88% of those

run folder: /kaggle/working/runs/dev
Task 5 completed.

------------------------------------------
STARTING TASK 6
------------------------------------------
2026-11-17 08:55:56,240 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
task 6 [dev] accuracy 93.3% (56/60, 95% range 84.1%..97.4%), ECE 0.035
    score            93% of inputs, accuracy 96.4%
    think_vote       7% of inputs, accuracy 50.0%
    calibration: stack on 150 examples, held-out accuracy 74.0% -> 94.7%; sure on 95% of held-out inputs, right on 95% of those

run folder: /kaggle/working/runs/dev
Task 6 completed.

------------------------------------------
STARTING TASK 7
------------------------------------------
2026-11-17 09:02:45,481 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
task 7 [dev] accuracy 20.0% (12/60, 95% range 11.8%..31.8%), ECE 0.633
    generate         68% of inputs, accuracy 19.5%
    think_vote       32% of inputs, accuracy 21.1%

run folder: /kaggle/working/runs/dev
Task 7 completed.

------------------------------------------
STARTING TASK 8
------------------------------------------
2026-11-17 09:18:28,002 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
task 8 [dev] static check OK 96%, runs 0%, matches PyTorch 0%, consistency score 0.000 (n=24)
    execution outcomes: {'not_run': 24}

run folder: /kaggle/working/runs/dev
Task 8 completed.

==================================================
 FINAL EVALUATION SUMMARY
==================================================

Task 1: Accuracy 100.0% (n=60, 95% CI 94%..100%) | 282s
Program: accepted=True attempts=2 best_verify=1.0

Task 2: Accuracy 45.0% (n=60, 95% CI 33%..58%) | 143s

Task 3: Accuracy 100.0% (n=60, 95% CI 94%..100%) | 398s
Program: accepted=True attempts=3 best_verify=1.0

Task 4: Accuracy 80.0% (n=60, 95% CI 68%..88%) | 3831s
Program: accepted=False attempts=8 best_verify=0.0

Task 5: Accuracy 85.0% (n=60, 95% CI 74%..92%) | 128s
Calibration: singleton_rate=0.76 singleton_accuracy=0.8842 trusted=True

Task 6: Accuracy 93.3% (n=60, 95% CI 84%..97%) | 408s
Calibration: singleton_rate=0.9533 singleton_accuracy=0.951 trusted=True

Task 7: Accuracy 20.0% (n=60, 95% CI 12%..32%) | 941s

Task 8: Consistency 0.000 | Callable 0% | PyTorch Match 0% | 1315s
Outcomes: {'not_run': 24}

Reports directory: /kaggle/working/runs/dev
Summary completed.

==================================================
 EVALUATION FINISHED
==================================================
Main logs: /kaggle/working/cairn_run.log
vLLM attempt A logs: /kaggle/working/vllm_A.log
vLLM attempt B logs: /kaggle/working/vllm_B.log
Reports: /kaggle/working/runs/dev
Tue Nov 17 09:40:23 AM UTC 2026