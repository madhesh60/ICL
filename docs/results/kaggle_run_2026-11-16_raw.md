THE CODE IN NOTEBOOK : 
%%bash
# ============================================================
# CAIRN + Qwen3-4B | FULL KAGGLE EVALUATION
# ============================================================

# SETTINGS
TASKS="1 2 3 4 5 6 7 8"
LIMIT=20
BRANCH="claude/ai-project-enhancement-z6ct2n"

# PATHS
W="/kaggle/working"
REPO="$W/cairn"
MODEL_DIR="/tmp/Qwen3-4B"
RUNS="$W/runs/dev"
PORT=2026

set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export CAIRN_TRITON_INTERPRET=0

say() {
    printf '\n==================================================\n'
    printf ' %s\n' "$*"
    printf '==================================================\n'
}

up() {
    curl -sf --connect-timeout 3 \
        "http://127.0.0.1:$PORT/v1/models" >/dev/null
}

# ============================================================
# 1. GPU CHECK
# ============================================================

say "1/7 GPU CHECK"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU unavailable."
    echo "Enable GPU in Kaggle Notebook settings."
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv

NGPU=$(nvidia-smi -L | wc -l)

if [ "$NGPU" -lt 1 ]; then
    echo "ERROR: No GPU detected."
    exit 1
fi

echo "Detected GPUs: $NGPU"

# ============================================================
# 2. CUDA DRIVER LIBRARY FIX
# ============================================================

say "2/7 CUDA DRIVER LIBRARY CHECK"

CUDA_DRIVER=$(ldconfig -p 2>/dev/null \
    | awk '/libcuda\.so\.1 / {print $NF; exit}' || true)

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

echo "Found CUDA driver: $CUDA_DRIVER"

CUDA_LINK_DIR="$W/cuda-link"
mkdir -p "$CUDA_LINK_DIR"

ln -sf "$CUDA_DRIVER" "$CUDA_LINK_DIR/libcuda.so"

export LIBRARY_PATH="$CUDA_LINK_DIR:/usr/local/nvidia/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_LINK_DIR:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_LIBCUDA_PATH="$CUDA_LINK_DIR"

echo "CUDA library paths configured."

ls -l "$CUDA_LINK_DIR/libcuda.so"

if ! gcc -print-file-name=libcuda.so \
    | grep -q '^/'; then
    echo "ERROR: Linker cannot locate libcuda.so."
    exit 1
fi

echo "CUDA linker check passed."

# ============================================================
# 3. GET PROJECT CODE
# ============================================================

say "3/7 GET PROJECT CODE"

rm -rf "$REPO"

git clone --depth 1 \
    -b "$BRANCH" \
    https://github.com/madhesh60/ICL.git \
    "$REPO"

cd "$REPO"

echo "Repository downloaded."
echo "Branch: $BRANCH"

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

print("vLLM:", vllm.__version__)
print("CAIRN import: successful")
PY

# ============================================================
# 5. DOWNLOAD AND CONFIGURE QWEN3-4B
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
    echo "ERROR: Model config.json missing."
    exit 1
fi

# Apply YaRN configuration while preserving original config.
python - "$MODEL_DIR/config.json" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path) as f:
    config = json.load(f)

config["rope_scaling"] = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}

with open(path, "w") as f:
    json.dump(config, f, indent=2)

print("Qwen3-4B configuration updated.")
print("RoPE scaling: YaRN")
print("Scaling factor: 4.0")
PY

# ============================================================
# 6. START vLLM SERVER
# ============================================================

say "6/7 START MODEL SERVER"

SERVER_PID=""

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

    SERVER_PID=$!

    for i in $(seq 1 90); do

        if up; then
            echo "Server is ready."
            echo "PID: $SERVER_PID"
            return 0
        fi

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then

            echo "Server crashed: $name"
            echo "Relevant error logs:"

            grep -iE \
                "error|exception|memory|not supported|nccl|cuda|raise|insufficient|failed" \
                "$LOG" | tail -40 || true

            echo "Full log: $LOG"
            return 1
        fi

        sleep 10

        if [ $((i % 6)) -eq 0 ]; then
            echo "Loading... $((i / 6)) minutes"
        fi

    done

    echo "Server startup timed out."

    kill "$SERVER_PID" 2>/dev/null || true

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
        echo "Check:"
        echo "$W/vllm_A.log"
        echo "$W/vllm_B.log"

        exit 1
    }

else
    echo "Existing model server detected."
fi

if ! up; then
    echo "ERROR: Model API is not responding."
    exit 1
fi

echo "Testing CAIRN environment..."

cairn --config configs/default.yaml doctor

# ============================================================
# 7. RUN EVALUATION TASKS
# ============================================================

say "7/7 RUN EVALUATION"

mkdir -p "$RUNS"

for t in $TASKS; do

    echo
    echo "------------------------------------------"
    echo "Running task: $t"
    echo "------------------------------------------"

    if cairn --config configs/default.yaml -v run \
        --task "$t" \
        --split dev \
        --limit "$LIMIT" \
        --run-dir "$RUNS"; then

        echo "Task $t completed."

    else
        echo "Task $t FAILED. Continuing."
    fi

done

# ============================================================
# FINAL RESULTS
# ============================================================

say "FINAL EVALUATION RESULTS"

python - "$RUNS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])

reports = sorted(
    root.glob("task*/dev.report.json")
)

if not reports:
    print("No evaluation reports found.")
    sys.exit(0)

for path in reports:

    try:

        report = json.loads(path.read_text())

        task = report.get("task", path.parent.name)
        seconds = report.get("seconds", 0)

        if "eval" in report:

            e = report["eval"]

            print(
                f"Task {task}: "
                f"Accuracy {e['accuracy']:.1%} | "
                f"N={e['n']} | "
                f"95% CI "
                f"{e['ci95'][0]:.0%}..{e['ci95'][1]:.0%} | "
                f"Time {seconds:.0f}s"
            )

        elif "code_quality" in report:

            q = report["code_quality"]

            print(
                f"Task {task}: "
                f"Consistency {q['consistency']:.3f} | "
                f"Callable {q['callable']:.0%} | "
                f"PyTorch Match {q['matches_torch']:.0%} | "
                f"N={q['n']} | "
                f"Time {seconds:.0f}s"
            )

        else:
            print(
                f"Task {task}: Report format not recognized."
            )

    except Exception as error:

        print(f"Could not read {path}: {error}")

print("\nReports directory:", root)
print("Evaluation summary completed.")
PY

say "ALL REQUESTED TASKS FINISHED"


RESULTS:
=================================================
 1/7 GPU CHECK
==================================================
index, name, memory.total [MiB], memory.used [MiB]
0, Tesla T4, 15360 MiB, 0 MiB
1, Tesla T4, 15360 MiB, 0 MiB
Detected GPUs: 2

==================================================
 2/7 CUDA DRIVER LIBRARY CHECK
==================================================
Found CUDA driver: /usr/local/nvidia/lib64/libcuda.so.1
CUDA library paths configured.
lrwxrwxrwx 1 root root 36 Nov 16 11:55 /kaggle/working/cuda-link/libcuda.so -> /usr/local/nvidia/lib64/libcuda.so.1
CUDA linker check passed.

==================================================
 3/7 GET PROJECT CODE
==================================================
Repository downloaded.
Branch: claude/ai-project-enhancement-z6ct2n

==================================================
 4/7 INSTALL DEPENDENCIES
==================================================
Installing CAIRN and vLLM...
vLLM: 0.12.0
CAIRN import: successful

==================================================
 5/7 MODEL SETUP
==================================================
/tmp/Qwen3-4B
Qwen3-4B configuration updated.
RoPE scaling: YaRN
Scaling factor: 4.0

==================================================
 6/7 START MODEL SERVER
==================================================
Starting attempt: A
Arguments: --tensor-parallel-size 2 --disable-custom-all-reduce --gpu-memory-utilization 0.85 --max-num-seqs 16
Loading... 1 minutes
Loading... 2 minutes
Loading... 3 minutes
Server is ready.
PID: 195
Testing CAIRN environment...
checking http://127.0.0.1:2026/v1 (model Qwen3-4B)

[PASS] server reachable: models: Qwen3-4B
[PASS] plain answer (thinking off): got 'OK'
[PASS] token probabilities (logprobs): 5 candidates returned
[PASS] constrained answers (structured outputs): works with structured_api: structured_outputs
[PASS] thinking mode: 1323 characters of reasoning, answer '17 + 26 = 43\n\n**Step'
[PASS] prefix cache: second call: 4016 cached tokens, 1.09s -> 0.14s

==================================================
 7/7 RUN EVALUATION
==================================================

------------------------------------------
Running task: 1
------------------------------------------
task 1 [dev] accuracy 70.0% (14/20, 95% range 48.1%..85.5%), ECE 0.225
    generate         15% of inputs, accuracy 0.0%
    think_vote       85% of inputs, accuracy 82.3%

run folder: /kaggle/working/runs/dev
Task 1 completed.

------------------------------------------
Running task: 2
------------------------------------------
task 2 [dev] accuracy 70.0% (14/20, 95% range 48.1%..85.5%), ECE 0.201
    score            35% of inputs, accuracy 28.6%
    think_vote       65% of inputs, accuracy 92.3%

run folder: /kaggle/working/runs/dev
Task 2 completed.

------------------------------------------
Running task: 3
------------------------------------------
task 3 [dev] accuracy 100.0% (20/20, 95% range 83.9%..100.0%), ECE 0.000
    think_vote       100% of inputs, accuracy 100.0%

run folder: /kaggle/working/runs/dev
Task 3 completed.

------------------------------------------
Running task: 4
------------------------------------------
task 4 [dev] accuracy 70.0% (14/20, 95% range 48.1%..85.5%), ECE 0.237
    think_vote       100% of inputs, accuracy 70.0%

run folder: /kaggle/working/runs/dev
Task 4 completed.

------------------------------------------
Running task: 5
------------------------------------------
task 5 [dev] accuracy 70.0% (14/20, 95% range 48.1%..85.5%), ECE 0.154
    score            75% of inputs, accuracy 86.7%
    think_vote       25% of inputs, accuracy 20.0%

run folder: /kaggle/working/runs/dev
Task 5 completed.

------------------------------------------
Running task: 6
------------------------------------------
task 6 [dev] accuracy 95.0% (19/20, 95% range 76.4%..99.1%), ECE 0.044
    score            90% of inputs, accuracy 100.0%
    think_vote       10% of inputs, accuracy 50.0%

run folder: /kaggle/working/runs/dev
Task 6 completed.

------------------------------------------
Running task: 7
------------------------------------------
task 7 [dev] accuracy 20.0% (4/20, 95% range 8.1%..41.6%), ECE 0.536
    generate         55% of inputs, accuracy 18.2%
    think_vote       45% of inputs, accuracy 22.2%

run folder: /kaggle/working/runs/dev
Task 7 completed.

------------------------------------------
Running task: 8
------------------------------------------
task 8 [dev] static check OK 95%, runs 0%, matches PyTorch 0%, consistency score 0.000 (n=20)

run folder: /kaggle/working/runs/dev
Task 8 completed.

==================================================
 FINAL EVALUATION RESULTS
==================================================
Task 1: Accuracy 70.0% | N=20 | 95% CI 48%..85% | Time 1609s
Task 2: Accuracy 70.0% | N=20 | 95% CI 48%..85% | Time 719s
Task 3: Accuracy 100.0% | N=20 | 95% CI 84%..100% | Time 1422s
Task 4: Accuracy 70.0% | N=20 | 95% CI 48%..85% | Time 1302s
Task 5: Accuracy 70.0% | N=20 | 95% CI 48%..85% | Time 322s
Task 6: Accuracy 95.0% | N=20 | 95% CI 76%..99% | Time 302s
Task 7: Accuracy 20.0% | N=20 | 95% CI 8%..42% | Time 599s
Task 8: Consistency 0.000 | Callable 0% | PyTorch Match 0% | N=20 | Time 1006s

Reports directory: /kaggle/working/runs/dev
Evaluation summary completed.

==================================================
 ALL REQUESTED TASKS FINISHED
==================================================
Cloning into '/kaggle/working/cairn'...
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gradio-client 2.7.2 requires huggingface-hub<3.0,>=1.16.0, but you have huggingface-hub 0.36.2 which is incompatible.
gradio 6.30.0 requires huggingface-hub<3.0,>=1.16.0, but you have huggingface-hub 0.36.2 which is incompatible.
gradio 6.30.0 requires pillow<13.0,>=8.0, but you have pillow 13.0.0 which is incompatible.
diffusers 0.42.0 requires huggingface-hub<3.0,>=1.32.0, but you have huggingface-hub 0.36.2 which is incompatible.
Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]Downloading 'merges.txt' to '/tmp/Qwen3-4B/.cache/huggingface/download/PtHk0z_I45atnj23IIRhTExwT3w=.31349551d90c7606f325fe0f11bbb8bd5fa0d7c7.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/merges.txt
Downloading '.gitattributes' to '/tmp/Qwen3-4B/.cache/huggingface/download/wPaCkH-WbT7GsmxMKKrNZTV4nSM=.52373fe24473b1aa44333d318f578ae6bf04b49b.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/.gitattributes
Fetching 13 files:   8%|▊         | 1/13 [00:00<00:01,  6.54it/s]Downloading 'model-00003-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/NmVQgThqhsRkG2FlXLh-M51MilA=.e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1.incomplete'
Downloading 'model.safetensors.index.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/yVzAsSxRSINSz-tQbpx-TLpfkLU=.95c0a0059df040d75dc6c396b174382cf61d2f91.incomplete'
Downloading 'LICENSE' to '/tmp/Qwen3-4B/.cache/huggingface/download/DhCjcNQuMpl4FL346qr3tvNUCgY=.6634c8cc3133b3848ec74b9f275acaaa1ea618ab.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/model.safetensors.index.json
Download complete. Moving file to /tmp/Qwen3-4B/LICENSE
Downloading 'model-00001-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/Y6g195DXbsCiN_Ka1NwH6jyLkvc=.328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223.incomplete'
Downloading 'model-00002-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/D5-QTaCbl9VUFv9Wr0C28eFewRc=.6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5.incomplete'
Downloading 'config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/8_PA_wEVGiVa2goH2H4KQOQpvVY=.e49eccdc32f36da9c09cfa0e737084f9e0105e5e.incomplete'
Downloading 'generation_config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/3EVKVggOldJcKSsGjSdoUCN1AyQ=.20a8a9156fc8c3f25295ca067f61fdf120d517c5.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/config.json
Download complete. Moving file to /tmp/Qwen3-4B/generation_config.json
Downloading 'tokenizer.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/HgM_lKo9sdSCfRtVg7MMFS7EKqo=.aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4.incomplete'
Downloading 'tokenizer_config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/vzaExXFZNBay89bvlQv-ZcI6BTg=.417d038a63fa3de29cfde265caedae14d1a58d92.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/tokenizer_config.json
Downloading 'README.md' to '/tmp/Qwen3-4B/.cache/huggingface/download/Xn7B-BWUGOee2Y6hCZtEhtFu4BE=.2de5ee7eee214bb55ea33ec7505c5838a7adf7f6.incomplete'
Downloading 'vocab.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/j3m-Hy6QvBddw8RXA1uSWl1AJ0c=.4783fe10ac3adce15ac8f358ef5462739852c569.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/README.md
Fetching 13 files:  23%|██▎       | 3/13 [00:00<00:01,  9.90it/s]Download complete. Moving file to /tmp/Qwen3-4B/vocab.json
Download complete. Moving file to /tmp/Qwen3-4B/tokenizer.json
Download complete. Moving file to /tmp/Qwen3-4B/model-00003-of-00003.safetensors
Download complete. Moving file to /tmp/Qwen3-4B/model-00002-of-00003.safetensors
Download complete. Moving file to /tmp/Qwen3-4B/model-00001-of-00003.safetensors
Fetching 13 files: 100%|██████████| 13/13 [00:33<00:00,  2.54s/it]
2026-11-16 12:05:51,383 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-16 12:32:41,900 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-16 12:44:42,054 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-16 13:08:25,477 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-16 13:30:08,468 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-16 13:35:31,359 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-16 13:40:34,302 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-16 13:50:34,396 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
