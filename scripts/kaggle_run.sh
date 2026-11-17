#!/usr/bin/env bash
# Cairn + Qwen3-4B on a Kaggle notebook with 2 x Tesla T4 (free tier).
#
# Paste this into ONE notebook cell starting with `%%bash`, or run `bash scripts/kaggle_run.sh`
# after cloning.  Every stage is safe to re-run: the repo is re-cloned, the model download is
# skipped when present, model answers are cached on disk, and the run journal resumes where
# it stopped.  To make a run survive the session, "Save Version" the notebook: only
# /kaggle/working is kept.
#
# Settings you will actually change are at the top.  Runs so far: E000 (LIMIT=20, 2026-11-16)
# and E001 (LIMIT=60, 2026-11-17); raw logs in docs/results/.  60 inputs gives a 95% range of
# about +/- 12 points.  The exact tasks (1 3 4) go first: with a verified program they take
# minutes.  Task 8 is always measured on TEST inputs (LIMIT8 of them): its metric needs no
# gold answer, and the demonstration kernels carry no wrapper signature, so dev inputs can
# never be executed (E001: 24/24 "not_run").

TASKS="${TASKS:-1 3 4 2 5 6 7 8}"
LIMIT="${LIMIT:-60}"
LIMIT8="${LIMIT8:-24}"              # task 8 in a dev pass: this many test inputs, scored without gold
SPLIT="${SPLIT:-dev}"               # dev = scored on held-out examples; test = submission files
BRANCH="${BRANCH:-claude/ai-project-enhancement-z6ct2n}"
REPO_URL="${REPO_URL:-https://github.com/madhesh60/ICL.git}"

W="/kaggle/working"
REPO="$W/cairn"
MODEL_DIR="${MODEL_DIR:-/tmp/Qwen3-4B}"
RUNS="$W/runs/$SPLIT"
PORT=2026

set -Eeuo pipefail
export PYTHONUNBUFFERED=1
# Generated Triton kernels (task 8) run on the real GPU in a child process.
export CAIRN_TRITON_INTERPRET=0

say() { printf '\n==================================================\n %s\n==================================================\n' "$*"; }
up()  { curl -sf --connect-timeout 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null; }

# ------------------------------------------------------------------ 1. GPU
say "1/7 GPU CHECK"
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no NVIDIA GPU. Enable GPU in the notebook settings."; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
NGPU=$(nvidia-smi -L | wc -l)
echo "Detected GPUs: $NGPU"

# ------------------------------------------------------------------ 2. CUDA driver library
# Triton needs libcuda.so at link time; on Kaggle it is only present as libcuda.so.1.
say "2/7 CUDA DRIVER LIBRARY CHECK"
CUDA_DRIVER=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 / {print $NF; exit}' || true)
if [ -z "${CUDA_DRIVER:-}" ] || [ ! -f "$CUDA_DRIVER" ]; then
  for candidate in /usr/local/nvidia/lib64/libcuda.so.1 /usr/local/nvidia/lib64/libcuda.so \
                   /usr/lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so.1; do
    [ -f "$candidate" ] && { CUDA_DRIVER="$candidate"; break; }
  done
fi
[ -n "${CUDA_DRIVER:-}" ] && [ -f "$CUDA_DRIVER" ] || { echo "ERROR: NVIDIA driver library not found; restart the session."; exit 1; }
CUDA_LINK_DIR="$W/cuda-link"; mkdir -p "$CUDA_LINK_DIR"; ln -sf "$CUDA_DRIVER" "$CUDA_LINK_DIR/libcuda.so"
export LIBRARY_PATH="$CUDA_LINK_DIR:/usr/local/nvidia/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_LINK_DIR:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_LIBCUDA_PATH="$CUDA_LINK_DIR"
gcc -print-file-name=libcuda.so | grep -q '^/' || { echo "ERROR: linker cannot find libcuda.so"; exit 1; }
echo "CUDA driver: $CUDA_DRIVER"

# ------------------------------------------------------------------ 3. code
say "3/7 GET PROJECT CODE ($BRANCH)"
rm -rf "$REPO"
git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$REPO"
cd "$REPO"
git log --oneline -1

# ------------------------------------------------------------------ 4. dependencies
say "4/7 INSTALL DEPENDENCIES"
if ! python -c "import vllm, cairn" >/dev/null 2>&1; then
  pip install -q -e ".[dev]" "vllm==0.12.0"
fi
python -c "import vllm, cairn; print('vLLM', vllm.__version__, '| cairn import OK')"

# ------------------------------------------------------------------ 5. model
say "5/7 MODEL SETUP"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  mkdir -p "$MODEL_DIR"
  hf download Qwen/Qwen3-4B --local-dir "$MODEL_DIR"
fi
# Prompts exceed the 32K the model was trained on: YaRN position scaling x4.
python - "$MODEL_DIR/config.json" <<'PY'
import json, sys
path = sys.argv[1]
cfg = json.load(open(path))
want = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
if cfg.get("rope_scaling") != want:
    cfg["rope_scaling"] = want
    json.dump(cfg, open(path, "w"), indent=2)
print("rope_scaling:", cfg["rope_scaling"])
PY

# ------------------------------------------------------------------ 6. server
say "6/7 START MODEL SERVER"
start_server() {
  local name="$1"; shift
  local LOG="$W/vllm_${name}.log"
  echo "attempt $name: $*"
  up && { echo "server already running"; return 0; }
  nohup vllm serve "$MODEL_DIR" --served-model-name Qwen3-4B --host 127.0.0.1 --port "$PORT" \
    --dtype half --max-model-len 40960 --enforce-eager --enable-prefix-caching \
    --reasoning-parser qwen3 --enable-prompt-tokens-details "$@" > "$LOG" 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    up && { echo "server ready (pid $pid)"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "server crashed ($name); last errors:"; grep -iE "error|exception|memory|cuda|failed" "$LOG" | tail -30 || true; return 1; }
    sleep 10; [ $((i % 6)) -eq 0 ] && echo "loading... $((i / 6)) min"
  done
  echo "server start timed out"; kill "$pid" 2>/dev/null || true; return 1
}
if ! up; then
  start_server A --tensor-parallel-size "$NGPU" --disable-custom-all-reduce --gpu-memory-utilization 0.85 --max-num-seqs 16 \
  || start_server B --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 8 \
  || { echo "both server attempts failed; see $W/vllm_A.log and $W/vllm_B.log"; exit 1; }
fi
cairn --config configs/default.yaml doctor

# ------------------------------------------------------------------ 7. run
say "7/7 RUN ($SPLIT, limit $LIMIT): tasks $TASKS"
mkdir -p "$RUNS"
for t in $TASKS; do
  echo; echo "------------------------------------------ task $t"
  if [ "$SPLIT" = "test" ]; then
    cairn --config configs/default.yaml -v run --task "$t" --split test --run-dir "$RUNS" || echo "task $t FAILED; continuing"
  elif [ "$t" = "8" ]; then
    cairn --config configs/default.yaml -v run --task 8 --split test --limit "$LIMIT8" --run-dir "$RUNS" || echo "task 8 FAILED; continuing"
  else
    cairn --config configs/default.yaml -v run --task "$t" --split "$SPLIT" --limit "$LIMIT" --run-dir "$RUNS" || echo "task $t FAILED; continuing"
  fi
done

say "SUMMARY"
python - "$RUNS" "$SPLIT" <<'PY'
import json, pathlib, sys
root, split = pathlib.Path(sys.argv[1]), sys.argv[2]
paths = sorted(root.glob(f"task*/{split}.report.json"))
if split != "test":
    paths += sorted(root.glob("task8/test.report.json"))   # task 8 is measured on test inputs
for path in paths:
    r = json.loads(path.read_text())
    t, secs = r.get("task", path.parent.name), r.get("seconds", 0)
    if "eval" in r:
        e = r["eval"]
        print(f"task {t}: accuracy {e['accuracy']:.1%} (n={e['n']}, 95% {e['ci95'][0]:.0%}..{e['ci95'][1]:.0%}) | {secs:.0f}s | sources {r['sources']}")
    elif "code_quality" in r:
        q = r["code_quality"]
        print(f"task {t}: consistency {q['consistency']:.3f} | callable {q['callable']:.0%} | torch match {q['matches_torch']:.0%} | outcomes {q.get('statuses')} | {secs:.0f}s")
    else:
        print(f"task {t}: wrote {r['n']} predictions | sources {r['sources']} | {secs:.0f}s")
    notes = r.get("notes", {})
    prog = notes.get("program")
    if isinstance(prog, dict):
        print(f"        program accepted={prog['accepted']} attempts={prog['attempts']} best_verify={prog['best_verify_accuracy']:.1%}")
    pol = notes.get("policy")
    if isinstance(pol, dict):
        print(f"        policy {pol['chosen']}: cheap {pol['cheap_accuracy']:.0%} vs thinking {pol['costly_accuracy']:.0%} on {pol['probe_examples']} held-out inputs")
    cal = notes.get("calibration")
    if isinstance(cal, dict) and isinstance(cal.get("conformal"), dict):
        c = cal["conformal"]
        print(f"        scoring sure on {c['singleton_rate']:.0%}, right on {c['singleton_accuracy']:.0%} of those, trusted={c.get('singletons_trusted')}")
print("\nreports:", root)
PY
if [ "$SPLIT" = "test" ]; then
  cairn submit --run-dir "$RUNS" --out "$W/submission.zip" && echo "submission: $W/submission.zip"
fi
say "DONE"
