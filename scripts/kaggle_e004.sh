#!/usr/bin/env bash
# Paired E003 comparison on a fresh Kaggle T4 x2 session.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
export PYTHONUNBUFFERED=1
OUT=/kaggle/working/cairn-e004
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

# Tie resumable responses to the packaged source, config, official data and baseline.
sha256sum --check --quiet e004-files.sha256
if [[ -e "$OUT/e004-files.sha256" ]]; then
  cmp e004-files.sha256 "$OUT/e004-files.sha256"
elif [[ -e "$OUT/task2" || -e "$OUT/task7" || -e "$OUT/task8" ]]; then
  echo "Existing task output has no snapshot identity. Use a fresh notebook session." >&2
  exit 1
else
  cp e004-files.sha256 "$OUT/"
fi

CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES CAIRN_TRITON_INTERPRET=0
unset CAIRN_TOKENIZER CAIRN_TRITON_PYTHON TRITON_INTERPRET
export CAIRN_DATA_DIR="$ROOT/OpenSeek/openseek/competition/LongContext-ICL-Annotation/data"
export CAIRN_QWEN_VOCAB="$ROOT/OpenSeek/hf_openseek/tokenizer/qwen.tiktoken"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpus.txt"
if [[ $(grep -c 'T4' "$OUT/gpus.txt") != 2 ]]; then
  echo "Select GPU T4 x2 to match E003." >&2
  exit 1
fi
RUNTIME=/tmp/cairn-e004-runtime
mkdir -p "$RUNTIME/lib"
driver=$(ldconfig -p | awk '/libcuda\.so\.1 / && !found { print $NF; found=1 }')
if [[ ! -f "$driver" ]]; then
  for driver in /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
    [[ -f "$driver" ]] && break
  done
fi
[[ -f "$driver" ]] || { echo "Cannot find the NVIDIA driver library" >&2; exit 1; }
ln -sfn "$driver" "$RUNTIME/lib/libcuda.so"
export LIBRARY_PATH="$RUNTIME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$RUNTIME/lib:${LD_LIBRARY_PATH:-}"
export TRITON_LIBCUDA_PATH="$RUNTIME/lib"

FS_REV=f16a47f205966d3f058ea626f5f01f63cd5ff52a
export MODEL_REV=1cfa9a7208912126459214e8b04321603b3df60c
FRAMEWORK="$RUNTIME/FlagScale"
if [[ ! -d "$FRAMEWORK/.git" ]]; then
  git clone --quiet --depth 1 --branch v1.0.0 https://github.com/FlagOpen/FlagScale.git "$FRAMEWORK"
fi
[[ $(git -C "$FRAMEWORK" rev-parse HEAD) == "$FS_REV" ]]
printf 'FlagScale %s\nQwen/Qwen3-4B %s\n' "$FS_REV" "$MODEL_REV" | tee "$OUT/revisions.txt"
if [[ ! -x "$RUNTIME/venv/bin/python" ]]; then
  python -m venv "$RUNTIME/venv"
fi
source "$RUNTIME/venv/bin/activate"
export PYTHONPATH="$ROOT:$FRAMEWORK"
python -m pip install --disable-pip-version-check --no-cache-dir -q \
  'vllm==0.12.0' 'transformers==4.57.3' 'omegaconf==2.3.0' tiktoken aiohttp tqdm pyyaml requests
python -m pip freeze > "$OUT/environment.txt"
python scripts/compare_e003.py
python - <<'PY'
import socket
import torch
import triton
from cairn.tokens import get_counter
if torch.cuda.device_count() != 2 or get_counter().name != "tiktoken-qwen":
    raise RuntimeError("Expected two CUDA devices and the same bundled Qwen token counter as E003")
with socket.socket() as port:
    port.bind(("127.0.0.1", 2026))
print("GPU executor ready", torch.__version__, triton.__version__)
PY

export MODEL_DIR="$RUNTIME/Qwen3-4B"
python - "$OUT" <<'PY'
import json
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml

model = Path(os.environ["MODEL_DIR"])
snapshot_download("Qwen/Qwen3-4B", revision=os.environ["MODEL_REV"], local_dir=model)
path = model / "config.json"
config = json.loads(path.read_text())
config["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
path.write_text(json.dumps(config, indent=2))

serve = yaml.safe_load(Path("configs/serve/flagscale_qwen3_4b.yaml").read_text())
serve["serve"][0]["engine_args"].update(
    model=str(model), host="127.0.0.1", dtype="half", tensor_parallel_size=2,
    gpu_memory_utilization=0.85, max_model_len=40960, max_num_seqs=16,
    enforce_eager=True, disable_custom_all_reduce=True, enable_prompt_tokens_details=True,
)
serve["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1"
(Path(sys.argv[1]) / "flagscale.yaml").write_text(yaml.safe_dump(serve, sort_keys=False))
PY

# FlagScale's vLLM backend chooses this engine entry point for use_fs_serve: false.
# Launch in a dedicated process group so that only this notebook's server is stopped.
server_pid=""
stop_server() {
  [[ -n "$server_pid" ]] || return 0
  kill -TERM -- "-$server_pid" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 -- "-$server_pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid python -m flagscale.serve.run_inference_engine \
  --config-path "$OUT/flagscale.yaml" --log-dir "$OUT" > "$OUT/flagscale.log" 2>&1 &
server_pid=$!
ready=0
for _ in {1..120}; do
  kill -0 "$server_pid" 2>/dev/null || break
  if curl -fsS --max-time 3 http://127.0.0.1:2026/v1/models 2>/dev/null | \
     python -c 'import json,sys; sys.exit(not any(m["id"] == "Qwen3-4B" for m in json.load(sys.stdin)["data"]))' 2>/dev/null; then
    ready=1
    break
  fi
  sleep 6
done
if [[ "$ready" != 1 ]]; then
  tail -n 90 "$OUT/flagscale.log"
  exit 1
fi

cli=(python -m cairn.cli --config configs/default.yaml -v)
"${cli[@]}" doctor
for task in 2 7; do
  "${cli[@]}" run --task "$task" --split dev --limit 60 --run-dir "$OUT"
done
# A single repair worker keeps concurrent Triton child processes off the reserved vLLM memory.
"${cli[@]}" run --task 8 --split test --limit 24 --workers 1 --run-dir "$OUT"
stop_server
python scripts/compare_e003.py --run-dir "$OUT"
