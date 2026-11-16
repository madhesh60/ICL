#!/usr/bin/env bash
# End-to-end run on a Linux machine with one NVIDIA GPU (24 GB or more).
#
#   bash scripts/gpu_run.sh            # everything: download, serve, check, dev, test, zip
#   STAGE=dev bash scripts/gpu_run.sh  # stop after scoring the dev split
#
# Every stage is safe to re-run: downloads are skipped when present, model answers are
# cached on disk, and the run journal resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${MODEL_DIR:-$PWD/Qwen3-4B}"
CONFIG="${CONFIG:-configs/default.yaml}"
RUN_DIR="${RUN_DIR:-runs/gpu}"
STAGE="${STAGE:-all}"
PORT="${PORT:-2026}"

step() { printf '\n==> %s\n' "$*"; }

mkdir -p "$(dirname "$RUN_DIR")"

step "1/7 install"
python3 -m pip install -q -e ".[dev]" "vllm>=0.12"
# Triton and torch come with vLLM, so generated kernels can be executed on the real GPU.
export CAIRN_TRITON_INTERPRET=0

step "2/7 model weights"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  huggingface-cli download Qwen/Qwen3-4B --local-dir "$MODEL_DIR"
fi
# Prompts are longer than the 32K the model was trained for, so turn on YaRN position scaling.
python3 - "$MODEL_DIR/config.json" <<'PY'
import json, sys
path = sys.argv[1]
cfg = json.load(open(path))
want = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
if cfg.get("rope_scaling") != want:
    cfg["rope_scaling"] = want
    json.dump(cfg, open(path, "w"), indent=2)
    print("patched rope_scaling -> YaRN x4")
PY

step "3/7 server"
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  MODEL_DIR="$MODEL_DIR" PORT="$PORT" nohup bash configs/serve/vllm.sh > "$RUN_DIR.server.log" 2>&1 &
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
    sleep 5
  done
fi
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "server did not start; see $RUN_DIR.server.log"; exit 1; }

step "4/7 doctor"
cairn --config "$CONFIG" doctor || echo "some checks failed: read the advice above (the run continues with fallbacks)"

step "5/7 dev split (scored)"
cairn --config "$CONFIG" -v run --task all --split dev --run-dir "$RUN_DIR"
[ "$STAGE" = "dev" ] && exit 0

step "6/7 test split"
cairn --config "$CONFIG" -v run --task all --split test --run-dir "$RUN_DIR"

step "7/7 package"
cairn submit --run-dir "$RUN_DIR" --out "$RUN_DIR/submission.zip"
