#!/usr/bin/env bash
# Serve Qwen3-4B with plain vLLM, set up the way Cairn wants it.
#   usage: MODEL_DIR=./Qwen3-4B bash configs/serve/vllm.sh
#
# What each flag is for:
#   --max-model-len 65536         room for a 30K+ example block plus thinking tokens.
#                                 Above 32K the model needs the YaRN patch in config.json
#                                 (see README, "Serving the model").
#   --enable-prefix-caching       reuse the server's notes for repeated prompt beginnings.
#                                 This is what makes 30K-token prompts affordable.
#   --reasoning-parser qwen3      return the <think> part separately from the answer.
#   --enable-prompt-tokens-details  report how many prompt tokens came from the cache.
#                                 Some vLLM versions return nothing for it; Cairn copes.
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-./Qwen3-4B}"
PORT="${PORT:-2026}"

exec vllm serve "$MODEL_DIR" \
  --served-model-name Qwen3-4B \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --trust-remote-code
