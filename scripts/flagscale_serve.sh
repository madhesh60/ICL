#!/usr/bin/env bash
# Run in the competition FlagScale environment with a YaRN-configured Qwen3-4B.
set -euo pipefail
action=${1:-run}
case "$action" in
  run|stop) ;;
  *) echo "usage: bash scripts/flagscale_serve.sh [run|stop]" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "$0")/.." && pwd)
: "${FLAGSCALE_DIR:?Set FLAGSCALE_DIR to the competition FlagScale checkout containing run.py}"
framework=$(cd "$FLAGSCALE_DIR" && pwd)
[[ -f "$framework/run.py" ]] || { echo "Missing $framework/run.py" >&2; exit 1; }
export MODEL_DIR=${MODEL_DIR:-$root/Qwen3-4B}
if [[ "$action" == run ]]; then
  [[ -f "$MODEL_DIR/config.json" ]] || { echo "Missing Qwen3-4B config in $MODEL_DIR" >&2; exit 1; }
  MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
fi
cd "$framework"
exec python run.py --config-path "$root/configs/serve" --config-name flagscale_qwen3_4b "action=$action"
