#!/usr/bin/env bash
set -euo pipefail

drive_root=${FISH9B_DRIVE_ROOT:-/content/drive/MyDrive/Fish-9B}
model_path=${FISH9B_MODEL_PATH:-$drive_root/model_cache/Qwen3.5-9B}
: "${FISH9B_SGLANG_API_KEY:?Set FISH9B_SGLANG_API_KEY before starting SGLang}"

if [[ ! -f "$model_path/config.json" ]]; then
    printf 'Missing model configuration at %s/config.json\n' "$model_path" >&2
    exit 1
fi

exec python -m sglang.launch_server \
    --model-path "$model_path" \
    --served-model-name Qwen3.5-9B \
    --host 127.0.0.1 \
    --port "${FISH9B_SGLANG_PORT:-30000}" \
    --dtype bfloat16 \
    --context-length "${FISH9B_SGLANG_CONTEXT_LENGTH:-16384}" \
    --mem-fraction-static "${FISH9B_SGLANG_MEM_FRACTION:-0.88}" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --api-key "$FISH9B_SGLANG_API_KEY"
