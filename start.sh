#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-unsloth/gemma-4-31B-it-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-31B-it-Q8_0.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
CTX_SIZE="${CTX_SIZE:-8192}"
LLAMA_PORT="${LLAMA_PORT:-8080}"

# Use the persistent RunPod volume for model caching when available,
# so the 33 GB model doesn't re-download on every cold start.
if [ -d "/runpod-volume" ] && [ -w "/runpod-volume" ]; then
    export LLAMA_CACHE="/runpod-volume/llama-cache"
    echo "Using persistent cache: $LLAMA_CACHE"
fi
mkdir -p "$LLAMA_CACHE"

echo "Starting llama-server: repo=${MODEL_REPO} file=${MODEL_FILE}"
echo "  GPU layers: $N_GPU_LAYERS  CTX: $CTX_SIZE  Port: $LLAMA_PORT  Cache: $LLAMA_CACHE"

# llama-server downloads the GGUF from HuggingFace on first run and
# caches it locally (controlled by LLAMA_CACHE env var).
/app/llama-server \
    --hf-repo "$MODEL_REPO" \
    --hf-file "$MODEL_FILE" \
    --host 0.0.0.0 \
    --port "$LLAMA_PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --flash-attn on \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    &

SERVER_PID=$!

echo "Waiting for llama-server (pid $SERVER_PID) ..."
READY=0
for _ in $(seq 1 360); do          # up to 12 min for download + load
    if curl -sf "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; then
        echo "llama-server is ready"
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: llama-server exited unexpectedly"
        exit 1
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: llama-server did not become healthy within timeout"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# ---- start RunPod handler ----
exec python3 -u /handler.py
