#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-unsloth/gemma-4-31B-it-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-31B-it-Q8_0.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
CTX_SIZE="${CTX_SIZE:-8192}"
LLAMA_PORT="${LLAMA_PORT:-8080}"

# ---- helper: download a file from HF and print its local path ----
hf_download() {
    python3 -c "
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2]))
" "$1" "$2"
}

# ---- ensure HF cache directory exists ----
# RunPod mounts /runpod-volume when model caching is active.
# Fall back to a local directory if it isn't available.
if [ ! -d "/runpod-volume" ]; then
    echo "WARN: /runpod-volume not mounted, using /tmp/hf-cache"
    export HF_HUB_CACHE="/tmp/hf-cache"
fi
mkdir -p "$HF_HUB_CACHE"

# ---- download model (uses RunPod's HF cache) ----
echo "Resolving model ${MODEL_REPO} / ${MODEL_FILE} ..."
MODEL_PATH=$(hf_download "$MODEL_REPO" "$MODEL_FILE") || {
    echo "ERROR: model download failed (check repo name, filename, HF_TOKEN)"
    exit 1
}

# Optionally download the multimodal projector
MMPROJ_ARGS=()
if [ -n "$MMPROJ_FILE" ]; then
    echo "Resolving mmproj: ${MMPROJ_FILE} ..."
    MMPROJ_PATH=$(hf_download "$MODEL_REPO" "$MMPROJ_FILE")
    MMPROJ_ARGS=(--mmproj "$MMPROJ_PATH")
fi

echo "Model:  $MODEL_PATH"
echo "Layers: $N_GPU_LAYERS  CTX: $CTX_SIZE  Port: $LLAMA_PORT"

# ---- start llama-server ----
/app/llama-server \
    --model "$MODEL_PATH" \
    ${MMPROJ_ARGS[@]+"${MMPROJ_ARGS[@]}"} \
    --host 0.0.0.0 \
    --port "$LLAMA_PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --flash-attn \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    &

SERVER_PID=$!

echo "Waiting for llama-server (pid $SERVER_PID) ..."
READY=0
for _ in $(seq 1 180); do          # up to 6 min for large model load
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
