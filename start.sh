#!/usr/bin/env bash
set -uo pipefail

MODEL_REPO="${MODEL_REPO:-unsloth/gemma-4-31B-it-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-31B-it-Q8_0.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
CTX_SIZE="${CTX_SIZE:-8192}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_LOG="/tmp/llama.log"

if [ -d "/runpod-volume" ] && [ -w "/runpod-volume" ]; then
    export LLAMA_CACHE="/runpod-volume/llama-cache"
fi
mkdir -p "${LLAMA_CACHE:-/tmp/llama-cache}"

MODEL_ARGS=(--hf-repo "$MODEL_REPO" --hf-file "$MODEL_FILE")
echo "Starting llama-server: ${MODEL_ARGS[*]}  (GPU=$N_GPU_LAYERS CTX=$CTX_SIZE)" | tee "$LLAMA_LOG"

# Launch llama-server, tee all output to a log file so the handler can surface
# the real error if it fails to come up.
/app/llama-server \
    "${MODEL_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$LLAMA_PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --flash-attn on \
    --jinja \
    >>"$LLAMA_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for llama-server (pid $SERVER_PID) ..."
for _ in $(seq 1 360); do
    if curl -sf "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; then
        echo "llama-server is ready" | tee -a "$LLAMA_LOG"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "llama-server EXITED early — see log below" | tee -a "$LLAMA_LOG"
        break
    fi
    sleep 2
done

# IMPORTANT: never exit. Always start the handler so failures are reported
# back through the RunPod job response instead of an invisible crash loop.
exec python3 -u /handler.py
