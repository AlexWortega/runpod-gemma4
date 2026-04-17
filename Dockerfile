# syntax=docker/dockerfile:1
#
# RunPod worker serving Gemma-4-31B-it via llama.cpp
# Based on the official llama.cpp CUDA server image.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /requirements.txt

COPY handler.py /handler.py
COPY start.sh   /start.sh
RUN chmod +x /start.sh

# ----- configuration (override via RunPod template env vars) -----
ENV MODEL_REPO="unsloth/gemma-4-31B-it-GGUF"
ENV MODEL_FILE="gemma-4-31B-it-Q8_0.gguf"
ENV N_GPU_LAYERS="999"
ENV CTX_SIZE="8192"
ENV LLAMA_PORT="8080"
# llama.cpp's own download cache — start.sh overrides this to
# /runpod-volume/llama-cache when the volume is mounted.
ENV LLAMA_CACHE="/tmp/llama-cache"

# Override the base image's ENTRYPOINT (which is /app/llama-server)
ENTRYPOINT []
CMD ["/start.sh"]
