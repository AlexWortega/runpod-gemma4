# syntax=docker/dockerfile:1
#
# RunPod worker serving Gemma-4-31B-it via llama.cpp
# Build llama.cpp from a pinned tag so upgrades are explicit.

# ---- build stage ----
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG LLAMA_CPP_TAG=b8648

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} \
        https://github.com/ggml-org/llama.cpp.git

WORKDIR /build/llama.cpp
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc) \
        --target llama-server llama-cli

# ---- runtime stage ----
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# cuBLAS + cuBLASLt are needed by the CUDA backend at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcublas-12-4 \
        python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-cli    /usr/local/bin/

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY handler.py /handler.py
COPY start.sh   /start.sh
RUN chmod +x /start.sh

# ----- configuration (override via RunPod template env vars) -----
ENV MODEL_REPO="unsloth/gemma-4-31B-it-GGUF"
ENV MODEL_FILE="gemma-4-31B-it-Q8_0.gguf"
# Set to a multimodal-projector GGUF to enable vision (e.g. mmproj-BF16.gguf)
ENV MMPROJ_FILE=""
ENV N_GPU_LAYERS="999"
ENV CTX_SIZE="8192"
ENV LLAMA_PORT="8080"
# Use RunPod's model cache path so the platform can pre-cache and reuse models
# across workers.  See https://docs.runpod.io/serverless/endpoints/model-caching
ENV HF_HUB_CACHE="/runpod-volume/huggingface-cache/hub"

CMD ["/start.sh"]
