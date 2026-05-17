# runpod-gemma4

RunPod serverless worker that serves **Gemma-4-31B-it** (Q8_0, 8-bit GGUF from
[Unsloth](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)) via
[llama.cpp](https://github.com/ggml-org/llama.cpp).

Based on the official
[`ghcr.io/ggml-org/llama.cpp:server-cuda`](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp)
image. At runtime it starts `llama-server` with an OpenAI-compatible API and
fronts it with a thin RunPod handler.

## How it works

```
RunPod job ──► handler.py ──► llama-server (:8080, OpenAI-compat)
```

`start.sh` runs on container boot:

1. Downloads the GGUF model via `huggingface-cli` (no-op when already cached).
2. Launches `llama-server` with full GPU offload and flash attention.
3. Waits for the health endpoint, then starts the RunPod handler.

## Deploying to RunPod

### 1. Fork this repo

### 2. Create a serverless template

In the RunPod console under **Serverless**:
* click **+ New Endpoint**, 
* **Configure Endpoint**

| Field              | Value                                       |
| ------------------ | ------------------------------------------- |
| Endpoint Name      | `gemma-4-31b-llama-cpp`                     |
| Container Source   | **GitHub Repo** — select the repo above     |
| Dockerfile Path    | `Dockerfile`                                |
| Model              | `unsloth/gemma-4-31B-it-GGUF`              |

Setting the **Model** field tells RunPod to pre-cache the model on worker hosts
and schedule workers onto hosts that already have it.  Download time is not
billed.  See
[RunPod model caching docs](https://docs.runpod.io/serverless/endpoints/model-caching).

Pick a GPU tier — Q8_0 at ~33 GB needs at least 35 GB VRAM:

| GPU          | VRAM   | Notes                       |
| ------------ | ------ | --------------------------- |
| A6000        | 48 GB  | Minimum comfortable option  |
| A100 80 GB   | 80 GB  | Room for larger context     |
| H100         | 80 GB  | Fastest inference           |

## Configuration

All settings are environment variables, overridable in the RunPod template
without rebuilding.

| Variable          | Default                            | Description                                  |
| ----------------- | ---------------------------------- | -------------------------------------------- |
| `MODEL_REPO`      | `unsloth/gemma-4-31B-it-GGUF`     | HuggingFace repo for the GGUF                |
| `MODEL_FILE`      | `gemma-4-31B-it-Q8_0.gguf`        | Specific GGUF file to download               |
| `MMPROJ_FILE`     | *(empty)*                          | Set to `mmproj-BF16.gguf` to enable vision   |
| `N_GPU_LAYERS`    | `999`                              | Layers to offload to GPU (`999` = all)        |
| `CTX_SIZE`        | `8192`                             | Context window size in tokens                |
| `LLAMA_PORT`      | `8080`                             | Internal port for llama-server               |
| `REQUEST_TIMEOUT` | `300`                              | Handler timeout per request (seconds)        |

### Switching quantization

Change `MODEL_FILE` to any file in the
[Unsloth GGUF repo](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF), e.g.:

- `gemma-4-31B-it-Q4_K_M.gguf` — 4-bit, ~18 GB, fits on 24 GB GPUs
- `gemma-4-31B-it-Q6_K.gguf` — 6-bit, ~25 GB
- `gemma-4-31B-it-UD-Q4_K_XL.gguf` — Unsloth Dynamic 4-bit

## Usage

Send an OpenAI-compatible chat completion request as the job input:

```json
{
    "input": {
        "messages": [
            {"role": "user", "content": "Explain the Pythagorean theorem."}
        ],
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.95
    }
}
```

The handler forwards requests to llama-server's `/v1/chat/completions` by
default.  Set `"endpoint": "/v1/completions"` in the input to use the text
completions endpoint instead.

### Example with the RunPod CLI

```bash
runpodctl send \
  --endpoint YOUR_ENDPOINT_ID \
  --input '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

### Example with curl

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128
    }
  }'
```

## Files

| File               | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `Dockerfile`       | Layers Python + handler on official llama.cpp CUDA image   |
| `start.sh`         | Downloads model, starts llama-server, then handler         |
| `handler.py`       | RunPod handler — proxies requests to llama-server          |
| `requirements.txt` | Python dependencies                                        |
| `test_input.json`  | Sample input for local testing                             |
