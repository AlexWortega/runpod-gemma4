"""RunPod serverless handler – proxies to llama-server, surfaces startup errors."""

import os

import requests
import runpod

LLAMA_URL = "http://localhost:" + os.environ.get("LLAMA_PORT", "8080")
TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "600"))
LLAMA_LOG = "/tmp/llama.log"


def _log_tail(n=8000):
    try:
        with open(LLAMA_LOG, "r", errors="replace") as f:
            return f.read()[-n:]
    except Exception as e:  # noqa: BLE001
        return f"<no log: {e}>"


def _healthy():
    try:
        return requests.get(f"{LLAMA_URL}/health", timeout=5).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def handler(job):
    if not _healthy():
        return {"error": "llama-server is not healthy",
                "llama_log_tail": _log_tail()}

    body = job["input"]
    endpoint = body.pop("endpoint", "/v1/chat/completions")
    try:
        resp = requests.post(f"{LLAMA_URL}{endpoint}", json=body, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"llama request failed: {e}",
                "llama_log_tail": _log_tail()}


runpod.serverless.start({"handler": handler})
