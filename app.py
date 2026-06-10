import time
from typing import List
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Local SLM Benchmarking Engine")

# Allows Streamlit (on a different port) to call this API without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_MODELS = ["qwen2.5:1.5b", "phi3:mini", "llama3.2:3b"]

# ── Pydantic request schemas ──────────────────────────────────────────────────
# Pydantic validates what comes IN to your API.
# If a request is missing "query" or sends the wrong type, FastAPI rejects it
# automatically before it even reaches your function. That's why we use it.

class ChatRequest(BaseModel):
    query: str
    model: str

class CompareRequest(BaseModel):
    query: str
    models: List[str] = SUPPORTED_MODELS   # defaults to all 3 if not specified


# ── Core inference function ───────────────────────────────────────────────────
def run_inference(model_name: str, query: str) -> dict:
    """
    Sends a query to a local Ollama model and measures performance.

    Why time.perf_counter() instead of time.time()?
    perf_counter is the highest resolution timer available — it measures
    wall-clock time in fractional seconds with nanosecond precision.
    time.time() is lower resolution and can jump if the system clock is adjusted.

    Why eval_count?
    Ollama returns metadata about the generation in the response object.
    eval_count = number of tokens the model generated.
    Dividing by elapsed time gives tokens/sec — the standard throughput metric.
    """
    try:
        t0 = time.perf_counter()

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": query}]
        )

        elapsed = time.perf_counter() - t0

        text       = response["message"]["content"]
        eval_count = response.get("eval_count", None)   # tokens generated
        tokens_sec = round(eval_count / elapsed, 1) if eval_count else None

        return {
            "status":           "success",
            "model":            model_name,
            "response":         text,
            "latency_seconds":  round(elapsed, 3),
            "tokens_generated": eval_count,
            "tokens_per_second": tokens_sec,
        }

    except Exception as e:
        return {
            "status":          "failed",
            "model":           model_name,
            "error":           str(e),
            "latency_seconds": 0,
        }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/models")
def list_models():
    """Returns which supported models are actually installed and running."""
    try:
        local  = ollama.list()
        installed = [m["model"] for m in local.get("models", [])]
        # Only return models from our supported list
        available = [m for m in SUPPORTED_MODELS if any(m in name for name in installed)]
        return {"available": available, "all_installed": installed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama not reachable: {e}")


@app.post("/chat")
def chat(request: ChatRequest):
    """Single model chat — returns response + performance metrics."""
    result = run_inference(request.model, request.query)
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result)
    return result


@app.post("/compare")
def compare(request: CompareRequest):
    """
    Runs the same query across multiple models sequentially.
    Returns all responses side by side so you can compare quality and speed.
    """
    results = []
    for model in request.models:
        result = run_inference(model, request.query)
        results.append(result)

    return {
        "query":    request.query,
        "hardware": "Mac M1 8GB Unified Memory",
        "results":  results
    }