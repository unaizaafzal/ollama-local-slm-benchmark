 Local SLM Benchmarking Engine

> Compare local language models side-by-side — fully offline, zero API cost, your data never leaves your machine.

Built with **Python · FastAPI · Ollama · Streamlit · Pydantic**


## What This Does

Most LLM apps send your data to OpenAI or Anthropic's servers. This one doesn't.

This system runs large language models entirely on your local hardware using Ollama, exposes a benchmarking API via FastAPI, and lets you compare model responses and performance metrics through a Streamlit UI — with your WiFi off if you want.

You type a question. You pick which models to run it against. You get back the responses, latency, and tokens/sec for each model side by side. No API keys. No internet. No cost per query.



## Benchmark Results

Tested on **Mac M1 8GB Unified Memory** with the question:  
*"Explain the difference between supervised and unsupervised learning"*

| Model | Parameters | Latency | Tokens/sec | Tokens Generated |
|---|---|---|---|---|
| Qwen 2.5 | 1.5B | 8.52s | 14.1 | 120 |
| Phi-3 Mini | 3.8B | 20.65s | 8.0 | 166 |

**Key finding:** Phi-3 Mini is 142% slower than Qwen 2.5 while generating only 38% more tokens — a direct reflection of the parameter count difference on constrained hardware. Qwen 2.5 is the better choice for speed-critical tasks on 8GB machines. Phi-3 Mini produces longer, more detailed responses where answer depth matters more than latency.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                     │
│              localhost:8501  ·  ui.py               │
│  Query input · Model selector · Results + metrics   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP POST /compare
┌──────────────────────▼──────────────────────────────┐
│                   FastAPI Backend                   │
│              localhost:8000  ·  app.py              │
│  Pydantic validation · Inference · Metrics capture  │
└──────────────────────┬──────────────────────────────┘
                       │ ollama.chat()
┌──────────────────────▼──────────────────────────────┐
│                  Ollama Model Server                │
│                   localhost:11434                   │
│    Qwen 2.5 1.5B · Phi-3 Mini 3.8B · Llama 3.2 3B   │
└─────────────────────────────────────────────────────┘

Everything runs locally. No external calls.
```



## Why Local Inference Matters

Running models locally isn't just a technical choice, it's an architecture decision that maps to real business constraints:

**Privacy** — Healthcare (HIPAA), legal (attorney-client privilege), and finance (IP protection) cannot send sensitive documents to external APIs. Local inference is the only option.

**Cost** — No per-token billing. A team running 10,000 queries/day pays $0 in inference costs.

**Latency** — No network round trip. Useful for edge deployments, air-gapped environments, and offline-first applications.

**Control** — The model version doesn't change unless you change it. No silent API updates breaking your prompts.



## Tech Stack

| Layer | Tool | Version | Why |
|---|---|---|---|
| Model serving | Ollama | 0.6.2 | Runs GGUF quantized models locally with an OpenAI-compatible API |
| API backend | FastAPI | 0.136.3 | Async Python API framework with automatic Swagger docs at `/docs` |
| Request validation | Pydantic | 2.13.4 | Validates incoming API requests before they reach inference code |
| Frontend | Streamlit | 1.58.0 | Rapid Python UI — no JavaScript needed |
| API server | Uvicorn | 0.49.0 | ASGI server that runs the FastAPI app |
| Performance timing | `time.perf_counter()` | stdlib | Highest resolution wall-clock timer in Python |

---

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed

### 1. Clone the repo
```bash
git clone https://github.com/unaizaafzal/ollama-local-slm-benchmark.git
cd ollama-local-slm-benchmark
```

### 2. Create virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull the models (requires internet — one time only)
```bash
ollama pull qwen2.5:1.5b
ollama pull phi3:mini
ollama pull llama3.2:3b          # optional, needs ~2GB
```

### 4. Run (3 terminals, all with venv activated)

**Terminal 1 — Model server:**
```bash
ollama serve
```

**Terminal 2 — API backend:**
```bash
source .venv/bin/activate
uvicorn app:app --reload
```

**Terminal 3 — UI:**
```bash
source .venv/bin/activate
streamlit run ui.py
```

Open `http://localhost:8501` in your browser.

**At this point you can turn your WiFi off. Everything runs locally.**

---

## API Endpoints

The FastAPI backend runs at `http://localhost:8000`. Interactive docs available at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/models` | Lists which supported models are installed and ready |
| POST | `/chat` | Single model inference with latency metrics |
| POST | `/compare` | Runs the same query across multiple models sequentially |

Example `/compare` request:
```json
{
  "query": "Explain transformers in simple terms",
  "models": ["qwen2.5:1.5b", "phi3:mini"]
}
```

---

## Project Structure

```
ollama-local-slm-benchmark/
├── app.py              # FastAPI backend — inference engine and API endpoints
├── ui.py               # Streamlit frontend — query input, model selection, results
├── requirements.txt    # Python dependencies
└── .gitignore          # Excludes .venv, __pycache__, .DS_Store
```

---

## Hardware Notes

Tested on Mac M1 8GB. M1's unified memory architecture means the GPU and CPU share the same memory pool, which makes it more capable for local inference than equivalent Intel/AMD setups with the same RAM. Models larger than ~4B parameters will be slow or fail to load on 8GB.

Recommended models for 8GB machines:
- Qwen 2.5 1.5B — fastest, lowest memory footprint
- Phi-3 Mini 3.8B — better reasoning, ~2.5x slower
- Llama 3.2 3B — balanced, good general purpose performance

---

## Understanding the Metrics

**Latency (seconds)** — Total wall-clock time from sending the request to receiving the complete response. Measured with `time.perf_counter()` for nanosecond precision.

**Tokens/sec** — Generation throughput. How fast the model produces output tokens. Higher is faster. Calculated as `eval_count / elapsed_time` where `eval_count` comes from Ollama's response metadata.

**Tokens generated** — Total number of tokens in the model's response. Longer responses take more time — so a model with lower tokens/sec but fewer tokens may still finish faster.
