import streamlit as st
import requests

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title="Local SLM Benchmarker",
    page_icon="🖇",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Local SLM Benchmarker")
st.caption("Entirely offline · No API keys · No internet · Your data stays on your machine")
st.divider()

# ── Sidebar: backend status + model list ──────────────────────────────────────
with st.sidebar:
    st.subheader("System Status")

    try:
        r = requests.get(f"{BACKEND}/models", timeout=3)
        r.raise_for_status()
        available_models = r.json().get("available", [])
        st.success("Backend is running")
        if available_models:
            st.markdown("**Models ready:**")
            for m in available_models:
                st.markdown(f"- `{m}`")
        else:
            st.warning("No supported models found. Pull them first.")
            st.code("ollama pull qwen2.5:1.5b\nollama pull phi3:mini", language="bash")

    except Exception:
        available_models = []
        st.error("FastAPI backend not running")
        st.markdown("Open a terminal and run:")
        st.code("uvicorn app:app --reload", language="bash")

    st.divider()
    st.subheader("Privacy")
    st.markdown(
        "Everything runs on your hardware. "
        "Your queries never leave your machine. "
        "No tokens consumed. No logs sent anywhere."
    )

# ── Query input ───────────────────────────────────────────────────────────────
st.subheader("Your Query")
query = st.text_area(
    label="query",
    placeholder="Ask anything — e.g. Explain transformers in simple terms.",
    height=120,
    label_visibility="collapsed"
)

# ── Model selector ────────────────────────────────────────────────────────────
st.subheader("Select Models to Compare")

ALL_MODELS = ["qwen2.5:1.5b", "phi3:mini", "llama3.2:3b"]

cols = st.columns(3)
selected = []

for col, model in zip(cols, ALL_MODELS):
    with col:
        is_ready = model in available_models
        label    = f"**`{model}`**" if is_ready else f"~~`{model}`~~ — not installed"
        checked  = st.checkbox(label, value=is_ready, disabled=not is_ready, key=model)
        if checked:
            selected.append(model)

        if not is_ready:
            st.caption(f"`ollama pull {model}`")

st.markdown("")
run = st.button("⚡ Compare", type="primary", use_container_width=True, disabled=not available_models)

# ── Results ───────────────────────────────────────────────────────────────────
if run:
    if not query.strip():
        st.warning("Enter a query first.")
        st.stop()
    if not selected:
        st.warning("Select at least one model.")
        st.stop()

    with st.spinner(f"Running on {len(selected)} model(s)… this takes 20–60s on M1"):
        try:
            resp = requests.post(
                f"{BACKEND}/compare",
                json={"query": query, "models": selected},
                timeout=180
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Lost connection to backend. Is `uvicorn app:app --reload` still running?")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    results = data.get("results", [])
    st.divider()
    st.subheader("Results")

    result_cols = st.columns(len(results))
    successful  = [r for r in results if r.get("status") == "success"]

    # find fastest for badge
    fastest = min(successful, key=lambda r: r["latency_seconds"])["model"] if successful else None

    for col, result in zip(result_cols, results):
        with col:
            model   = result["model"]
            status  = result["status"]
            latency = result.get("latency_seconds", 0)
            tps     = result.get("tokens_per_second")
            tokens  = result.get("tokens_generated")

            badge = " ⚡ fastest" if (model == fastest and len(successful) > 1) else ""
            st.markdown(f"#### `{model}`{badge}")

            if status == "failed":
                st.error(f"Failed: {result.get('error', 'unknown error')}")
                continue

            # metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Latency",    f"{latency}s")
            m2.metric("Tokens/sec", f"{tps}" if tps else "–")
            m3.metric("Tokens out", f"{tokens}" if tokens else "–")

            # response
            st.markdown("**Response**")
            st.markdown(
                f"<div style='background:#1e1e2e;border:1px solid #313244;"
                f"border-radius:8px;padding:16px;font-size:0.9rem;line-height:1.7;"
                f"color:#cdd6f4'>{result['response']}</div>",
                unsafe_allow_html=True
            )

    # ── Summary table ──────────────────────────────────────────────────────────
    if len(successful) > 1:
        st.divider()
        st.subheader("📊 Summary")
        st.table([
            {
                "Model":       r["model"],
                "Latency (s)": r["latency_seconds"],
                "Tokens/sec":  r.get("tokens_per_second") or "–",
                "Tokens out":  r.get("tokens_generated") or "–",
            }
            for r in successful
        ])
        st.caption(
            "Latency = total wall-clock time · "
            "Tokens/sec = generation throughput (higher = faster) · "
            "Quality is subjective — read the responses above"
        )

    # raw JSON for debugging / learning
    with st.expander("Raw API response (good for learning)"):
        st.json(data)