import streamlit as st
import httpx
import pandas as pd

# Configure modern, wide page layout
st.set_page_config(
    page_title="Local SLM Benchmarking Dashboard",
    page_icon="",
    layout="wide"
)

# FastAPI Backend Target Base URL
FASTAPI_URL = "http://127.0.0.1:8000"

st.title("💿 Local SLM Performance & Inference Benchmarking")
st.markdown(
    "Run fully offline models entirely on your local hardware. Compare execution speeds, "
    "structural schema adherence, and latency constraints transparently."
)

st.sidebar.header(" Hardware & Engine Configuration")
st.sidebar.info(
    "**Current Hardware:**\nMac M1 8GB Unified Memory\n\n"
    "**Configured Models:**\n- Qwen 2.5 (1.5B)\n- Phi-3 Mini (3.8B)"
)

# Interactive Trigger Button
trigger_benchmark = st.button(" Fire Sequenced Inference Benchmark", type="primary")

if trigger_benchmark:
    with st.spinner("Executing extraction routines across local models... Watch Activity Monitor."):
        try:
            # Reaching out to your FastAPI server endpoint
            response = httpx.get(f"{FASTAPI_URL}/benchmark/all", timeout=180.0)
            
            if response.status_code == 200:
                results_data = response.json()
                raw_results = results_data.get("benchmark_results", [])
                
                st.success(" Benchmark Routine Completed Successfully!")
                
                # --- VISUAL CARD METRICS ---
                st.subheader(" Foundational Latency Profiles")
                col1, col2 = st.columns(2)
                
                for idx, run in enumerate(raw_results):
                    target_col = col1 if idx == 0 else col2
                    with target_col:
                        model_name = run.get("model", "Unknown")
                        status = run.get("status", "failed")
                        latency = run.get("latency_seconds", 0.0)
                        
                        if status == "success":
                            target_col.metric(
                                label=f" {model_name} (Success)", 
                                value=f"{latency}s",
                                delta="Optimal Target" if latency < 15.0 else "VRAM Overhead"
                            )
                        else:
                            target_col.metric(
                                label=f" {model_name} (Failed)", 
                                value=f"{latency}s",
                                delta="Schema Error",
                                delta_color="inverse"
                            )
                
                st.markdown("---")
                
                # --- PARSED VECTOR COMPARISON DATA TABLE ---
                st.subheader(" Structured Output Extraction Integrity")
                
                table_rows = []
                for run in raw_results:
                    model_name = run.get("model", "Unknown")
                    status = run.get("status", "failed")
                    latency = run.get("latency_seconds", 0.0)
                    
                    if status == "success":
                        extracted_data = run.get("data", {})
                        trend = extracted_data.get("market_trend_summary", "N/A")
                        companies = extracted_data.get("companies_mentioned", [])
                        companies_str = ", ".join([c.get("company_name", "") for c in companies])
                    else:
                        trend = "FAIL: Refused schema structure bounds."
                        companies_str = f"N/A ({run.get('error_type', 'Error')})"
                    
                    table_rows.append({
                        "Model Target": model_name,
                        "Status Check": " Passed" if status == "success" else " Failed",
                        "Inference Latency": f"{latency}s",
                        "Extracted Market Trend Summary": trend,
                        "Companies Extracted": companies_str
                    })
                
                df = pd.DataFrame(table_rows)
                st.dataframe(df, width="stretch", hide_index=True)
                
            else:
                st.error(f"Backend Engine returned an operational error code: {response.status_code}")
                
        except httpx.ConnectError:
            st.error(" Connection Failed! Make sure your FastAPI backend app is running on port 8000.")
        except Exception as e:
            st.error(f"An unhandled interface error occurred: {str(e)}")