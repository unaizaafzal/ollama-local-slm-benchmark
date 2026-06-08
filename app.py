import time
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
from fastapi import FastAPI, HTTPException
from openai import OpenAI

# Initialize FastAPI App
app = FastAPI(title="Local SLM Benchmarking Engine")

# 1. PYDANTIC TARGET SCHEMAS

class CompanySentiment(BaseModel):
    company_name: str = Field(
        ..., 
        description="The name of the company discussed.",
        validation_alias="Company Name"  # Accepts title case variation
    )
    ticker: Optional[str] = Field(
        None, 
        description="Stock ticker symbol if mentioned, uppercase.",
        validation_alias="Ticker"
    )
    sentiment_score: float = Field(
        ..., 
        description="Sentiment score ranging from -1.0 to 1.0.",
        validation_alias="Sentiment Score"
    )
    key_driver: str = Field(
        ..., 
        description="A one-sentence summary of what drove this specific sentiment.",
        validation_alias="Key Driver"
    )

class FinancialReportAnalysis(BaseModel):
    market_trend_summary: str = Field(
        ..., 
        description="Overall direction of the market described.",
        validation_alias="Market Trend Summary"  # Fixes Phi-3 mutation
    )
    companies_mentioned: List[CompanySentiment] = Field(
        ..., 
        description="List of companies identified and analyzed.",
        validation_alias="Companies Mentioned"  # Fixes Phi-3 mutation
    )

# 2. INSTRUCTOR CONFIGURATION & BENCHMARK CORE

# Initialize Instructor using the unified provider routing
# ---------------------------------------------------------
# 2. INSTRUCTOR CONFIGURATION & BENCHMARK CORE
# ---------------------------------------------------------
from openai import OpenAI

# Using the high-speed OpenAI-compatible local proxy routing
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ),
    mode=instructor.Mode.JSON
)

def run_inference_benchmark(model_name: str, test_prompt: str) -> dict:
    """
    Executes an extraction call against a specific local Ollama model
    and measures foundational latency performance.
    """
    start_time = time.perf_counter()
    print(f"\n Starting benchmark execution for model: {model_name}...")
    
    # We append a structural hint to the prompt to force small models (like Phi-3)
    # to output keys at the root instead of wrapping them in a class name block.
    optimized_prompt = (
        f"{test_prompt}\n\n"
        "CRITICAL INSTRUCTION: Output your JSON directly using the keys 'market_trend_summary' "
        "and 'companies_mentioned' at the root level. Do NOT wrap the JSON inside any parent key."
    )
    
    try:
        structured_data = client.chat.completions.create(
            model=model_name,
            response_model=FinancialReportAnalysis,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an advanced financial analyst engine. Extract information precisely into JSON schemas."
                },
                {
                    "role": "user", 
                    "content": optimized_prompt
                }
            ]
        )
        
        execution_time = time.perf_counter() - start_time
        print(f" Success! {model_name} finished in {execution_time:.3f}s")
        
        return {
            "status": "success",
            "model": model_name,
            "latency_seconds": round(execution_time, 3),
            "data": structured_data.model_dump()
        }
        
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        print(f" Failed execution for {model_name}. Error type: {type(e).__name__}")
        print(f" Error details: {str(e)}")
        
        return {
            "status": "failed",
            "model": model_name,
            "latency_seconds": round(execution_time, 3),
            "error_type": type(e).__name__,
            "error_msg": str(e)
        }

# ---------------------------------------------------------
# 3. MOCK DATA & ENDPOINTS
# ---------------------------------------------------------
MOCK_FINANCIAL_TEXT = r"""
Early morning trading showed major volatility. Nvidia Corp (NVDA) surged by 4.2% following 
unprecedented demand for their next-gen compute chips, highlighting massive enterprise AI spending. 
Conversely, Apple (AAPL) dropped slightly by 0.8% due to localized supply chain bottlenecks in East Asia 
that might restrict hardware shipments this quarter. Overall, tech sectors are leading a bullish rally 
while retail indices remain completely flat.
"""

@app.get("/benchmark/single")
def benchmark_model(model: str = "phi3:mini"):
    """
    Benchmark a single local model on the financial extraction task.
    Example query: /benchmark/single?model=phi3:mini
    """
    result = run_inference_benchmark(model, MOCK_FINANCIAL_TEXT)
    if result["status"] == "failed":
        # Returns the error explicitly to the Swagger UI instead of throwing an empty 500 error
        raise HTTPException(status_code=500, detail=result)
    return result

@app.get("/benchmark/all")
def benchmark_all_models():
    """
    Sequentially loops through the 3 targeted models to compare latency profiles.
    """
    target_models = ["qwen2.5:1.5b", "phi3:mini"]
    comparison_report = []
    
    for model in target_models:
        report = run_inference_benchmark(model, MOCK_FINANCIAL_TEXT)
        comparison_report.append(report)
        
    return {
        "hardware_context": "Mac M1 8GB Unified Memory",
        "benchmark_results": comparison_report
    }
    

