import json
import time
import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime


# Observability: A Tracing class that logs every step (Input -> LLM -> Output) to a JSONL file (simulating a system like LangSmith).

# Guardrails: A decorator @guardrail_pii that scans agent outputs for sensitive data (emails) and redacts them.

# Caching: A simple semantic cache simulation to avoid redundant LLM calls.

# End-to-End Flow: A secure, observable agent that processes a user request using these production modules.
# --- SETUP ---
try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError:
    print("Please install required packages: pip install openai python-dotenv")
    exit(1)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================================================
# MODULE 1: OBSERVABILITY (Tracing)
# =============================================================================
class Tracer:
    """
    Simulates a production tracing system (like LangSmith or Arize).
    Logs every 'span' (unit of work) to a local JSONL file.
    """
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.spans = []
        
    def start_span(self, name: str, input_data: Any):
        span = {
            "trace_id": self.trace_id,
            "span_name": name,
            "start_time": datetime.now().isoformat(),
            "input": input_data,
            "status": "RUNNING"
        }
        self.spans.append(span)
        return len(self.spans) - 1 # Return span index

    def end_span(self, span_index: int, output_data: Any):
        span = self.spans[span_index]
        span["end_time"] = datetime.now().isoformat()
        span["output"] = output_data
        span["status"] = "COMPLETED"
        self._log_to_file(span)

    def _log_to_file(self, span: Dict):
        # In prod, this would send an async HTTP request to a collector
        with open("agent_traces.jsonl", "a") as f:
            f.write(json.dumps(span) + "\n")
        print(f"   [Trace] Logged span: {span['span_name']}")

# =============================================================================
# MODULE 2: SECURITY (Guardrails)
# =============================================================================
def guardrail_pii(func):
    """
    Output Guardrail: Scans the function's return value for PII (emails).
    If found, it redacts them before returning to the user.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Regex for email detection
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        if isinstance(result, str):
            redacted_result = re.sub(email_pattern, "[REDACTED_EMAIL]", result)
            if redacted_result != result:
                print("   [Guardrail] 🛡️ ALERT: PII detected and redacted.")
            return redacted_result
        return result
    return wrapper

# =============================================================================
# MODULE 3: SCALING (Caching)
# =============================================================================
# In prod, use Redis. Here we use a simple dict.
CACHE_STORE = {}

def check_cache(prompt: str) -> Optional[str]:
    # In prod, we would embed the prompt and do a vector similarity search
    # Here we do exact match for simplicity
    return CACHE_STORE.get(prompt)

def update_cache(prompt: str, response: str):
    CACHE_STORE[prompt] = response

# =============================================================================
# CORE: THE PRODUCTION AGENT
# =============================================================================
@guardrail_pii
def secure_agent_executor(user_query: str, tracer: Tracer) -> str:
    """
    A robust function that runs the agent logic wrapped with 
    Tracing, Caching, and Guardrails.
    """
    
    # 1. Start Trace
    span_idx = tracer.start_span("secure_agent_execution", user_query)
    
    # 2. Check Cache
    cached = check_cache(user_query)
    if cached:
        print("   [Cache] ⚡ Hit! Returning cached response.")
        tracer.end_span(span_idx, {"source": "cache", "response": cached})
        return cached

    # 3. Call LLM (Simulated Work)
    if not client.api_key:
        # Mock response if no key
        response = f"Sure, I can help. Contact support at admin@company.com for details."
    else:
        try:
            # We explicitly ask the model to include an email to test the guardrail
            prompt = f"{user_query}. Include a contact email in your response."
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            response = completion.choices[0].message.content
        except Exception as e:
            response = f"Error: {e}"

    # 4. Update Cache
    update_cache(user_query, response)
    
    # 5. End Trace
    tracer.end_span(span_idx, {"source": "llm", "response": response})
    
    return response

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("--- Week 4: Production Agent Demo ---\n")
    
    # Initialize Observability
    trace_id = f"trace_{int(time.time())}"
    tracer = Tracer(trace_id)
    
    # Test 1: First Run (LLM Call + Guardrail Check)
    print(f"1. Processing Query (Trace ID: {trace_id})")
    query = "How do I reset my password?"
    result = secure_agent_executor(query, tracer)
    print(f"   [Agent Output]: {result}\n")
    
    # Test 2: Second Run (Cache Hit)
    print("2. Processing Same Query (Testing Cache)")
    result_cached = secure_agent_executor(query, tracer)
    print(f"   [Agent Output]: {result_cached}\n")
    
    print("--- Demo Complete. Check 'agent_traces.jsonl' for logs. ---")