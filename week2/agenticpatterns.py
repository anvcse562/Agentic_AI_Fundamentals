import os
import json
import asyncio
import time
from typing import List, Dict, Any

# --- SETUP ---
try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError:
    print("Please install required packages: pip install openai python-dotenv")
    exit(1)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Helper for colorized output
def print_step(pattern: str, step: str, content: str):
    print(f"\n[{pattern}] \033[92m{step}\033[0m: {content}")

# Wrapper for OpenAI calls
def call_llm(system: str, user: str, model: str = "gpt-4o", json_mode: bool = False) -> str:
    if not client.api_key: return "Simulated Output (No API Key)"
    
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.7
    }
    if json_mode: kwargs["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

# =============================================================================
# PATTERN 1: PROMPT CHAINING (The Saga Pattern)
# Sequential decomposition: Step A Output -> Step B Input
# =============================================================================
def pattern_prompt_chaining():
    print("\n--- PATTERN 1: PROMPT CHAINING ---")
    user_input = "I want to cook a romantic dinner for 2, vegetarian, under 30 mins."

    # Step 1: Ideation (Extract Criteria -> Generate Idea)
    step1_response = call_llm(
        system="You are a Chef. Suggest ONE meal name based on constraints.",
        user=user_input
    )
    print_step("Chain", "Step 1 (Idea)", step1_response)

    # Step 2: Recipe Generation (Idea -> Ingredients)
    step2_response = call_llm(
        system="You are a Sous Chef. List ingredients for the provided meal name as a CSV list.",
        user=step1_response
    )
    print_step("Chain", "Step 2 (Ingredients)", step2_response)

    # Step 3: Shopping List (Ingredients -> Formatted JSON)
    step3_response = call_llm(
        system="You are a Clerk. Convert CSV ingredients to a JSON object with 'aisle' and 'item'.",
        user=step2_response,
        json_mode=True
    )
    print_step("Chain", "Step 3 (JSON)", step3_response)

# =============================================================================
# PATTERN 2: ROUTING (Dynamic Dispatch)
# Input -> Classifier -> Specialized Worker
# =============================================================================
def pattern_routing():
    print("\n--- PATTERN 2: ROUTING ---")
    inputs = [
        "My bill is wrong, I was charged twice.",
        "How do I reset my password?",
        "Tell me a joke."
    ]

    ROUTER_SYSTEM = """
    Classify the user query into exactly one category: 
    - BILLING
    - TECHNICAL
    - GENERAL
    Output only the category name.
    """

    for query in inputs:
        # 1. The Router decides WHERE to go
        category = call_llm(ROUTER_SYSTEM, query).strip().upper()
        
        # 2. Dispatch to specialized logic
        if "BILLING" in category:
            response = call_llm("You are a Billing Agent. Be empathetic.", query)
            print_step("Router", f"Route: BILLING", response)
        elif "TECHNICAL" in category:
            response = call_llm("You are a Tech Support. Be precise.", query)
            print_step("Router", f"Route: TECHNICAL", response)
        else:
            response = call_llm("You are a Chatbot. Be witty.", query)
            print_step("Router", f"Route: GENERAL", response)

# =============================================================================
# PATTERN 3: PARALLELIZATION (Scatter-Gather)
# Input -> Multiple Parallel Sub-tasks -> Aggregator
# =============================================================================
async def async_call_llm(system, user, tag):
    """Async wrapper for OpenAI to demonstrate parallel execution"""
    # Note: In Python, OpenAI client is sync by default. 
    # For true async, use AsyncOpenAI. Here we simulate using threads/async wrapper.
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_llm, system, user)

async def pattern_parallelization():
    print("\n--- PATTERN 3: PARALLELIZATION (SCATTER-GATHER) ---")
    topic = "The Future of AI in 2030"

    # Scatter: Launch 3 distinct perspectives simultaneously
    print_step("Parallel", "Scatter", f"Generating 3 perspectives on '{topic}'...")
    
    task1 = async_call_llm("You are an Optimist.", f"Write 1 sentence on {topic}.", "Optimist")
    task2 = async_call_llm("You are a Pessimist.", f"Write 1 sentence on {topic}.", "Pessimist")
    task3 = async_call_llm("You are a Realist.", f"Write 1 sentence on {topic}.", "Realist")

    # Gather: Wait for all to finish
    results = await asyncio.gather(task1, task2, task3)
    
    combined_input = f"Optimist: {results[0]}\nPessimist: {results[1]}\nRealist: {results[2]}"
    
    # Aggregator: Synthesize
    final_summary = call_llm("You are a Synthesizer. Combine these views into a balanced conclusion.", combined_input)
    print_step("Parallel", "Gather (Synthesis)", final_summary)

# =============================================================================
# PATTERN 4: ORCHESTRATOR-WORKERS (Saga Orchestration)
# Orchestrator plans -> Delegates to Workers -> Compiles results
# =============================================================================
def pattern_orchestrator():
    print("\n--- PATTERN 4: ORCHESTRATOR-WORKERS ---")
    complex_task = "Write a blog post about coffee. Section 1: History. Section 2: Health Benefits."

    # 1. Orchestrator: Breakdown
    plan_response = call_llm(
        system="You are an Editor. Break the blog topic into exactly 2 sub-task prompts for writers. Return JSON list ['task1', 'task2'].",
        user=complex_task,
        json_mode=True
    )
    
    try:
        tasks = json.loads(plan_response).get("tasks", [])
        # Fallback if keys differ
        if not tasks: tasks = list(json.loads(plan_response).values())[0]
    except:
        tasks = ["Write history of coffee", "Write health benefits of coffee"]

    print_step("Orchestrator", "Plan", str(tasks))

    # 2. Workers: Execute sub-tasks
    results = []
    for i, task in enumerate(tasks):
        # In a real system, these could run in parallel
        worker_output = call_llm("You are a Blog Writer. Write 1 short paragraph.", task)
        print_step("Orchestrator", f"Worker {i+1}", worker_output)
        results.append(worker_output)

    # 3. Orchestrator: Final Compile
    final_doc = "\n\n".join(results)
    print_step("Orchestrator", "Final Output", final_doc[:100] + "...")

# =============================================================================
# PATTERN 5: EVALUATOR-OPTIMIZER (Reflect-Refine Loop)
# Generator -> Evaluator -> Loop if rejected
# =============================================================================
def pattern_evaluator_optimizer():
    print("\n--- PATTERN 5: EVALUATOR-OPTIMIZER ---")
    request = "Write a python function to add two numbers, but make the variable names terrible."
    
    context = request
    status = "REJECTED"
    attempt = 0
    
    while status == "REJECTED" and attempt < 3:
        attempt += 1
        
        # 1. Generator
        draft = call_llm("You are a Junior Coder. Write the code requested.", context)
        print_step("Loop", f"Draft {attempt}", draft)
        
        # 2. Evaluator
        eval_response = call_llm(
            system="You are a Senior Dev. Evaluate code. If variable names are bad/unreadable, say PASS. If they are clean/good, say FAIL (because prompt asked for terrible names). Return JSON {'status': 'PASS'/'FAIL', 'feedback': 'string'}",
            user=f"Req: {request}\nCode: {draft}",
            json_mode=True
        )
        
        try:
            eval_json = json.loads(eval_response)
            status = "APPROVED" if eval_json.get("status") == "PASS" else "REJECTED"
            feedback = eval_json.get("feedback")
        except:
            status = "APPROVED" # Fallback exit
            feedback = "JSON Error"

        print_step("Loop", f"Evaluation ({status})", feedback)
        
        if status == "REJECTED":
            # Refine context with feedback
            context = f"Original Request: {request}\nPrevious Draft: {draft}\nCritique: {feedback}\nFix it."

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    if not client.api_key:
        print("WARNING: No API Key found. Results will be simulated/mocked.")
    
    # 1. Prompt Chaining
    pattern_prompt_chaining()
    
    # 2. Routing
    pattern_routing()
    
    # 3. Parallelization (Async)
    asyncio.run(pattern_parallelization())
    
    # 4. Orchestrator
    pattern_orchestrator()
    
    # 5. Evaluator-Optimizer
    pattern_evaluator_optimizer()