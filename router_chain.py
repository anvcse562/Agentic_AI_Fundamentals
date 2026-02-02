

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colors and Env
init(autoreset=True)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. DEFINE SPECIALIZED HANDLERS ---
def handle_billing(query):
    """Simulates a specialized Billing Agent."""
    print(f"{Fore.YELLOW}⚡ Routing to BILLING System...{Style.RESET_ALL}")
    # In reality, this might query a SQL database or Stripe API
    return "Billing Agent: I see your last payment of $49.99 was processed on Jan 1st."

def handle_technical(query):
    """Simulates a specialized Tech Support Agent."""
    print(f"{Fore.CYAN}⚡ Routing to TECHNICAL Support...{Style.RESET_ALL}")
    # In reality, this might query Vector DB (RAG) docs
    return "Tech Agent: To reset your password, please go to Settings > Security."

def handle_general(query):
    """Fallback for general queries."""
    print(f"{Fore.GREEN}⚡ Routing to GENERAL Chat...{Style.RESET_ALL}")
    return "General Agent: I can help you with that. What specifically do you need?"

# --- 2. THE ROUTER (CLASSIFIER) ---
ROUTER_PROMPT = """
You are a Customer Support Router.
Classify the user query into one of these categories:
- BILLING (Payments, refunds, invoices)
- TECHNICAL (Bugs, login issues, configuration)
- GENERAL (Everything else)

Output ONLY the category name. Do not explain.
"""

def route_query(query):
    print(f"\n{Fore.WHITE}User Query: {query}{Style.RESET_ALL}")
    
    # 1. The Classification Step (Deterministically limiting output)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0.0  # Zero temp for strict classification
    )
    
    route = response.choices[0].message.content.strip().upper()
    print(f"{Fore.MAGENTA}🔍 Router Decision: {route}{Style.RESET_ALL}")
    
    # 2. The Branching Logic (Python Control Flow)
    if "BILLING" in route:
        return handle_billing(query)
    elif "TECHNICAL" in route:
        return handle_technical(query)
    else:
        return handle_general(query)

# --- 3. DEMO EXECUTION ---
if __name__ == "__main__":
    # Test Case 1: Billing
    print(route_query("I was charged twice for my subscription."))
    
    # Test Case 2: Technical
    print(route_query("I can't log in, I get a 403 error."))
    
    # Test Case 3: General
    print(route_query("Tell me a joke about AI."))
    

    