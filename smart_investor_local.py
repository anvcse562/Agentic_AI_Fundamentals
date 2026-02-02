


import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load API Key from .env file

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. DEFINE TOOLS ---
def get_stock_price(ticker):
    """Simulates fetching a stock price."""
    # In a real app, you would call Yahoo Finance or AlphaVantage API here
    return f"{ticker} is currently trading at $215.40"

def get_news(query):
    """Simulates a web search for news."""
    return f"Recent news for {query}: Analysts predict strong growth due to AI demand."

# Tool Registry
tools = {
    "get_stock_price": get_stock_price,
    "get_news": get_news
}

# --- 2. SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a Financial Research Agent.
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, output your Final Answer.

Use Action: tool_name[input] to use a tool.
Available tools: get_stock_price, get_news.

Example:
Thought: I need to check the price.
Action: get_stock_price[AAPL]
PAUSE
"""

# --- 3. THE AGENT LOOP ---
def run_agent(user_query):
    print(f"User: {user_query}\n")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    # Safety Limit: Max 5 iterations to prevent infinite loops
    for turn in range(5):
        print(f"--- Turn {turn + 1} ---")
        
        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        agent_text = response.choices[0].message.content
        print(f"Agent: {agent_text}")
        
        # Add agent response to history
        messages.append({"role": "assistant", "content": agent_text})
        
        # Check for Final Answer
        if "Final Answer:" in agent_text:
            print("\n✅ Task Complete!")
            return agent_text
            
        # Parse Action using Regex
        # Looking for: Action: tool_name[input]
        action_match = re.search(r"Action: (\w+)\[(.*)\]", agent_text)
        
        if action_match:
            tool_name = action_match.group(1)
            tool_input = action_match.group(2)
            
            # Execute Tool
            if tool_name in tools:
                print(f"⚙️ Executing {tool_name} with input: {tool_input}")
                try:
                    observation = tools[tool_name](tool_input)
                except Exception as e:
                    observation = f"Error: {str(e)}"
            else:
                observation = f"Error: Tool '{tool_name}' not found."
            
            print(f"👀 Observation: {observation}\n")
            
            # Feed Observation back to LLM
            messages.append({
                "role": "user", 
                "content": f"Observation: {observation}"
            })
        else:
            # If no action found and no Final Answer, prompt to continue
            print("No action detected. Continuing...\n")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # Ensure you have OPENAI_API_KEY in your .env file
    run_agent("Is Nvidia a good buy right now? Check price and news.")
    