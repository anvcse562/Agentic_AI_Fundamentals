import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. DEFINE TOOLS (The "Hands") ---
def get_stock_price(ticker):
    """Fetches the current stock price."""
    # Mock data - in prod this hits an API
    return json.dumps({"ticker": ticker, "price": 215.40, "currency": "USD"})

def get_news(ticker):
    """Searches for recent news about a company."""
    return json.dumps({"ticker": ticker, "news": "Earnings beat expectations. Analysts bullish."})

# --- 2. DEFINE TOOL SCHEMAS (The "API Definition") ---
# This JSON schema tells OpenAI exactly what functions are available
# It replaces the need for a complex System Prompt description
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker (e.g., AAPL)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get recent news for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker (e.g., NVDA)"}
                },
                "required": ["ticker"]
            }
        }
    }
]

# Map string names to actual functions for execution
available_functions = {
    "get_stock_price": get_stock_price,
    "get_news": get_news
}

# --- 3. THE AGENT LOOP ---
def run_native_agent(query):
    print(f"User: {query}\n")
    messages = [
        {"role": "system", "content": "You are a helpful financial assistant. Use tools to answer questions."},
        {"role": "user", "content": query}
    ]

    for turn in range(5): # Max 5 turns safety limit
        print(f"--- Turn {turn + 1} ---")
        
        # 1. Call LLM with Tools enabled
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto" # Let the model decide whether to use a tool or chat
        )
        
        msg = response.choices[0].message
        tool_calls = msg.tool_calls

        # 2. Check if the model wants to call a tool
        if tool_calls:
            print(f"🤖 Agent wants to call {len(tool_calls)} tool(s)...")
            
            # Important: Add the assistant's request to history
            messages.append(msg) 
            
            # 3. Execute Tools
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"⚙️ Calling: {function_name}({function_args})")
                
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                
                # 4. Feed Output back to LLM
                # We must include the 'tool_call_id' so the LLM knows which request this answer belongs to
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })
                print(f"👀 Observation: {function_response}")
        
        else:
            # No tool calls? The model has finished its task.
            print(f"✅ Final Answer: {msg.content}")
            break

if __name__ == "__main__":
    # This query forces the agent to use BOTH tools
    run_native_agent("What is the price of NVDA and is there any good news?")