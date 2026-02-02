import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from colorama import Fore, Style, init

init(autoreset=True)
load_dotenv()

# Use Async client for parallel execution
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. DEFINE THE WORKER ---
async def ask_agent(name, prompt, color):
    """Runs a single LLM call independently."""
    print(f"{color}🤖 {name} is thinking...{Style.RESET_ALL}")
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7 # High temp to encourage diversity of thought
    )
    
    content = response.choices[0].message.content
    print(f"{color}✅ {name} finished.{Style.RESET_ALL}")
    return f"{name}: {content}"

# --- 2. THE ORCHESTRATOR ---
async def run_consensus(topic):
    print(f"Topic: {topic}\n")
    
    prompt = f"Provide a brief, 1-sentence interesting fact about: {topic}"
    
    # 1. Parallel Execution (Scatter)
    # We launch 3 requests simultaneously, not sequentially
    results = await asyncio.gather(
        ask_agent("Agent A", prompt, Fore.YELLOW),
        ask_agent("Agent B", prompt, Fore.CYAN),
        ask_agent("Agent C", prompt, Fore.MAGENTA)
    )
    
    # 2. Aggregation (Gather)
    print(f"\n{Fore.WHITE}--- AGGREGATING RESULTS ---{Style.RESET_ALL}")
    combined_text = "\n".join(results)
    print(combined_text)
    
    # 3. Final Decision (Judge)
    print(f"\n{Fore.GREEN}--- FINAL JUDGMENT ---{Style.RESET_ALL}")
    final_verdict = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a Judge. Synthesize the following 3 facts into one definitive truth."},
            {"role": "user", "content": combined_text}
        ]
    )
    print(final_verdict.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(run_consensus("The origin of the Python programming language name"))
    
    