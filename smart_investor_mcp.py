import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
# install node js- brew install node for mac. install for windows accordinly

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- CONFIGURATION ---
# We will use the standard Filesystem MCP server provided by Anthropic/MCP community
# This requires Node.js installed.
# Command: npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/folder
ALLOWED_PATH = "./data"  # Ensure this folder exists locally
SERVER_PARAMS = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", ALLOWED_PATH],
    env=None
)

SYSTEM_PROMPT = """
You are an Agent connected to an MCP Filesystem Server.
You have access to external tools. 
When you need to use a tool, output a Function Call definition (or just describe it if using text mode).
For this demo, we will simply execute the tool if you ask for it.
"""

async def run_mcp_agent():
    # 1. Connect to the MCP Server
    print("🔌 Connecting to MCP Server...")
    
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            
            # 2. Dynamic Tool Discovery
            # The agent asks the server: "What can you do?"
            tools_list = await session.list_tools()
            print(f"✅ Connected! Found {len(tools_list.tools)} tools:")
            for tool in tools_list.tools:
                print(f"  - {tool.name}: {tool.description[:50]}...")

            # 3. Simulate Agent Logic (Simplified)
            # In a real app, the LLM would decide this. Here we hardcode the call to demo execution.
            print("\n--- Agent Task: Save a Research Report ---")
            
            file_name = "data/investment_report.txt"
            content = "Nvidia (NVDA) Analysis: Strong Buy based on AI infrastructure demand."
            
            # Check if 'write_file' is available
            tool_names = [t.name for t in tools_list.tools]
            if "write_file" in tool_names:
                print(f"🤖 Agent Decided: Call 'write_file' to save {file_name}")
                
                # 4. Execute Tool via MCP
                result = await session.call_tool(
                    "write_file",
                    arguments={
                        "path": file_name,
                        "content": content
                    }
                )
                
                print("✨ Tool Output:", result)
                print(f"📂 Check the '{ALLOWED_PATH}' folder for your file!")
            else:
                print("❌ Error: 'write_file' tool not found on server.")

if __name__ == "__main__":
    # Create data directory if not exists
    if not os.path.exists(ALLOWED_PATH):
        os.makedirs(ALLOWED_PATH)
        
    asyncio.run(run_mcp_agent())
    