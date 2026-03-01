import os
import re
import json
from typing import Annotated, TypedDict, Union, List

# --- 1. CONFIGURATION & ENVIRONMENT ---
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception:
    pass 

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


# howo to build and run  
# #
# docker build -t langgraph-agent .
# docker run --env-file .env -p 8000:8000 langgraph-agent
#  

# Try loading .env but ignore parsing errors to prevent container crashes


load_dotenv()
raw_key = os.getenv("OPENAI_API_KEY")
if raw_key:
    # Prints only the first 8 characters for safety
    print(f"DEBUG: Key found starting with: {raw_key[:8]}...")
else:
    print("DEBUG: No API Key found!")

llm = ChatOpenAI(model="gpt-4o", openai_api_key=raw_key)

# --- 2. SECURITY GUARDRAILS (PII Check) ---
def pii_guardrail(text: str) -> str:
    """
    Week 4 Requirement: Check to prevent PII leakage.
    Scans for emails and phone numbers to ensure safety before LLM processing.
    """
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    if re.search(email_pattern, text) or re.search(phone_pattern, text):
        # In a real app, you might raise an error or mask it.
        # Here we mask it to allow the agent to continue safely.
        text = re.sub(email_pattern, "[REDACTED_EMAIL]", text)
        text = re.sub(phone_pattern, "[REDACTED_PHONE]", text)
    return text

# --- 3. TOOLS (Simulated MCP Filesystem Server) ---
@tool
def write_to_filesystem(filename: str, content: str):
    """
    Simulates writing a file to a local filesystem via an MCP-like interface.
    Use this to save logs, reports, or data.
    """
    # In a real MCP setup, this would call an external MCP server via JSON-RPC
    # For the assignment, we simulate the side effect
    print(f"[MCP SERVER] Writing to {filename}...")
    return f"Successfully wrote {len(content)} bytes to {filename}."

@tool
def read_from_filesystem(filename: str):
    """Reads content from the simulated filesystem server."""
    return f"Content of {filename}: [Sample Data for Week 4 Assignment]"

tools = [write_to_filesystem, read_from_filesystem]
tool_node = ToolNode(tools)
model_with_tools = llm.bind_tools(tools)

# --- 4. LANGGRAPH STATE DEFINITION ---
class AgentState(TypedDict):
    # The 'add_messages' function tells LangGraph to append new messages to the list
    messages: Annotated[List[BaseMessage], add_messages]

# --- 5. AGENT LOGIC ---
def call_model(state: AgentState):
    messages = state['messages']
    
    # 1. Apply PII Guardrail
    if isinstance(messages[-1], HumanMessage):
        safe_content = pii_guardrail(messages[-1].content)
        messages[-1].content = safe_content
        print(f"DEBUG: Content sent to LLM: {safe_content}")
    
    # 2. IMPORTANT: You must call the model and RETURN the response
    response = model_with_tools.invoke(messages)
    
    # 3. Return the new message to be added to the state
    return {"messages": [response]}
    
def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- 6. COMPILE AND TEST RUN ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

if __name__ == "__main__":
    print("--- Starting Agent Test Run ---")
    # Test case: Includes PII (email) and a Tool request (write file)
    test_input = {
        "messages": [
            HumanMessage(content="My email is dev@test.com. Write a file called 'status.txt' saying 'ready'.")
        ]
    }
    
    for event in graph.stream(test_input):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)