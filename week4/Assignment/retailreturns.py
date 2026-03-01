import os
import re
import base64
from typing import Annotated, List, TypedDict, Union, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# --- 1. INITIALIZATION & CONFIGURATION ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

# Initialize the Brain (GPT-4o for multimodal and reasoning)
llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

# --- 2. SECURITY GUARDRAILS (Input Rail) ---
def pii_guardrail(content: Union[str, List[Dict]]) -> Union[str, List[Dict]]:
    """
    Week 4 Requirement: Scans and redacts PII before processing[cite: 134, 173].
    Safely handles multimodal list-based content[cite: 151].
    """
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    cc_pattern = r'\b(?:\d[ -]*?){13,16}\b' # Basic pattern for credit cards

    def redact(text: str) -> str:
        text = re.sub(email_pattern, "[REDACTED_EMAIL]", text)
        text = re.sub(cc_pattern, "[REDACTED_CC]", text)
        return text

    if isinstance(content, str):
        return redact(content)
    
    if isinstance(content, list):
        cleaned_content = []
        for item in content:
            new_item = item.copy()
            if new_item.get("type") == "text":
                new_item["text"] = redact(new_item["text"])
            cleaned_content.append(new_item)
        return cleaned_content
    return content

# --- 3. TOOLS (Simulated MCP Filesystem Server) ---
@tool
def write_return_manifest(order_id: str, decision: str, notes: str):
    """
    Saves the return decision to the local filesystem using MCP standard[cite: 133].
    Use this for finalizing returns.
    """
    # Simulated MCP side-effect [cite: 114, 116]
    print(f"[MCP SERVER] Writing return manifest for {order_id}.txt...")
    return f"Successfully recorded {decision} for Order {order_id}."

tools = [write_return_manifest]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# --- 4. STATE DEFINITION ---
class ReturnState(TypedDict):
    """Tracks conversation history and assessment metrics[cite: 132]."""
    messages: Annotated[List[BaseMessage], add_messages]
    item_condition: str
    refund_eligibility: bool
    final_action: str

# --- 5. AGENT NODES ---

def pii_gate_node(state: ReturnState):
    """Layer 1: Security Firewall[cite: 172, 173]."""
    last_msg = state["messages"][-1]
    last_msg.content = pii_guardrail(last_msg.content)
    print("DEBUG: PII input rail check complete.")
    return {"messages": [last_msg]}

def vision_node(state: ReturnState):
    """Vision Agent: Analyzes multimodal evidence (product photos)[cite: 144, 151]."""
    print("-> Vision Agent assessing item condition...")
    response = llm.invoke(state['messages'])
    return {"item_condition": response.content}

def orchestrator_node(state: ReturnState):
    """Orchestrator: Synthesizes data for autonomous decision or HITL[cite: 142, 203]."""
    prompt = f"""
    Review return request. 
    Evidence: {state.get('item_condition', 'No image provided')}
    Instruction: If item is damaged/defective, call 'write_return_manifest' to Approve.
    Otherwise, flag for Human Review.
    """
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
    decision = "Instant Refund" if response.tool_calls else "Human Review Required"
    return {"final_action": decision, "messages": [response]}

# --- 6. GRAPH CONSTRUCTION ---
workflow = StateGraph(ReturnState)

workflow.add_node("pii_gate", pii_gate_node)
workflow.add_node("vision_agent", vision_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("pii_gate")
workflow.add_edge("pii_gate", "vision_agent")
workflow.add_edge("vision_agent", "orchestrator")

def route_decision(state: ReturnState):
    """Conditional routing for tool execution[cite: 125]."""
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("orchestrator", route_decision)
workflow.add_edge("tools", END)

app = workflow.compile()

# --- 7. LOCAL EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Retail Return Agent Started ---")
    
    # Simulate a multimodal input (Text + Image)
    test_input = {
        "messages": [
            HumanMessage(content=[
                {"type": "text", "text": "Return order 999. My email is max@store.com. The item arrived broken."},
                # In actual run, replace with base64 encoded image string
                {"type": "text", "text": ""}
            ])
        ]
    }
    
    for output in app.stream(test_input):
        print(output)