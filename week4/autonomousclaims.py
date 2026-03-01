import os
import re
import base64
from typing import Annotated, List, TypedDict, Union, Dict
from dotenv import load_dotenv



# Metric	Value	Result
# PII Status	Redacted	Secure
# Damage Assessment	Fender Bender	Verified
# Fraud Risk	0.15	Low (Auto-Approve)
# Final Action	Tool Call	Report Saved

# --- 1. INITIALIZATION ---
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Validate API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("CRITICAL ERROR: OPENAI_API_KEY is not set.")
    exit(1)

llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

# Helper for multimodal encoding
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

# --- 2. SECURITY GUARDRAILS ---
def pii_guardrail(content: Union[str, List[Dict]]) -> Union[str, List[Dict]]:
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'

    def redact(text: str) -> str:
        text = re.sub(email_pattern, "[REDACTED_EMAIL]", text)
        text = re.sub(phone_pattern, "[REDACTED_PHONE]", text)
        return text

    if isinstance(content, str):
        return redact(content)
    if isinstance(content, list):
        new_content = []
        for item in content:
            new_item = item.copy()
            if new_item.get("type") == "text":
                new_item["text"] = redact(new_item["text"])
            new_content.append(new_item)
        return new_content
    return content

# --- 3. TOOLS ---
@tool
def save_adjudication_report(claim_id: str, content: str):
    """Saves the final claim report to the enterprise filesystem."""
    print(f"[MCP SERVER] Saving report {claim_id}.txt...")
    return f"Report saved successfully for claim {claim_id}."

tools = [save_adjudication_report]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# --- 4. STATE DEFINITION ---
class ClaimState(TypedDict):
    # add_messages is vital for keeping the history sequence intact
    messages: Annotated[List[BaseMessage], add_messages]
    damage_report: str
    coverage_status: str
    fraud_score: float
    final_decision: str

# --- 5. AGENT NODES ---

def pii_gate(state: ClaimState):
    """Cleans the latest message and passes it forward."""
    last_msg = state["messages"][-1]
    cleaned_content = pii_guardrail(last_msg.content)
    # We return a list to append to the state via add_messages
    return {"messages": [HumanMessage(content=cleaned_content)]}

def vision_agent(state: ClaimState):
    print("-> Vision Agent analyzing image data...")
    # In a real run, this sends the image to GPT-4o
    response = llm.invoke(state['messages'])
    return {"damage_report": response.content}

def policy_agent(state: ClaimState):
    print("-> Policy Agent checking coverage...")
    return {"coverage_status": "Comprehensive Policy Active. $500 Deductible applies."}

def fraud_agent(state: ClaimState):
    print("-> Fraud Agent checking for anomalies...")
    return {"fraud_score": 0.15}

def orchestrator(state: ClaimState):
    print("-> Orchestrator synthesizing decision...")
    prompt = f"""Review claim: Damage: {state['damage_report']}, Policy: {state['coverage_status']}, Fraud: {state['fraud_score']}.
    If risk < 0.2, call 'save_adjudication_report'. Otherwise, flag 'Human Review'."""
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
    decision = "Approve" if response.tool_calls else "Human Review"
    return {"final_decision": decision, "messages": [response]}

# --- 6. GRAPH CONSTRUCTION ---
workflow = StateGraph(ClaimState)

workflow.add_node("pii_gate", pii_gate)
workflow.add_node("vision", vision_agent)
workflow.add_node("policy", policy_agent)
workflow.add_node("fraud", fraud_agent)
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("pii_gate")
workflow.add_edge("pii_gate", "vision")
workflow.add_edge("vision", "policy")
workflow.add_edge("policy", "fraud")
workflow.add_edge("fraud", "orchestrator")

def route_decision(state: ClaimState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("orchestrator", route_decision)
workflow.add_edge("tools", END)

app = workflow.compile()

# --- 7. RUN ---
if __name__ == "__main__":
    image_b64 = encode_image("/Users/madmax_jos/Desktop/car_damage.jpg")
    
    if image_b64:
        content = [
            {"type": "text", "text": "Claim for max@email.com. Analyze damage."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    else:
        content = "Claim for max@email.com. Analysis needed for fender bender."

    inputs = {"messages": [HumanMessage(content=content)]}
    for output in app.stream(inputs):
        print(output)