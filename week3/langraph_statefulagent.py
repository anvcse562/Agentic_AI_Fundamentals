import os
import operator
from typing import Annotated, TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define State ---
# The state is the "Blackboard" passed between nodes.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    plan: List[str]
    final_answer: str

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# --- 2. Define Nodes ---

def planner_node(state: AgentState):
    print("--- PLANNER NODE ---")
    messages = state['messages']
    # Simple prompt to generate a plan
    plan_response = llm.invoke([
        HumanMessage(content="You are a planner. Generate a list of 3 steps to solve: " + messages[-1].content)
    ])
    # Mocking a parsed list for the demo
    plan = ["Step 1: Research", "Step 2: Draft", "Step 3: Review"] 
    return {"plan": plan, "current_step": "plan_created"}

def executor_node(state: AgentState):
    print("--- EXECUTOR NODE ---")
    plan = state['plan']
    # Simulate executing the plan
    execution_log = f"Executed {len(plan)} steps successfully."
    return {"messages": [AIMessage(content=execution_log)], "current_step": "executed"}

def reviewer_node(state: AgentState):
    print("--- REVIEWER NODE ---")
    # Simulate a review process
    return {"final_answer": "Task Completed and Verified.", "current_step": "finished"}

# --- 3. Define Conditional Logic (Edges) ---
def router(state: AgentState):
    step = state['current_step']
    if step == "plan_created":
        return "executor"
    elif step == "executed":
        return "reviewer"
    elif step == "finished":
        return "end"
    return "end"

# --- 4. Build the Graph ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reviewer", reviewer_node)

# Add edges
workflow.set_entry_point("planner")

# Conditional edges allow dynamic routing
workflow.add_conditional_edges(
    "planner",
    router,
    {"executor": "executor", "end": END}
)
workflow.add_conditional_edges(
    "executor",
    router,
    {"reviewer": "reviewer", "end": END}
)
workflow.add_conditional_edges(
    "reviewer",
    router,
    {"end": END}
)

# Compile
app = workflow.compile()

# --- 5. Run the Agent ---
if __name__ == "__main__":
    print("--- LangGraph Stateful Demo ---")
    inputs = {"messages": [HumanMessage(content="Write a report on AI.")]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            # print(f"Current State: {value}")