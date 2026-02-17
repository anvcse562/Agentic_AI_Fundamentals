import os
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# Configuration for the agents
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]

# 1. Define the Assistant Agent (The "Brain")
# This agent acts as the AI that generates code or answers.
assistant = AssistantAgent(
    name="Assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0.7,
    },
    system_message="You are a helpful coding assistant. Write Python code to solve the user's request."
)

# 2. Define the User Proxy Agent (The "Executor")
# This agent acts on behalf of the user. It can execute code locally.
user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",  # Set to "ALWAYS" for interactive mode
    max_consecutive_auto_reply=3, # Limit the loop to prevent infinite chatter
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding_output", # Where to save/run code
        "use_docker": False, # Set to True for sandboxed execution
    },
)

# 3. Start the Conversation
# The user proxy initiates the chat with a specific task.
if __name__ == "__main__":
    print("--- AutoGen Demo: Stock Plotter ---")
    user_proxy.initiate_chat(
        assistant,
        message="Write a python script to plot a sine wave and save it to 'sine_wave.png'. When done, output 'TERMINATE'."
    )