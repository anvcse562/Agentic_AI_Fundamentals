import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Define the LLM (OpenAI via LangChain)
llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define Agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI Agents',
    backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
    # tools=[search_tool] # Add tools here in production
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# 2. Define Tasks
task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI Agents in 2024.
    Identify key drivers, major breakthrough technologies, and potential industry impacts.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog post
    that highlights the most significant AI Agent advancements.
    Your post should be informative yet accessible to a general tech-savvy audience.
    Make it sound cool, avoid complex words.""",
    expected_output="Full blog post of at least 3 paragraphs",
    agent=writer
)


crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True,  # <-- this is causing the error
)

# 4. Kickoff
if __name__ == "__main__":
    print("--- CrewAI Demo: Research Team ---")
    result = crew.kickoff()
    print("\n\n########################")
    print("## Final Result ##")
    print("########################\n")
    print(result)