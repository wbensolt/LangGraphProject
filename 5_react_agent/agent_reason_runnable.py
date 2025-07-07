import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import tool, create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool
from datetime import datetime
from langchain import hub

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Generative AI model
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key = os.getenv("GROQ_API_KEY"))

search_tool = TavilySearchResults(search_depth="basic", max_results=3)

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Return the current system time in the specified format."""
    current_time = datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


# Define the tools to be used by the agent
tools = [search_tool, get_system_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt= react_prompt)