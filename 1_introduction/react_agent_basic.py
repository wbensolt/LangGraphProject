from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

search_tool = TavilySearchResults(search_depth="basic", max_results=3)

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Return the current system time in the specified format."""
    current_time = datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


# Define the tools to be used by the agent
tools = [search_tool, get_system_time]

# Initialize the agent with the specified tools and model
agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

#agent.invoke("Give me a tweet about today's weather in Paris, France.")
agent.invoke("When was SpaceX's last launch and how many days ago was that from today?")

#results = agent.invoke("Give me a tweet about today's weather in Paris, France.")

#print(results)