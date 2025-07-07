from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain   
# Load environment variables from .env file
load_dotenv()

LLM_MODEL = "llama3.2:latest"
llm = ChatOllama(model=LLM_MODEL)

llm.invoke("Hello, world!")