from datetime import datetime
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool

LLM_MODEL = "llama3.2:latest"
llm = ChatOllama(model=LLM_MODEL)

test_message = [('system', "You are a fact expert who knows facts about {animal}"),
                ('human', "Tell me {fact_number} facts"),]

query = "What is the current time?"
#prompt_template = ChatPromptTemplate.from_messages(test_message)
prompt_template = hub.pull("hwchase17/react")#ChatPromptTemplate.from_template("{input}")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Return the current system time in the specified format."""
    current_time = datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [get_system_time]

agent = create_react_agent(llm, tools, prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#chain = prompt_template | llm | StrOutputParser()

"""prompt = prompt_template.invoke({
    "content": "football",
    "content_number": "2"
})"""
agent_executor.invoke({
    "input": query
})
#print(Resultat)