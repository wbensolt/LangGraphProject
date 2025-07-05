import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_google_genai import GoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a twitter techie influncer assistant tasked with writing excellent twitter"
        "posts. Generate the best twitter post possible for the user's request. If the user "
        "provides critique, respond with a revised version of your previous attempts."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a viral twitter influncer grading a twet. Generate critique and "
        "recommandations for the user's tweet. Always provide detailed recommandations, including"
        "requests for length, virality, style, etc."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
#llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

LLM_MODEL = "llama3.2:latest"
llm = ChatOllama(model=LLM_MODEL)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

