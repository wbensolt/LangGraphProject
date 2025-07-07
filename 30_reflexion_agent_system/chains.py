from typing import Any
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import re
from dotenv import load_dotenv
import os

load_dotenv()

# Extraction manuelle des search queries depuis le texte
def extract_search_queries(text: str) -> list[str]:
    match = re.search(r"Search Queries:\s*((?:- .+\n?)*)", text)
    if not match:
        return []
    lines = match.group(1).strip().splitlines()
    return [line.strip().lstrip("- ").strip() for line in lines if line.strip()]

# Traitement de la première réponse
def process_response(response):
    content = response.content
    print("=== RAW RESPONSE ===")
    print(content)

    queries = extract_search_queries(content)
    print("\n✅ Extracted Search Queries:")
    for q in queries:
        print("-", q)

    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": "AnswerQuestion",
                "args": {
                    "search_queries": queries,
                    "reflections": {"missing": "", "superfluous": ""},
                    "answer": content
                },
                "id": "call_auto_1"
            }
        ]
    )

# Traitement de la version révisée
def process_revisor_response(response):
    content = response.content
    print("=== RAW REVISED RESPONSE ===")
    print(content)

    queries = extract_search_queries(content)
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": "ReviseAnswer",
                "args": {
                    "search_queries": queries,
                    "reflections": {"missing": "", "superfluous": ""},
                    "answer": content
                },
                "id": "call_auto_2"
            }
        ]
    )

# Prompt system
base_prompt = """You are an expert AI assistant.
Current time: {time}

{first_instruction}

Format your output strictly like this:

Answer:
<Your answer here>

Reflections:
- Missing: <what was missing>
- Superfluous: <what was unnecessary>

Search Queries:
- <query 1>
- <query 2>
- <query 3>
"""

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", base_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Write a ~250 word blog post answering the user question."
)

revisor_instruction = """Revise the previous answer using the reflection. 
- Add missing parts and remove superfluous content.
- Keep it under 250 words.
- Then propose 1-3 relevant search queries under 'Search Queries:'.
"""

revisor_prompt_template = actor_prompt_template.partial(
    first_instruction=revisor_instruction
)

# LLM config
#LLM_MODEL = "llama3.2:latest"
#llm = ChatOllama(model=LLM_MODEL)


llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.getenv("GROQ_API_KEY"))


# Chains
first_responder_chain = first_responder_prompt_template | llm | process_response
revisor_chain = revisor_prompt_template | llm | process_revisor_response
