from typing import Any, Dict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_ollama import ChatOllama
import re
from schema import AnswerQuestion, ReviewAnswer
from langchain_core.messages import HumanMessage
import json
from dotenv import load_dotenv

load_dotenv()


def parse_json_response_(response_content: str) -> Dict[str, Any]:
    try:
        print("=== RAW RESPONSE ===")
        print(response_content)

        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON object found.")

        cleaned_json_str = json_match.group()
        print("=== CLEANED RESPONSE ===")
        print(cleaned_json_str)

        parsed_json = json.loads(cleaned_json_str)

        # ✅ S'assurer que les champs existent avant validation Pydantic
        for key in ['answer', 'reflections', 'search_queries']:
            if key not in parsed_json:
                raise ValueError(f"Missing key in JSON: {key}")

        parsed = ReviewAnswer.model_validate(parsed_json)

        print("=== PARSED ===")
        print(parsed)
        return parsed

    except json.JSONDecodeError as e:
        print("❌ JSON Decode Error:", e)
        raise
    except Exception as e:
        print("❌ Parsing Error:", e)
        raise


def parse_json_response(response_content: str) -> Dict[str, Any]:
    try:
        # Afficher la réponse brute pour vérification
        print("=== RAW RESPONSE ===")
        print(response_content)

        # Nettoyer la réponse
        cleaned_json_str = response_content.strip()
        cleaned_json_str = re.sub(r'[^\x20-\x7E]', '', cleaned_json_str)  # Supprimer les caractères non ASCII
        cleaned_json_str = cleaned_json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # Afficher la réponse nettoyée pour vérification
        print("=== CLEANED RESPONSE ===")
        print(cleaned_json_str)

        # Parser la réponse nettoyée
        parsed_json = json.loads(cleaned_json_str)
        parsed = AnswerQuestion.model_validate(parsed_json)
        return parsed
    except json.JSONDecodeError as e:
        print("\n❌ Erreur de décodage JSON :", e)
        raise
    except Exception as e:
        print("\n❌ Erreur de parsing Pydantic :", e)
        raise

# Actor agent prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI researcher.
Current time: {time}
1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, ***list 1-3 search queries separately*** for researching improvements. Do not include them inside the reflection.

IMPORTANT:
You MUST output your response in STRICT JSON format. Do NOT include any markdown, explanations, or additional commentary.
The required JSON format is strictly as follows:
{{
  "answer": "Your detailed response as a single string without any newlines or special characters...",
  "reflections": {{
    "missing": "...",
    "superfluous": "..."
  }},
  "search_queries": [
    "...",
    "..."
  ]
}}

Example:
{{
  "answer": "This is an example answer.",
  "reflections": {{
    "missing": "Example missing reflection",
    "superfluous": "Example superfluous reflection"
  }},
  "search_queries": [
    "Example search query 1",
    "Example search query 2"
  ]
}}

DO NOT return anything other than this JSON.
Ensure the output is valid JSON: do not forget commas between fields.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

LLM_MODEL = "llama3.2:latest"
llm = ChatOllama(model=LLM_MODEL)

# Define the chain with parsing
def process_response(response):
    response_content = response.content
    parsed_response = parse_json_response(response_content)
    print("\n=== PARSED ===")
    print(parsed_response)
    return parsed_response

first_responder_chain = first_responder_prompt_template | llm | process_response

# Revisor section
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count toward the word limit). In the form of:
        - [1] https://example.com
        - [2] https://example.com
    You should use the previous critique to remove superfluous information from your answer and make sure it is not more than 250 words.
    You MUST include the following key **exactly in JSON**:
            "references": ["https://...", "https://..."]
            If no sources are available, return an empty list: "references": []
    You MUST return ONE single JSON object with the following keys:
        - answer
        - reflections
        - search_queries
        - references

        Do NOT return multiple JSON objects or any text outside the JSON.

"""

def process_revisor_response(response):
    response_content = response.content
    parsed_response = parse_json_response_(response_content)
    print("\n=== REVISED RESPONSE ===")
    print(parsed_response)
    return parsed_response

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm | process_revisor_response

# Invoke the first responder chain
response = first_responder_chain.invoke({
    "messages": [HumanMessage(content="Write me a blog post on how small business can leverage AI to grow")]
})

# Invoke the revisor chain with the initial response
revisor_response = revisor_chain.invoke({
    "messages": [HumanMessage(content=response.answer)]
})
