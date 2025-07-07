from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_ollama import ChatOllama 
from schema import AnswerQuestion, ReviewAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
import json
from dotenv import load_dotenv

load_dotenv()

pydantic_parser = PydanticToolsParser(tools=AnswerQuestion)



#Actor agent prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """You are expert AI researcher.
Current time: {time}

1.{first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvment.
3. After the reflection, ***list 1-3 search queries seperately** for 
researching improvments. Do not include them inside the reflection.

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
""",    
        ),
        MessagesPlaceholder(variable_name="messages"),
        #("system","Answer the user's question above using the required format.",),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

LLM_MODEL = "llama3.2:latest"
llm = ChatOllama(model=LLM_MODEL)

first_responder_chain = first_responder_prompt_template | llm
#.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion') | pydantic_parser

#Revisor section

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to you
    answer.
        - You MUST include numerical citations in your revised answer to ensure
        it can be verified.
        - Add a "References" section to the bottom of your answer (which does 
        not count toward the world limit). In form of:
            -[1] https://example.com
            -[2] https://example.com
    You should use the previous critique to remove superfluous information from
    your answer and make SURE it is not more the 250 words.

"""

revisor_chain = actor_prompt_template.partial(
    first_instruction = revise_instructions
) | llm

response = first_responder_chain.invoke({
    "messages": [HumanMessage(content="Write me a blog post on how small business" \
    "can leverage AI to grow")]
})

# Vérifier le contenu texte brut
print("=== RAW RESPONSE ===")
print(response.content)

# Nettoyer et parser la réponse JSON
try:
    # Supprimer les espaces et caractères invisibles en début et fin de chaîne
    cleaned_json_str = response.content.strip()
    # Remplacer les caractères non valides
    cleaned_json_str = cleaned_json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    parsed_json = json.loads(cleaned_json_str)
    parsed = AnswerQuestion.model_validate(parsed_json)
    print("\n=== PARSED ===")
    print(parsed)
except json.JSONDecodeError as e:
    print("\n❌ Erreur de décodage JSON :", e)
except Exception as e:
    print("\n❌ Erreur de parsing Pydantic :", e)