
import traceback
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool

MODEL_NAME = "llama3.2:latest"
llm = ChatOllama(model=MODEL_NAME)
prompt = hub.pull("hwchase17/react")
tools = []

def create_agent_executor(tools_list):
    agent = create_react_agent(llm=llm, tools=tools_list, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools_list, verbose=True)

def try_agent_with_repair(query):
    agent_executor = create_agent_executor(tools)

    try:
        result = agent_executor.invoke({"input": query})
        print(f"🟡 Réponse brute :\n{result}")

        if "Agent stopped" in result.get("output", "") or "not a valid tool" in result.get("output", ""):
            raise RuntimeError("Agent bloqué : outil manquant ou boucle infinie")

        print(f"\n✅ Réponse finale : {result['output']}")

    except Exception as e:
        print("❌ L'agent a échoué. Tentative d'auto-réparation...\n")
        error_trace = traceback.format_exc()
        repair_and_retry(query, error_trace)

def repair_and_retry(query, error_trace):
    print("🤖 Génération d’un outil Python pour corriger le problème...")

    tool_request = f"""
Tu es un assistant développeur LangChain.
Tu dois générer un outil Python capable de répondre à la question suivante :
'{query}'

L’agent LangChain a échoué avec cette erreur :
{error_trace}

Retourne une seule fonction Python comme ci-dessous :

```python
def get_current_time():
    from datetime import datetime
    return f"The current time is {{datetime.now().strftime('%H:%M:%S')}}"
```

Ne mets que la fonction.
    """
    tool_code = llm.invoke(tool_request)
    print(f"\n🛠️ Code généré :\n{tool_code}\n")

    local_vars = {}
    try:
        exec(tool_code, {}, local_vars)
        tool_func = list(local_vars.values())[0]
    except Exception as ex:
        print("⚠️ Erreur lors de la génération de l’outil :", ex)
        return

    new_tool = Tool.from_function(
        func=tool_func,
        name="AutoTool",
        description="Outil généré automatiquement par l'agent pour répondre à la question."
    )

    updated_tools = tools + [new_tool]
    print("♻️ Redémarrage de l’agent avec le nouvel outil...\n")
    agent_executor = create_agent_executor(updated_tools)
    result = agent_executor.invoke({"input": query})

    print(f"\n✅ Nouvelle réponse : {result['output']}")

if __name__ == "__main__":
    user_query = "What is the current time?"
    try_agent_with_repair(user_query)
