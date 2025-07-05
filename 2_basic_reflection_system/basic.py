from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain

# Load environment variables from .env file
load_dotenv()

graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"

def generate_node(state):
    print(f"State before generation: {state}")
    response = generation_chain.invoke(
        {
            "messages": state
        }
    )
    return state + [response]

def reflect_node(state):
    print(f"State before reflection: {state}")
    response = reflection_chain.invoke(
        {
            "messages": state
        }
    )
    return state + [HumanMessage(content=response.content)]

graph.add_node(
    GENERATE,
    generate_node
)

graph.add_node(
    REFLECT,
    reflect_node
)

graph.set_entry_point(GENERATE)

"""def should_continue(state: List[BaseMessage]):
    if(len(state) > 16):
        return END
    return REFLECT"""

def should_continue(state: List[BaseMessage]):
    print(f"State before checking continuation: {state}")
    # Par exemple : boucle 3 fois max
    num_cycles = sum(1 for m in state if isinstance(m, HumanMessage))
    if num_cycles >= 3:
        return END
    return REFLECT


graph.add_conditional_edges(GENERATE, should_continue)

graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()


print("\n=== LANCEMENT DU GRAPHE ===\n")
final_state = app.invoke([
    HumanMessage(content="Fais-moi un tweet viral sur l'IA.")
])
print("\n=== RÉSULTAT FINAL ===\n")
for msg in final_state:
    print(f"{msg.type}: {msg.content}")