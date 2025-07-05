from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Définir un type d'état (par exemple la liste de messages)
class MessagesState(list):
    pass

REFLECT = "reflect"
GENERATE = "generate"

def generate_node(state: MessagesState) -> MessagesState:
    # Simuler la génération IA
    last_human = state[-1].content if state else "..."
    ai_msg = AIMessage(content=f"Génération basée sur : {last_human}")
    return state + [ai_msg]

def reflect_node(state: MessagesState) -> MessagesState:
    # Simuler la réflexion
    last_ai = state[-1].content if state else "..."
    human_msg = HumanMessage(content=f"Réflexion sur : {last_ai}")
    return state + [human_msg]

def should_continue(state: MessagesState):
    num_cycles = sum(1 for m in state if isinstance(m, HumanMessage))
    if num_cycles >= 3:
        return END
    return REFLECT

builder = StateGraph(MessagesState)
builder.add_node(GENERATE, generate_node)
builder.add_node(REFLECT, reflect_node)
builder.set_entry_point(GENERATE)

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()
