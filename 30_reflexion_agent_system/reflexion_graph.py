from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

graph = MessageGraph()
MAX_ITERATIONS = 2

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revisor_chain)

graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

def event_loop(state: list[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits >= MAX_ITERATIONS:
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor", event_loop)
graph.set_entry_point("draft")

# Compilation et exécution
app = graph.compile()

print("\n✅ Graphe Mermaid:")
print(app.get_graph().draw_mermaid())

response = app.invoke(
    "Write about how small businesses can leverage AI to grow"
)

print("\n✅ Final Answer:")
print(response[-1].content)
