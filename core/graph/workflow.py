from langgraph.graph import StateGraph, START, END

from .types import GraphState
from .nodes.retrieval import retrieve
from .nodes.generation import generate, llm_fallback
from .nodes.web_search import web_search
from .nodes.routing import route_question
from .nodes.grading import grade_generation

def build_workflow(vector_db, web_search_tool):
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", lambda state: retrieve(state, vector_db))
    workflow.add_node("generate", lambda state: generate(state, vector_db))
    workflow.add_node(
        "web_search",
        lambda state: web_search(state, web_search_tool)
    )



    workflow.add_node("llm_fallback", llm_fallback)

    workflow.add_conditional_edges(
        START,
        lambda state: route_question(state, vector_db),
        {"retrieve":"retrieve", "web_search":"web_search", "llm_fallback":"llm_fallback"}
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {"useful":END, "not useful":"web_search", "not supported":"generate"}
    )
    workflow.add_edge("llm_fallback", END)
    workflow.add_edge("web_search", END)

    return workflow.compile()
