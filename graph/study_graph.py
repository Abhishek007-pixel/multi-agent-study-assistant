"""LangGraph state, nodes, conditional edges, compiled workflow."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    query: str
    intent: str
    context: str
    explanation: str
    quiz: str
    final_output: str
    vector_store: Any
    bm25_store: Any


def _scaffold_reply(state: AgentState) -> dict:
    return {
        "final_output": (
            "Scaffold graph: implement planner -> research -> explain/quiz "
            f"per plan. Your query: {state['query']!r}"
        ),
    }


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("scaffold", _scaffold_reply)
    workflow.set_entry_point("scaffold")
    workflow.add_edge("scaffold", END)
    return workflow.compile()


study_graph = build_graph()
