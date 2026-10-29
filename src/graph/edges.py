"""Conditional edges routing the execution graph."""
from .state import AgentState
from ..config import settings


def route_after_input_guard(state: AgentState) -> str:
    """Decides whether to stop due to safety violation or proceed to router."""
    if state.get("route") == "blocked":
        return "blocked"
    return "router"


def route_after_router(state: AgentState) -> str:
    """Decides between direct response or RAG retrieval."""
    if state.get("route") == "direct":
        return "direct"
    return "retrieve"


def decide_after_grading(state: AgentState) -> str:
    """CRAG logic: If relevant docs exist -> generate. Else if retries remain -> rewrite. Else -> generate fallback."""
    if state.get("has_relevant_docs", False):
        return "generate"
    
    if state.get("retry_count", 0) < settings.max_rewrite_cycles:
        return "rewrite"
    
    return "generate"


def decide_after_generation_eval(state: AgentState) -> str:
    """Self-RAG logic: If answer is grounded and useful -> output guard. Else if retries remain -> rewrite."""
    is_grounded = state.get("hallucination_grade") == "grounded"
    is_useful = state.get("answer_grade") == "useful"

    if is_grounded and is_useful:
        return "output_guardrail"

    if state.get("retry_count", 0) < settings.max_rewrite_cycles:
        return "rewrite"

    return "output_guardrail"
