"""LangGraph StateGraph workflow definition and compiled execution engine."""
from typing import Any, Dict
from .state import AgentState
from .nodes import GraphNodes
from .edges import (
    route_after_input_guard,
    route_after_router,
    decide_after_grading,
    decide_after_generation_eval,
)
from ..rag.retriever import HybridRetriever


class AgenticRAGRunner:
    """Deterministic, resilient workflow orchestrator modeled on LangGraph's cyclical StateGraph."""

    def __init__(self, nodes: GraphNodes):
        self.nodes = nodes

    def invoke(self, initial_state: Dict[str, Any]) -> AgentState:
        """Executes the cyclical agentic workflow with safety boundaries."""
        state: AgentState = {
            "query": initial_state.get("query", ""),
            "sanitized_query": "",
            "rewritten_query": "",
            "route": "",
            "documents": [],
            "relevant_documents": [],
            "generation": "",
            "citations": [],
            "retry_count": 0,
            "has_relevant_docs": False,
            "hallucination_grade": "grounded",
            "answer_grade": "useful",
            "guardrail_status": {},
            "trace": [],
            "error": None,
        }

        # Step 1: Input Guardrail
        state = self.nodes.input_guardrail_node(state)
        if route_after_input_guard(state) == "blocked":
            return state

        # Step 2: Router
        state = self.nodes.router_node(state)
        if route_after_router(state) == "direct":
            return state

        # Step 3: Retrieval & Corrective Loops
        while True:
            state = self.nodes.retrieve_node(state)
            state = self.nodes.grade_documents_node(state)

            action = decide_after_grading(state)
            if action == "rewrite":
                state = self.nodes.rewrite_query_node(state)
                continue
            break

        # Step 4: Generation & Verification
        while True:
            state = self.nodes.generate_node(state)
            state = self.nodes.grade_hallucination_node(state)
            state = self.nodes.grade_answer_quality_node(state)

            gen_action = decide_after_generation_eval(state)
            if gen_action == "rewrite" and state.get("retry_count", 0) < 2:
                state = self.nodes.rewrite_query_node(state)
                state = self.nodes.retrieve_node(state)
                state = self.nodes.grade_documents_node(state)
                continue
            break

        # Step 5: Output Guardrail
        state = self.nodes.output_guardrail_node(state)
        return state


def create_rag_graph(retriever: HybridRetriever) -> AgenticRAGRunner:
    """Factory creating the production Agentic RAG workflow runner."""
    nodes = GraphNodes(retriever=retriever)
    return AgenticRAGRunner(nodes=nodes)
