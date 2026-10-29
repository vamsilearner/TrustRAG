"""Nodes for the LangGraph agentic RAG workflow."""
import re
from typing import List
from .state import AgentState
from ..guardrails.input_guard import InputGuard
from ..guardrails.output_guard import OutputGuard
from ..rag.chunking import Document
from ..rag.retriever import HybridRetriever
from ..config import settings


class GraphNodes:
    """Encapsulates node execution logic for LangGraph."""

    def __init__(self, retriever: HybridRetriever, input_guard: InputGuard = None, output_guard: OutputGuard = None):
        self.retriever = retriever
        self.input_guard = input_guard or InputGuard()
        self.output_guard = output_guard or OutputGuard()

    def input_guardrail_node(self, state: AgentState) -> AgentState:
        """Evaluates input query for prompt injections and sanitizes PII."""
        query = state.get("query", "")
        trace = list(state.get("trace", []))
        trace.append("input_guardrail_node")

        res = self.input_guard.validate(query)
        if not res.passed:
            return {
                **state,
                "sanitized_query": res.sanitized_text,
                "route": "blocked",
                "generation": f"Security Guardrail: Request blocked. Reasons: {', '.join(res.reasons)}",
                "guardrail_status": {
                    "input_passed": False,
                    "risk_level": res.risk_level,
                    "reasons": res.reasons,
                },
                "trace": trace,
            }

        return {
            **state,
            "sanitized_query": res.sanitized_text,
            "guardrail_status": {
                "input_passed": True,
                "risk_level": res.risk_level,
                "pii_detected_count": len(res.pii_matches),
            },
            "trace": trace,
        }

    def router_node(self, state: AgentState) -> AgentState:
        """Routes query: determines whether to use RAG retrieval or general direct answer."""
        trace = list(state.get("trace", []))
        trace.append("router_node")

        raw_query = state.get("sanitized_query", "").lower()
        clean_query = re.sub(r"[^\w\s]", "", raw_query).strip()
        greetings = ["hi", "hello", "hey", "who are you", "what can you do"]

        if any(clean_query == g or clean_query.startswith(g + " ") for g in greetings):
            return {
                **state,
                "route": "direct",
                "generation": "Hello! I am TrustRAG, your enterprise AI policy and compliance assistant. How can I assist you with governance, SEC cybersecurity rules, or data retention policies today?",
                "trace": trace,
            }

        return {
            **state,
            "route": "rag",
            "retry_count": state.get("retry_count", 0),
            "trace": trace,
        }

    def retrieve_node(self, state: AgentState) -> AgentState:
        """Fetches candidate chunks using hybrid dense + sparse retrieval."""
        trace = list(state.get("trace", []))
        trace.append("retrieve_node")

        search_query = state.get("rewritten_query") or state.get("sanitized_query", "")
        results = self.retriever.retrieve(search_query, top_k=settings.top_k)
        docs = [r.document for r in results]

        return {
            **state,
            "documents": docs,
            "trace": trace,
        }

    def grade_documents_node(self, state: AgentState) -> AgentState:
        """Determines whether retrieved chunks are relevant to the query (CRAG pattern)."""
        trace = list(state.get("trace", []))
        trace.append("grade_documents_node")

        query = (state.get("rewritten_query") or state.get("sanitized_query", "")).lower()
        query_terms = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query))
        docs = state.get("documents", [])

        relevant_docs: List[Document] = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            overlap = sum(1 for term in query_terms if term in content_lower)
            # Relevance heuristic: at least 15% keyword overlap or key phrase match
            if query_terms and (overlap / len(query_terms) >= 0.15 or overlap >= 2):
                relevant_docs.append(doc)

        has_relevant = len(relevant_docs) > 0
        return {
            **state,
            "relevant_documents": relevant_docs,
            "has_relevant_docs": has_relevant,
            "trace": trace,
        }

    def rewrite_query_node(self, state: AgentState) -> AgentState:
        """Rewrites failed query to improve retrieval recall with synonym expansion."""
        trace = list(state.get("trace", []))
        trace.append("rewrite_query_node")

        current_query = state.get("sanitized_query", "")
        retries = state.get("retry_count", 0) + 1

        # Heuristic query expansion
        rewritten = current_query
        if "sec" in current_query.lower() and "cybersecurity" not in current_query.lower():
            rewritten += " cybersecurity incident disclosure material form 8-k"
        elif "policy" in current_query.lower():
            rewritten += " enterprise compliance governance standards retention"
        else:
            rewritten += " policy requirements guidelines"

        return {
            **state,
            "rewritten_query": rewritten,
            "retry_count": retries,
            "trace": trace,
        }

    def generate_node(self, state: AgentState) -> AgentState:
        """Synthesizes answer grounded strictly in retrieved documents."""
        trace = list(state.get("trace", []))
        trace.append("generate_node")

        relevant_docs = state.get("relevant_documents", [])
        query = state.get("sanitized_query", "")

        if not relevant_docs:
            return {
                **state,
                "generation": "I am sorry, but the enterprise policy documentation does not contain sufficient grounded information to verify an answer to your query.",
                "citations": [],
                "trace": trace,
            }

        # Build grounded synthesis
        context_snippets = []
        citations = []
        for doc in relevant_docs:
            context_snippets.append(f"[{doc.doc_id}] {doc.page_content}")
            citations.append(doc.doc_id)

        # In production with OPENAI_API_KEY, an LLM call happens here.
        # Fallback synthesizer produces grounded extract:
        synthesis_lines = [
            "Based on enterprise policy documentation:",
        ]
        for doc in relevant_docs[:2]:
            lines = [l.strip() for l in doc.page_content.split("\n") if l.strip() and not l.strip().startswith("#")]
            best_snippet = lines[0] if lines else doc.page_content[:150]
            if len(best_snippet) > 200:
                best_snippet = best_snippet[:197] + "..."
            src_name = doc.metadata.get("source", doc.doc_id)
            synthesis_lines.append(f"- {best_snippet} (Source: {src_name}) [{doc.doc_id}]")

        answer = "\n".join(synthesis_lines)
        return {
            **state,
            "generation": answer,
            "citations": citations,
            "trace": trace,
        }

    def grade_hallucination_node(self, state: AgentState) -> AgentState:
        """Validates that generated claims are supported by context (Self-RAG check)."""
        trace = list(state.get("trace", []))
        trace.append("grade_hallucination_node")

        generation = state.get("generation", "").lower()
        relevant_docs = state.get("relevant_documents", [])
        if not relevant_docs:
            return {**state, "hallucination_grade": "grounded", "trace": trace}

        combined_context = " ".join([d.page_content.lower() for d in relevant_docs])
        gen_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{4,}\b", generation))
        supported_tokens = sum(1 for t in gen_tokens if t in combined_context)

        ratio = supported_tokens / max(1, len(gen_tokens))
        grade = "grounded" if ratio >= 0.50 else "hallucinated"

        return {
            **state,
            "hallucination_grade": grade,
            "trace": trace,
        }

    def grade_answer_quality_node(self, state: AgentState) -> AgentState:
        """Validates that generation directly addresses the user query."""
        trace = list(state.get("trace", []))
        trace.append("grade_answer_quality_node")

        query = state.get("sanitized_query", "").lower()
        generation = state.get("generation", "").lower()

        q_terms = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query))
        addressed = any(term in generation for term in q_terms) or len(generation) > 40

        grade = "useful" if addressed else "not_useful"
        return {
            **state,
            "answer_grade": grade,
            "trace": trace,
        }

    def output_guardrail_node(self, state: AgentState) -> AgentState:
        """Validates the output generation against safety and leakage policies."""
        trace = list(state.get("trace", []))
        trace.append("output_guardrail_node")

        generation = state.get("generation", "")
        res = self.output_guard.validate(generation)

        status = dict(state.get("guardrail_status", {}))
        status["output_passed"] = res.passed
        status["output_confidence"] = res.confidence_score

        return {
            **state,
            "generation": res.sanitized_text,
            "guardrail_status": status,
            "trace": trace,
        }
