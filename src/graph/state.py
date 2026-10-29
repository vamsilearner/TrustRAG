"""State schema for the Agentic RAG Graph."""
from typing import Any, Dict, List, Optional
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from ..rag.chunking import Document


class AgentState(TypedDict, total=False):
    """Represents the mutable state throughout the LangGraph execution flow."""
    query: str
    sanitized_query: str
    rewritten_query: str
    route: str  # "rag", "direct", "blocked"
    documents: List[Document]
    relevant_documents: List[Document]
    generation: str
    citations: List[str]
    retry_count: int
    has_relevant_docs: bool
    hallucination_grade: str  # "grounded", "hallucinated"
    answer_grade: str  # "useful", "not_useful"
    guardrail_status: Dict[str, Any]
    trace: List[str]
    error: Optional[str]
