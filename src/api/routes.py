"""FastAPI application endpoints."""
from fastapi import APIRouter, HTTPException, Depends
from .schemas import QueryRequest, QueryResponse, GuardrailAudit, HealthResponse
from ..graph.workflow import AgenticRAGRunner

router = APIRouter()


def get_runner() -> AgenticRAGRunner:
    # Injected by app state
    from .app import global_runner
    if global_runner is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return global_runner


@router.get("/health", response_model=HealthResponse)
def health_check(runner: AgenticRAGRunner = Depends(get_runner)):
    doc_count = len(runner.nodes.retriever.documents)
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        documents_indexed=doc_count,
    )


@router.post("/query", response_model=QueryResponse)
def query_agent(req: QueryRequest, runner: AgenticRAGRunner = Depends(get_runner)):
    try:
        state = runner.invoke({"query": req.query})
        guard_data = state.get("guardrail_status", {})

        return QueryResponse(
            query=req.query,
            answer=state.get("generation", ""),
            citations=state.get("citations", []),
            route=state.get("route", "rag"),
            retry_count=state.get("retry_count", 0),
            hallucination_grade=state.get("hallucination_grade", "grounded"),
            answer_grade=state.get("answer_grade", "useful"),
            guardrails=GuardrailAudit(
                input_passed=guard_data.get("input_passed", True),
                output_passed=guard_data.get("output_passed", True),
                risk_level=guard_data.get("risk_level", "LOW"),
                reasons=guard_data.get("reasons", []),
            ),
            trace=state.get("trace", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
