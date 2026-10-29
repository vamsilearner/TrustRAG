"""Pydantic request and response schemas for REST API."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query submitted to TrustRAG agent", example="What is the SEC deadline for reporting material cybersecurity incidents?")
    enable_guardrails: bool = Field(default=True, description="Toggle input/output safety filters")


class Citation(BaseModel):
    doc_id: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class GuardrailAudit(BaseModel):
    input_passed: bool = True
    output_passed: bool = True
    risk_level: str = "LOW"
    reasons: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    route: str
    retry_count: int = 0
    hallucination_grade: str
    answer_grade: str
    guardrails: GuardrailAudit
    trace: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    documents_indexed: int
