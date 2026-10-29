"""Application configuration and environment settings."""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    """Central configuration for TrustRAG."""
    app_name: str = "TrustRAG Enterprise"
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Models
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # RAG parameters
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    rrf_k: int = 60

    # Agentic LangGraph settings
    max_rewrite_cycles: int = 2
    relevance_threshold: float = 0.65
    hallucination_threshold: float = 0.70

    # Guardrails
    enable_input_guardrails: bool = True
    enable_output_guardrails: bool = True
    blocked_topics: List[str] = field(default_factory=lambda: [
        "weapon", "illicit substance", "hate speech", "malware synthesis", "credential theft"
    ])


settings = Settings()
