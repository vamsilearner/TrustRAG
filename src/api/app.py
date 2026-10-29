from typing import List, Optional
"""FastAPI server application initialization."""
import os
import glob
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config import settings
from ..rag.chunking import Document, TextSplitter
from ..rag.retriever import HybridRetriever
from ..graph.workflow import create_rag_graph, AgenticRAGRunner
from .routes import router

global_runner: AgenticRAGRunner = None


def load_knowledge_base(data_dir: str = "data/knowledge_base") -> List[Document]:
    """Reads markdown policy files from knowledge base directory."""
    documents: List[Document] = []
    splitter = TextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    search_path = os.path.join(data_dir, "*.md")
    files = glob.glob(search_path)
    if not files:
        # Fallback if running from a different root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        files = glob.glob(os.path.join(project_root, data_dir, "*.md"))

    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        raw_doc = Document(page_content=content, metadata={"source": fname, "title": fname.replace(".md", "")})
        chunks = splitter.split_documents([raw_doc])
        documents.extend(chunks)

    return documents


def create_app() -> FastAPI:
    global global_runner
    app = FastAPI(
        title="TrustRAG: Enterprise Self-Reflective Agentic RAG API",
        description="Production Agentic RAG combining LangChain, LangGraph, Guardrails, and Ragas Evaluation",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize retrieval & graph
    docs = load_knowledge_base()
    retriever = HybridRetriever(rrf_k=settings.rrf_k)
    retriever.index_documents(docs)
    global_runner = create_rag_graph(retriever=retriever)

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
