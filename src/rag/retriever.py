"""Hybrid Retriever combining BM25 keyword matching and Dense Vector similarity via RRF."""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .chunking import Document
from .embeddings import EmbeddingProvider


@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank_dense: int
    rank_sparse: int


class BM25Index:
    """Lightweight in-memory BM25 Okapi searcher."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def index(self, texts: List[str]):
        self.corpus = [self._tokenize(t) for t in texts]
        self.doc_lens = [len(c) for c in self.corpus]
        total_docs = len(texts)
        self.avg_dl = sum(self.doc_lens) / max(1, total_docs)

        self.df = {}
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> List[float]:
        q_tokens = self._tokenize(query)
        scores = [0.0] * len(self.corpus)
        for i, doc in enumerate(self.corpus):
            dl = self.doc_lens[i]
            doc_counts = {}
            for t in doc:
                doc_counts[t] = doc_counts.get(t, 0) + 1

            doc_score = 0.0
            for t in q_tokens:
                if t in doc_counts and t in self.idf:
                    tf = doc_counts[t]
                    num = tf * (self.k1 + 1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (dl / max(1e-5, self.avg_dl)))
                    doc_score += self.idf[t] * (num / denom)
            scores[i] = doc_score
        return scores


class HybridRetriever:
    """Performs hybrid retrieval using BM25 and Vector Search with Reciprocal Rank Fusion (RRF)."""

    def __init__(self, rrf_k: int = 60, embedding_provider: EmbeddingProvider = None):
        self.rrf_k = rrf_k
        self.embedder = embedding_provider or EmbeddingProvider()
        self.documents: List[Document] = []
        self.bm25 = BM25Index()
        self.doc_vectors: List[List[float]] = []

    def index_documents(self, documents: List[Document]):
        self.documents = documents
        texts = [d.page_content for d in documents]
        self.bm25.index(texts)
        self.doc_vectors = self.embedder.embed_documents(texts)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievalResult]:
        if not self.documents:
            return []

        # 1. Sparse BM25 ranking
        sparse_scores = self.bm25.score(query)
        sparse_ranked_indices = np.argsort(sparse_scores)[::-1]
        sparse_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(sparse_ranked_indices)}

        # 2. Dense Vector ranking
        q_vec = self.embedder.embed_query(query)
        dense_scores = [self._cosine_similarity(q_vec, d_vec) for d_vec in self.doc_vectors]
        dense_ranked_indices = np.argsort(dense_scores)[::-1]
        dense_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(dense_ranked_indices)}

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[int, float] = {}
        for idx in range(len(self.documents)):
            r_sparse = sparse_ranks[idx]
            r_dense = dense_ranks[idx]
            # RRF formula: 1 / (k + rank_sparse) + 1 / (k + rank_dense)
            rrf_score = (1.0 / (self.rrf_k + r_sparse)) + (1.0 / (self.rrf_k + r_dense))
            fused_scores[idx] = rrf_score

        sorted_by_rrf = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in sorted_by_rrf:
            results.append(
                RetrievalResult(
                    document=self.documents[idx],
                    score=score,
                    rank_dense=dense_ranks[idx],
                    rank_sparse=sparse_ranks[idx],
                )
            )
        return results
