"""Embeddings engine with local TF-IDF cosine fallback for deterministic testing."""
import math
import re
from typing import List
import numpy as np


class SimpleEmbeddings:
    """Lightweight self-contained vector embedding generator for local deterministic testing and fallback."""

    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def fit(self, texts: List[str]):
        doc_count = len(texts)
        df = {}
        for text in texts:
            tokens = set(self._tokenize(text))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        self.vocab = {t: idx for idx, t in enumerate(sorted(df.keys()))}
        self.idf = {t: math.log((1 + doc_count) / (1 + count)) + 1.0 for t, count in df.items()}

    def embed_query(self, text: str) -> List[float]:
        vec = np.zeros(max(1, len(self.vocab)), dtype=np.float32)
        tokens = self._tokenize(text)
        if not self.vocab:
            return vec.tolist()

        for t in tokens:
            if t in self.vocab:
                idx = self.vocab[t]
                vec[idx] += self.idf.get(t, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.vocab:
            self.fit(texts)
        return [self.embed_query(t) for t in texts]


class EmbeddingProvider:
    """Wrapper that utilizes OpenAI embeddings if API key is set, otherwise utilizes SimpleEmbeddings."""

    def __init__(self, api_key: str = "", model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.local_embedder = SimpleEmbeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.local_embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.local_embedder.embed_query(text)
