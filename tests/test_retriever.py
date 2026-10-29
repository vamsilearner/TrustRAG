"""Unit tests for Hybrid Retriever (BM25 + Vector + RRF)."""
import unittest
from src.rag.chunking import Document, TextSplitter
from src.rag.retriever import HybridRetriever, BM25Index


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(page_content="SEC Form 8-K requires reporting material cybersecurity incidents within four business days.", metadata={"source": "sec.md"}),
            Document(page_content="Cloud data retention rules mandate TLS 1.3 for data in transit and AES-256 at rest.", metadata={"source": "cloud.md"}),
            Document(page_content="AI systems must obtain Human-In-The-Loop authorization for transactions over $10,000.", metadata={"source": "ai.md"}),
        ]
        self.retriever = HybridRetriever(rrf_k=60)
        self.retriever.index_documents(self.docs)

    def test_bm25_exact_match(self):
        bm25 = BM25Index()
        bm25.index([d.page_content for d in self.docs])
        scores = bm25.score("cybersecurity Form 8-K incident")
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_hybrid_rrf_retrieval(self):
        results = self.retriever.retrieve("How quickly must SEC cybersecurity incidents be reported?", top_k=2)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn("four business days", top_match.document.page_content)
        self.assertGreater(top_match.score, 0.0)

    def test_text_splitter_overlap(self):
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
        chunks = splitter.split_text(text)
        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
