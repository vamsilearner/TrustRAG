"""Document representations and semantic-aware text chunking."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class TextSplitter:
    """Splits long text documents into overlapping semantic chunks with metadata preservation."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for idx, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta["chunk_index"] = str(idx)
                meta["parent_id"] = doc.doc_id
                split_docs.append(Document(page_content=chunk, metadata=meta, doc_id=f"{doc.doc_id}_c{idx}"))
        return split_docs
