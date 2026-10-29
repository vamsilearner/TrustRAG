"""Formulation of automated RAG evaluation metrics."""
import re
from dataclasses import dataclass
from typing import List, Set


@dataclass
class MetricScore:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    average: float


class RAGEvalMetrics:
    """Calculates Ragas and RAG Triad inspired metrics with deterministic grading."""

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower()))

    @classmethod
    def calculate_faithfulness(cls, answer: str, contexts: List[str]) -> float:
        """Measures whether the claims made in the answer can be directly inferred from the retrieved contexts."""
        if not answer.strip():
            return 0.0
        if not contexts:
            return 0.0

        ans_tokens = cls._tokenize(answer)
        if not ans_tokens:
            return 1.0

        combined_context = " ".join(contexts).lower()
        supported = sum(1 for token in ans_tokens if token in combined_context)
        return round(supported / len(ans_tokens), 4)

    @classmethod
    def calculate_answer_relevance(cls, question: str, answer: str) -> float:
        """Measures how well the generated answer addresses the user question."""
        q_tokens = cls._tokenize(question)
        a_tokens = cls._tokenize(answer)

        if not q_tokens or not a_tokens:
            return 0.0

        overlap = q_tokens.intersection(a_tokens)
        precision = len(overlap) / len(q_tokens)
        return round(min(1.0, precision * 1.5), 4)

    @classmethod
    def calculate_context_precision(cls, retrieved_contexts: List[str], ground_truth: str) -> float:
        """Measures whether all ground-truth relevant items retrieved are ranked at the top."""
        if not retrieved_contexts:
            return 0.0
        gt_tokens = cls._tokenize(ground_truth)
        if not gt_tokens:
            return 1.0

        precisions = []
        hits = 0
        for i, ctx in enumerate(retrieved_contexts):
            ctx_tokens = cls._tokenize(ctx)
            if gt_tokens.intersection(ctx_tokens):
                hits += 1
                precisions.append(hits / (i + 1))

        if not precisions:
            return 0.0
        return round(sum(precisions) / len(precisions), 4)

    @classmethod
    def calculate_context_recall(cls, retrieved_contexts: List[str], ground_truth: str) -> float:
        """Measures the extent to which retrieved context aligns with the reference answer."""
        gt_tokens = cls._tokenize(ground_truth)
        if not gt_tokens:
            return 1.0
        combined_ctx = " ".join(retrieved_contexts).lower()
        covered = sum(1 for token in gt_tokens if token in combined_ctx)
        return round(covered / len(gt_tokens), 4)

    @classmethod
    def evaluate(cls, question: str, answer: str, retrieved_contexts: List[str], ground_truth: str = "") -> MetricScore:
        faith = cls.calculate_faithfulness(answer, retrieved_contexts)
        rel = cls.calculate_answer_relevance(question, answer)
        prec = cls.calculate_context_precision(retrieved_contexts, ground_truth) if ground_truth else 1.0
        rec = cls.calculate_context_recall(retrieved_contexts, ground_truth) if ground_truth else 1.0
        avg = round((faith + rel + prec + rec) / 4.0, 4)

        return MetricScore(
            faithfulness=faith,
            answer_relevance=rel,
            context_precision=prec,
            context_recall=rec,
            average=avg,
        )
