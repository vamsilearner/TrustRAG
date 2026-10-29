"""Automated batch benchmarking and evaluation pipeline."""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from .metrics import RAGEvalMetrics, MetricScore
from ..graph.workflow import AgenticRAGRunner


@dataclass
class BenchmarkCase:
    id: str
    query: str
    ground_truth: str
    expected_route: str = "rag"


@dataclass
class BenchmarkResult:
    case_id: str
    query: str
    generation: str
    metrics: MetricScore
    latency_ms: float
    guardrail_status: Dict[str, Any]
    trace: List[str]


@dataclass
class BenchmarkSummary:
    total_cases: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_precision: float
    mean_context_recall: float
    overall_mean: float
    average_latency_ms: float
    results: List[BenchmarkResult] = field(default_factory=list)


class RAGEvaluator:
    """Runs automated regression benchmarks against golden test datasets."""

    def __init__(self, runner: AgenticRAGRunner):
        self.runner = runner

    def run_benchmark(self, dataset: List[Dict[str, Any]]) -> BenchmarkSummary:
        results: List[BenchmarkResult] = []
        latencies = []

        for item in dataset:
            start_t = time.perf_counter()
            response_state = self.runner.invoke({"query": item["query"]})
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            latencies.append(elapsed_ms)

            gen = response_state.get("generation", "")
            docs = [d.page_content for d in response_state.get("relevant_documents", [])]
            gt = item.get("ground_truth", "")

            scores = RAGEvalMetrics.evaluate(
                question=item["query"],
                answer=gen,
                retrieved_contexts=docs,
                ground_truth=gt,
            )

            results.append(
                BenchmarkResult(
                    case_id=item.get("id", "case"),
                    query=item["query"],
                    generation=gen,
                    metrics=scores,
                    latency_ms=round(elapsed_ms, 2),
                    guardrail_status=response_state.get("guardrail_status", {}),
                    trace=response_state.get("trace", []),
                )
            )

        n = max(1, len(results))
        summary = BenchmarkSummary(
            total_cases=len(results),
            mean_faithfulness=round(sum(r.metrics.faithfulness for r in results) / n, 4),
            mean_answer_relevance=round(sum(r.metrics.answer_relevance for r in results) / n, 4),
            mean_context_precision=round(sum(r.metrics.context_precision for r in results) / n, 4),
            mean_context_recall=round(sum(r.metrics.context_recall for r in results) / n, 4),
            overall_mean=round(sum(r.metrics.average for r in results) / n, 4),
            average_latency_ms=round(sum(latencies) / n, 2),
            results=results,
        )
        return summary
