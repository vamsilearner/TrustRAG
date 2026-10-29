"""Generates clean Markdown & CLI reports of evaluation runs."""
from .evaluator import BenchmarkSummary


def generate_markdown_report(summary: BenchmarkSummary) -> str:
    lines = [
        "# TrustRAG Automated Evaluation & Benchmark Report",
        "",
        "## Executive Summary",
        f"- **Total Test Cases Evaluated**: {summary.total_cases}",
        f"- **Overall System Score**: {summary.overall_mean * 100:.1f}%",
        f"- **Mean Latency**: {summary.average_latency_ms:.2f} ms",
        "",
        "## Core RAG Triad & Ragas Metric Breakdown",
        "| Metric | Score | Target Threshold | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Faithfulness** | `{summary.mean_faithfulness:.4f}` | `> 0.8500` | {'✅ PASS' if summary.mean_faithfulness >= 0.80 else '⚠️ WARN'} |",
        f"| **Answer Relevance** | `{summary.mean_answer_relevance:.4f}` | `> 0.8000` | {'✅ PASS' if summary.mean_answer_relevance >= 0.75 else '⚠️ WARN'} |",
        f"| **Context Precision** | `{summary.mean_context_precision:.4f}` | `> 0.7500` | {'✅ PASS' if summary.mean_context_precision >= 0.70 else '⚠️ WARN'} |",
        f"| **Context Recall** | `{summary.mean_context_recall:.4f}` | `> 0.8000` | {'✅ PASS' if summary.mean_context_recall >= 0.75 else '⚠️ WARN'} |",
        "",
        "## Detailed Case-by-Case Log",
        "| ID | Query | Faithfulness | Relevance | Latency (ms) | Guardrail Risk |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for r in summary.results:
        q_trunc = r.query if len(r.query) < 40 else r.query[:37] + "..."
        risk = r.guardrail_status.get("risk_level", "LOW")
        lines.append(
            f"| `{r.case_id}` | {q_trunc} | `{r.metrics.faithfulness:.2f}` | `{r.metrics.answer_relevance:.2f}` | `{r.latency_ms}` | `{risk}` |"
        )

    lines.append("")
    lines.append("---")
    lines.append("*Report generated automatically by TrustRAG CI/CD Evaluation Suite.*")
    return "\n".join(lines)
