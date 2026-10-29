"""CLI runner to execute benchmark evaluation against the golden dataset."""
import os
import json
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import load_knowledge_base
from src.rag.retriever import HybridRetriever
from src.graph.workflow import create_rag_graph
from src.eval.evaluator import RAGEvaluator
from src.eval.benchmark_report import generate_markdown_report


def main():
    print("=" * 65)
    print(" TrustRAG CI/CD Benchmark & Evaluation Suite")
    print("=" * 65)

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden_eval_dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"[*] Loaded {len(dataset)} evaluation cases from golden dataset.")
    print("[*] Initializing knowledge base and hybrid indexing...")

    docs = load_knowledge_base()
    retriever = HybridRetriever()
    retriever.index_documents(docs)
    runner = create_rag_graph(retriever)

    print(f"[*] Indexed {len(docs)} document chunks.")
    print("[*] Running automated benchmark evaluation...")

    evaluator = RAGEvaluator(runner)
    summary = evaluator.run_benchmark(dataset)

    report_md = generate_markdown_report(summary)
    print("\n" + report_md + "\n")

    # Save benchmark report to file
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    out_file = os.path.join(out_dir, "benchmark_report.md")
    with open(out_file, "w") as f:
        f.write(report_md)
    print(f"[✓] Benchmark report exported to: {out_file}")


if __name__ == "__main__":
    main()
