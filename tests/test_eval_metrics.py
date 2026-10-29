"""Unit tests for Ragas evaluation metrics calculations."""
import unittest
from src.eval.metrics import RAGEvalMetrics


class TestEvalMetrics(unittest.TestCase):
    def test_faithfulness_high_when_grounded(self):
        context = ["SEC Form 8-K requires reporting within four business days."]
        answer = "SEC Form 8-K requires reporting in four business days."
        score = RAGEvalMetrics.calculate_faithfulness(answer, context)
        self.assertGreaterEqual(score, 0.8)

    def test_faithfulness_low_when_hallucinated(self):
        context = ["SEC Form 8-K requires reporting within four business days."]
        answer = "The company declared bankruptcy and sold its offshore submarine fleet."
        score = RAGEvalMetrics.calculate_faithfulness(answer, context)
        self.assertLess(score, 0.3)

    def test_answer_relevance(self):
        question = "What is the cybersecurity incident deadline?"
        answer = "The cybersecurity incident deadline is four business days."
        score = RAGEvalMetrics.calculate_answer_relevance(question, answer)
        self.assertGreater(score, 0.5)

    def test_context_recall(self):
        contexts = ["Registrants must file Form 8-K within four business days."]
        ground_truth = "Registrants must file Form 8-K within four business days."
        recall = RAGEvalMetrics.calculate_context_recall(contexts, ground_truth)
        self.assertGreaterEqual(recall, 0.9)


if __name__ == "__main__":
    unittest.main()
