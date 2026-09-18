import unittest

from rag.evaluation import evaluate_retrieval
from rag.retrieval import HybridSearchEngine
from rag.types import Document
from test_retrieval import KeywordEmbedder


class EvaluationTest(unittest.TestCase):
    def test_retrieval_metrics(self) -> None:
        documents = [
            Document("python", "Python", "Python code and Python functions."),
            Document("database", "Database", "Database indexes answer queries."),
        ]
        engine = HybridSearchEngine(documents, KeywordEmbedder())
        metrics = evaluate_retrieval(
            engine,
            [
                {"query": "Python functions", "target_id": "python"},
                {"query": "database query", "target_id": "database"},
            ],
            top_k=1,
        )
        self.assertEqual(metrics["hit_rate_at_1"], 1.0)
        self.assertEqual(metrics["mrr_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
