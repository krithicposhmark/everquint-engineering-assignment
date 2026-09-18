import unittest

import numpy as np
from rag.retrieval import BM25Index, HybridSearchEngine
from rag.types import Document


class KeywordEmbedder:
    vocabulary = ("python", "database", "football")

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.asarray(
                [text.lower().count(term) for term in self.vocabulary], dtype=np.float32
            )
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors)


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document("python", "Python", "Python code uses functions and classes."),
            Document(
                "database",
                "Databases",
                "A database stores records and supports queries.",
            ),
            Document(
                "football", "Football", "A football match has two teams and a referee."
            ),
        ]

    def test_bm25_prefers_matching_document(self) -> None:
        scores = BM25Index(self.documents).score("database query")
        self.assertEqual(int(np.argmax(scores)), 1)

    def test_hybrid_search_returns_relevant_document(self) -> None:
        engine = HybridSearchEngine(self.documents, KeywordEmbedder())
        result = engine.search("database queries", top_k=1)[0]
        self.assertEqual(result.document.document_id, "database")
        self.assertEqual(result.bm25_rank, 1)
        self.assertEqual(result.dense_rank, 1)

    def test_invalid_query_is_rejected(self) -> None:
        engine = HybridSearchEngine(self.documents, KeywordEmbedder())
        with self.assertRaises(ValueError):
            engine.search(" ")


if __name__ == "__main__":
    unittest.main()
