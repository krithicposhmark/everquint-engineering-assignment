from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag.types import Document, SearchResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(
        self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75
    ) -> None:
        if not documents:
            raise ValueError("at least one document is required")
        self.k1 = k1
        self.b = b
        self.term_frequencies = [
            Counter(tokenize(document.text)) for document in documents
        ]
        self.lengths = np.asarray(
            [sum(counts.values()) for counts in self.term_frequencies]
        )
        self.average_length = float(self.lengths.mean())
        document_frequency: Counter[str] = Counter()
        for counts in self.term_frequencies:
            document_frequency.update(counts.keys())
        document_count = len(documents)
        self.inverse_document_frequency = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.term_frequencies), dtype=np.float32)
        for term in tokenize(query):
            inverse_frequency = self.inverse_document_frequency.get(term)
            if inverse_frequency is None:
                continue
            for index, counts in enumerate(self.term_frequencies):
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.average_length
                )
                scores[index] += (
                    inverse_frequency * frequency * (self.k1 + 1) / denominator
                )
        return scores


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
            ),
            dtype=np.float32,
        )


class HybridSearchEngine:
    def __init__(
        self,
        documents: Sequence[Document],
        embedder: Embedder,
        embeddings: np.ndarray | None = None,
        bm25_weight: float = 0.45,
        dense_weight: float = 0.55,
        rrf_constant: int = 60,
    ) -> None:
        if not documents:
            raise ValueError("at least one document is required")
        if bm25_weight < 0 or dense_weight < 0 or bm25_weight + dense_weight == 0:
            raise ValueError("retrieval weights must be non-negative and not both zero")
        self.documents = list(documents)
        self.embedder = embedder
        self.bm25 = BM25Index(self.documents)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_constant = rrf_constant
        self.embeddings = (
            np.asarray(embeddings, dtype=np.float32)
            if embeddings is not None
            else embedder.encode([document.text for document in self.documents])
        )
        if self.embeddings.shape[0] != len(self.documents):
            raise ValueError("embedding count does not match document count")

    @staticmethod
    def _ranks(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        bm25_scores = self.bm25.score(query)
        query_embedding = self.embedder.encode([query])[0]
        dense_scores = self.embeddings @ query_embedding
        bm25_ranks = self._ranks(bm25_scores)
        dense_ranks = self._ranks(dense_scores)
        fused_scores = self.bm25_weight / (
            self.rrf_constant + bm25_ranks
        ) + self.dense_weight / (self.rrf_constant + dense_ranks)
        best_indices = np.argsort(-fused_scores, kind="stable")[
            : min(top_k, len(self.documents))
        ]
        return [
            SearchResult(
                document=self.documents[index],
                score=float(fused_scores[index]),
                bm25_rank=int(bm25_ranks[index]),
                dense_rank=int(dense_ranks[index]),
            )
            for index in best_indices
        ]
