from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    bm25_rank: int
    dense_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document.document_id,
            "title": self.document.title,
            "category": self.document.category,
            "excerpt": self.document.text[:400],
            "score": self.score,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
        }
