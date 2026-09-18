from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.retrieval import HybridSearchEngine
from rag.summarization import Summarizer


def evaluate_retrieval(
    engine: HybridSearchEngine,
    queries: Sequence[dict[str, object]],
    top_k: int = 5,
) -> dict[str, float | int]:
    if not queries:
        raise ValueError("at least one query is required")
    hits = 0
    reciprocal_rank = 0.0
    for case in queries:
        results = engine.search(str(case["query"]), top_k=top_k)
        ranked_ids = [result.document.document_id for result in results]
        target_id = case["target_id"]
        if target_id in ranked_ids:
            hits += 1
            reciprocal_rank += 1 / (ranked_ids.index(target_id) + 1)
    return {
        "query_count": len(queries),
        f"hit_rate_at_{top_k}": hits / len(queries),
        f"mrr_at_{top_k}": reciprocal_rank / len(queries),
    }


def evaluate_summaries(
    summarizer: Summarizer,
    cases: Sequence[dict[str, Any]],
    length: str = "short",
) -> tuple[dict[str, float | int], list[dict[str, str]]]:
    from rouge_score import rouge_scorer

    if not cases:
        raise ValueError("at least one summary case is required")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    outputs = []
    for case in cases:
        candidate = summarizer.summarize(case["text"], length)
        scores = scorer.score(case["reference"], candidate)
        for metric in totals:
            totals[metric] += scores[metric].fmeasure
        outputs.append(
            {
                "title": case["title"],
                "reference": case["reference"],
                "generated": candidate,
            }
        )
    metrics: dict[str, float | int] = {"case_count": len(cases)}
    metrics.update({metric: total / len(cases) for metric, total in totals.items()})
    return metrics, outputs
