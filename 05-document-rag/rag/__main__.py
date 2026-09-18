from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from rag.corpus import load_20_newsgroups, read_documents, read_jsonl, write_jsonl
from rag.evaluation import evaluate_retrieval, evaluate_summaries
from rag.retrieval import HybridSearchEngine, SentenceTransformerEmbedder
from rag.summarization import build_context, build_summarizer


def _engine(args: argparse.Namespace) -> HybridSearchEngine:
    documents = read_documents(args.corpus)
    embedder = SentenceTransformerEmbedder(args.embedding_model)
    embeddings = np.load(args.embeddings) if args.embeddings.exists() else None
    engine = HybridSearchEngine(documents, embedder, embeddings)
    if embeddings is None:
        args.embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embeddings, engine.embeddings)
    return engine


def prepare(args: argparse.Namespace) -> None:
    documents, queries = load_20_newsgroups(args.limit, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "corpus.jsonl", [document.to_dict() for document in documents]
    )
    random.Random(args.seed).shuffle(queries)
    write_jsonl(args.output_dir / "queries.jsonl", queries[: args.evaluation_queries])
    print(
        f"prepared {len(documents)} documents and {min(len(queries), args.evaluation_queries)} queries"
    )


def search(args: argparse.Namespace) -> None:
    engine = _engine(args)
    results = engine.search(args.query, args.top_k)
    summarizer = build_summarizer(args.summarizer, args.summary_model)
    output = {
        "query": args.query,
        "results": [result.to_dict() for result in results],
        "summary": summarizer.summarize(build_context(results), args.summary_length),
    }
    print(json.dumps(output, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    engine = _engine(args)
    queries = read_jsonl(args.queries)[: args.query_limit]
    retrieval_metrics = {}
    for name, weights in {
        "bm25": (1.0, 0.0),
        "dense": (0.0, 1.0),
        "hybrid": (0.45, 0.55),
    }.items():
        engine.bm25_weight, engine.dense_weight = weights
        retrieval_metrics[name] = evaluate_retrieval(engine, queries, args.top_k)
    summary_cases = json.loads(args.summary_cases.read_text(encoding="utf-8"))
    summarizer = build_summarizer(args.summarizer, args.summary_model)
    summary_metrics, summary_outputs = evaluate_summaries(summarizer, summary_cases)
    output = {
        "corpus_size": len(engine.documents),
        "embedding_model": args.embedding_model,
        "summarizer": args.summarizer,
        "summary_model": args.summary_model,
        "retrieval": retrieval_metrics,
        "summarization": summary_metrics,
        "summary_outputs": summary_outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def add_index_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid document search and summarization."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, default=Path("data"))
    prepare_parser.add_argument("--limit", type=int)
    prepare_parser.add_argument("--evaluation-queries", type=int, default=200)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.set_defaults(function=prepare)

    search_parser = subparsers.add_parser("search")
    add_index_arguments(search_parser)
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument(
        "--summarizer",
        choices=("extractive", "transformers", "openai"),
        default="transformers",
    )
    search_parser.add_argument("--summary-model")
    search_parser.add_argument(
        "--summary-length", choices=("short", "medium", "long"), default="short"
    )
    search_parser.set_defaults(function=search)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_index_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--queries", type=Path, default=Path("data/queries.jsonl")
    )
    evaluate_parser.add_argument("--query-limit", type=int, default=100)
    evaluate_parser.add_argument("--top-k", type=int, default=5)
    evaluate_parser.add_argument(
        "--summary-cases", type=Path, default=Path("tests/fixtures/summaries.json")
    )
    evaluate_parser.add_argument(
        "--summarizer",
        choices=("extractive", "transformers", "openai"),
        default="transformers",
    )
    evaluate_parser.add_argument("--summary-model")
    evaluate_parser.add_argument(
        "--output", type=Path, default=Path("results/metrics.json")
    )
    evaluate_parser.set_defaults(function=evaluate)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
