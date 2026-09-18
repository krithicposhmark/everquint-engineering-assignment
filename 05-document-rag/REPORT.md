# Search and summarization report

## Corpus and preparation

The corpus is a deterministic 3,000-document sample from 20 Newsgroups (seed 42). The preparation command can ingest the full dataset by omitting `--limit`.

Messages are parsed with their email structure rather than split on blank lines. Subject headers become evaluation queries but are not included in the indexed body. Quoted replies, signatures, URLs and email addresses are removed, whitespace is normalized, and very short subjects or bodies are dropped. This avoids an easy subject-to-subject match and reduces duplicate quoted content.

## Search and summarization

The lexical index is BM25 with `k1=1.5` and `b=0.75`. The semantic index uses normalized `all-MiniLM-L6-v2` embeddings. Their raw scores are not directly comparable, so reciprocal-rank fusion combines them with weights 0.45 and 0.55 and a rank constant of 60.

The default local summarizer is `sshleifer/distilbart-cnn-6-6`. It runs without an API key and uses deterministic beam search. An OpenAI Responses API adapter and an extractive fallback use the same interface. Summary length maps to approximate 60, 120 or 220-word budgets, and the retrieved context is capped before tokenization.

## Evaluation

Retrieval was measured on 300 fixed query cases. Each query is a document subject and the expected result is its cleaned body. Results are from the same fixed corpus and query files.

| Retriever | Hit@5 | MRR@5 |
| --- | ---: | ---: |
| BM25 | 0.450 | 0.377 |
| MiniLM | 0.500 | 0.407 |
| Hybrid | 0.500 | 0.408 |

Hybrid search improved Hit@5 by 5 percentage points over BM25, matched dense retrieval on Hit@5 and produced the strongest reciprocal rank.

Summary evaluation used five separate short articles with human-written references. The small set is intended as a regression check, not a general model benchmark.

| Summarizer | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| Extractive | 0.520 | 0.261 | 0.414 |
| DistilBART | 0.435 | 0.188 | 0.348 |

The extractive baseline scores higher because the references reuse source wording. DistilBART is shorter and easier to scan, but sometimes drops a key number or follow-up action. A single-reviewer check of its five outputs scored fidelity 5.0/5, coverage 3.8/5 and readability 5.0/5. The case-level notes are in `results/human-review.json`; generated outputs and raw metrics are in `results/metrics.json`.

## Challenges and trade-offs

- Newsgroup messages contain signatures and long quoted threads. Removing them improved the signal but can remove context from reply-only messages.
- BM25 and cosine similarity have different scales. Rank fusion avoided dataset-specific score calibration and performed better in this run.
- Local generation keeps the project usable without credentials, but the small model loses more detail than the extractive baseline.
- The current dense search is an exact NumPy scan and BM25 is held in memory. This is simple for tens of thousands of documents, not millions.

For a larger deployment I would chunk long documents, persist a proper inverted index, move embeddings to HNSW or FAISS, batch incremental ingestion, and add latency and cost measurements. I would also expand the summary test set and use multiple human reviewers before tuning prompts or models against ROUGE.
