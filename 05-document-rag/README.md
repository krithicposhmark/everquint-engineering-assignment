# Document search and summarization

This project indexes the 20 Newsgroups corpus and combines BM25 with MiniLM embeddings through reciprocal-rank fusion. Search results can be summarized locally with DistilBART, through OpenAI, or with a lightweight extractive fallback.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v

python -m rag prepare --limit 3000 --evaluation-queries 300
python -m rag search "disk controller performance" --summary-length short
python -m rag evaluate --query-limit 300
```

Omit `--limit` to prepare the full corpus. The first search builds and caches document embeddings. `--summarizer extractive` avoids downloading a generation model; `--summarizer openai` uses `OPENAI_API_KEY`. `short`, `medium` and `long` summaries target roughly 60, 120 and 220 words.

## Design

- `corpus.py` removes headers, quoted replies, signatures, URLs and email addresses.
- `retrieval.py` keeps lexical and semantic retrieval independent, then fuses ranks.
- `summarization.py` contains interchangeable summarizers.
- `evaluation.py` reports Hit@K, MRR and ROUGE-1/2/L.
- `REPORT.md` records the measured run, trade-offs and manual review.

The NumPy dense index is intentionally simple and works well for this corpus size. For a much larger collection, I would move the same normalized embeddings to an approximate-nearest-neighbor index and persist the BM25 postings instead of rebuilding them in memory.
