from __future__ import annotations

import hashlib
import html
import json
import random
import re
from email import message_from_string
from pathlib import Path

from rag.types import Document


def clean_text(value: str) -> str:
    # Decode HTML entities before applying the text cleanup rules.
    value = html.unescape(value)

    # Remove URLs and email addresses that do not help document retrieval.
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\b", " ", value)

    # Drop email signatures and quoted lines copied from earlier messages.
    value = re.split(r"\n--\s*\n", value, maxsplit=1)[0]
    value = "\n".join(
        line for line in value.splitlines() if not line.lstrip().startswith(">")
    )

    # Collapse newlines and repeated spaces into a single normalized space.
    return re.sub(r"\s+", " ", value).strip()


def _parse_message(raw_message: str, category: str) -> tuple[Document, str] | None:
    message = message_from_string(raw_message)
    title = clean_text(message.get("Subject", ""))
    body = message.get_payload()
    if not isinstance(body, str):
        return None
    body = clean_text(body)
    if len(title) < 4 or len(body) < 120:
        return None
    document_id = hashlib.sha1(f"{category}\n{title}\n{body}".encode()).hexdigest()[:16]
    return Document(document_id, title, body, category), title


def load_20_newsgroups(
    limit: int | None = None, seed: int = 42
) -> tuple[list[Document], list[dict[str, str]]]:
    from sklearn.datasets import fetch_20newsgroups

    dataset = fetch_20newsgroups(subset="all", remove=())
    records: list[tuple[Document, str]] = []
    for raw_message, target in zip(dataset.data, dataset.target):
        parsed = _parse_message(raw_message, dataset.target_names[target])
        if parsed is not None:
            records.append(parsed)

    random.Random(seed).shuffle(records)
    if limit is not None:
        records = records[:limit]
    documents = [record[0] for record in records]
    queries = [
        {"query": record[1], "target_id": record[0].document_id} for record in records
    ]
    return documents, queries


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_documents(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as input_file:
        return [Document(**json.loads(line)) for line in input_file if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]
