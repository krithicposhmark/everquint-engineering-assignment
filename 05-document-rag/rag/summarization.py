from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from rag.types import SearchResult

SUMMARY_WORDS = {"short": 60, "medium": 120, "long": 220}


class Summarizer(Protocol):
    def summarize(self, text: str, length: str) -> str: ...


def build_context(results: Sequence[SearchResult], character_limit: int = 12000) -> str:
    sections = [
        f"{result.document.title}\n{result.document.text}" for result in results
    ]
    return "\n\n".join(sections)[:character_limit]


class ExtractiveSummarizer:
    def summarize(self, text: str, length: str) -> str:
        word_limit = SUMMARY_WORDS[length]
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        selected: list[str] = []
        word_count = 0
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            if selected and word_count + len(words) > word_limit:
                break
            selected.append(sentence)
            word_count += len(words)
        return " ".join(selected)


class TransformersSummarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-6-6") -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def summarize(self, text: str, length: str) -> str:
        word_limit = SUMMARY_WORDS[length]
        maximum_tokens = max(30, round(word_limit * 1.4))
        minimum_tokens = max(10, round(maximum_tokens * 0.35))
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )
        generated = self.model.generate(
            **inputs,
            max_length=maximum_tokens,
            min_length=minimum_tokens,
            do_sample=False,
            num_beams=4,
        )
        summary = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return re.sub(r"\s+([.,!?;:])", r"\1", summary)


class OpenAISummarizer:
    def __init__(self, model: str = "gpt-5-mini") -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def summarize(self, text: str, length: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                f"Summarize the supplied search results in about {SUMMARY_WORDS[length]} words. "
                "Keep the claims grounded in the text and do not add outside facts."
            ),
            input=text,
        )
        return response.output_text.strip()


def build_summarizer(name: str, model: str | None = None) -> Summarizer:
    if name == "extractive":
        return ExtractiveSummarizer()
    if name == "transformers":
        return TransformersSummarizer(model or "sshleifer/distilbart-cnn-6-6")
    if name == "openai":
        return OpenAISummarizer(model or "gpt-5-mini")
    raise ValueError(f"unknown summarizer: {name}")
