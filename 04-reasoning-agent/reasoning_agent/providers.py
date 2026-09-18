from __future__ import annotations

from typing import Protocol


class CompletionProvider(Protocol):
    def complete(self, system_prompt: str, payload: str) -> str: ...


class OpenAICompletionProvider:
    def __init__(self, model: str = "gpt-5-mini") -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def complete(self, system_prompt: str, payload: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=payload,
        )
        return response.output_text
