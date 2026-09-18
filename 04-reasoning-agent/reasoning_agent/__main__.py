from __future__ import annotations

import argparse
import json

from reasoning_agent.agent import ReasoningAgent
from reasoning_agent.providers import OpenAICompletionProvider


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve and verify a structured word problem."
    )
    parser.add_argument("question", nargs="+")
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    result = ReasoningAgent(OpenAICompletionProvider(args.model)).solve(
        " ".join(args.question)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
