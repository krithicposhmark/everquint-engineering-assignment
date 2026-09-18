from __future__ import annotations

import argparse
import json
from pathlib import Path

from reasoning_agent.agent import ReasoningAgent
from reasoning_agent.providers import OpenAICompletionProvider

CASES = (
    ("easy", "What is 12 + 7?", "19", "Adding 12 and 7 gives 19."),
    ("easy", "What is 25 - 9?", "16", "Subtracting 9 from 25 gives 16."),
    ("easy", "What is 8 * 6?", "48", "Multiplying 8 by 6 gives 48."),
    ("easy", "What is 81 / 9?", "9", "Dividing 81 by 9 gives 9."),
    ("easy", "What is 15% of 200?", "30", "Fifteen percent of 200 is 30."),
    (
        "easy",
        "What is the average of 4, 8, and 12?",
        "8",
        "The values total 24, and 24 divided by 3 is 8.",
    ),
    (
        "easy",
        "A train leaves at 14:30 and arrives at 18:05. How long is the journey?",
        "3 hours 35 minutes",
        "The elapsed time is 3 hours and 35 minutes.",
    ),
    (
        "easy",
        "Alice has 3 red apples and twice as many green apples. How many apples does she have?",
        "9",
        "There are 3 red and 6 green apples, for 9 in total.",
    ),
    (
        "tricky",
        "What is (12 + 8) * 3?",
        "60",
        "The value in parentheses is 20, and 20 multiplied by 3 is 60.",
    ),
    (
        "tricky",
        "A train leaves at 23:40 and arrives at 01:10. How long is the journey?",
        "1 hour 30 minutes",
        "Accounting for midnight, the elapsed time is 1 hour and 30 minutes.",
    ),
    (
        "tricky",
        "A meeting needs 60 minutes. Free slots are 09:00-09:30, 09:45-10:30, and 11:00-12:00. Which slots fit?",
        "11:00-12:00",
        "Only 11:00-12:00 is at least 60 minutes long.",
    ),
    (
        "tricky",
        "A shop discounts an $80 item by 25%, then adds 10% tax. What is the final price?",
        "$66",
        "The discounted price is $60, and 10% tax adds $6.",
    ),
)


class FixtureProvider:
    def __init__(self, answer: str, explanation: str) -> None:
        self.answer = answer
        self.explanation = explanation

    def complete(self, system_prompt: str, payload: str) -> str:
        del payload
        if system_prompt.startswith("[planner]"):
            return json.dumps(
                {"steps": ["extract the given values", "calculate", "verify"]}
            )
        if system_prompt.startswith("[executor]"):
            return json.dumps(
                {"answer": self.answer, "explanation": self.explanation, "facts": []}
            )
        return json.dumps(
            {
                "passed": True,
                "checks": [
                    {
                        "check_name": "expected_answer",
                        "passed": True,
                        "details": "answer matches the evaluation fixture",
                    }
                ],
                "feedback": "",
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/run-log.jsonl"))
    parser.add_argument("--live", action="store_true", help="run cases through OpenAI")
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()
    with args.output.open("w", encoding="utf-8") as output_file:
        for difficulty, question, expected_answer, explanation in CASES:
            provider = (
                OpenAICompletionProvider(args.model)
                if args.live
                else FixtureProvider(expected_answer, explanation)
            )
            agent = ReasoningAgent(provider)
            result = agent.solve(question)
            output_file.write(
                json.dumps(
                    {
                        "provider": args.model if args.live else "fixture",
                        "difficulty": difficulty,
                        "question": question,
                        "expected_answer": expected_answer,
                        "matches_expected": result["answer"] == expected_answer,
                        "result": result,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
