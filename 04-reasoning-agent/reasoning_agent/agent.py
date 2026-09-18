from __future__ import annotations

import json
from typing import Any

from reasoning_agent.prompts import EXECUTOR_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from reasoning_agent.providers import CompletionProvider


class ReasoningAgent:
    def __init__(self, provider: CompletionProvider, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.provider = provider
        self.max_retries = max_retries

    @staticmethod
    def _parse_object(raw_response: str, stage: str) -> dict[str, Any]:
        parsed = json.loads(raw_response)
        if not isinstance(parsed, dict):
            raise TypeError(f"{stage} returned a non-object JSON value")
        return parsed

    def solve(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question cannot be empty")

        retries = 0
        last_checks: list[dict[str, Any]] = []
        last_plan: list[str] = []
        feedback = ""

        while retries <= self.max_retries:
            try:
                plan_payload = json.dumps(
                    {"question": question, "previous_feedback": feedback}
                )
                plan = self._parse_object(
                    self.provider.complete(PLANNER_PROMPT, plan_payload), "planner"
                )
                steps = plan.get("steps")
                if not isinstance(steps, list) or not all(
                    isinstance(step, str) and step.strip() for step in steps
                ):
                    raise ValueError("planner returned invalid steps")
                last_plan = steps

                execution_payload = json.dumps(
                    {"question": question, "plan": steps, "previous_feedback": feedback}
                )
                solution = self._parse_object(
                    self.provider.complete(EXECUTOR_PROMPT, execution_payload),
                    "executor",
                )
                if not isinstance(solution.get("answer"), str) or not isinstance(
                    solution.get("explanation"), str
                ):
                    raise TypeError("executor returned an invalid answer")

                verification_payload = json.dumps(
                    {"question": question, "proposed_solution": solution}
                )
                verification = self._parse_object(
                    self.provider.complete(VERIFIER_PROMPT, verification_payload),
                    "verifier",
                )
                last_checks = verification.get("checks", [])
                if verification.get("passed") is True:
                    return {
                        "answer": solution["answer"],
                        "status": "success",
                        "reasoning_visible_to_user": solution["explanation"],
                        "metadata": {
                            "plan": " -> ".join(last_plan),
                            "checks": last_checks,
                            "retries": retries,
                        },
                    }
                feedback = str(verification.get("feedback", "verification failed"))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                feedback = f"{type(exc).__name__}: {exc}"
                last_checks = [
                    {
                        "check_name": "valid_stage_output",
                        "passed": False,
                        "details": feedback,
                    }
                ]

            retries += 1

        return {
            "answer": "Unable to produce a verified answer.",
            "status": "failed",
            "reasoning_visible_to_user": feedback,
            "metadata": {
                "plan": " -> ".join(last_plan),
                "checks": last_checks,
                "retries": self.max_retries,
            },
        }
