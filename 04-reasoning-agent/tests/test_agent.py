import json
import unittest

from reasoning_agent.agent import ReasoningAgent


class SuccessfulProvider:
    def complete(self, system_prompt: str, payload: str) -> str:
        del payload
        if system_prompt.startswith("[planner]"):
            return json.dumps({"steps": ["calculate", "check"]})
        if system_prompt.startswith("[executor]"):
            return json.dumps(
                {"answer": "25", "explanation": "Adding 18 and 7 gives 25."}
            )
        return json.dumps(
            {
                "passed": True,
                "checks": [
                    {
                        "check_name": "arithmetic",
                        "passed": True,
                        "details": "18 + 7 equals 25",
                    }
                ],
                "feedback": "",
            }
        )


class RetryProvider:
    def __init__(self, always_fail: bool = False) -> None:
        self.verifications = 0
        self.always_fail = always_fail

    def complete(self, system_prompt: str, payload: str) -> str:
        if system_prompt.startswith("[planner]"):
            return json.dumps({"steps": ["calculate", "check"]})
        if system_prompt.startswith("[executor]"):
            return json.dumps({"answer": "4", "explanation": "Two plus two is four."})
        self.verifications += 1
        passed = not self.always_fail and self.verifications > 1
        return json.dumps(
            {
                "passed": passed,
                "checks": [
                    {"check_name": "arithmetic", "passed": passed, "details": "checked"}
                ],
                "feedback": "try again" if not passed else "",
            }
        )


class MalformedProvider:
    def complete(self, system_prompt: str, payload: str) -> str:
        del system_prompt, payload
        return "not json"


class ReasoningAgentTest(unittest.TestCase):
    def test_successful_result_matches_schema(self) -> None:
        result = ReasoningAgent(SuccessfulProvider()).solve("What is 18 + 7?")
        self.assertEqual(result["answer"], "25")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metadata"]["retries"], 0)
        self.assertTrue(result["metadata"]["checks"][0]["passed"])

    def test_failed_verification_triggers_retry(self) -> None:
        result = ReasoningAgent(RetryProvider(), max_retries=2).solve("What is 2 + 2?")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metadata"]["retries"], 1)

    def test_retry_limit_returns_failed_status(self) -> None:
        result = ReasoningAgent(RetryProvider(always_fail=True), max_retries=1).solve(
            "What is 2 + 2?"
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metadata"]["retries"], 1)

    def test_malformed_provider_output_returns_failed_status(self) -> None:
        result = ReasoningAgent(MalformedProvider(), max_retries=0).solve(
            "What is 2 + 2?"
        )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["metadata"]["checks"][0]["passed"])

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReasoningAgent(SuccessfulProvider()).solve("  ")


if __name__ == "__main__":
    unittest.main()
