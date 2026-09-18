import unittest

from rag.summarization import ExtractiveSummarizer


class SummarizationTest(unittest.TestCase):
    def test_length_setting_changes_summary_budget(self) -> None:
        text = " ".join(
            f"Sentence {index} contains several useful words about the document."
            for index in range(1, 30)
        )
        summarizer = ExtractiveSummarizer()
        short = summarizer.summarize(text, "short")
        long = summarizer.summarize(text, "long")
        self.assertLess(len(short.split()), len(long.split()))


if __name__ == "__main__":
    unittest.main()
