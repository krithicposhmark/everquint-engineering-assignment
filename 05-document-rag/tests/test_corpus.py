import unittest

from rag.corpus import clean_text


class CleanTextTest(unittest.TestCase):
    def test_removes_contact_details_and_quoted_replies(self) -> None:
        text = "Contact user@internal or https://example.com.\n> quoted reply\nUseful body."
        cleaned = clean_text(text)
        self.assertNotIn("user@internal", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("quoted reply", cleaned)
        self.assertIn("Useful body.", cleaned)


if __name__ == "__main__":
    unittest.main()
