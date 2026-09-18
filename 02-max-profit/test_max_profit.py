import unittest

from max_profit import maximize_profit


class MaximizeProfitTest(unittest.TestCase):
    def test_assignment_case_seven(self) -> None:
        result = maximize_profit(7)
        self.assertEqual(result.earnings, 3000)
        self.assertEqual(set(result.mixes), {(1, 0, 0), (0, 1, 0)})

    def test_assignment_case_eight(self) -> None:
        result = maximize_profit(8)
        self.assertEqual(result.earnings, 4500)
        self.assertEqual(result.mixes, ((1, 0, 0),))

    def test_assignment_case_thirteen(self) -> None:
        result = maximize_profit(13)
        self.assertEqual(result.earnings, 16500)
        self.assertEqual(result.mixes, ((2, 0, 0),))

    def test_time_before_first_profitable_completion(self) -> None:
        self.assertEqual(maximize_profit(4).earnings, 0)
        self.assertEqual(maximize_profit(4).mixes, ((0, 0, 0),))

    def test_rejects_invalid_time(self) -> None:
        with self.assertRaises(ValueError):
            maximize_profit(-1)
        with self.assertRaises(TypeError):
            maximize_profit(7.5)


if __name__ == "__main__":
    unittest.main()
