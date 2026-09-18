import unittest

from water_tank import calculate_trapped_water


class CalculateTrappedWaterTest(unittest.TestCase):
    def test_assignment_example(self) -> None:
        self.assertEqual(calculate_trapped_water([0, 4, 0, 0, 0, 6, 0, 6, 4, 0]), 18)

    def test_common_example(self) -> None:
        self.assertEqual(calculate_trapped_water([3, 0, 2, 0, 4]), 7)

    def test_empty_and_short_inputs(self) -> None:
        self.assertEqual(calculate_trapped_water([]), 0)
        self.assertEqual(calculate_trapped_water([4]), 0)
        self.assertEqual(calculate_trapped_water([4, 0]), 0)

    def test_monotonic_heights_do_not_hold_water(self) -> None:
        self.assertEqual(calculate_trapped_water([1, 2, 3, 4]), 0)
        self.assertEqual(calculate_trapped_water([4, 3, 2, 1]), 0)

    def test_rejects_invalid_heights(self) -> None:
        with self.assertRaises(ValueError):
            calculate_trapped_water([2, -1, 2])
        with self.assertRaises(TypeError):
            calculate_trapped_water([2, 1.5, 2])


if __name__ == "__main__":
    unittest.main()
