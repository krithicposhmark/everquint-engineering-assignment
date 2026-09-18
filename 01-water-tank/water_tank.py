from __future__ import annotations

import argparse
from collections.abc import Sequence


def calculate_trapped_water(heights: Sequence[int]) -> int:
    """Return the amount of water held by the supplied block heights."""
    if any(
        isinstance(height, bool) or not isinstance(height, int) for height in heights
    ):
        raise TypeError("block heights must be integers")
    if any(height < 0 for height in heights):
        raise ValueError("block heights cannot be negative")

    left = 0
    right = len(heights) - 1
    left_max = 0
    right_max = 0
    water = 0

    while left < right:
        if heights[left] <= heights[right]:
            left_max = max(left_max, heights[left])
            water += left_max - heights[left]
            left += 1
        else:
            right_max = max(right_max, heights[right])
            water += right_max - heights[right]
            right -= 1

    return water


def _parse_height(value: str) -> int:
    try:
        height = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if height < 0:
        raise argparse.ArgumentTypeError("block heights cannot be negative")
    return height


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate trapped rain water.")
    parser.add_argument("heights", nargs="+", type=_parse_height)
    args = parser.parse_args()
    print(calculate_trapped_water(args.heights))


if __name__ == "__main__":
    main()
