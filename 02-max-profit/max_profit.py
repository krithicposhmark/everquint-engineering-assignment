from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class Property:
    code: str
    build_time: int
    earnings_per_unit: int


@dataclass(frozen=True)
class Result:
    earnings: int
    mixes: tuple[tuple[int, ...], ...]


PROPERTIES = (
    Property("T", build_time=5, earnings_per_unit=1500),
    Property("P", build_time=4, earnings_per_unit=1000),
    Property("C", build_time=10, earnings_per_unit=2000),
)


def maximize_profit(total_time: int) -> Result:
    if isinstance(total_time, bool) or not isinstance(total_time, int):
        raise TypeError("time must be an integer")
    if total_time < 0:
        raise ValueError("time cannot be negative")

    @cache
    def solve(elapsed: int) -> tuple[int, frozenset[tuple[int, ...]]]:
        best_earnings = 0
        best_mixes = {tuple(0 for _ in PROPERTIES)}

        for index, property_type in enumerate(PROPERTIES):
            completion_time = elapsed + property_type.build_time
            if completion_time >= total_time:
                continue

            future_earnings, future_mixes = solve(completion_time)
            operating_earnings = property_type.earnings_per_unit * (
                total_time - completion_time
            )
            candidate_earnings = operating_earnings + future_earnings
            candidate_mixes = set()
            for mix in future_mixes:
                updated = list(mix)
                updated[index] += 1
                candidate_mixes.add(tuple(updated))

            if candidate_earnings > best_earnings:
                best_earnings = candidate_earnings
                best_mixes = candidate_mixes
            elif candidate_earnings == best_earnings:
                best_mixes.update(candidate_mixes)

        return best_earnings, frozenset(best_mixes)

    earnings, mixes = solve(0)
    return Result(earnings=earnings, mixes=tuple(sorted(mixes, reverse=True)))


def format_result(result: Result) -> str:
    lines = [f"Earnings: ${result.earnings}"]
    for position, mix in enumerate(result.mixes, start=1):
        counts = " ".join(
            f"{property_type.code}: {count}"
            for property_type, count in zip(PROPERTIES, mix)
        )
        lines.append(f"{position}. {counts}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the most profitable build mix.")
    parser.add_argument("time", type=int, help="available units of time")
    args = parser.parse_args()
    print(format_result(maximize_profit(args.time)))


if __name__ == "__main__":
    main()
