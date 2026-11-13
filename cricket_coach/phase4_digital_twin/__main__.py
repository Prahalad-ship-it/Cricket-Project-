"""Standalone digital-twin package checks."""

from cricket_coach.phase4_digital_twin.pro_comparator import ProComparator


if __name__ == "__main__":
    assert ProComparator.sequence_distance(
        [[0.0], [1.0]], [[0.0], [1.0]], use_dtaidistance=False
    ) == 0.0
    print("Digital-twin package checks passed.")
