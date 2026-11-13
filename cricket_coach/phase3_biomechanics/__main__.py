"""Standalone biomechanics package checks."""

from cricket_coach.config import ICC_CREASE_DEPTH_M, ICC_CREASE_WIDTH_M


if __name__ == "__main__":
    assert ICC_CREASE_WIDTH_M > 0 and ICC_CREASE_DEPTH_M > 0
    print("Biomechanics package checks passed.")
