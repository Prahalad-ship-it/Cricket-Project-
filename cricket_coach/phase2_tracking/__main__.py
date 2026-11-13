"""Standalone tracking package checks."""

from cricket_coach.config import KEYPOINT_COUNT


if __name__ == "__main__":
    assert KEYPOINT_COUNT == 17
    print("Tracking package checks passed.")
