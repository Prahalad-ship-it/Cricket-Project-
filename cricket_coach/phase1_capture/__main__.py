"""Standalone capture package checks."""

from cricket_coach.phase1_capture.calibrator import PitchCalibrator


if __name__ == "__main__":
    assert len(PitchCalibrator.LABELS) == 4
    print("Capture package checks passed.")
