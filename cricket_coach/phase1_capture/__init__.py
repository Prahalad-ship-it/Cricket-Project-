"""Video ingestion, pitch calibration, and delivery slicing."""

if __name__ == "__main__":
    from cricket_coach.phase1_capture.calibrator import PitchCalibrator

    assert len(PitchCalibrator.LABELS) == 4
    print("Capture package checks passed.")
