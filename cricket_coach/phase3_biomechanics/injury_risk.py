"""Bowling-action screening metrics; not a medical assessment."""

from pathlib import Path
from typing import Any

import numpy as np

from cricket_coach.config import (
    HIP_SHOULDER_SEPARATION_WARNING_DEG,
    LEFT_HIP,
    LEFT_SHOULDER,
    MIXED_ACTION_MAX_SEPARATION_DEG,
    MIXED_ACTION_MIN_SEPARATION_DEG,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from cricket_coach.data_contract import load_phase_payload, save_phase_payload


class InjuryRiskAnalyzer:
    """Measure relative trunk/hip rotation as non-diagnostic action indicators."""

    def analyze(self, poses_3d: np.ndarray) -> dict[str, Any]:
        if poses_3d.ndim != 3 or poses_3d.shape[1] < 17 or poses_3d.shape[2] < 3:
            raise ValueError("Expected 3D pose shaped (frames, 17, at least 3).")
        hips = self._axis_angle(poses_3d[:, LEFT_HIP, :3], poses_3d[:, RIGHT_HIP, :3])
        shoulders = self._axis_angle(poses_3d[:, LEFT_SHOULDER, :3], poses_3d[:, RIGHT_SHOULDER, :3])
        separation = np.abs(np.arctan2(np.sin(shoulders - hips), np.cos(shoulders - hips)))
        separation_deg = np.degrees(separation)
        finite = separation_deg[np.isfinite(separation_deg)]
        peak = float(np.max(finite)) if len(finite) else None
        return {
            "hip_shoulder_separation_peak_deg": peak,
            "separation_warning": None if peak is None else peak > HIP_SHOULDER_SEPARATION_WARNING_DEG,
            "mixed_action_screen": None if peak is None else MIXED_ACTION_MIN_SEPARATION_DEG <= peak <= MIXED_ACTION_MAX_SEPARATION_DEG,
            "status": "screening_only_not_medical_advice",
        }

    @staticmethod
    def _axis_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        direction = right - left
        length = np.linalg.norm(direction[:, [0, 2]], axis=1)
        angle = np.arctan2(direction[:, 2], direction[:, 0])
        angle[length <= np.finfo(np.float32).eps] = np.nan
        angle[~np.isfinite(direction).all(axis=1)] = np.nan
        return angle

    def process(self, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        metadata, arrays = load_phase_payload(metadata_path)
        if "poses_3d" not in arrays:
            raise ValueError("3D poses missing; run pose processing first.")
        metadata["injury_screen"] = self.analyze(arrays["poses_3d"])
        metadata["phase_status"] = {**metadata.get("phase_status", {}), "injury_screen": "complete"}
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays


if __name__ == "__main__":
    poses = np.zeros((2, 17, 3), dtype=np.float32)
    assert InjuryRiskAnalyzer().analyze(poses)["hip_shoulder_separation_peak_deg"] is None
    print("Injury screening checks passed.")
