"""3D joint geometry and bowling technique screening measurements."""

from pathlib import Path
from typing import Any

import numpy as np

from cricket_coach.config import (
    ELBOW_EXTENSION_WARNING_DEG,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    KNEE_BRACE_MIN_ANGLE_DEG,
    TRUNK_LEAN_WARNING_DEG,
)
from cricket_coach.data_contract import load_phase_payload, save_phase_payload


class KinematicsAnalyzer:
    """Compute interpretable joint angles from metric or lifted 3D landmarks."""

    def analyze(self, poses: np.ndarray) -> dict[str, Any]:
        """Analyze pose frames; unknown/NaN depth yields unknown 3D measurements."""
        if poses.ndim != 3 or poses.shape[1] < 17 or poses.shape[2] < 3:
            raise ValueError("Expected pose array shaped (frames, 17, at least 3).")
        knee_angles = np.column_stack(
            (
                self._angle(poses[:, LEFT_HIP, :3], poses[:, LEFT_KNEE, :3], poses[:, LEFT_ANKLE, :3]),
                self._angle(poses[:, RIGHT_HIP, :3], poses[:, RIGHT_KNEE, :3], poses[:, RIGHT_ANKLE, :3]),
            )
        )
        elbow_angles = np.column_stack(
            (
                self._angle(poses[:, LEFT_SHOULDER, :3], poses[:, LEFT_ELBOW, :3], poses[:, LEFT_WRIST, :3]),
                self._angle(poses[:, RIGHT_SHOULDER, :3], poses[:, RIGHT_ELBOW, :3], poses[:, RIGHT_WRIST, :3]),
            )
        )
        hips_center = (poses[:, LEFT_HIP, :3] + poses[:, RIGHT_HIP, :3]) / 2
        shoulders_center = (poses[:, LEFT_SHOULDER, :3] + poses[:, RIGHT_SHOULDER, :3]) / 2
        trunk_vector = shoulders_center - hips_center
        trunk_tilt = np.degrees(np.arctan2(np.linalg.norm(trunk_vector[:, [0, 2]], axis=1), np.abs(trunk_vector[:, 1])))
        metrics = {
            "knee_angle_deg": self._summary(knee_angles),
            "elbow_angle_deg": self._summary(elbow_angles),
            "trunk_tilt_deg": self._summary(trunk_tilt),
            "knee_brace_warning": self._flag_min(knee_angles, KNEE_BRACE_MIN_ANGLE_DEG),
            "trunk_tilt_warning": self._flag_max(trunk_tilt, TRUNK_LEAN_WARNING_DEG),
            "elbow_extension_screen": self._flag_elbow(elbow_angles),
            "interpretation_note": "Screening estimates only; camera calibration and validated 3D pose are required for meaningful biomechanics.",
        }
        return metrics

    @staticmethod
    def _angle(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> np.ndarray:
        a, b = first - vertex, last - vertex
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.divide(np.sum(a * b, axis=1), denominator, out=np.full(len(a), np.nan), where=denominator > 0)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @staticmethod
    def _summary(values: np.ndarray) -> dict[str, float | None]:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return {"min": None, "mean": None, "max": None}
        return {"min": float(finite.min()), "mean": float(finite.mean()), "max": float(finite.max())}

    @staticmethod
    def _flag_min(values: np.ndarray, threshold: float) -> bool | None:
        finite = values[np.isfinite(values)]
        return bool(np.min(finite) < threshold) if len(finite) else None

    @staticmethod
    def _flag_max(values: np.ndarray, threshold: float) -> bool | None:
        finite = values[np.isfinite(values)]
        return bool(np.max(finite) > threshold) if len(finite) else None

    @staticmethod
    def _flag_elbow(values: np.ndarray) -> bool | None:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return None
        extension = 180.0 - finite
        return bool(np.max(extension) > ELBOW_EXTENSION_WARNING_DEG)

    def process(self, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        metadata, arrays = load_phase_payload(metadata_path)
        if "poses_3d" not in arrays:
            raise ValueError("3D pose array missing; run pose processing first.")
        metadata["biomechanics"] = self.analyze(arrays["poses_3d"])
        metadata["phase_status"] = {**metadata.get("phase_status", {}), "biomechanics": "complete"}
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays


if __name__ == "__main__":
    collinear = np.zeros((1, 17, 4), dtype=np.float32)
    collinear[0, LEFT_HIP, :3] = [0, 2, 0]
    collinear[0, LEFT_KNEE, :3] = [0, 1, 0]
    collinear[0, LEFT_ANKLE, :3] = [0, 0, 0]
    assert np.isclose(KinematicsAnalyzer._angle(collinear[:, LEFT_HIP, :3], collinear[:, LEFT_KNEE, :3], collinear[:, LEFT_ANKLE, :3])[0], 180)
    print("Kinematics checks passed.")
