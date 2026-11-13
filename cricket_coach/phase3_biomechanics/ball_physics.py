"""Calibrated pitch mapping and ball-speed calculation when metric tracks exist."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cricket_coach.config import ICC_CREASE_DEPTH_M, ICC_CREASE_WIDTH_M
from cricket_coach.data_contract import load_phase_payload, save_phase_payload
from cricket_coach.phase1_capture.calibrator import PitchCalibrator


class BallPhysicsAnalyzer:
    """Convert ground-plane image tracks and metric trajectories into estimates."""

    @staticmethod
    def pixel_track_to_pitch(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """Project image points to the calibrated pitch plane (valid for ground-plane points)."""
        if homography.shape != (3, 3) or points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("Expected 3x3 homography and Nx2 image coordinates.")
        mapped = np.full((len(points_xy), 2), np.nan, dtype=np.float32)
        valid = np.isfinite(points_xy).all(axis=1)
        if valid.any():
            mapped[valid] = cv2.perspectiveTransform(points_xy[valid].astype(np.float32).reshape(1, -1, 2), homography)[0]
        return mapped

    @staticmethod
    def release_speed_kmh(metric_xyz: np.ndarray, fps: float, release_frame: int) -> float | None:
        """Estimate speed from two adjacent 3D metric samples, when available."""
        if metric_xyz.ndim != 2 or metric_xyz.shape[1] != 3 or fps <= 0:
            raise ValueError("Expected Nx3 metric positions and positive FPS.")
        if release_frame <= 0 or release_frame >= len(metric_xyz):
            return None
        pair = metric_xyz[release_frame - 1 : release_frame + 1]
        if not np.isfinite(pair).all():
            return None
        meters_per_second = float(np.linalg.norm(pair[1] - pair[0]) * fps)
        return meters_per_second * 3.6

    @staticmethod
    def pitch_zone(point_m: tuple[float, float] | None, width_m: float, depth_m: float) -> dict[str, Any]:
        if point_m is None or not np.isfinite(point_m).all():
            return {"status": "unavailable", "x_m": None, "y_m": None}
        x, y = point_m
        return {
            "status": "inside_calibrated_box" if 0 <= x <= width_m and 0 <= y <= depth_m else "outside_calibrated_box",
            "x_m": float(x),
            "y_m": float(y),
        }

    def process(self, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        metadata, arrays = load_phase_payload(metadata_path)
        matrix_path = metadata.get("calibration_matrix_path")
        calibration_status = metadata.get("calibration_status", "unknown")
        if (
            matrix_path
            and Path(matrix_path).is_file()
            and "ball_xy_pixels" in arrays
            and calibration_status != "identity_fallback"
        ):
            homography = np.load(matrix_path, allow_pickle=False)
            if PitchCalibrator.is_identity_fallback(homography):
                calibration_status = "identity_fallback"
                pitch_points = np.empty((0, 2), dtype=np.float32)
            else:
                pitch_points = self.pixel_track_to_pitch(arrays["ball_xy_pixels"], homography)
            arrays["ball_pitch_xy_m"] = pitch_points
            bounce = metadata.get("ball_tracking", {}).get("bounce_fit", {})
            bounce_frame = bounce.get("frame")
            if (
                calibration_status != "identity_fallback"
                and isinstance(bounce_frame, int)
                and 0 <= bounce_frame < len(pitch_points)
            ):
                point = pitch_points[bounce_frame]
                zone = self.pitch_zone((float(point[0]), float(point[1])), ICC_CREASE_WIDTH_M, ICC_CREASE_DEPTH_M)
            else:
                zone = self.pitch_zone(None, ICC_CREASE_WIDTH_M, ICC_CREASE_DEPTH_M)
        else:
            pitch_points = np.empty((0, 2), dtype=np.float32)
            arrays["ball_pitch_xy_m"] = pitch_points
            zone = self.pitch_zone(None, ICC_CREASE_WIDTH_M, ICC_CREASE_DEPTH_M)
        metadata["ball_physics"] = {
            "release_speed_kmh": None,
            "pitch_track_status": (
                "unavailable_identity_calibration"
                if calibration_status == "identity_fallback"
                else "image_to_ground_plane_projection_only"
            ),
            "bounce_pitch_zone_candidate": zone,
            "note": "A ground-plane homography cannot recover airborne ball depth or metric release speed.",
        }
        metadata["phase_status"] = {**metadata.get("phase_status", {}), "ball_physics": "partial"}
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays


if __name__ == "__main__":
    assert BallPhysicsAnalyzer.release_speed_kmh(np.zeros((3, 3)), 30, 0) is None
    print("Ball physics checks passed.")
