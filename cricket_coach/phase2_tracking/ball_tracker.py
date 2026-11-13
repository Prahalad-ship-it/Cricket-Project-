"""TrackNet-compatible ball heatmap detection and bounce trajectory fitting."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cricket_coach.config import BALL_CONFIDENCE_THRESHOLD, MIN_BALL_FIT_DETECTIONS
from cricket_coach.data_contract import load_phase_payload, save_phase_payload


class BallTracker:
    """Extract image-space ball centers from a TrackNet heatmap provider."""

    def __init__(self, heatmap_model: Any = None, confidence_threshold: float = BALL_CONFIDENCE_THRESHOLD) -> None:
        self.model = heatmap_model
        self.confidence_threshold = confidence_threshold

    def track(self, video_path: str | Path, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Track detected ball centers and save arrays; no-model frames remain NaN."""
        metadata, arrays = load_phase_payload(metadata_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open delivery video: {video_path}")
        points: list[tuple[float, float]] = []
        scores: list[float] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                point, confidence = self._detect(frame)
                points.append(point)
                scores.append(confidence)
        finally:
            capture.release()
        ball_xy = np.asarray(points, dtype=np.float32).reshape((-1, 2))
        ball_confidence = np.asarray(scores, dtype=np.float32)
        bounce = self.fit_bounce(ball_xy, float(metadata.get("original_fps", 0.0)))
        arrays.update({"ball_xy_pixels": ball_xy, "ball_confidence": ball_confidence})
        metadata["ball_tracking"] = {
            "status": "tracked" if self.model is not None and np.isfinite(ball_xy).any() else "unavailable",
            "bounce_fit": bounce,
        }
        metadata["phase_status"] = {**metadata.get("phase_status", {}), "tracking_ball": metadata["ball_tracking"]["status"]}
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays

    def _detect(self, frame: np.ndarray) -> tuple[tuple[float, float], float]:
        if self.model is None:
            return (float("nan"), float("nan")), 0.0
        prediction = self.model(frame)
        if hasattr(prediction, "detach"):
            prediction = prediction.detach()
        if hasattr(prediction, "cpu"):
            prediction = prediction.cpu()
        if hasattr(prediction, "numpy"):
            prediction = prediction.numpy()
        heatmap = np.asarray(prediction)
        if heatmap.ndim == 3:
            heatmap = heatmap.squeeze()
        if heatmap.ndim != 2 or not np.isfinite(heatmap).any():
            return (float("nan"), float("nan")), 0.0
        _, confidence, _, location = cv2.minMaxLoc(heatmap.astype(np.float32))
        if confidence < self.confidence_threshold:
            return (float("nan"), float("nan")), float(confidence)
        height, width = frame.shape[:2]
        x = location[0] * width / heatmap.shape[1]
        y = location[1] * height / heatmap.shape[0]
        return (float(x), float(y)), float(confidence)

    @staticmethod
    def fit_bounce(points_xy: np.ndarray, fps: float) -> dict[str, float | int | None]:
        """Fit image-space vertical motion; bounce estimate is only a candidate."""
        valid = np.isfinite(points_xy).all(axis=1)
        indices = np.flatnonzero(valid)
        if len(indices) < MIN_BALL_FIT_DETECTIONS or fps <= 0:
            return {"status": "insufficient_detections", "frame": None, "time_sec": None, "x_pixels": None, "fit_rmse_pixels": None}
        x = indices.astype(np.float64) / fps
        y = points_xy[indices, 1].astype(np.float64)
        coefficients = np.polyfit(x, y, 2)
        a, b, c = coefficients
        vertex_time = -b / (2 * a) if abs(a) > np.finfo(float).eps else float("nan")
        fitted = np.polyval(coefficients, x)
        rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
        if not np.isfinite(vertex_time) or not (x.min() <= vertex_time <= x.max()):
            return {"status": "no_in_range_vertex", "frame": None, "time_sec": None, "x_pixels": None, "fit_rmse_pixels": rmse}
        nearest = int(np.argmin(np.abs(x - vertex_time)))
        return {
            "status": "candidate_image_space_vertex",
            "frame": int(indices[nearest]),
            "time_sec": float(vertex_time),
            "x_pixels": float(points_xy[indices[nearest], 0]),
            "fit_rmse_pixels": rmse,
        }


if __name__ == "__main__":
    times = np.arange(9, dtype=np.float64)
    points = np.column_stack((times, -(times - 4) ** 2)).astype(np.float32)
    fit = BallTracker.fit_bounce(points, 30)
    assert fit["status"] == "candidate_image_space_vertex"
    print("Ball trajectory fit checks passed.")
