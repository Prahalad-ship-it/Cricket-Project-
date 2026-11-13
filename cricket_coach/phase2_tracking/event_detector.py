"""Detect coarse bowling action phases from pose sequences."""

from pathlib import Path
from typing import Any

import numpy as np

from cricket_coach.config import LEFT_ANKLE, LEFT_WRIST, MIN_EVENT_FRAMES, RIGHT_ANKLE, RIGHT_WRIST
from cricket_coach.data_contract import load_phase_payload, save_phase_payload


class EventDetector:
    """Estimate BFC, FFC, and release candidates from observed keypoint motion."""

    def detect(self, poses: np.ndarray, fps: float) -> dict[str, Any]:
        """Return event frame/time candidates with confidence and method labels."""
        if poses.ndim != 3 or poses.shape[1] < 17 or poses.shape[2] < 2:
            raise ValueError("Expected pose array shaped (frames, 17, dimensions).")
        if fps <= 0 or not np.isfinite(fps):
            raise ValueError("FPS must be finite and positive.")
        if len(poses) < MIN_EVENT_FRAMES:
            return {"status": "insufficient_frames", "events": {}}
        velocity = np.full((len(poses), 2), np.nan, dtype=np.float32)
        for col, ankle in enumerate((LEFT_ANKLE, RIGHT_ANKLE)):
            positions = poses[:, ankle, :2]
            valid = np.isfinite(positions).all(axis=1)
            for frame in range(1, len(poses)):
                if valid[frame] and valid[frame - 1]:
                    velocity[frame, col] = np.linalg.norm(positions[frame] - positions[frame - 1]) * fps
        if not np.isfinite(velocity).any():
            return {"status": "insufficient_keypoints", "events": {}}
        ankle_speed = np.nan_to_num(velocity, nan=-np.inf)
        lead_foot = np.argmax(ankle_speed, axis=1)
        ffc_index = int(np.argmax(np.max(ankle_speed, axis=1)))
        ffc_confidence = float(np.nanmax(ankle_speed)) if np.isfinite(ankle_speed).any() else 0.0
        release_speed = np.zeros(len(poses), dtype=np.float32)
        for wrist in (LEFT_WRIST, RIGHT_WRIST):
            delta = np.diff(poses[:, wrist, :2], axis=0)
            speed = np.linalg.norm(delta, axis=1) * fps
            release_speed[1:] = np.maximum(release_speed[1:], np.nan_to_num(speed, nan=0.0, posinf=0.0))
        release_index = int(np.argmax(release_speed))
        if ffc_index >= 1:
            bfc_index = int(np.unravel_index(np.argmax(ankle_speed[: ffc_index + 1, :]), ankle_speed[: ffc_index + 1, :].shape)[0])
        else:
            bfc_index = 0
        events = {}
        for name, frame, confidence, method in (
            ("BFC", bfc_index, min(ffc_confidence / max(fps, 1), 1.0), "peak ankle speed before FFC proxy"),
            ("FFC", ffc_index, min(ffc_confidence / max(fps, 1), 1.0), "peak ankle image-space speed proxy"),
            ("release", release_index, 0.0, "peak wrist image-space speed candidate"),
        ):
            events[name] = {
                "frame": frame,
                "time_sec": frame / fps,
                "confidence": float(confidence),
                "method": method,
            }
        events["FFC"]["lead_foot_index"] = int(lead_foot[ffc_index])
        return {"status": "estimated_candidates", "events": events}

    def process(self, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Read stored pose arrays, detect events, and update delivery JSON."""
        metadata, arrays = load_phase_payload(metadata_path)
        if "poses_2d" not in arrays:
            raise ValueError("Pose arrays are missing; run pose extraction first.")
        result = self.detect(arrays["poses_2d"], float(metadata["original_fps"]))
        metadata["events"] = result
        metadata["phase_status"] = {**metadata.get("phase_status", {}), "event_detection": result["status"]}
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays


if __name__ == "__main__":
    poses = np.zeros((5, 17, 3), dtype=np.float32)
    poses[:, LEFT_ANKLE, 0] = np.arange(5)
    result = EventDetector().detect(poses, 30)
    assert "FFC" in result["events"]
    print("Event detection checks passed.")
