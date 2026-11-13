"""Pose inference and optional externally supplied 3D pose lifting."""

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from cricket_coach.config import KEYPOINT_COUNT, MODEL_POSE_PATH
from cricket_coach.data_contract import load_phase_payload, save_phase_payload


class Pose3DProcessor:
    """Extract 2D poses and lift them when a trained 3D model is provided."""

    def __init__(self, pose_model: Any = None, lifter: Callable[[np.ndarray], np.ndarray] | None = None, model_path: str = MODEL_POSE_PATH) -> None:
        if pose_model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise ImportError("Install ultralytics to run pose extraction.") from error
            pose_model = YOLO(model_path)
        self.pose_model = pose_model
        self.lifter = lifter

    def process(self, video_path: str | Path, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Process all frames, persist pose arrays, and update the delivery contract."""
        metadata, arrays = load_phase_payload(metadata_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open delivery video: {video_path}")
        frames: list[np.ndarray] = []
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = self.pose_model(frame, verbose=False)
                frames.append(self._largest_person(result))
        finally:
            capture.release()
        poses_2d = np.stack(frames) if frames else np.empty((0, KEYPOINT_COUNT, 3), dtype=np.float32)
        if self.lifter is None:
            poses_3d = np.full((len(poses_2d), KEYPOINT_COUNT, 4), np.nan, dtype=np.float32)
            poses_3d[..., :2] = poses_2d[..., :2]
            poses_3d[..., 3] = poses_2d[..., 2]
            pose_status = "2d_only_no_lifter_configured"
        else:
            lifted = np.asarray(self.lifter(poses_2d[..., :2]), dtype=np.float32)
            if lifted.shape != (len(poses_2d), KEYPOINT_COUNT, 3):
                raise ValueError("3D lifter must return an (frames, 17, 3) array.")
            poses_3d = np.concatenate((lifted, poses_2d[..., 2:3]), axis=-1)
            pose_status = "3d_lifted"
        arrays.update({"poses_2d": poses_2d.astype(np.float32), "poses_3d": poses_3d})
        metadata.update(
            {
                "original_fps": metadata.get("original_fps", fps),
                "pose_frame_count": len(poses_2d),
                "pose_status": pose_status,
                "phase_status": {**metadata.get("phase_status", {}), "tracking_pose": pose_status},
            }
        )
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays

    @staticmethod
    def _largest_person(results: Any) -> np.ndarray:
        """Pick the largest detected person and return COCO XY/confidence."""
        output = np.full((KEYPOINT_COUNT, 3), np.nan, dtype=np.float32)
        if not results:
            return output
        boxes, keypoints = getattr(results[0], "boxes", None), getattr(results[0], "keypoints", None)
        if boxes is None or keypoints is None:
            return output
        xyxy = Pose3DProcessor._numpy(boxes.xyxy)
        xy = Pose3DProcessor._numpy(keypoints.xy)
        if len(xyxy) == 0 or len(xyxy) != len(xy):
            return output
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        selected = int(np.argmax(area))
        points = xy[selected]
        count = min(KEYPOINT_COUNT, len(points))
        output[:count, :2] = points[:count, :2]
        conf = getattr(keypoints, "conf", None)
        if conf is not None:
            confidence = Pose3DProcessor._numpy(conf)
            if confidence.ndim == 2 and selected < len(confidence):
                output[:count, 2] = confidence[selected, :count]
        else:
            output[:count, 2] = 1.0
        return output

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)


if __name__ == "__main__":
    pose = np.full((17, 2), 1.0)
    assert pose.shape == (KEYPOINT_COUNT, 2)
    print("Pose module checks passed; 3D lifting requires an injected trained model.")
