"""YOLO 2D keypoints and MediaPipe world-landmark 3D pose lifting."""

from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from cricket_coach.config import (
    KEYPOINT_COUNT,
    MEDIAPIPE_COCO_LANDMARK_INDICES,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_MODEL_COMPLEXITY,
    MODEL_POSE_PATH,
)
from cricket_coach.data_contract import load_phase_payload, save_phase_payload


class FramePoseLifter(Protocol):
    """Interface for a stateful image-to-world-pose model."""

    def lift_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return COCO-ordered world landmarks as x, y, z, confidence."""

    def close(self) -> None:
        """Release model resources."""


class MediaPipeWorldPoseLifter:
    """Run MediaPipe Pose tracking and map its world landmarks to COCO order."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise ImportError("Install the supported MediaPipe version to lift 2D poses to 3D.") from error
        pose_api = getattr(getattr(mp, "solutions", None), "pose", None)
        if pose_api is None:
            raise ImportError(
                "The installed MediaPipe package does not expose mediapipe.solutions.pose. "
                "Install mediapipe>=0.10.14,<0.10.22."
            )
        self._pose = pose_api.Pose(
            static_image_mode=False,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )

    def lift_frame(self, frame: np.ndarray) -> np.ndarray:
        """Infer pose_world_landmarks for a BGR frame."""
        output = np.full((KEYPOINT_COUNT, 4), np.nan, dtype=np.float32)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        landmarks = getattr(result, "pose_world_landmarks", None)
        if landmarks is None:
            return output
        world_landmarks = landmarks.landmark
        if len(world_landmarks) <= max(MEDIAPIPE_COCO_LANDMARK_INDICES):
            return output
        for coco_index, mediapipe_index in enumerate(MEDIAPIPE_COCO_LANDMARK_INDICES):
            landmark = world_landmarks[mediapipe_index]
            visibility = float(getattr(landmark, "visibility", 1.0))
            output[coco_index] = (
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
                visibility,
            )
        return output

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._pose.close()


class Pose3DProcessor:
    """Extract 2D poses and lift them to world coordinates by default."""

    def __init__(
        self,
        pose_model: Any = None,
        lifter: Callable[[np.ndarray], np.ndarray] | FramePoseLifter | None = None,
        model_path: str = MODEL_POSE_PATH,
    ) -> None:
        if pose_model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise ImportError("Install ultralytics to run pose extraction.") from error
            pose_model = YOLO(model_path)
        self.pose_model = pose_model
        self.lifter = lifter if lifter is not None else MediaPipeWorldPoseLifter()

    def process(self, video_path: str | Path, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Process all frames, persist pose arrays, and update the delivery contract."""
        metadata, arrays = load_phase_payload(metadata_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open delivery video: {video_path}")
        frames: list[np.ndarray] = []
        world_poses: list[np.ndarray] = []
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = self.pose_model(frame, verbose=False)
                frames.append(self._largest_person(result))
                if hasattr(self.lifter, "lift_frame"):
                    world_poses.append(self.lifter.lift_frame(frame))
        finally:
            capture.release()
        poses_2d = np.stack(frames) if frames else np.empty((0, KEYPOINT_COUNT, 3), dtype=np.float32)
        if hasattr(self.lifter, "lift_frame"):
            poses_3d = (
                np.stack(world_poses).astype(np.float32)
                if world_poses
                else np.empty((0, KEYPOINT_COUNT, 4), dtype=np.float32)
            )
            has_world_landmarks = bool(
                len(poses_3d) and np.isfinite(poses_3d[..., :3]).all(axis=-1).any()
            )
            pose_status = (
                "mediapipe_world_landmarks"
                if has_world_landmarks
                else "mediapipe_no_pose_detected"
            )
        else:
            lifted = np.asarray(self.lifter(poses_2d[..., :2]), dtype=np.float32)
            if lifted.shape != (len(poses_2d), KEYPOINT_COUNT, 3):
                raise ValueError("3D lifter must return an (frames, 17, 3) array.")
            poses_3d = np.concatenate((lifted, poses_2d[..., 2:3]), axis=-1)
            pose_status = "external_3d_lifter"
        valid_world_frames = (
            int(np.isfinite(poses_3d[..., :3]).all(axis=-1).any(axis=1).sum())
            if len(poses_3d)
            else 0
        )
        arrays.update({"poses_2d": poses_2d.astype(np.float32), "poses_3d": poses_3d})
        metadata.update(
            {
                "original_fps": metadata.get("original_fps", fps),
                "pose_frame_count": len(poses_2d),
                "pose_3d_valid_frame_count": valid_world_frames,
                "pose_status": pose_status,
                "phase_status": {**metadata.get("phase_status", {}), "tracking_pose": pose_status},
            }
        )
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays

    def close(self) -> None:
        """Release resources owned by the configured pose lifter."""
        close = getattr(self.lifter, "close", None)
        if callable(close):
            close()

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
    assert len(MEDIAPIPE_COCO_LANDMARK_INDICES) == KEYPOINT_COUNT
    assert len(set(MEDIAPIPE_COCO_LANDMARK_INDICES)) == KEYPOINT_COUNT
    print("Pose mapping checks passed.")
