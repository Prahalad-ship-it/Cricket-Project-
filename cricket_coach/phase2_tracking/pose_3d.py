"""YOLO 2D keypoints and MediaPipe world-landmark 3D pose lifting."""

import logging
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

LOGGER = logging.getLogger(__name__)


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
        self._uses_mediapipe = lifter is None
        self.pose_error: str | None = None
        if lifter is None:
            try:
                self.lifter = MediaPipeWorldPoseLifter()
            except Exception as error:
                self.lifter = None
                self.pose_error = self._exception_text(error)
                LOGGER.exception(
                    "MediaPipe pose initialization failed; using YOLO 2D keypoints: %s",
                    self.pose_error,
                )
        else:
            self.lifter = lifter

    def process(self, video_path: str | Path, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Process all frames, persist pose arrays, and update the delivery contract."""
        metadata, arrays = load_phase_payload(metadata_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open delivery video: {video_path}")
        frames: list[np.ndarray] = []
        candidate_poses_by_frame: list[np.ndarray] = []
        candidate_boxes_by_frame: list[np.ndarray] = []
        candidate_counts: list[int] = []
        world_poses: list[np.ndarray] = []
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        pose_error = self.pose_error
        has_frame_lifter = callable(getattr(self.lifter, "lift_frame", None))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = self.pose_model(frame, verbose=False)
                candidate_poses, candidate_boxes = self._person_detections(result)
                candidate_poses_by_frame.append(candidate_poses)
                candidate_boxes_by_frame.append(candidate_boxes)
                candidate_counts.append(len(candidate_poses))
                frames.append(
                    candidate_poses[0]
                    if len(candidate_poses)
                    else np.full((KEYPOINT_COUNT, 3), np.nan, dtype=np.float32)
                )
                if has_frame_lifter and pose_error is None:
                    try:
                        world_pose = np.asarray(self.lifter.lift_frame(frame), dtype=np.float32)
                        if world_pose.shape != (KEYPOINT_COUNT, 4):
                            raise ValueError(
                                "MediaPipe lifter must return a (17, 4) landmark array."
                            )
                        world_poses.append(world_pose)
                    except Exception as error:
                        pose_error = self._exception_text(error)
                        world_poses.clear()
                        LOGGER.exception(
                            "MediaPipe pose inference failed; using YOLO 2D keypoints: %s",
                            pose_error,
                        )
        finally:
            capture.release()
        poses_2d = np.stack(frames) if frames else np.empty((0, KEYPOINT_COUNT, 3), dtype=np.float32)
        max_candidates = max(candidate_counts, default=0)
        poses_2d_candidates = np.full(
            (len(frames), max_candidates, KEYPOINT_COUNT, 3),
            np.nan,
            dtype=np.float32,
        )
        person_boxes_xyxy = np.full(
            (len(frames), max_candidates, 4),
            np.nan,
            dtype=np.float32,
        )
        for frame_index, (candidate_poses, candidate_boxes) in enumerate(
            zip(candidate_poses_by_frame, candidate_boxes_by_frame)
        ):
            count = len(candidate_poses)
            poses_2d_candidates[frame_index, :count] = candidate_poses
            person_boxes_xyxy[frame_index, :count] = candidate_boxes
        fallback_poses = self._fallback_poses_3d(poses_2d)
        fallback_frames = np.ones(len(poses_2d), dtype=bool)
        if not len(poses_2d):
            poses_3d = fallback_poses.copy()
            fallback_frames[:] = False
        elif pose_error is None and has_frame_lifter and len(world_poses) == len(poses_2d):
            poses_3d = np.stack(world_poses).astype(np.float32)
            valid_landmarks = np.isfinite(poses_3d[..., :3]).all(axis=-1).sum(axis=1)
            fallback_frames = valid_landmarks < 3
            poses_3d[fallback_frames] = fallback_poses[fallback_frames]
        elif pose_error is None and self.lifter is not None:
            try:
                lifted = np.asarray(self.lifter(poses_2d[..., :2]), dtype=np.float32)
                if lifted.shape != (len(poses_2d), KEYPOINT_COUNT, 3):
                    raise ValueError("3D lifter must return an (frames, 17, 3) array.")
                poses_3d = np.concatenate((lifted, poses_2d[..., 2:3]), axis=-1)
                valid_landmarks = np.isfinite(poses_3d[..., :3]).all(axis=-1).sum(axis=1)
                fallback_frames = valid_landmarks < 3
                poses_3d[fallback_frames] = fallback_poses[fallback_frames]
            except Exception as error:
                pose_error = self._exception_text(error)
                LOGGER.exception(
                    "3D pose lifting failed; using YOLO 2D keypoints: %s",
                    pose_error,
                )
                poses_3d = fallback_poses.copy()
        else:
            poses_3d = fallback_poses.copy()

        if pose_error is not None:
            pose_status = (
                "mediapipe_error_2d_fallback"
                if self._uses_mediapipe
                else "external_lifter_error_2d_fallback"
            )
            tracking_pose_error = pose_error
            fallback_frames[:] = True
        elif not len(poses_2d):
            pose_status = "no_frames"
            tracking_pose_error = "No frames were decoded from the delivery video."
        elif fallback_frames.any():
            pose_status = (
                "yolov8_2d_fallback"
                if fallback_frames.all()
                else "mediapipe_world_landmarks_with_2d_fallback"
                if self._uses_mediapipe
                else "external_3d_lifter_with_2d_fallback"
            )
            tracking_pose_error = (
                f"3D landmarks were unavailable in {int(fallback_frames.sum())} frame(s); "
                "used YOLO 2D keypoints for those frames."
            )
        else:
            pose_status = (
                "mediapipe_world_landmarks"
                if self._uses_mediapipe
                else "external_3d_lifter"
            )
            tracking_pose_error = None

        valid_frames = (
            int(
                (
                    np.isfinite(poses_3d[..., :3]).all(axis=-1).sum(axis=1) >= 3
                ).sum()
            )
            if len(poses_3d)
            else 0
        )
        phase_errors = metadata.get("phase_errors")
        phase_errors = dict(phase_errors) if isinstance(phase_errors, dict) else {}
        if tracking_pose_error:
            phase_errors["tracking_pose"] = tracking_pose_error
        else:
            phase_errors.pop("tracking_pose", None)
        if phase_errors:
            metadata["phase_errors"] = phase_errors
        else:
            metadata.pop("phase_errors", None)

        arrays.update(
            {
                "poses_2d": poses_2d.astype(np.float32),
                "poses_2d_candidates": poses_2d_candidates,
                "person_boxes_xyxy": person_boxes_xyxy,
                "person_candidate_counts": np.asarray(candidate_counts, dtype=np.int16),
                "poses_3d": poses_3d,
                "pose_3d_fallback_frames": fallback_frames,
            }
        )
        metadata.update(
            {
                "original_fps": metadata.get("original_fps", fps),
                "pose_frame_count": len(poses_2d),
                "pose_3d_valid_frame_count": valid_frames,
                "pose_3d_fallback_frame_count": int(fallback_frames.sum()),
                "pose_status": pose_status,
                "phase_status": {**metadata.get("phase_status", {}), "tracking_pose": pose_status},
            }
        )
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays

    @staticmethod
    def _fallback_poses_3d(poses_2d: np.ndarray) -> np.ndarray:
        """Represent YOLO image-space keypoints in the 3D contract for downstream use."""
        fallback = np.full((len(poses_2d), KEYPOINT_COUNT, 4), np.nan, dtype=np.float32)
        fallback[..., :2] = poses_2d[..., :2]
        fallback[..., 2] = 0.0
        if poses_2d.shape[-1] >= 3:
            fallback[..., 3] = poses_2d[..., 2]
        else:
            fallback[..., 3] = np.isfinite(poses_2d[..., :2]).all(axis=-1).astype(np.float32)
        return fallback

    @staticmethod
    def _exception_text(error: Exception) -> str:
        return f"{type(error).__name__}: {error}"

    def close(self) -> None:
        """Release resources owned by the configured pose lifter."""
        close = getattr(self.lifter, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _largest_person(results: Any) -> np.ndarray:
        """Pick the largest detected person and return COCO XY/confidence."""
        output = np.full((KEYPOINT_COUNT, 3), np.nan, dtype=np.float32)
        candidates, _ = Pose3DProcessor._person_detections(results)
        if len(candidates):
            output[:] = candidates[0]
        return output

    @staticmethod
    def _person_detections(results: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return detected people ordered by descending box area."""
        if not results:
            return (
                np.empty((0, KEYPOINT_COUNT, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )
        boxes, keypoints = getattr(results[0], "boxes", None), getattr(results[0], "keypoints", None)
        if boxes is None or keypoints is None:
            return (
                np.empty((0, KEYPOINT_COUNT, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )
        xyxy = Pose3DProcessor._numpy(boxes.xyxy)
        xy = Pose3DProcessor._numpy(keypoints.xy)
        if (
            xyxy.ndim != 2
            or xyxy.shape[1] < 4
            or xy.ndim != 3
            or len(xyxy) == 0
            or len(xyxy) != len(xy)
        ):
            return (
                np.empty((0, KEYPOINT_COUNT, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )
        boxes_xyxy = xyxy[:, :4].astype(np.float32, copy=False)
        area = np.maximum(0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * np.maximum(
            0,
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        )
        order = np.argsort(area)[::-1]
        conf = getattr(keypoints, "conf", None)
        confidence = Pose3DProcessor._numpy(conf) if conf is not None else None
        candidates = np.full(
            (len(order), KEYPOINT_COUNT, 3),
            np.nan,
            dtype=np.float32,
        )
        count = min(KEYPOINT_COUNT, xy.shape[1])
        candidates[:, :count, :2] = xy[order, :count, :2]
        if confidence is not None and confidence.ndim == 2 and len(confidence) == len(xy):
            confidence_count = min(count, confidence.shape[1])
            candidates[:, :confidence_count, 2] = confidence[
                order, :confidence_count
            ]
        else:
            candidates[:, :count, 2] = 1.0
        return candidates, boxes_xyxy[order]

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
