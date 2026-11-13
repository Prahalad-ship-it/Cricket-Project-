"""Rolling-buffer delivery slicer driven by pose-based overarm motion."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import find_peaks

from cricket_coach.config import (
    COOLDOWN_SEC,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    MODEL_POSE_PATH,
    OUTPUT_HEIGHT_PX,
    OUTPUT_WIDTH_PX,
    POSE_CONFIDENCE_THRESHOLD,
    POST_TRIGGER_SEC,
    PRE_TRIGGER_SEC,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SHORT_CLIP_FALLBACK_SEC,
)
from cricket_coach.data_contract import save_phase_payload


class DeliverySlicer:
    """Find normalized wrist-speed peaks and extract one clip per delivery."""

    def __init__(self, model: Any = None, model_path: str = MODEL_POSE_PATH) -> None:
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise ImportError("Install ultralytics to run pose-based delivery slicing.") from error
            model = YOLO(model_path)
        self.model = model

    def slice_video(self, video_path: str | Path, output_dir: str | Path, calibration_matrix_path: str | Path) -> list[Path]:
        """Analyze pose motion, then write peak or short-clip fallback deliveries."""
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        fps, frame_count, velocities, detected = self._analyze_video(video_path)
        if frame_count == 0:
            return []

        peaks = self._find_release_peaks(velocities, fps)
        detected_indices = np.flatnonzero(detected)
        short_clip = frame_count / fps < SHORT_CLIP_FALLBACK_SEC
        if len(detected_indices) and (short_clip or not len(peaks)):
            start = int(detected_indices[0])
            end = int(detected_indices[-1])
            method = "short_clip_fallback" if short_clip else "no_release_peak_fallback"
            windows = [(start, end, start, method)]
        else:
            pre_count = max(0, int(round(fps * PRE_TRIGGER_SEC)))
            post_count = max(1, int(round(fps * POST_TRIGGER_SEC)))
            windows = [
                (max(0, int(peak) - pre_count), min(frame_count - 1, int(peak) + post_count), int(peak), "adaptive_velocity_peak")
                for peak in peaks
            ]
        return self._write_windows(
            video_path,
            destination,
            calibration_matrix_path,
            fps,
            velocities,
            windows,
        )

    def _analyze_video(self, video_path: str | Path) -> tuple[float, int, np.ndarray, np.ndarray]:
        """Run pose inference once and collect normalized wrist-speed signals."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError(f"Invalid source FPS: {fps}")
        velocities: list[float] = []
        detected: list[bool] = []
        previous_pose: np.ndarray | None = None
        frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                pose = self._active_pose(frame)
                detected.append(pose is not None)
                velocities.append(
                    self._normalized_wrist_velocity(pose, previous_pose, fps)
                    if pose is not None and previous_pose is not None
                    else 0.0
                )
                previous_pose = pose
                frame_count += 1
        finally:
            capture.release()
        return fps, frame_count, np.asarray(velocities, dtype=np.float64), np.asarray(detected, dtype=bool)

    @staticmethod
    def _find_release_peaks(velocities: np.ndarray, fps: float) -> np.ndarray:
        """Find prominent velocity peaks, enforcing the configured cooldown."""
        if not len(velocities):
            return np.empty(0, dtype=np.int64)
        prominence = float(np.std(velocities))
        if not np.isfinite(prominence) or prominence <= np.finfo(np.float64).eps:
            return np.empty(0, dtype=np.int64)
        minimum_distance = max(1, int(fps * COOLDOWN_SEC))
        peaks, _ = find_peaks(velocities, distance=minimum_distance, prominence=prominence)
        return peaks

    def _active_pose(self, frame: np.ndarray) -> np.ndarray | None:
        """Return the largest-person pose with low-confidence keypoints masked."""
        results = self.model(frame, verbose=False)
        if not results:
            return None
        boxes, keypoints = getattr(results[0], "boxes", None), getattr(results[0], "keypoints", None)
        if boxes is None or keypoints is None:
            return None
        xyxy, pose = self._numpy(boxes.xyxy), self._numpy(keypoints.xy)
        if xyxy.ndim != 2 or pose.ndim != 3 or len(xyxy) == 0 or len(xyxy) != len(pose):
            return None
        areas = np.maximum(0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
        selected = pose[int(np.argmax(areas))].astype(np.float32, copy=True)
        confidence_value = getattr(keypoints, "conf", None)
        if confidence_value is not None:
            confidences = self._numpy(confidence_value)
            if confidences.ndim == 2 and len(confidences) == len(pose):
                selected[confidences[int(np.argmax(areas))] < POSE_CONFIDENCE_THRESHOLD] = np.nan
        return selected

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _normalized_wrist_velocity(pose: np.ndarray, previous: np.ndarray, fps: float) -> float:
        """Measure the fastest eligible wrist in torso lengths per second."""
        shoulder_points = pose[[LEFT_SHOULDER, RIGHT_SHOULDER], :2]
        hip_points = pose[[LEFT_HIP, RIGHT_HIP], :2]
        if not np.isfinite(shoulder_points).all() or not np.isfinite(hip_points).all():
            return 0.0
        shoulder_center = shoulder_points.mean(axis=0)
        hip_center = hip_points.mean(axis=0)
        torso_height = float(np.linalg.norm(shoulder_center - hip_center))
        if not np.isfinite(torso_height) or torso_height <= np.finfo(np.float32).eps:
            return 0.0
        velocities: list[float] = []
        for wrist, shoulder in ((LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)):
            points = pose[[wrist, shoulder], :2]
            old_wrist = previous[wrist, :2]
            if not np.isfinite(points).all() or not np.isfinite(old_wrist).all():
                continue
            if points[0, 1] < points[1, 1]:
                velocities.append(float(np.linalg.norm(points[0] - old_wrist) / torso_height * fps))
        return max(velocities, default=0.0)

    def _write_windows(
        self,
        video_path: str | Path,
        output_dir: Path,
        calibration_matrix_path: str | Path,
        fps: float,
        velocities: np.ndarray,
        windows: list[tuple[int, int, int, str]],
    ) -> list[Path]:
        """Decode only selected clip windows on a second pass and save each delivery."""
        if not windows:
            return []
        ordered = sorted(windows, key=lambda window: window[0])
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to reopen video: {video_path}")
        frames_by_window: list[list[np.ndarray]] = [[] for _ in ordered]
        outputs: list[Path] = []
        index = 0
        next_window = 0
        active_windows: list[int] = []
        try:
            while next_window < len(ordered) or active_windows:
                ok, frame = capture.read()
                if not ok:
                    break
                while next_window < len(ordered) and ordered[next_window][0] <= index:
                    active_windows.append(next_window)
                    next_window += 1
                for window_index in active_windows:
                    start, end, _, _ = ordered[window_index]
                    if start <= index <= end:
                        frames_by_window[window_index].append(frame.copy())
                finished = []
                for window_index in active_windows:
                    start, end, trigger, method = ordered[window_index]
                    if index >= end:
                        window_frames = frames_by_window[window_index]
                        if window_frames:
                            outputs.append(
                                self._write_delivery(
                                    window_frames,
                                    output_dir,
                                    len(outputs) + 1,
                                    fps,
                                    start,
                                    trigger,
                                    calibration_matrix_path,
                                    velocities[start : end + 1],
                                    method,
                                )
                            )
                        finished.append(window_index)
                active_windows = [window_index for window_index in active_windows if window_index not in finished]
                index += 1
        finally:
            capture.release()
        return outputs

    @staticmethod
    def _write_delivery(
        frames: list[np.ndarray],
        output_dir: Path,
        number: int,
        fps: float,
        start: int,
        trigger: int,
        matrix_path: str | Path,
        normalized_velocity: np.ndarray,
        trigger_method: str,
    ) -> Path:
        """Write resized MP4 plus standard JSON and per-frame NumPy payload."""
        matrix_file = Path(matrix_path)
        calibration_status = "unavailable"
        if matrix_file.is_file():
            from cricket_coach.phase1_capture.calibrator import PitchCalibrator

            calibration_matrix = PitchCalibrator.load_homography(matrix_file)
            calibration_status = (
                "identity_fallback"
                if PitchCalibrator.is_identity_fallback(calibration_matrix)
                else "calibrated"
            )
        stem = f"delivery_{number:03d}"
        clip_path = output_dir / f"{stem}.mp4"
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUTPUT_WIDTH_PX, OUTPUT_HEIGHT_PX))
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Unable to create video: {clip_path}")
        try:
            for frame in frames:
                writer.write(DeliverySlicer._resize_letterboxed(frame))
        finally:
            writer.release()
        payload = {
            "schema_version": 1,
            "delivery_id": stem,
            "video_path": str(clip_path.resolve()),
            "original_fps": fps,
            "start_frame": start,
            "trigger_frame": trigger,
            "frame_count": len(frames),
            "trigger_method": trigger_method,
            "calibration_matrix_path": str(Path(matrix_path).resolve()),
            "calibration_status": calibration_status,
            "phase_status": {"capture": "complete"},
        }
        save_phase_payload(
            output_dir / f"{stem}_data.json",
            payload,
            {
                "source_frame_indices": np.arange(start, start + len(frames), dtype=np.int64),
                "normalized_wrist_velocity_torso_lengths_per_second": normalized_velocity[: len(frames)],
            },
        )
        return clip_path

    @staticmethod
    def _resize_letterboxed(frame: np.ndarray) -> np.ndarray:
        """Fit a frame to output dimensions without changing its aspect ratio."""
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Frame dimensions must be positive.")
        scale = min(OUTPUT_WIDTH_PX / width, OUTPUT_HEIGHT_PX / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        output = np.zeros((OUTPUT_HEIGHT_PX, OUTPUT_WIDTH_PX, frame.shape[2]), dtype=frame.dtype)
        offset_x = (OUTPUT_WIDTH_PX - resized_width) // 2
        offset_y = (OUTPUT_HEIGHT_PX - resized_height) // 2
        output[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
        return output


if __name__ == "__main__":
    pose = np.zeros((17, 3), dtype=np.float32)
    previous = pose.copy()
    pose[LEFT_SHOULDER, :2] = [50, 80]
    pose[RIGHT_SHOULDER, :2] = [70, 80]
    pose[LEFT_HIP, :2] = [50, 180]
    pose[RIGHT_HIP, :2] = [70, 180]
    pose[LEFT_WRIST, :2] = [70, 20]
    previous[LEFT_WRIST, :2] = [50, 20]
    assert DeliverySlicer._normalized_wrist_velocity(pose, previous, 30) > 0
    signal = np.zeros(100, dtype=np.float64)
    signal[[20, 60]] = [3.0, 5.0]
    peaks = DeliverySlicer._find_release_peaks(signal, 30)
    assert peaks.tolist() == [20, 60]
    assert not DeliverySlicer._find_release_peaks(np.zeros(20), 30).size
    rate_signal = np.zeros(400, dtype=np.float64)
    rate_signal[[50, 100, 300]] = [4.0, 5.0, 8.0]
    assert DeliverySlicer._find_release_peaks(rate_signal, 30).tolist() == [50, 100, 300]
    assert DeliverySlicer._find_release_peaks(rate_signal, 120).tolist() == [100, 300]
    assert np.isclose(
        DeliverySlicer._normalized_wrist_velocity(pose * 2, previous * 2, 30),
        DeliverySlicer._normalized_wrist_velocity(pose, previous, 30),
    )
    portrait = np.ones((720, 480, 3), dtype=np.uint8)
    letterboxed = DeliverySlicer._resize_letterboxed(portrait)
    assert letterboxed.shape == (OUTPUT_HEIGHT_PX, OUTPUT_WIDTH_PX, 3)
    assert not letterboxed[:, 0, :].any()
    print("Slicer trigger checks passed.")
