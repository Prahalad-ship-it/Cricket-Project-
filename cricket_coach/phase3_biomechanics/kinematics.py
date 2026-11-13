"""3D joint geometry and bowling technique screening measurements."""

from pathlib import Path
from typing import Any

import numpy as np

from cricket_coach.config import (
    ELBOW_EXTENSION_WARNING_DEG,
    EVENT_EXACT_CONFIDENCE_THRESHOLD,
    KINEMATICS_PEAK_WINDOW_FRAMES,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    POSE_CONFIDENCE_THRESHOLD,
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
    """Compute 3D action metrics around contact events or peak wrist speed."""

    def analyze(
        self,
        poses: np.ndarray,
        events: dict[str, Any] | None = None,
        poses_2d: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Analyze 3D landmarks, using a ten-frame peak-speed window without exact contacts."""
        if poses.ndim != 3 or poses.shape[1] < 17 or poses.shape[2] < 3:
            raise ValueError("Expected pose array shaped (frames, 17, at least 3).")
        points = poses[..., :3].astype(np.float32, copy=True)
        if poses.shape[2] >= 4:
            low_confidence = ~np.isfinite(poses[..., 3]) | (poses[..., 3] < POSE_CONFIDENCE_THRESHOLD)
            points[low_confidence] = np.nan
        joint_series = self._joint_angle_series(points)
        knee_angles = joint_series[:, :2]
        elbow_angles = joint_series[:, 2:4]
        trunk_tilt = joint_series[:, 4]

        peak_frame, bowling_side = self._peak_wrist(
            points,
            poses_2d,
        )
        event_map = self._event_map(events)
        exact_bfc = self._exact_event_frame(event_map.get("BFC"), len(points))
        exact_ffc = self._exact_event_frame(event_map.get("FFC"), len(points))
        exact_contacts_available = exact_bfc is not None and exact_ffc is not None
        if exact_contacts_available:
            analysis_frames = np.unique([exact_bfc, exact_ffc])
            window_start = int(analysis_frames.min())
            window_end = int(analysis_frames.max())
            selection_method = "exact_contact_frames"
        elif peak_frame is not None:
            window_start, window_end = self._centered_window(peak_frame, len(points))
            analysis_frames = np.arange(window_start, window_end, dtype=np.int64)
            selection_method = "peak_wrist_speed"
        else:
            visible_frame = self._best_pose_frame(joint_series)
            window_start, window_end = self._centered_window(visible_frame, len(points))
            analysis_frames = np.arange(window_start, window_end, dtype=np.int64)
            selection_method = "best_pose_visibility"

        side_selection_frame = (
            peak_frame
            if peak_frame is not None
            else (int(analysis_frames[len(analysis_frames) // 2]) if len(analysis_frames) else 0)
        )
        front_side = self._front_leg_side(event_map.get("FFC"), knee_angles, side_selection_frame)
        selected_knee_angles = knee_angles[analysis_frames, front_side] if len(analysis_frames) else np.empty(0)
        selected_elbow_angles = elbow_angles[analysis_frames, bowling_side] if len(analysis_frames) else np.empty(0)
        selected_trunk_tilt = trunk_tilt[analysis_frames] if len(analysis_frames) else np.empty(0)
        extension = 180.0 - selected_elbow_angles
        knee_summary = self._summary(selected_knee_angles)
        elbow_summary = self._summary(selected_elbow_angles)
        trunk_summary = self._summary(selected_trunk_tilt)

        metrics = {
            "knee_angle_deg": knee_summary,
            "elbow_angle_deg": elbow_summary,
            "trunk_tilt_deg": trunk_summary,
            "front_knee_angle_deg": knee_summary,
            "bowling_elbow_angle_deg": elbow_summary,
            "bowling_elbow_extension_deg": self._summary(extension),
            "lateral_trunk_tilt_deg": trunk_summary,
            "front_leg": "left" if front_side == 0 else "right",
            "bowling_arm": "left" if bowling_side == 0 else "right",
            "peak_wrist_speed_frame": int(peak_frame) if peak_frame is not None else None,
            "analysis_window": {
                "start_frame": int(window_start),
                "end_frame_exclusive": int(window_end),
                "frame_count": int(len(analysis_frames)),
                "selection_method": selection_method,
                "used_peak_wrist_fallback": selection_method == "peak_wrist_speed",
            },
            "knee_brace_warning": self._flag_min(selected_knee_angles, KNEE_BRACE_MIN_ANGLE_DEG),
            "trunk_tilt_warning": self._flag_max(selected_trunk_tilt, TRUNK_LEAN_WARNING_DEG),
            "elbow_extension_screen": self._flag_elbow(selected_elbow_angles),
            "interpretation_note": (
                "Angles use MediaPipe pose_world_landmarks in estimated meters. "
                "The ten-frame peak-wrist window is a fallback when exact foot-contact frames "
                "are unavailable. Elbow extension above 15 degrees is only a video screen; "
                "it is not an official legality measurement or medical assessment."
            ),
        }
        return metrics

    @staticmethod
    def _event_map(events: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(events, dict):
            return {}
        nested = events.get("events")
        return nested if isinstance(nested, dict) else events

    @staticmethod
    def _exact_event_frame(event: Any, frame_count: int) -> int | None:
        if not isinstance(event, dict):
            return None
        confidence = event.get("confidence")
        is_exact = event.get("exact") is True or event.get("is_exact") is True
        if not is_exact:
            return None
        if isinstance(confidence, (int, float)) and confidence < EVENT_EXACT_CONFIDENCE_THRESHOLD:
            return None
        frame = event.get("frame")
        if isinstance(frame, int) and 0 <= frame < frame_count:
            return frame
        return None

    @staticmethod
    def _centered_window(center: int, frame_count: int) -> tuple[int, int]:
        if frame_count <= 0:
            return 0, 0
        window_size = min(KINEMATICS_PEAK_WINDOW_FRAMES, frame_count)
        start = max(0, center - window_size // 2)
        start = min(start, frame_count - window_size)
        return start, start + window_size

    @staticmethod
    def _best_pose_frame(joint_series: np.ndarray) -> int:
        """Choose the frame with the most valid angle estimates if no wrist peak exists."""
        if not len(joint_series):
            return 0
        return int(np.argmax(np.isfinite(joint_series).sum(axis=1)))

    @staticmethod
    def _peak_wrist(points_3d: np.ndarray, poses_2d: np.ndarray | None) -> tuple[int | None, int]:
        """Find the fastest wrist while it is above its same-side shoulder."""
        frame_count = len(points_3d)
        if frame_count < 2:
            return 0, 0
        if poses_2d is not None and poses_2d.ndim == 3 and poses_2d.shape[1] >= 17 and poses_2d.shape[2] >= 2:
            source = poses_2d[..., :2]
            use_image_gate = True
        else:
            source = points_3d
            use_image_gate = False

        best_speed = -1.0
        best_frame = 0
        best_side = 0
        for side, (wrist, shoulder) in enumerate(
            ((LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER))
        ):
            wrist_positions = source[:, wrist]
            valid = np.isfinite(wrist_positions).all(axis=1)
            if use_image_gate:
                shoulder_positions = source[:, shoulder]
                valid &= np.isfinite(shoulder_positions).all(axis=1)
                valid &= wrist_positions[:, 1] < shoulder_positions[:, 1]
                if poses_2d.shape[2] >= 3:
                    confidence = poses_2d[..., 2]
                    valid &= (
                        np.isfinite(confidence[:, wrist])
                        & np.isfinite(confidence[:, shoulder])
                        & (confidence[:, wrist] >= POSE_CONFIDENCE_THRESHOLD)
                        & (confidence[:, shoulder] >= POSE_CONFIDENCE_THRESHOLD)
                    )
            differences = np.linalg.norm(np.diff(wrist_positions, axis=0), axis=1)
            valid_pairs = valid[1:] & np.isfinite(wrist_positions[:-1]).all(axis=1)
            differences[~valid_pairs] = np.nan
            if not np.isfinite(differences).any():
                continue
            local_frame = int(np.nanargmax(differences)) + 1
            local_speed = float(differences[local_frame - 1])
            if local_speed > best_speed:
                best_speed = local_speed
                best_frame = local_frame
                best_side = side
        return (best_frame, best_side) if best_speed >= 0 else (None, 0)

    @staticmethod
    def _front_leg_side(ffc: Any, knee_angles: np.ndarray, peak_frame: int) -> int:
        """Use a detected FFC foot when available; otherwise infer the braced leg."""
        if isinstance(ffc, dict):
            side = ffc.get("front_leg") or ffc.get("lead_foot")
            if side in ("left", "right"):
                return 0 if side == "left" else 1
            foot_index = ffc.get("lead_foot_index")
            if foot_index in (0, 1):
                return int(foot_index)
        if len(knee_angles):
            angles = knee_angles[min(max(peak_frame, 0), len(knee_angles) - 1)]
            valid = np.isfinite(angles)
            if valid.any():
                return int(np.nanargmax(angles))
        return 0

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
        metadata["biomechanics"] = self.analyze(
            arrays["poses_3d"],
            events=metadata.get("event_detection", metadata.get("events")),
            poses_2d=arrays.get("poses_2d"),
        )
        metadata["joint_angle_series_columns"] = [
            "left_knee_included_angle_deg",
            "right_knee_included_angle_deg",
            "left_elbow_included_angle_deg",
            "right_elbow_included_angle_deg",
            "lateral_trunk_tilt_deg",
        ]
        arrays["joint_angles_deg"] = self._joint_angle_series(arrays["poses_3d"])
        metadata["phase_status"] = {**metadata.get("phase_status", {}), "biomechanics": "complete"}
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays

    @classmethod
    def _joint_angle_series(cls, points: np.ndarray) -> np.ndarray:
        """Return framewise left/right knee and elbow angles plus lateral trunk tilt."""
        source = np.asarray(points)
        points = source[..., :3].astype(np.float32, copy=True)
        if source.shape[-1] >= 4:
            low_confidence = ~np.isfinite(source[..., 3]) | (source[..., 3] < POSE_CONFIDENCE_THRESHOLD)
            points[low_confidence] = np.nan
        knees = np.column_stack(
            (
                cls._angle(points[:, LEFT_HIP], points[:, LEFT_KNEE], points[:, LEFT_ANKLE]),
                cls._angle(points[:, RIGHT_HIP], points[:, RIGHT_KNEE], points[:, RIGHT_ANKLE]),
            )
        )
        elbows = np.column_stack(
            (
                cls._angle(points[:, LEFT_SHOULDER], points[:, LEFT_ELBOW], points[:, LEFT_WRIST]),
                cls._angle(points[:, RIGHT_SHOULDER], points[:, RIGHT_ELBOW], points[:, RIGHT_WRIST]),
            )
        )
        hips_center = (points[:, LEFT_HIP] + points[:, RIGHT_HIP]) / 2
        shoulders_center = (points[:, LEFT_SHOULDER] + points[:, RIGHT_SHOULDER]) / 2
        trunk_vector = shoulders_center - hips_center
        trunk_tilt = np.degrees(
            np.arctan2(np.abs(trunk_vector[:, 0]), np.abs(trunk_vector[:, 1]))
        )
        return np.column_stack((knees, elbows, trunk_tilt)).astype(np.float32)


if __name__ == "__main__":
    collinear = np.zeros((12, 17, 4), dtype=np.float32)
    collinear[:, :, 3] = 1.0
    for side, hip, knee, ankle, shoulder, elbow, wrist in (
        (0, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        (1, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ):
        del side
        collinear[:, hip, :3] = [0, 2, 0]
        collinear[:, knee, :3] = [0, 1, 0]
        collinear[:, ankle, :3] = [0, 0, 0]
        collinear[:, shoulder, :3] = [0, 3, 0]
        collinear[:, elbow, :3] = [0, 2, 0]
        collinear[:, wrist, :3] = [0, 1, 0]
    collinear[6, LEFT_WRIST, 0] = 1.0
    collinear[6, LEFT_WRIST, 3] = 1.0
    collinear[6, LEFT_SHOULDER, 3] = 1.0
    collinear_2d = np.zeros((12, 17, 3), dtype=np.float32)
    collinear_2d[:, LEFT_WRIST, 1] = 0
    collinear_2d[:, LEFT_SHOULDER, 1] = 10
    collinear_2d[..., 2] = 1.0
    result = KinematicsAnalyzer().analyze(collinear, poses_2d=collinear_2d)
    assert result["analysis_window"]["used_peak_wrist_fallback"]
    assert result["analysis_window"]["frame_count"] == KINEMATICS_PEAK_WINDOW_FRAMES
    assert result["front_knee_angle_deg"]["min"] is not None
    assert result["bowling_elbow_angle_deg"]["min"] is not None
    assert result["lateral_trunk_tilt_deg"]["max"] is not None
    no_overhead = collinear_2d.copy()
    no_overhead[:, LEFT_WRIST, 1] = 20
    no_overhead[:, LEFT_SHOULDER, 1] = 10
    no_peak_result = KinematicsAnalyzer().analyze(collinear, poses_2d=no_overhead)
    assert no_peak_result["peak_wrist_speed_frame"] is None
    assert no_peak_result["analysis_window"]["selection_method"] == "best_pose_visibility"
    proxy_event = {"frame": 6, "confidence": 1.0}
    assert KinematicsAnalyzer._exact_event_frame(proxy_event, len(collinear)) is None
    exact_event = {"frame": 6, "confidence": 1.0, "exact": True}
    assert KinematicsAnalyzer._exact_event_frame(exact_event, len(collinear)) == 6
    assert KinematicsAnalyzer._exact_event_frame(
        {"frame": 6, "confidence": 0.1, "exact": True}, len(collinear)
    ) is None
    low_confidence = collinear.copy()
    low_confidence[:, LEFT_WRIST, 3] = 0.0
    assert np.isnan(KinematicsAnalyzer._joint_angle_series(low_confidence)[6, 2])
    print("Kinematics checks passed.")
