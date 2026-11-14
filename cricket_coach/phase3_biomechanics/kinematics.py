"""3D joint geometry and bowling technique screening measurements."""

from pathlib import Path
from typing import Any

import numpy as np

from cricket_coach.config import (
    ELBOW_EXTENSION_WARNING_DEG,
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
    """Compute 3D action metrics around the global peak wrist speed."""

    def analyze(
        self,
        poses: np.ndarray,
        events: dict[str, Any] | None = None,
        poses_2d: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Analyze landmarks in a ten-frame window centered on the global wrist-speed peak."""
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

        peak_frame, _ = self._peak_wrist(points)
        if peak_frame is not None:
            window_start, window_end = self._centered_window(peak_frame, len(points))
            analysis_frames = np.arange(window_start, window_end, dtype=np.int64)
            selection_method = "peak_wrist_speed"
        else:
            window_start = window_end = 0
            analysis_frames = np.empty(0, dtype=np.int64)
            selection_method = "no_valid_wrist_speed"

        bowling_side = (
            self._bowling_side(points, poses_2d, analysis_frames)
            if peak_frame is not None
            else None
        )
        front_side = 1 - bowling_side if bowling_side is not None else None
        selected_knee_angles = (
            knee_angles[analysis_frames, front_side]
            if len(analysis_frames) and front_side is not None
            else np.empty(0)
        )
        release_frames = (
            analysis_frames[analysis_frames <= peak_frame]
            if peak_frame is not None
            else np.empty(0, dtype=np.int64)
        )
        selected_elbow_angles = (
            elbow_angles[release_frames, bowling_side]
            if len(release_frames) and bowling_side is not None
            else np.empty(0)
        )
        selected_trunk_tilt = trunk_tilt[analysis_frames] if len(analysis_frames) else np.empty(0)
        extension = 180.0 - selected_elbow_angles
        knee_summary = self._summary(selected_knee_angles)
        elbow_summary = self._summary(selected_elbow_angles)
        trunk_summary = self._summary(selected_trunk_tilt)
        front_leg = ("left", "right")[front_side] if front_side is not None else None
        bowling_arm = ("left", "right")[bowling_side] if bowling_side is not None else None

        metrics = {
            "knee_angle_deg": knee_summary,
            "elbow_angle_deg": elbow_summary,
            "trunk_tilt_deg": trunk_summary,
            "front_knee_angle_deg": knee_summary,
            "bowling_elbow_angle_deg": elbow_summary,
            "bowling_elbow_extension_deg": self._summary(extension),
            "lateral_trunk_tilt_deg": trunk_summary,
            "front_leg": front_leg,
            "bowling_arm": bowling_arm,
            "peak_wrist_speed_frame": int(peak_frame) if peak_frame is not None else None,
            "analysis_window": {
                "start_frame": int(window_start),
                "end_frame_exclusive": int(window_end),
                "frame_count": int(len(analysis_frames)),
                "selection_method": selection_method,
            },
            "knee_brace_warning": self._flag_min(selected_knee_angles, KNEE_BRACE_MIN_ANGLE_DEG),
            "trunk_tilt_warning": self._flag_max(selected_trunk_tilt, TRUNK_LEAN_WARNING_DEG),
            "elbow_extension_screen": self._flag_max(extension, ELBOW_EXTENSION_WARNING_DEG),
            "interpretation_note": (
                "Angles use MediaPipe pose_world_landmarks in estimated meters. "
                "The ten-frame window is centered on the global valid wrist-speed peak; "
                "elbow extension is screened only through that release frame. Values above "
                "15 degrees are only a video screen, not an official legality measurement "
                "or medical assessment."
            ),
        }
        return metrics

    @staticmethod
    def _centered_window(center: int, frame_count: int) -> tuple[int, int]:
        if frame_count <= 0:
            return 0, 0
        window_size = min(KINEMATICS_PEAK_WINDOW_FRAMES, frame_count)
        start = max(0, center - window_size // 2)
        end = min(frame_count, center + window_size - window_size // 2)
        return start, end

    @staticmethod
    def _wrist_velocity(points_3d: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
        """Return valid 3D wrist speeds and their destination frame indices."""
        frame_count = len(points_3d)
        if frame_count < 2:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        wrist, hip = (
            (LEFT_WRIST, LEFT_HIP) if side == 0 else (RIGHT_WRIST, RIGHT_HIP)
        )
        wrist_xyz = points_3d[:, wrist]
        hip_xyz = points_3d[:, hip]
        valid = (
            np.isfinite(wrist_xyz).all(axis=1)
            & np.isfinite(hip_xyz).all(axis=1)
            & (wrist_xyz[:, 1] >= hip_xyz[:, 1])
        )
        velocity = np.linalg.norm(np.diff(wrist_xyz, axis=0), axis=1)
        valid_pairs = valid[:-1] & valid[1:] & np.isfinite(velocity)
        velocity[~valid_pairs] = np.nan
        return velocity, np.arange(1, frame_count, dtype=np.int64)

    @classmethod
    def _peak_wrist(cls, points_3d: np.ndarray) -> tuple[int | None, int | None]:
        """Find the global maximum valid 3D wrist speed over the complete clip."""
        best_speed = -np.inf
        best_frame = None
        best_side = None
        for side in range(2):
            speeds, frames = cls._wrist_velocity(points_3d, side)
            if not np.isfinite(speeds).any():
                continue
            local_index = int(np.nanargmax(speeds))
            speed = float(speeds[local_index])
            if speed > best_speed:
                best_speed = speed
                best_frame = int(frames[local_index])
                best_side = side
        return best_frame, best_side

    @classmethod
    def _bowling_side(
        cls,
        points_3d: np.ndarray,
        poses_2d: np.ndarray | None,
        frames: np.ndarray,
    ) -> int | None:
        """Compare each wrist's peak reach and speed in the selected delivery window."""
        if not len(frames):
            return None
        valid_2d = (
            poses_2d is not None
            and poses_2d.ndim == 3
            and poses_2d.shape[1] >= 17
            and poses_2d.shape[2] >= 2
        )
        reach: list[float] = []
        speeds_in_window: list[float] = []
        for side, wrist in enumerate((LEFT_WRIST, RIGHT_WRIST)):
            world_wrist = points_3d[frames, wrist]
            world_hip = points_3d[frames, LEFT_HIP if side == 0 else RIGHT_HIP]
            above_hip = (
                np.isfinite(world_wrist).all(axis=1)
                & np.isfinite(world_hip).all(axis=1)
                & (world_wrist[:, 1] >= world_hip[:, 1])
            )
            if valid_2d:
                image_wrist = poses_2d[frames, wrist]
                image_valid = np.isfinite(image_wrist[:, :2]).all(axis=1)
                if poses_2d.shape[2] >= 3:
                    confidence = image_wrist[:, 2]
                    image_valid &= (
                        np.isfinite(confidence)
                        & (confidence >= POSE_CONFIDENCE_THRESHOLD)
                    )
                valid_reach = above_hip & image_valid
                reach_value = (
                    -float(np.min(image_wrist[valid_reach, 1]))
                    if valid_reach.any()
                    else np.nan
                )
            else:
                reach_value = (
                    float(np.max(world_wrist[above_hip, 1]))
                    if above_hip.any()
                    else np.nan
                )
            reach.append(reach_value)

            velocity, velocity_frames = cls._wrist_velocity(points_3d, side)
            inside = np.isin(velocity_frames, frames) & np.isin(
                velocity_frames - 1, frames
            )
            valid_speeds = velocity[inside]
            speeds_in_window.append(
                float(np.nanmax(valid_speeds))
                if np.isfinite(valid_speeds).any()
                else np.nan
            )

        if not np.isfinite(reach).any() and not np.isfinite(speeds_in_window).any():
            return None
        reach_values = np.nan_to_num(reach, nan=-np.inf)
        speed_values = np.nan_to_num(speeds_in_window, nan=-np.inf)
        # Prefer the arm that wins both comparisons; use speed as the tie-breaker.
        wins = [
            int(reach_values[side] > reach_values[1 - side])
            + int(speed_values[side] > speed_values[1 - side])
            for side in range(2)
        ]
        return max(
            range(2),
            key=lambda side: (wins[side], speed_values[side], reach_values[side]),
        )

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
        collinear[:, hip, :3] = [0, 0, 0]
        collinear[:, knee, :3] = [0, -1, 0]
        collinear[:, ankle, :3] = [0, -2, 0]
        collinear[:, shoulder, :3] = [0, 1, 0]
        collinear[:, elbow, :3] = [0, 2, 0]
        collinear[:, wrist, :3] = [0, 3, 0]
    collinear[6, LEFT_WRIST, 0] = 1.0
    collinear[6, LEFT_WRIST, 3] = 1.0
    collinear[6, LEFT_SHOULDER, 3] = 1.0
    collinear_2d = np.zeros((12, 17, 3), dtype=np.float32)
    collinear_2d[:, LEFT_WRIST, 1] = 0
    collinear_2d[:, LEFT_SHOULDER, 1] = 10
    collinear_2d[..., 2] = 1.0
    result = KinematicsAnalyzer().analyze(collinear, poses_2d=collinear_2d)
    assert result["analysis_window"]["selection_method"] == "peak_wrist_speed"
    assert result["analysis_window"]["frame_count"] == KINEMATICS_PEAK_WINDOW_FRAMES
    assert result["front_knee_angle_deg"]["min"] is not None
    assert result["bowling_elbow_angle_deg"]["min"] is not None
    assert result["lateral_trunk_tilt_deg"]["max"] is not None
    reach_2d = np.zeros_like(collinear_2d)
    reach_2d[..., 2] = 1.0
    reach_2d[:, LEFT_WRIST, 1] = 5
    reach_2d[:, RIGHT_WRIST, 1] = 10
    assert KinematicsAnalyzer._bowling_side(
        collinear, reach_2d, np.arange(1, 5)
    ) == 0
    no_peak_poses = collinear.copy()
    no_peak_poses[:, LEFT_WRIST, 1] = -1
    no_peak_poses[:, RIGHT_WRIST, 1] = -1
    no_peak_result = KinematicsAnalyzer().analyze(no_peak_poses, poses_2d=collinear_2d)
    assert no_peak_result["peak_wrist_speed_frame"] is None
    assert no_peak_result["analysis_window"]["selection_method"] == "no_valid_wrist_speed"
    assert no_peak_result["analysis_window"]["frame_count"] == 0

    delivery = np.repeat(collinear[:1], 30, axis=0)
    delivery[2, LEFT_WRIST, :3] = [100, -5, 0]
    delivery[3, LEFT_WRIST, :3] = [-100, -5, 0]
    delivery[20, RIGHT_WRIST, 1] = 5
    delivery[21, RIGHT_WRIST, 1] = 3
    delivery[22:, RIGHT_WRIST, :3] = [1, 2, 0]
    delivery_result = KinematicsAnalyzer().analyze(
        delivery,
        events={
            "BFC": {"frame": 2, "confidence": 1.0, "exact": True},
            "FFC": {"frame": 3, "confidence": 1.0, "exact": True},
        },
    )
    assert delivery_result["peak_wrist_speed_frame"] == 20
    assert delivery_result["analysis_window"]["start_frame"] == 15
    assert delivery_result["analysis_window"]["end_frame_exclusive"] == 25
    assert delivery_result["bowling_arm"] == "right"
    assert delivery_result["front_leg"] == "left"
    assert delivery_result["bowling_elbow_extension_deg"]["max"] == 0.0
    assert delivery_result["elbow_extension_screen"] is False

    low_confidence = collinear.copy()
    low_confidence[:, LEFT_WRIST, 3] = 0.0
    assert np.isnan(KinematicsAnalyzer._joint_angle_series(low_confidence)[6, 2])
    print("Kinematics checks passed.")
