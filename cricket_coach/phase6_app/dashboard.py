"""Streamlit dashboard for previously analyzed cricket delivery clips."""

from pathlib import Path
from typing import Any

import numpy as np

from cricket_coach.config import POSE_CONFIDENCE_THRESHOLD
from cricket_coach.data_contract import load_phase_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DELIVERY_DIR = PROJECT_ROOT / "data" / "sliced_clips" / "deliveries"
SKELETON_BONES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def discover_deliveries(delivery_dir: str | Path = DEFAULT_DELIVERY_DIR) -> list[Path]:
    """Find delivery metadata files, preferring delivery_001 for first launch."""
    root = Path(delivery_dir)
    if not root.is_dir():
        return []
    files = sorted(root.glob("delivery_*_data.json"))
    return sorted(files, key=lambda path: (path.name != "delivery_001_data.json", path.name))


def resolve_delivery_video(metadata_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Resolve the clip beside its JSON, with metadata path as a fallback."""
    source = Path(metadata_path)
    delivery_id = source.name.removesuffix("_data.json")
    adjacent_video = source.with_name(f"{delivery_id}.mp4")
    if adjacent_video.is_file():
        return adjacent_video
    metadata_video = metadata.get("video_path")
    if metadata_video:
        candidate = Path(metadata_video)
        if candidate.is_file():
            return candidate
    return adjacent_video


def _format_metric(value: Any, suffix: str = "") -> str:
    """Format finite numeric values for metric cards, marking unavailable values."""
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        return f"{float(value):.1f}{suffix}"
    return "N/A"


def _render_skeleton(st: Any, arrays: dict[str, np.ndarray], selected_frame: int = 0) -> None:
    """Render the best available 3D pose, falling back to a clearly labeled 2D view."""
    poses_3d = arrays.get("poses_3d")
    if (
        poses_3d is not None
        and poses_3d.ndim == 3
        and len(poses_3d)
        and poses_3d.shape[1] >= 17
        and poses_3d.shape[2] >= 3
    ):
        pose_points = poses_3d[..., :3].copy()
        if poses_3d.shape[2] >= 4:
            low_confidence = (
                ~np.isfinite(poses_3d[..., 3])
                | (poses_3d[..., 3] < POSE_CONFIDENCE_THRESHOLD)
            )
            pose_points[low_confidence] = np.nan
        available = np.isfinite(pose_points).all(axis=2).sum(axis=1)
        frame_index = min(max(selected_frame, 0), len(available) - 1) if len(available) else 0
        if len(available) and int(available[frame_index]) < 3:
            frame_index = int(np.argmax(available))
        frame = pose_points[frame_index]
        if int(available[frame_index]) >= 3:
            _render_3d_pose(st, frame, frame_index, len(poses_3d))
            return

    poses_2d = arrays.get("poses_2d")
    if poses_2d is None or poses_2d.ndim != 3 or poses_2d.shape[1] < 17 or poses_2d.shape[2] < 2:
        st.info("No pose landmarks are available for a skeleton view.")
        return
    pose_points_2d = poses_2d[..., :2].copy()
    if poses_2d.shape[2] >= 3:
        low_confidence = (
            ~np.isfinite(poses_2d[..., 2])
            | (poses_2d[..., 2] < POSE_CONFIDENCE_THRESHOLD)
        )
        pose_points_2d[low_confidence] = np.nan
    available = np.isfinite(pose_points_2d).all(axis=-1).sum(axis=1)
    if not len(available) or int(np.max(available)) < 3:
        st.info("Pose landmarks are unavailable for a skeleton preview.")
        return
    frame_index = min(max(selected_frame, 0), len(available) - 1)
    if int(available[frame_index]) < 3:
        frame_index = int(np.argmax(available))
    _render_2d_pose(st, pose_points_2d[frame_index], frame_index, len(poses_2d))
    st.caption(
        "Showing a 2D pose preview because 3D landmarks were unavailable for this delivery."
    )


def _render_3d_pose(st: Any, frame: np.ndarray, frame_index: int, frame_count: int) -> None:
    """Draw available 3D landmarks and bones in an interactive Plotly card."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install Plotly to view the 3D skeleton.")
        return

    figure = go.Figure()
    for first, second in SKELETON_BONES:
        if np.isfinite(frame[[first, second]]).all():
            figure.add_trace(
                go.Scatter3d(
                    x=frame[[first, second], 0].tolist(),
                    y=frame[[first, second], 1].tolist(),
                    z=frame[[first, second], 2].tolist(),
                    mode="lines",
                    line={"width": 6},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    valid = np.isfinite(frame).all(axis=1)
    points = frame[valid]
    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0].tolist(),
            y=points[:, 1].tolist(),
            z=points[:, 2].tolist(),
            mode="markers",
            marker={"size": 4},
            text=np.flatnonzero(valid),
            name="Joints",
        )
    )
    figure.update_layout(
        title=f"3D pose — frame {frame_index + 1} of {frame_count}",
        height=520,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene={"aspectmode": "data", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
    )
    st.plotly_chart(figure, width="stretch")


def _render_2d_pose(st: Any, frame: np.ndarray, frame_index: int, frame_count: int) -> None:
    """Draw an image-coordinate skeleton when depth is not available."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install Plotly to view the 2D pose preview.")
        return

    figure = go.Figure()
    for first, second in SKELETON_BONES:
        if np.isfinite(frame[[first, second]]).all():
            figure.add_trace(
                go.Scatter(
                    x=frame[[first, second], 0].tolist(),
                    y=frame[[first, second], 1].tolist(),
                    mode="lines",
                    line={"width": 3},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    valid = np.isfinite(frame).all(axis=1)
    points = frame[valid]
    figure.add_trace(
        go.Scatter(
            x=points[:, 0].tolist(),
            y=points[:, 1].tolist(),
            mode="markers",
            marker={"size": 7},
            text=np.flatnonzero(valid),
            name="Joints",
        )
    )
    figure.update_layout(
        title=f"2D pose — frame {frame_index + 1} of {frame_count}",
        height=480,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        yaxis={"autorange": "reversed", "scaleanchor": "x", "scaleratio": 1},
        xaxis_title="Image X (pixels)",
        yaxis_title="Image Y (pixels)",
    )
    st.plotly_chart(figure, width="stretch")


def _render_dashboard(st: Any, metadata_path: Path) -> None:
    """Load one delivery contract and render the video and analysis cards."""
    try:
        metadata, arrays = load_phase_payload(metadata_path)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load delivery data: {error}")
        return

    delivery_id = metadata.get("delivery_id", metadata_path.stem.removesuffix("_data"))
    st.header(delivery_id)
    st.caption(
        f"Source FPS: {_format_metric(metadata.get('original_fps'))} | "
        f"Frames: {metadata.get('frame_count', 'N/A')} | "
        f"Calibration: {metadata.get('calibration_status', 'unknown')}"
    )

    video_path = resolve_delivery_video(metadata_path, metadata)
    if video_path.is_file():
        st.video(str(video_path))
    else:
        st.warning(f"Delivery video was not found beside its metadata: {video_path}")

    pose_frames = arrays.get("poses_3d")
    if pose_frames is None or pose_frames.ndim != 3:
        pose_frames = arrays.get("poses_2d")
    selected_frame = 0
    if pose_frames is not None and pose_frames.ndim == 3 and len(pose_frames):
        suggested_frame = metadata.get("biomechanics", {}).get("peak_wrist_speed_frame", 0)
        if not isinstance(suggested_frame, (int, float, np.number)) or not np.isfinite(suggested_frame):
            angle_series = arrays.get("joint_angles_deg")
            suggested_frame = (
                int(np.argmax(np.isfinite(angle_series).sum(axis=1)))
                if angle_series is not None and angle_series.ndim == 2 and len(angle_series)
                else 0
            )
        default_frame = min(max(int(suggested_frame), 0), len(pose_frames) - 1)
        selected_frame = st.slider(
            "Pose and angle frame",
            min_value=1,
            max_value=len(pose_frames),
            value=default_frame + 1,
            key=f"pose-frame-{delivery_id}",
        ) - 1

    st.subheader("Biomechanics")
    biomechanics = metadata.get("biomechanics", {})
    angle_series = arrays.get("joint_angles_deg")
    if angle_series is not None and angle_series.ndim == 2 and selected_frame < len(angle_series):
        front_leg = biomechanics.get("front_leg", "left")
        bowling_arm = biomechanics.get("bowling_arm", "right")
        knee_index = 0 if front_leg == "left" else 1
        elbow_index = 2 if bowling_arm == "left" else 3
        knee_value = angle_series[selected_frame, knee_index]
        elbow_value = angle_series[selected_frame, elbow_index]
        trunk_value = angle_series[selected_frame, 4] if angle_series.shape[1] > 4 else np.nan
        knee_summary = biomechanics.get("front_knee_angle_deg", biomechanics.get("knee_angle_deg", {}))
        elbow_summary = biomechanics.get("bowling_elbow_angle_deg", biomechanics.get("elbow_angle_deg", {}))
        trunk_summary = biomechanics.get("lateral_trunk_tilt_deg", biomechanics.get("trunk_tilt_deg", {}))
        extension_summary = biomechanics.get("bowling_elbow_extension_deg", {})
        knee_label = f"{front_leg.title()} knee · frame angle"
        elbow_label = f"{bowling_arm.title()} elbow · frame angle"
        trunk_label = "Lateral trunk tilt · frame"
        extension_label = "Elbow extension · frame estimate"
        if not np.isfinite(knee_value):
            knee_value = knee_summary.get("min")
            knee_label = f"{front_leg.title()} knee · window minimum"
        if not np.isfinite(elbow_value):
            elbow_value = elbow_summary.get("min")
            elbow_label = f"{bowling_arm.title()} elbow · window minimum"
        if not np.isfinite(trunk_value):
            trunk_value = trunk_summary.get("max")
            trunk_label = "Lateral trunk tilt · window maximum"
        extension_value = 180.0 - elbow_value if isinstance(elbow_value, (int, float, np.number)) and np.isfinite(elbow_value) else extension_summary.get("max")
        if not np.isfinite(angle_series[selected_frame, elbow_index]):
            extension_label = "Elbow extension · window maximum"
        metrics = (
            (knee_label, knee_value, "°"),
            (elbow_label, elbow_value, "°"),
            (trunk_label, trunk_value, "°"),
            (extension_label, extension_value, "°"),
        )
    else:
        knee = biomechanics.get("front_knee_angle_deg", biomechanics.get("knee_angle_deg", {}))
        elbow = biomechanics.get("bowling_elbow_angle_deg", biomechanics.get("elbow_angle_deg", {}))
        trunk = biomechanics.get("lateral_trunk_tilt_deg", biomechanics.get("trunk_tilt_deg", {}))
        extension = biomechanics.get("bowling_elbow_extension_deg", {})
        metrics = (
            ("Front knee · window minimum", knee.get("min"), "°"),
            ("Bowling elbow · window minimum", elbow.get("min"), "°"),
            ("Lateral trunk tilt · window maximum", trunk.get("max"), "°"),
            ("Elbow extension · window maximum", extension.get("max"), "°"),
        )
    metric_columns = st.columns(len(metrics))
    for column, (label, value, suffix) in zip(metric_columns, metrics):
        column.metric(label, _format_metric(value, suffix))
    window = biomechanics.get("analysis_window", {})
    if window:
        method = {
            "exact_contact_frames": "Exact contact frames",
            "peak_wrist_speed": "10-frame peak-wrist fallback",
            "best_pose_visibility": "10-frame best-pose fallback (no valid wrist-speed peak)",
        }.get(window.get("selection_method"), "Biomechanics analysis window")
        st.caption(
            f"{method}; frames {window.get('start_frame', 'N/A')}–"
            f"{max(window.get('start_frame', 0), window.get('end_frame_exclusive', 1) - 1)} | "
            f"front leg: {biomechanics.get('front_leg', 'unknown')} | "
            f"bowling arm: {biomechanics.get('bowling_arm', 'unknown')}"
        )
    if biomechanics.get("interpretation_note"):
        st.caption(biomechanics["interpretation_note"])

    st.subheader("Injury screening")
    injury = metadata.get("injury_screen", {})
    alert_columns = st.columns(3)
    alerts = (
        ("Hip-shoulder separation", injury.get("separation_warning")),
        ("Mixed-action screen", injury.get("mixed_action_screen")),
        ("Knee-brace screen", biomechanics.get("knee_brace_warning")),
    )
    for column, (label, value) in zip(alert_columns, alerts):
        if value is True:
            column.warning(f"{label}: review")
        elif value is False:
            column.success(f"{label}: not flagged")
        else:
            column.info(f"{label}: unavailable")
    st.caption(injury.get("status", "Screening indicators only; not a medical assessment."))

    st.subheader("Coach feedback")
    coach = metadata.get("coach", {})
    report = coach.get("report", coach.get("feedback"))
    if isinstance(report, dict):
        sections = (
            ("Root-cause diagnosis", "root_cause_diagnosis"),
            ("#1 Priority fix", "priority_fix"),
            ("Prescribed net drill", "prescribed_net_drill"),
        )
        for heading, key in sections:
            st.markdown(f"**{heading}**")
            st.info(report.get(key) or "No coaching guidance was returned for this section.")
        st.caption(f"Generated by: {coach.get('generated_by', 'unknown')}")
    elif report:
        st.info(report)
    else:
        st.info("Coaching feedback has not been generated for this delivery.")
    observations = coach.get("observations") or []
    if observations:
        st.markdown("**Observations**")
        for observation in observations:
            st.markdown(f"- {observation}")
    if coach.get("safety_note"):
        st.caption(coach["safety_note"])

    st.subheader("3D digital twin")
    _render_skeleton(st, arrays, selected_frame)


def main() -> None:
    """Automatically display delivery_001 and allow switching between saved clips."""
    try:
        import streamlit as st
    except ImportError as error:
        raise ImportError("Install streamlit to use the dashboard.") from error

    st.set_page_config(page_title="Cricket Bowling Analyzer", layout="wide")
    st.title("Cricket Bowling Analyzer")
    st.caption("Saved delivery analysis | Estimates are screening aids, not medical or legality decisions.")

    delivery_dir = Path(st.text_input("Delivery folder", str(DEFAULT_DELIVERY_DIR))).expanduser()
    delivery_files = discover_deliveries(delivery_dir)
    if not delivery_files:
        st.warning(f"No saved delivery metadata found in {delivery_dir}. Run the pipeline first.")
        return

    default_index = next(
        (index for index, path in enumerate(delivery_files) if path.name == "delivery_001_data.json"),
        0,
    )
    selected_file = st.selectbox(
        "Delivery",
        delivery_files,
        index=default_index,
        format_func=lambda path: path.name.removesuffix("_data.json"),
        key=f"delivery-{delivery_dir.resolve()}",
    )
    _render_dashboard(st, selected_file)


if __name__ == "__main__":
    main()
