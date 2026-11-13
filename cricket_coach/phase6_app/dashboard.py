"""Streamlit dashboard for previously analyzed cricket delivery clips."""

from pathlib import Path
from typing import Any

import numpy as np

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


def _render_skeleton(st: Any, arrays: dict[str, np.ndarray]) -> None:
    """Render the best available 3D pose, falling back to a clearly labeled 2D view."""
    poses_3d = arrays.get("poses_3d")
    if poses_3d is not None and poses_3d.ndim == 3 and poses_3d.shape[1] >= 17 and poses_3d.shape[2] >= 3:
        available = np.isfinite(poses_3d[..., :3]).all(axis=2).sum(axis=1)
        frame_index = int(np.argmax(available)) if len(available) else 0
        frame = poses_3d[frame_index, :, :3]
        if int(available[frame_index]) >= 3:
            _render_3d_pose(st, frame, frame_index, len(poses_3d))
            return

    poses_2d = arrays.get("poses_2d")
    if poses_2d is None or poses_2d.ndim != 3 or poses_2d.shape[1] < 17 or poses_2d.shape[2] < 2:
        st.info("No pose landmarks are available for a skeleton view.")
        return
    available = np.isfinite(poses_2d[..., :2]).all(axis=-1).sum(axis=1)
    if not len(available) or int(np.max(available)) < 3:
        st.info("Pose landmarks are unavailable. A trained pose model is needed for a skeleton preview.")
        return
    frame_index = int(np.argmax(available))
    _render_2d_pose(st, poses_2d[frame_index, :, :2], frame_index, len(poses_2d))
    st.caption(
        "Showing a 2D pose preview. A validated 3D pose-lifting model is required for a real 3D skeleton."
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

    st.subheader("Biomechanics")
    biomechanics = metadata.get("biomechanics", {})
    metric_columns = st.columns(3)
    metrics = (
        ("Knee angle · minimum", biomechanics.get("knee_angle_deg", {}).get("min"), "°"),
        ("Elbow angle · minimum", biomechanics.get("elbow_angle_deg", {}).get("min"), "°"),
        ("Trunk tilt · maximum", biomechanics.get("trunk_tilt_deg", {}).get("max"), "°"),
    )
    for column, (label, value, suffix) in zip(metric_columns, metrics):
        column.metric(label, _format_metric(value, suffix))
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
    feedback = coach.get("feedback")
    if feedback:
        st.info(feedback)
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
    visualization = metadata.get("digital_twin_visualization")
    if isinstance(visualization, str):
        visual_path = Path(visualization)
        if visual_path.is_file():
            try:
                import streamlit.components.v1 as components

                components.html(visual_path.read_text(encoding="utf-8"), height=560, scrolling=True)
            except OSError as error:
                st.warning(f"Could not load saved 3D view: {error}")
            return
    _render_skeleton(st, arrays)


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
    )
    _render_dashboard(st, selected_file)


if __name__ == "__main__":
    main()
