"""Streamlit dashboard for previously analyzed cricket delivery clips."""

import sqlite3
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cricket_coach.config import POSE_CONFIDENCE_THRESHOLD
from cricket_coach.data_contract import load_phase_payload
from cricket_coach.training.mechanics_model import (
    mechanics_training_readiness,
    predict_active_mechanics,
    train_mechanics_model,
)
from cricket_coach.training.pose_dataset import (
    MIN_PLAYER_GROUPS,
    MIN_POSE_REVIEWS,
    export_reviewed_pose_dataset,
)
from cricket_coach.training.review_store import (
    KEYPOINT_NAMES,
    MECHANICS_LABELS,
    PLAYER_ROLES,
    list_reviews,
    review_stats,
    save_review,
)
from cricket_coach.training.train_pose import train_pose_model

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
    fallback_frames = arrays.get("pose_3d_fallback_frames")
    frame_uses_2d_fallback = (
        fallback_frames is not None
        and fallback_frames.ndim == 1
        and 0 <= selected_frame < len(fallback_frames)
        and bool(fallback_frames[selected_frame])
    )
    if (
        not frame_uses_2d_fallback
        and poses_3d is not None
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
    if frame_uses_2d_fallback:
        st.caption("Showing YOLOv8 2D keypoints because 3D landmarks were unavailable.")
    else:
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


def _needs_pipeline_diagnostics(metadata: dict[str, Any]) -> bool:
    """Open diagnostics automatically when one or more biomechanics values are unavailable."""
    biomechanics = metadata.get("biomechanics", {})
    if not isinstance(biomechanics, dict):
        return True
    metric_fields = (
        (
            biomechanics.get("front_knee_angle_deg", biomechanics.get("knee_angle_deg", {})),
            "min",
        ),
        (
            biomechanics.get("bowling_elbow_angle_deg", biomechanics.get("elbow_angle_deg", {})),
            "min",
        ),
        (
            biomechanics.get("lateral_trunk_tilt_deg", biomechanics.get("trunk_tilt_deg", {})),
            "max",
        ),
        (biomechanics.get("bowling_elbow_extension_deg", {}), "max"),
    )
    return any(
        not isinstance(summary, dict) or _format_metric(summary.get(key)) == "N/A"
        for summary, key in metric_fields
    )


def _render_pipeline_health(st: Any, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """Expose saved pipeline status and pose errors without opening the metadata JSON."""
    pose_count = metadata.get("pose_3d_valid_frame_count")
    if not isinstance(pose_count, (int, np.integer)):
        poses_3d = arrays.get("poses_3d")
        if poses_3d is not None and poses_3d.ndim == 3 and poses_3d.shape[2] >= 3:
            landmark_counts = np.isfinite(poses_3d[..., :3]).all(axis=-1).sum(axis=1)
            pose_count = int((landmark_counts >= 3).sum())
        else:
            pose_count = "N/A"

    coach = metadata.get("coach")
    coach = coach if isinstance(coach, dict) else {}
    phase_status = metadata.get("phase_status")
    phase_status = phase_status if isinstance(phase_status, dict) else {}
    phase_errors = metadata.get("phase_errors")
    phase_errors = phase_errors if isinstance(phase_errors, dict) else {}
    with st.expander(
        "Pipeline Health & Debug Status",
        expanded=_needs_pipeline_diagnostics(metadata) or bool(phase_errors),
    ):
        st.write("Phase status")
        st.json(phase_status)
        st.metric("Valid pose frames", pose_count)
        st.write("Pose status", metadata.get("pose_status", "not recorded"))
        st.write("Coach generated by", coach.get("generated_by", "not recorded"))
        if phase_errors:
            st.error("Pipeline errors or fallbacks were recorded.")
            st.json(phase_errors)
        else:
            st.success("No pipeline errors recorded.")
        if metadata.get("pose_3d_fallback_frame_count"):
            st.warning(
                f"2D keypoints were used as fallback in "
                f"{metadata['pose_3d_fallback_frame_count']} frame(s)."
            )


def _read_review_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open the saved delivery video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Unable to decode frame {frame_index + 1} for annotation.")
        height, width = frame.shape[:2]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width, height
    finally:
        capture.release()


def _draw_review_keypoints(frame: np.ndarray, pose: np.ndarray) -> np.ndarray:
    preview = frame.copy()
    height, width = frame.shape[:2]

    def image_point(point: np.ndarray) -> tuple[int, int]:
        clipped = np.clip(np.rint(point), (0, 0), (width - 1, height - 1)).astype(int)
        return int(clipped[0]), int(clipped[1])

    for first, second in SKELETON_BONES:
        endpoints = pose[[first, second], :2]
        if np.isfinite(endpoints).all():
            start = image_point(endpoints[0])
            end = image_point(endpoints[1])
            cv2.line(preview, start, end, (0, 255, 0), 2)
    for point in pose:
        if np.isfinite(point[:2]).all():
            center = image_point(point[:2])
            cv2.circle(preview, center, 4, (255, 200, 0), -1)
    return preview


def _draw_person_candidates(
    frame: np.ndarray,
    boxes: np.ndarray,
    selected_index: int,
    pose: np.ndarray,
) -> np.ndarray:
    preview = frame.copy()
    height, width = frame.shape[:2]
    for index, box in enumerate(boxes):
        if not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip((x1, x2), 0, width - 1)
        y1, y2 = np.clip((y1, y2), 0, height - 1)
        color = (0, 255, 0) if index == selected_index else (255, 200, 0)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            preview,
            f"Person {index + 1}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return _draw_review_keypoints(preview, pose)


def _history_rows(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for review in reviews:
        metrics = review.get("metrics", {})
        knee = metrics.get("front_knee_angle_deg", {})
        trunk = metrics.get("lateral_trunk_tilt_deg", {})
        rows.append(
            {
                "date": review["created_at"][:10],
                "delivery": review["delivery_id"],
                "frame": int(review["frame_index"]) + 1,
                "person role": review.get("player_role", "unknown"),
                "mechanics tags": ", ".join(review["mechanics_labels"]),
                "coach rating": review["coach_rating"],
                "drill completed": review["drill_completed"],
                "drill helpful": review.get("drill_helpful"),
                "prescribed drill": review.get("drill_text", ""),
                "front knee min (deg)": knee.get("min"),
                "trunk tilt max (deg)": trunk.get("max"),
            }
        )
    return rows


def _render_training_review(
    st: Any,
    metadata_path: Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    video_path: Path,
    selected_frame: int,
) -> None:
    """Capture human-reviewed pose labels, coaching feedback, and player history."""
    with st.expander("Training data review & player progress", expanded=False):
        st.caption(
            "Corrections are saved locally for supervised batch training. "
            "Nothing updates model weights automatically."
        )
        st.info(
            "Choosing a detected person and labeling their role saves feedback for future "
            "person-selection training. It does not change this delivery's existing "
            "biomechanics or automatically retrain a model."
        )
        st.info(
            "Only review videos you have permission to use. Enter an anonymous ID, not a person's name. "
            "This tool is not a medical assessment or an official action-legality test."
        )
        try:
            stats = review_stats()
        except (OSError, RuntimeError, sqlite3.Error) as error:
            st.error(f"Could not read the local review database: {error}")
            return
        columns = st.columns(3)
        columns[0].metric("Reviewed frames", stats["reviews"])
        columns[1].metric("Anonymous players", stats["players"])
        columns[2].metric("Mechanics-tagged examples", stats["tagged_reviews"])
        st.caption(
            f"Pose fine-tuning stays disabled until at least {MIN_POSE_REVIEWS} reviewed frames "
            f"from {MIN_PLAYER_GROUPS} different player IDs are available."
        )
        st.subheader("Train and evaluate models")
        st.caption(
            "Training runs locally only when you click a button. Candidates are not promoted "
            "unless you opt in and validation improves; the test split is report-only."
        )
        pose_ready = (
            stats["reviews"] >= MIN_POSE_REVIEWS
            and stats["players"] >= MIN_PLAYER_GROUPS
        )
        pose_epochs = st.number_input(
            "Pose model epochs",
            min_value=1,
            max_value=200,
            value=30,
            step=5,
            key="pose-training-epochs",
        )
        promote_pose = st.checkbox(
            "Promote pose candidate if validation improves (test remains report-only)",
            key="promote-pose-candidate",
        )
        if st.button("Train/evaluate pose model", disabled=not pose_ready):
            dataset_root = Path("data") / "training"
            version = 1
            while (dataset_root / f"pose_dataset_v{version}").exists():
                version += 1
            dataset_path = dataset_root / f"pose_dataset_v{version}"
            try:
                with st.spinner("Exporting reviewed frames and training the pose model..."):
                    dataset = export_reviewed_pose_dataset(dataset_path)
                    training_report = train_pose_model(
                        dataset_yaml=dataset["dataset_yaml"],
                        epochs=int(pose_epochs),
                        promote=promote_pose,
                    )
            except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as error:
                st.error(f"Pose training did not complete: {type(error).__name__}: {error}")
            else:
                st.success(
                    "Pose candidate trained. "
                    + ("Promoted for future pipeline runs." if training_report["promoted"] else "Baseline model remains active.")
                )
                st.json(training_report["metrics"])

        try:
            mechanics_readiness = mechanics_training_readiness()
        except (OSError, RuntimeError, sqlite3.Error) as error:
            st.error(f"Could not check mechanics-label coverage: {error}")
            mechanics_readiness = {"ready": False, "counts": {}, "player_counts": {}}
        if mechanics_readiness["counts"]:
            st.dataframe(
                [
                    {
                        "mechanics tag": label,
                        "reviewed deliveries": count,
                        "player IDs": mechanics_readiness["player_counts"][label],
                        "minimum examples": mechanics_readiness["minimum_examples_per_label"],
                        "trainable": label in mechanics_readiness["ready_labels"],
                    }
                    for label, count in mechanics_readiness["counts"].items()
                ],
                use_container_width=True,
                hide_index=True,
            )
        mechanics_epochs = st.number_input(
            "Mechanics model epochs",
            min_value=1,
            max_value=300,
            value=100,
            step=10,
            key="mechanics-training-epochs",
        )
        promote_mechanics = st.checkbox(
            "Promote mechanics model if validation improves (test remains report-only)",
            key="promote-mechanics-candidate",
        )
        if st.button(
            "Train/evaluate mechanics model",
            disabled=not mechanics_readiness["ready"],
        ):
            try:
                with st.spinner("Training the reviewed-delivery sequence model..."):
                    training_report = train_mechanics_model(
                        epochs=int(mechanics_epochs),
                        promote=promote_mechanics,
                    )
            except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as error:
                st.error(f"Mechanics training did not complete: {type(error).__name__}: {error}")
            else:
                st.success(
                    "Mechanics candidate trained. "
                    + ("Promoted for future pipeline runs." if training_report["promoted"] else "No model was promoted.")
                )
                st.json(training_report["metrics"])

        poses_2d = arrays.get("poses_2d")
        if (
            poses_2d is None
            or poses_2d.ndim != 3
            or len(poses_2d) == 0
            or poses_2d.shape[1] < len(KEYPOINT_NAMES)
        ):
            st.warning("This delivery has no editable 2D keypoints, so it cannot add a pose training example.")
        elif not video_path.is_file():
            st.warning("The saved clip is missing; a frame cannot be reviewed or labeled.")
        else:
            frame_index = min(max(selected_frame, 0), len(poses_2d) - 1)
            try:
                frame_image, image_width, image_height = _read_review_frame(video_path, frame_index)
            except (OSError, ValueError, cv2.error) as error:
                st.error(f"Could not load the frame for review: {error}")
                frame_image = None

            if frame_image is not None:
                candidate_poses = arrays.get("poses_2d_candidates")
                candidate_boxes = arrays.get("person_boxes_xyxy")
                candidate_counts = arrays.get("person_candidate_counts")
                candidate_count = 0
                if (
                    candidate_poses is not None
                    and candidate_poses.ndim == 4
                    and candidate_poses.shape[0] == len(poses_2d)
                    and candidate_poses.shape[2] >= len(KEYPOINT_NAMES)
                    and candidate_boxes is not None
                    and candidate_boxes.ndim == 3
                    and candidate_boxes.shape[:2] == candidate_poses.shape[:2]
                    and candidate_counts is not None
                    and candidate_counts.ndim == 1
                    and frame_index < len(candidate_counts)
                ):
                    candidate_count = min(
                        int(candidate_counts[frame_index]),
                        candidate_poses.shape[1],
                        candidate_boxes.shape[1],
                    )

                selected_person_index: int | None = None
                person_box: np.ndarray | None = None
                if candidate_count:
                    selected_person_index = st.selectbox(
                        "Person to label in this frame",
                        options=list(range(candidate_count)),
                        format_func=lambda index: (
                            f"Person {index + 1}"
                            + (" (largest detection)" if index == 0 else "")
                        ),
                        help=(
                            "Choose the box around the bowler. Detection order is by box size "
                            "for this frame only and may change in another frame."
                        ),
                        key=(
                            f"review-person-{metadata.get('delivery_id', metadata_path.stem)}-"
                            f"{frame_index}"
                        ),
                    )
                    pose = candidate_poses[frame_index, selected_person_index]
                    person_box = candidate_boxes[frame_index, selected_person_index]
                    review_preview = _draw_person_candidates(
                        frame_image,
                        candidate_boxes[frame_index, :candidate_count],
                        selected_person_index,
                        pose,
                    )
                    st.caption(
                        "Choose the highlighted person deliberately; the automatic pose may "
                        "have selected a different player."
                    )
                    if selected_person_index != 0:
                        st.caption(
                            "Biomechanics values below were computed from the automatic "
                            "largest-person selection, not this manually selected person."
                        )
                else:
                    pose = poses_2d[frame_index]
                    review_preview = _draw_review_keypoints(frame_image, pose)
                    st.warning(
                        "This clip has no saved per-person detections. Re-run the pipeline "
                        "with the current version to manually choose among detected people."
                    )
                st.image(
                    review_preview,
                    caption=f"Frame {frame_index + 1}; edit the selected person's keypoints below.",
                    use_container_width=True,
                )
                editor_rows = []
                for index, name in enumerate(KEYPOINT_NAMES):
                    x, y = float(pose[index, 0]), float(pose[index, 1])
                    confidence = (
                        float(pose[index, 2])
                        if pose.shape[1] >= 3 and np.isfinite(pose[index, 2])
                        else None
                    )
                    point_is_finite = np.isfinite((x, y)).all()
                    editor_rows.append(
                        {
                            "index": index,
                            "keypoint": name,
                            "x_px": round(x, 2) if point_is_finite else None,
                            "y_px": round(y, 2) if point_is_finite else None,
                            "visible": bool(
                                point_is_finite
                                and (confidence is None or confidence >= POSE_CONFIDENCE_THRESHOLD)
                            ),
                            "model confidence": round(confidence, 3) if confidence is not None else None,
                        }
                    )

                with st.form(key=f"review-form-{metadata.get('delivery_id', metadata_path.stem)}-{frame_index}"):
                    player_id = st.text_input(
                        "Anonymous player ID",
                        placeholder="player_001",
                        help="Use a stable alias such as player_001; do not enter a person's name.",
                    )
                    role_labels = {
                        "bowler": "Bowler",
                        "batter": "Batter",
                        "wicketkeeper": "Wicketkeeper",
                        "other": "Other",
                    }
                    player_role = st.selectbox(
                        "Selected person's role",
                        options=PLAYER_ROLES,
                        format_func=lambda role: role_labels[role],
                        index=0,
                        help="This role label is saved as feedback for future person-selection training.",
                        key=(
                            f"review-role-{metadata.get('delivery_id', metadata_path.stem)}-"
                            f"{frame_index}-{selected_person_index}"
                        ),
                    )
                    consent = st.checkbox(
                        "I have permission to use this player's video and labels for local analysis and training."
                    )
                    edited_keypoints = st.data_editor(
                        editor_rows,
                        num_rows="fixed",
                        hide_index=True,
                        use_container_width=True,
                        disabled=["index", "keypoint", "model confidence"],
                        key=(
                            f"keypoints-{metadata.get('delivery_id', metadata_path.stem)}-"
                            f"{frame_index}-{selected_person_index}"
                        ),
                    )
                    mechanics_labels = st.multiselect(
                        "Coach-reviewed mechanics tags (optional)",
                        MECHANICS_LABELS,
                        help=(
                            "Use descriptive technique observations, not legal or medical diagnoses. "
                            "The 'other' tag is stored but is not a mechanics-model training class."
                        ),
                    )
                    rating_choice = st.selectbox(
                        "How helpful was the coach feedback?",
                        ("Not rated", "1", "2", "3", "4", "5"),
                        key=f"coach-rating-{metadata.get('delivery_id', metadata_path.stem)}-{frame_index}",
                    )
                    drill_completed = st.checkbox("Player completed the prescribed drill.")
                    helpful_choice = st.selectbox(
                        "If completed, was the drill helpful?",
                        ("Not rated", "Yes", "No"),
                        disabled=not drill_completed,
                        key=f"drill-helpful-{metadata.get('delivery_id', metadata_path.stem)}-{frame_index}",
                    )
                    notes = st.text_area("Review notes (optional; do not include personal or medical details).")
                    submitted = st.form_submit_button("Save reviewed example")

                if submitted:
                    rating = int(rating_choice) if rating_choice != "Not rated" else None
                    drill_helpful = (
                        helpful_choice == "Yes"
                        if drill_completed and helpful_choice != "Not rated"
                        else None
                    )
                    coach = metadata.get("coach", {})
                    coach_report = coach.get("report", coach.get("feedback", {}))
                    drill_text = (
                        coach_report.get("prescribed_net_drill", "")
                        if isinstance(coach_report, dict)
                        else ""
                    )
                    try:
                        review_metrics = (
                            metadata.get("biomechanics", {})
                            if selected_person_index in (None, 0)
                            else {}
                        )
                        review_id = save_review(
                            player_id=player_id,
                            delivery_id=str(metadata.get("delivery_id", metadata_path.stem)),
                            metadata_path=metadata_path,
                            video_path=video_path,
                            frame_index=frame_index,
                            image_width=image_width,
                            image_height=image_height,
                            keypoint_rows=edited_keypoints,
                            mechanics_labels=mechanics_labels,
                            coach_rating=rating,
                            drill_completed=drill_completed,
                            drill_helpful=drill_helpful,
                            notes=notes,
                            metrics=review_metrics,
                            consent_confirmed=consent,
                            drill_text=drill_text,
                            player_role=player_role,
                            person_detection_index=selected_person_index,
                            person_box=person_box,
                        )
                    except (OSError, PermissionError, ValueError, sqlite3.Error) as error:
                        st.error(f"Could not save this reviewed example: {error}")
                    else:
                        st.success(f"Saved reviewed example {review_id[:8]}.")

        history_player_id = st.text_input(
            "View history for anonymous player ID",
            key="player-history-id",
        ).strip()
        if history_player_id:
            try:
                reviews = list_reviews(player_id=history_player_id, limit=1000)
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
                st.error(f"Could not load player history: {error}")
            else:
                latest_by_delivery: dict[tuple[str, str], dict[str, Any]] = {}
                for review in reviews:
                    history_key = (review["metadata_path"], review["delivery_id"])
                    if history_key not in latest_by_delivery:
                        latest_by_delivery[history_key] = review
                history = list(latest_by_delivery.values())
                if history:
                    completed_drills = sum(review["drill_completed"] for review in history)
                    rated_drills = [
                        review for review in history if review["drill_helpful"] is not None
                    ]
                    summary_columns = st.columns(2)
                    summary_columns[0].metric(
                        "Drills completed",
                        f"{completed_drills}/{len(history)}",
                    )
                    if rated_drills:
                        helpful_drills = sum(review["drill_helpful"] for review in rated_drills)
                        summary_columns[1].metric(
                            "Drills rated helpful",
                            f"{helpful_drills}/{len(rated_drills)}",
                        )
                    else:
                        summary_columns[1].metric("Drills rated helpful", "Not rated yet")
                    rated_by_drill: dict[str, list[bool]] = {}
                    for review in history:
                        drill = review.get("drill_text", "").strip()
                        if drill and review["drill_helpful"] is not None:
                            rated_by_drill.setdefault(drill, []).append(review["drill_helpful"])
                    consistent_drills = [
                        (text, outcomes)
                        for text, outcomes in rated_by_drill.items()
                        if len(outcomes) >= 3
                    ]
                    if consistent_drills:
                        best_drill, outcomes = max(
                            consistent_drills,
                            key=lambda item: (sum(item[1]) / len(item[1]), len(item[1])),
                        )
                        st.info(
                            f"Your highest-rated saved drill was marked helpful "
                            f"{sum(outcomes)}/{len(outcomes)} times: {best_drill}"
                        )
                    trend_rows = []
                    for review in reversed(history):
                        metrics = review.get("metrics", {})
                        knee_value = metrics.get("front_knee_angle_deg", {}).get("min")
                        trunk_value = metrics.get("lateral_trunk_tilt_deg", {}).get("max")
                        if knee_value is not None or trunk_value is not None:
                            trend_rows.append(
                                {
                                    "session": review["created_at"][:10],
                                    "front knee min (deg)": knee_value,
                                    "trunk tilt max (deg)": trunk_value,
                                }
                            )
                    if trend_rows:
                        st.line_chart(
                            trend_rows,
                            x="session",
                            y=["front knee min (deg)", "trunk tilt max (deg)"],
                        )
                    st.dataframe(_history_rows(history), use_container_width=True, hide_index=True)
                else:
                    st.info("No reviewed sessions are saved for this anonymous player ID yet.")


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
    st.caption("Displaying saved delivery analysis; opening this page does not rerun the pipeline.")
    _render_pipeline_health(st, metadata, arrays)

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

    _render_training_review(st, metadata_path, metadata, arrays, video_path, selected_frame)

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
            "peak_wrist_speed": "10-frame global peak-wrist window",
            "no_valid_wrist_speed": "No valid wrist-speed peak; metrics unavailable",
        }.get(window.get("selection_method"), "Biomechanics analysis window")
        st.caption(
            f"{method}; frames {window.get('start_frame', 'N/A')}–"
            f"{max(window.get('start_frame', 0), window.get('end_frame_exclusive', 1) - 1)} | "
            f"front leg: {biomechanics.get('front_leg') or 'unknown'} | "
            f"bowling arm: {biomechanics.get('bowling_arm') or 'unknown'}"
        )
    if biomechanics.get("interpretation_note"):
        st.caption(biomechanics["interpretation_note"])

    learned_mechanics = metadata.get("learned_mechanics", {})
    if isinstance(learned_mechanics, dict) and learned_mechanics.get("status") == "complete":
        st.subheader("Learned mechanics suggestions")
        probabilities = learned_mechanics.get("probabilities", {})
        if probabilities:
            st.dataframe(
                [
                    {"mechanics tag": label, "model confidence": float(score)}
                    for label, score in probabilities.items()
                ],
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            "Experimental model predictions from reviewed examples; confirm them with a coach. "
            "They are not legal, medical, or injury-risk conclusions."
        )
    elif isinstance(learned_mechanics, dict) and learned_mechanics.get("status") == "error":
        st.warning("The trained mechanics model could not evaluate this delivery; see Pipeline Health.")

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
