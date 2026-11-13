"""End-to-end orchestration for the modular cricket analysis phases."""

import argparse
import logging
from pathlib import Path
from typing import Any

from cricket_coach.config import PipelinePaths
from cricket_coach.phase1_capture.calibrator import PitchCalibrator
from cricket_coach.phase1_capture.slicer import DeliverySlicer
from cricket_coach.phase2_tracking.ball_tracker import BallTracker
from cricket_coach.phase2_tracking.event_detector import EventDetector
from cricket_coach.phase2_tracking.pose_3d import Pose3DProcessor
from cricket_coach.phase3_biomechanics.ball_physics import BallPhysicsAnalyzer
from cricket_coach.phase3_biomechanics.injury_risk import InjuryRiskAnalyzer
from cricket_coach.phase3_biomechanics.kinematics import KinematicsAnalyzer
from cricket_coach.phase4_digital_twin.pro_comparator import ProComparator
from cricket_coach.phase4_digital_twin.visualizer_3d import SkeletonVisualizer
from cricket_coach.phase5_ai_coach.rag_coach import RAGCoach
from cricket_coach.data_contract import load_phase_payload, save_phase_payload

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    matrix_path: str | Path | None = None,
    skip_interactive_calibration: bool = False,
    slicer: DeliverySlicer | None = None,
    pose_processor: Pose3DProcessor | None = None,
    ball_tracker: BallTracker | None = None,
    coach: RAGCoach | None = None,
    reference_path: str | Path | None = None,
    interactive_calibration: bool = False,
) -> dict[str, Any]:
    """Calibrate or load, slice deliveries, then run each configured analysis phase."""
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"Video file does not exist: {source}")
    paths = PipelinePaths.create(output_dir, matrix_path)
    if paths.calibration_matrix.is_file():
        matrix = PitchCalibrator.load_homography(paths.calibration_matrix)
        if PitchCalibrator.is_identity_fallback(matrix):
            LOGGER.warning(
                "Using identity fallback calibration at %s; pitch measurements are not metric.",
                paths.calibration_matrix,
            )
    elif skip_interactive_calibration:
        raise FileNotFoundError(f"Calibration matrix not found: {paths.calibration_matrix}")
    else:
        PitchCalibrator().calibrate(source, paths.calibration_matrix, interactive=interactive_calibration)

    clip_paths = (slicer or DeliverySlicer()).slice_video(source, paths.delivery_dir, paths.calibration_matrix)
    pose_analyzer = pose_processor
    ball_analyzer = ball_tracker or BallTracker()
    coach_module = coach or RAGCoach()
    results: list[dict[str, Any]] = []
    try:
        for clip_path in clip_paths:
            metadata_path = clip_path.with_name(f"{clip_path.stem}_data.json")
            metadata: dict[str, Any] = {"delivery_id": clip_path.stem, "video_path": str(clip_path.resolve())}
            try:
                if pose_analyzer is None:
                    pose_analyzer = Pose3DProcessor()
                metadata, arrays = pose_analyzer.process(clip_path, metadata_path)
                metadata, arrays = ball_analyzer.track(clip_path, metadata_path)
                if "poses_2d" in arrays:
                    EventDetector().process(metadata_path)
                if "poses_3d" in arrays:
                    KinematicsAnalyzer().process(metadata_path)
                    InjuryRiskAnalyzer().process(metadata_path)
                    metadata, arrays = load_phase_payload(metadata_path)
                    if reference_path:
                        reference = ProComparator.load_reference(reference_path)
                        comparison = ProComparator().compare(arrays["poses_3d"], reference)
                        metadata["pro_comparison"] = comparison
                        metadata["phase_status"] = {**metadata.get("phase_status", {}), "pro_comparison": "complete"}
                        save_phase_payload(metadata_path, metadata, arrays)
                    try:
                        visual_path = paths.output_dir / "digital_twin" / f"{clip_path.stem}_pose.html"
                        SkeletonVisualizer().export_html(arrays["poses_3d"], visual_path)
                        metadata["digital_twin_visualization"] = str(visual_path.resolve())
                        metadata["phase_status"] = {**metadata.get("phase_status", {}), "visualization": "complete"}
                        save_phase_payload(metadata_path, metadata, arrays)
                    except (ImportError, ValueError) as error:
                        metadata["digital_twin_visualization"] = {"status": "unavailable", "reason": str(error)}
                        metadata["phase_status"] = {**metadata.get("phase_status", {}), "visualization": "unavailable"}
                        save_phase_payload(metadata_path, metadata, arrays)
                BallPhysicsAnalyzer().process(metadata_path)
                coach_module.process(metadata_path)
                metadata, _ = load_phase_payload(metadata_path)
            except ImportError as error:
                metadata, arrays = load_phase_payload(metadata_path)
                metadata["analysis_warning"] = str(error)
                metadata["phase_status"] = {**metadata.get("phase_status", {}), "analysis": "dependency_unavailable"}
                save_phase_payload(metadata_path, metadata, arrays)
            results.append(metadata)
    finally:
        close_pose_analyzer = getattr(pose_analyzer, "close", None)
        if callable(close_pose_analyzer):
            close_pose_analyzer()
    return {
        "video_path": str(source.resolve()),
        "calibration_matrix_path": str(paths.calibration_matrix.resolve()),
        "delivery_count": len(clip_paths),
        "deliveries": results,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the cricket bowling analysis pipeline.")
    parser.add_argument("--video_path", required=True, help="Path to raw session video.")
    parser.add_argument("--output_dir", default="data/sliced_clips", help="Directory for analysis outputs.")
    parser.add_argument(
        "--matrix_path",
        help="Existing calibration .npy file; otherwise automatic calibration runs by default.",
    )
    parser.add_argument(
        "--skip_interactive_calibration",
        action="store_true",
        help="Fail if no existing matrix is available instead of running automatic calibration.",
    )
    parser.add_argument(
        "--interactive_calibration",
        action="store_true",
        help="Open the manual point-selection window (automatic calibration is the default).",
    )
    parser.add_argument("--reference_path", help="Optional trusted pro pose reference (.npy or .npz).")
    args = parser.parse_args()
    result = run_pipeline(
        args.video_path,
        args.output_dir,
        args.matrix_path,
        args.skip_interactive_calibration,
        reference_path=args.reference_path,
        interactive_calibration=args.interactive_calibration,
    )
    print(f"Extracted {result['delivery_count']} delivery clip(s).")
    for delivery in result["deliveries"]:
        print(f"- {delivery.get('delivery_id', 'delivery')}: {delivery.get('phase_status', {})}")


if __name__ == "__main__":
    main()
