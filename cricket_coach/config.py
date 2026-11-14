"""Centralized pipeline settings and thresholds."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ICC_CREASE_WIDTH_M = 2.64
ICC_CREASE_DEPTH_M = 1.22

PRE_TRIGGER_SEC = 0.5
POST_TRIGGER_SEC = 0.5
COOLDOWN_SEC = 1.0
SHORT_CLIP_FALLBACK_SEC = 8.0
WRIST_VELOCITY_THRESHOLD_PX_PER_FRAME_720P = 10.0
POSE_CONFIDENCE_THRESHOLD = 0.25
BALL_CONFIDENCE_THRESHOLD = 0.2
REFERENCE_FRAME_HEIGHT_PX = 720.0
MIN_BALL_FIT_DETECTIONS = 3
MIN_EVENT_FRAMES = 2
KINEMATICS_PEAK_WINDOW_FRAMES = 10
MEDIAPIPE_MODEL_COMPLEXITY = 2
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
EVENT_EXACT_CONFIDENCE_THRESHOLD = 0.8
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_COACH_MODEL = "meta/muse-glimmer-30b"
NVIDIA_COACH_TEMPERATURE = 0.4
NVIDIA_COACH_TOP_P = 0.95
NVIDIA_COACH_MAX_TOKENS = 1500

OUTPUT_WIDTH_PX = 1280
OUTPUT_HEIGHT_PX = 720
CALIBRATION_FRAME_SKIP = 1
CALIBRATION_WINDOW_TITLE = "Pitch calibration"
CALIBRATION_WHITE_SATURATION_MAX = 80
CALIBRATION_WHITE_VALUE_MIN = 170
CALIBRATION_WHITE_VALUE_MAX = 255
CALIBRATION_CANNY_LOW_THRESHOLD = 50
CALIBRATION_CANNY_HIGH_THRESHOLD = 150
CALIBRATION_HOUGH_THRESHOLD = 30
CALIBRATION_HOUGH_RHO_RESOLUTION_PX = 1.0
CALIBRATION_HOUGH_ANGLE_RESOLUTION_DEG = 1.0
CALIBRATION_HOUGH_MIN_LINE_LENGTH_PX = 12
CALIBRATION_HOUGH_MAX_LINE_GAP_PX = 4
CALIBRATION_HOUGH_MIN_LINE_LENGTH_RATIO = 0.04
CALIBRATION_HOUGH_MAX_LINE_GAP_RATIO = 0.02
CALIBRATION_ORIENTATION_BIN_COUNT = 36
CALIBRATION_ORIENTATION_TOLERANCE_DEG = 18.0
CALIBRATION_MIN_FAMILY_ANGLE_SEPARATION_DEG = 30.0
CALIBRATION_LINE_MERGE_DISTANCE_RATIO = 0.015
CALIBRATION_MIN_LINE_SEPARATION_RATIO = 0.03
CALIBRATION_MIN_QUADRILATERAL_AREA_RATIO = 0.005
CALIBRATION_MAX_CANDIDATE_LINES_PER_FAMILY = 8
CALIBRATION_MIN_INTERSECTION_DETERMINANT = 1e-6
CALIBRATION_IDENTITY_MATRIX_TOLERANCE = 1e-8
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_POSE_MODEL_PATH = _PROJECT_ROOT / "models" / "pose" / "active.pt"
MODEL_POSE_PATH = os.environ.get(
    "CRICKET_POSE_MODEL_PATH",
    str(_ACTIVE_POSE_MODEL_PATH) if _ACTIVE_POSE_MODEL_PATH.is_file() else "yolov8n-pose.pt",
)
MODEL_BALL_PATH = "tracknetv3.pt"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v"})

# COCO-compatible YOLOv8-pose keypoint indices.
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
KEYPOINT_COUNT = 17

# MediaPipe Pose landmark indices corresponding to the COCO keypoint order above.
MEDIAPIPE_COCO_LANDMARK_INDICES = (
    0,
    2,
    5,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
)

# These are coaching-screening thresholds, not medical or legal diagnoses.
KNEE_BRACE_MIN_ANGLE_DEG = 150.0
TRUNK_LEAN_WARNING_DEG = 30.0
ELBOW_EXTENSION_WARNING_DEG = 15.0
HIP_SHOULDER_SEPARATION_WARNING_DEG = 40.0
MIXED_ACTION_MIN_SEPARATION_DEG = 20.0
MIXED_ACTION_MAX_SEPARATION_DEG = 50.0
MIN_VALID_BONE_LENGTHS = 3


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved paths used by one pipeline run."""

    output_dir: Path
    calibration_matrix: Path
    delivery_dir: Path

    @classmethod
    def create(cls, output_dir: str | Path, matrix_path: str | Path | None = None) -> "PipelinePaths":
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        matrix = Path(matrix_path) if matrix_path else root / "pitch_homography.npy"
        delivery_dir = root / "deliveries"
        delivery_dir.mkdir(parents=True, exist_ok=True)
        return cls(root, matrix, delivery_dir)


if __name__ == "__main__":
    assert ICC_CREASE_WIDTH_M == 2.64
    assert ICC_CREASE_DEPTH_M == 1.22
    print("Configuration checks passed.")
