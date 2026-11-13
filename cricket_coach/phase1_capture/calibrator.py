"""Automatic four-point pitch calibration with an optional interactive tool."""

import logging
import math
from pathlib import Path
from itertools import combinations
from typing import Any

import cv2
import numpy as np

from cricket_coach.config import (
    CALIBRATION_FRAME_SKIP,
    CALIBRATION_HOUGH_ANGLE_RESOLUTION_DEG,
    CALIBRATION_CANNY_HIGH_THRESHOLD,
    CALIBRATION_CANNY_LOW_THRESHOLD,
    CALIBRATION_HOUGH_MAX_LINE_GAP_RATIO,
    CALIBRATION_HOUGH_MAX_LINE_GAP_PX,
    CALIBRATION_HOUGH_MIN_LINE_LENGTH_RATIO,
    CALIBRATION_HOUGH_MIN_LINE_LENGTH_PX,
    CALIBRATION_HOUGH_RHO_RESOLUTION_PX,
    CALIBRATION_HOUGH_THRESHOLD,
    CALIBRATION_IDENTITY_MATRIX_TOLERANCE,
    CALIBRATION_LINE_MERGE_DISTANCE_RATIO,
    CALIBRATION_MAX_CANDIDATE_LINES_PER_FAMILY,
    CALIBRATION_MIN_FAMILY_ANGLE_SEPARATION_DEG,
    CALIBRATION_MIN_INTERSECTION_DETERMINANT,
    CALIBRATION_MIN_LINE_SEPARATION_RATIO,
    CALIBRATION_MIN_QUADRILATERAL_AREA_RATIO,
    CALIBRATION_ORIENTATION_BIN_COUNT,
    CALIBRATION_ORIENTATION_TOLERANCE_DEG,
    CALIBRATION_WHITE_SATURATION_MAX,
    CALIBRATION_WHITE_VALUE_MAX,
    CALIBRATION_WHITE_VALUE_MIN,
    CALIBRATION_WINDOW_TITLE,
    ICC_CREASE_DEPTH_M,
    ICC_CREASE_WIDTH_M,
)

LOGGER = logging.getLogger(__name__)


class PitchCalibrator:
    """Map visible crease corners to the pitch plane without requiring a GUI."""

    LABELS = ("Popping-Left", "Popping-Right", "Bowling-Left", "Bowling-Right")

    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []
        self.frame: np.ndarray | None = None

    @staticmethod
    def sample_frames(video_path: str | Path, every_n_frames: int = CALIBRATION_FRAME_SKIP) -> list[tuple[int, np.ndarray]]:
        """Return frame-indexed samples without retaining the complete source video."""
        if every_n_frames < 1:
            raise ValueError("Frame-skip interval must be at least one.")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Unable to open video: {video_path}")
        sampled: list[tuple[int, np.ndarray]] = []
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % every_n_frames == 0:
                    sampled.append((index, frame))
                index += 1
        finally:
            capture.release()
        return sampled

    def calibrate(
        self,
        video_path: str | Path,
        matrix_path: str | Path,
        interactive: bool = False,
    ) -> np.ndarray:
        """Calibrate automatically by default; GUI point selection is explicit opt-in."""
        if not interactive:
            return self.calibrate_automatically(video_path, matrix_path)
        return self.calibrate_interactively(video_path, matrix_path)

    def calibrate_automatically(self, video_path: str | Path, matrix_path: str | Path) -> np.ndarray:
        """Detect white crease lines or save an identity fallback matrix with a warning."""
        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise ValueError(f"Unable to read first frame from {video_path}")
        points = self._detect_crease_corners(frame)
        if points is None:
            matrix = np.eye(3, dtype=np.float64)
            LOGGER.warning(
                "Could not confidently detect a crease-box quadrilateral in %s; "
                "saving identity fallback matrix. Pitch measurements will not be metric.",
                video_path,
            )
            path = Path(matrix_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
            return matrix
        try:
            return self.compute_homography(points, matrix_path)
        except ValueError:
            matrix = np.eye(3, dtype=np.float64)
            LOGGER.warning(
                "Detected crease lines in %s but could not form a stable homography; "
                "saving identity fallback matrix. Pitch measurements will not be metric.",
                video_path,
            )
            path = Path(matrix_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
            return matrix

    def calibrate_interactively(self, video_path: str | Path, matrix_path: str | Path) -> np.ndarray:
        """Prompt for four corners in order and persist a pixel-to-meter homography."""
        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise ValueError(f"Unable to read first frame from {video_path}")
        self.frame = frame
        self.points.clear()
        cv2.namedWindow(CALIBRATION_WINDOW_TITLE)
        cv2.setMouseCallback(CALIBRATION_WINDOW_TITLE, self._on_click)
        try:
            while True:
                cv2.imshow(CALIBRATION_WINDOW_TITLE, self._draw())
                key = cv2.waitKey(20) & 0xFF
                if key == ord("r"):
                    self.points.clear()
                elif key in (10, 13) and len(self.points) == len(self.LABELS):
                    break
                elif key == 27:
                    raise RuntimeError("Pitch calibration cancelled.")
        finally:
            cv2.destroyWindow(CALIBRATION_WINDOW_TITLE)
        return self.compute_homography(self.points, matrix_path)

    @classmethod
    def _detect_crease_corners(cls, frame: np.ndarray) -> np.ndarray | None:
        """Find a plausible four-line crease box using white-pixel Hough segments."""
        height, width = frame.shape[:2]
        min_dimension = min(height, width)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.asarray([0, 0, CALIBRATION_WHITE_VALUE_MIN], dtype=np.uint8),
            np.asarray([179, CALIBRATION_WHITE_SATURATION_MAX, CALIBRATION_WHITE_VALUE_MAX], dtype=np.uint8),
        )
        edges = cv2.Canny(
            white_mask,
            CALIBRATION_CANNY_LOW_THRESHOLD,
            CALIBRATION_CANNY_HIGH_THRESHOLD,
        )
        segments = cv2.HoughLinesP(
            edges,
            rho=CALIBRATION_HOUGH_RHO_RESOLUTION_PX,
            theta=np.deg2rad(CALIBRATION_HOUGH_ANGLE_RESOLUTION_DEG),
            threshold=CALIBRATION_HOUGH_THRESHOLD,
            minLineLength=max(
                CALIBRATION_HOUGH_MIN_LINE_LENGTH_PX,
                int(min_dimension * CALIBRATION_HOUGH_MIN_LINE_LENGTH_RATIO),
            ),
            maxLineGap=max(
                CALIBRATION_HOUGH_MAX_LINE_GAP_PX,
                int(min_dimension * CALIBRATION_HOUGH_MAX_LINE_GAP_RATIO),
            ),
        )
        if segments is None:
            return None

        records: list[tuple[float, float, float, float, float]] = []
        histogram = np.zeros(CALIBRATION_ORIENTATION_BIN_COUNT, dtype=np.float64)
        for segment in segments.reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in segment)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < min_dimension * CALIBRATION_HOUGH_MIN_LINE_LENGTH_RATIO:
                continue
            angle = math.atan2(dy, dx)
            if angle >= math.pi / 2:
                angle -= math.pi
            elif angle < -math.pi / 2:
                angle += math.pi
            normal_x, normal_y = -math.sin(angle), math.cos(angle)
            rho = normal_x * (x1 + x2) / 2 + normal_y * (y1 + y2) / 2
            records.append((angle, rho, length, x1, y1))
            bin_index = int((angle % math.pi) / math.pi * CALIBRATION_ORIENTATION_BIN_COUNT) % CALIBRATION_ORIENTATION_BIN_COUNT
            histogram[bin_index] += length
        if len(records) < 4:
            return None

        first_bin = int(np.argmax(histogram))
        orientation_bins = sorted(
            range(len(histogram)),
            key=lambda index: histogram[index],
            reverse=True,
        )
        tolerance = math.radians(CALIBRATION_ORIENTATION_TOLERANCE_DEG)
        second_bin = next(
            (
                index
                for index in orientation_bins
                if histogram[index] > 0
                and cls._orientation_distance(
                    (index + 0.5) * math.pi / len(histogram),
                    (first_bin + 0.5) * math.pi / len(histogram),
                )
                >= math.radians(CALIBRATION_MIN_FAMILY_ANGLE_SEPARATION_DEG)
            ),
            None,
        )
        if second_bin is None:
            return None
        centers = (
            (first_bin + 0.5) * math.pi / len(histogram),
            (second_bin + 0.5) * math.pi / len(histogram),
        )
        families = [
            cls._group_parallel_lines(records, center, tolerance, min_dimension * CALIBRATION_LINE_MERGE_DISTANCE_RATIO)
            for center in centers
        ]
        if any(len(family) < 2 for family in families):
            return None

        candidates: list[tuple[float, np.ndarray]] = []
        min_separation = min_dimension * CALIBRATION_MIN_LINE_SEPARATION_RATIO
        max_lines = CALIBRATION_MAX_CANDIDATE_LINES_PER_FAMILY
        family_a, family_b = families[0][:max_lines], families[1][:max_lines]
        for pair_a in combinations(family_a, 2):
            if abs(pair_a[0][2] - pair_a[1][2]) < min_separation:
                continue
            for pair_b in combinations(family_b, 2):
                if abs(pair_b[0][2] - pair_b[1][2]) < min_separation:
                    continue
                corners = []
                for line_a, line_b in (
                    (pair_a[0], pair_b[0]),
                    (pair_a[0], pair_b[1]),
                    (pair_a[1], pair_b[1]),
                    (pair_a[1], pair_b[0]),
                ):
                    intersection = cls._line_intersection(line_a, line_b)
                    if intersection is None:
                        break
                    corners.append(intersection)
                if len(corners) != 4:
                    continue
                polygon = np.asarray(corners, dtype=np.float32)
                if (
                    np.any(polygon[:, 0] < 0)
                    or np.any(polygon[:, 0] >= width)
                    or np.any(polygon[:, 1] < 0)
                    or np.any(polygon[:, 1] >= height)
                    or not cv2.isContourConvex(polygon.reshape(-1, 1, 2).astype(np.int32))
                ):
                    continue
                area_ratio = abs(cv2.contourArea(polygon)) / (width * height)
                if area_ratio < CALIBRATION_MIN_QUADRILATERAL_AREA_RATIO:
                    continue
                support = sum(line[3] for line in pair_a + pair_b)
                candidates.append((support * math.sqrt(area_ratio), polygon))
        if not candidates:
            return None
        best = max(candidates, key=lambda candidate: candidate[0])[1]
        return cls._order_corners(best)

    @staticmethod
    def _orientation_distance(first: float, second: float) -> float:
        """Smallest angle between two undirected line orientations."""
        difference = abs(first - second) % math.pi
        return min(difference, math.pi - difference)

    @classmethod
    def _group_parallel_lines(
        cls,
        records: list[tuple[float, float, float, float, float]],
        center_angle: float,
        tolerance: float,
        merge_distance: float,
    ) -> list[tuple[float, float, float, float]]:
        """Merge near-collinear Hough segments into weighted (a,b,c) lines."""
        selected = [record for record in records if cls._orientation_distance(record[0], center_angle) <= tolerance]
        selected.sort(key=lambda record: record[1])
        clusters: list[list[tuple[float, float, float, float, float]]] = []
        for record in selected:
            if not clusters or abs(record[1] - np.average([part[1] for part in clusters[-1]], weights=[part[2] for part in clusters[-1]])) > merge_distance:
                clusters.append([record])
            else:
                clusters[-1].append(record)
        lines: list[tuple[float, float, float, float]] = []
        for cluster in clusters:
            weights = np.asarray([record[2] for record in cluster], dtype=np.float64)
            angles = np.asarray([record[0] for record in cluster], dtype=np.float64)
            angle = (
                math.atan2(
                    float(np.average(np.sin(2 * angles), weights=weights)),
                    float(np.average(np.cos(2 * angles), weights=weights)),
                )
                / 2
            )
            normal_x, normal_y = -math.sin(angle), math.cos(angle)
            rho = float(np.average([record[1] for record in cluster], weights=weights))
            lines.append((normal_x, normal_y, rho, float(weights.sum())))
        return sorted(lines, key=lambda line: line[3], reverse=True)

    @staticmethod
    def _line_intersection(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> tuple[float, float] | None:
        """Intersect two lines represented as a*x + b*y = c."""
        matrix = np.asarray([[first[0], first[1]], [second[0], second[1]]], dtype=np.float64)
        if abs(np.linalg.det(matrix)) < CALIBRATION_MIN_INTERSECTION_DETERMINANT:
            return None
        point = np.linalg.solve(matrix, np.asarray([first[2], second[2]], dtype=np.float64))
        return float(point[0]), float(point[1])

    @staticmethod
    def _order_corners(points: np.ndarray) -> np.ndarray:
        """Order a detected quadrilateral as popping-left/right, bowling-left/right."""
        by_y = points[np.argsort(points[:, 1])]
        far_pair = by_y[:2][np.argsort(by_y[:2, 0])]
        near_pair = by_y[2:][np.argsort(by_y[2:, 0])]
        return np.asarray([far_pair[0], far_pair[1], near_pair[0], near_pair[1]], dtype=np.float32)

    @staticmethod
    def compute_homography(points: list[tuple[float, float]] | np.ndarray, matrix_path: str | Path | None = None) -> np.ndarray:
        """Calculate a validated 3x3 pixel-to-meter homography."""
        source = np.asarray(points, dtype=np.float32)
        if source.shape != (4, 2) or not np.isfinite(source).all():
            raise ValueError("Exactly four finite (x, y) points are required.")
        target = np.asarray(
            [
                [0.0, 0.0],
                [ICC_CREASE_WIDTH_M, 0.0],
                [0.0, ICC_CREASE_DEPTH_M],
                [ICC_CREASE_WIDTH_M, ICC_CREASE_DEPTH_M],
            ],
            dtype=np.float32,
        )
        matrix, _ = cv2.findHomography(source, target, 0)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("The four points do not produce a valid homography.")
        if matrix_path is not None:
            path = Path(matrix_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        return matrix

    @staticmethod
    def load_homography(matrix_path: str | Path) -> np.ndarray:
        """Read a finite 3x3 homography from a NumPy file."""
        matrix = np.load(matrix_path, allow_pickle=False)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid calibration matrix: {matrix_path}")
        return matrix.astype(np.float64, copy=False)

    @staticmethod
    def is_identity_fallback(matrix: np.ndarray) -> bool:
        """Return whether a matrix is the explicitly non-metric identity fallback."""
        return matrix.shape == (3, 3) and bool(
            np.allclose(
                matrix,
                np.eye(3, dtype=matrix.dtype),
                atol=CALIBRATION_IDENTITY_MATRIX_TOLERANCE,
            )
        )

    def pixel_to_meters(self, u: float, v: float, homography: np.ndarray) -> tuple[float, float]:
        """Project one image point to the calibrated pitch plane."""
        if homography.shape != (3, 3):
            raise ValueError("Homography must be a 3x3 matrix.")
        point = cv2.perspectiveTransform(np.asarray([[[u, v]]], dtype=np.float64), homography)[0, 0]
        return float(point[0]), float(point[1])

    def _on_click(self, event: int, x: int, y: int, flags: int, user_data: Any) -> None:
        del flags, user_data
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def _draw(self) -> np.ndarray:
        if self.frame is None:
            raise RuntimeError("No calibration frame is loaded.")
        output = self.frame.copy()
        for index, point in enumerate(self.points):
            cv2.circle(output, point, 7, (0, 255, 0), -1)
            cv2.putText(output, self.LABELS[index], (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for first, second in ((0, 1), (1, 3), (3, 2), (2, 0)):
            if first < len(self.points) and second < len(self.points):
                cv2.line(output, self.points[first], self.points[second], (255, 200, 0), 2)
        instruction = "Enter: confirm | R: reset | Esc: cancel"
        cv2.putText(output, instruction, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return output


if __name__ == "__main__":
    example = np.asarray([[0, 0], [264, 0], [0, 122], [264, 122]], dtype=np.float32)
    matrix = PitchCalibrator.compute_homography(example)
    assert matrix.shape == (3, 3)
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.polylines(
        synthetic_frame,
        [np.asarray([[150, 150], [490, 150], [470, 330], [170, 330]], dtype=np.int32)],
        True,
        (255, 255, 255),
        5,
    )
    detected_points = PitchCalibrator._detect_crease_corners(synthetic_frame)
    assert detected_points is not None and detected_points.shape == (4, 2)
    detected_homography = PitchCalibrator.compute_homography(detected_points)
    assert np.allclose(
        cv2.perspectiveTransform(detected_points.reshape(1, -1, 2), detected_homography)[0],
        np.asarray([[0, 0], [ICC_CREASE_WIDTH_M, 0], [0, ICC_CREASE_DEPTH_M], [ICC_CREASE_WIDTH_M, ICC_CREASE_DEPTH_M]]),
        atol=0.02,
    )
    assert PitchCalibrator._detect_crease_corners(np.zeros_like(synthetic_frame)) is None
    print("Calibration checks passed.")
