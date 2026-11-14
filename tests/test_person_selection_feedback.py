import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cricket_coach.data_contract import load_phase_payload, save_phase_payload
from cricket_coach.phase2_tracking.pose_3d import Pose3DProcessor
from cricket_coach.training.review_store import list_reviews, save_review


def _keypoint_rows() -> list[dict]:
    names = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )
    return [
        {
            "index": index,
            "keypoint": name,
            "x_px": 10.0 + index,
            "y_px": 20.0 + index,
            "visible": True,
        }
        for index, name in enumerate(names)
    ]


class PersonDetectionTests(unittest.TestCase):
    def test_candidates_are_ordered_by_box_area_and_keep_boxes(self) -> None:
        boxes = SimpleNamespace(
            xyxy=np.array([[0, 0, 10, 10], [5, 5, 45, 55]], dtype=np.float32)
        )
        keypoints = SimpleNamespace(
            xy=np.stack(
                (
                    np.full((17, 2), 1.0, dtype=np.float32),
                    np.full((17, 2), 2.0, dtype=np.float32),
                )
            ),
            conf=np.ones((2, 17), dtype=np.float32),
        )

        poses, ordered_boxes = Pose3DProcessor._person_detections(
            [SimpleNamespace(boxes=boxes, keypoints=keypoints)]
        )

        self.assertEqual(poses.shape, (2, 17, 3))
        np.testing.assert_array_equal(poses[0, :, 0], np.full(17, 2.0))
        np.testing.assert_array_equal(ordered_boxes[0], np.array([5, 5, 45, 55]))

    def test_pipeline_persists_person_candidates_for_review(self) -> None:
        class Capture:
            def __init__(self) -> None:
                self.frames = [np.zeros((60, 80, 3), dtype=np.uint8)]

            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                return 30.0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if not self.frames:
                    return False, None
                return True, self.frames.pop(0)

            def release(self) -> None:
                pass

        class PoseModel:
            def __call__(self, frame: np.ndarray, verbose: bool = False) -> list:
                xy = np.zeros((2, 17, 2), dtype=np.float32)
                xy[0] = 1.0
                xy[1] = 2.0
                return [
                    SimpleNamespace(
                        boxes=SimpleNamespace(
                            xyxy=np.array(
                                [[0, 0, 10, 10], [10, 5, 50, 55]],
                                dtype=np.float32,
                            )
                        ),
                        keypoints=SimpleNamespace(
                            xy=xy,
                            conf=np.ones((2, 17), dtype=np.float32),
                        ),
                    )
                ]

        processor = Pose3DProcessor(
            pose_model=PoseModel(),
            lifter=lambda poses: np.zeros((len(poses), 17, 3), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "delivery_data.json"
            save_phase_payload(metadata_path, {"delivery_id": "delivery"})
            with patch(
                "cricket_coach.phase2_tracking.pose_3d.cv2.VideoCapture",
                return_value=Capture(),
            ):
                metadata, arrays = processor.process("unused.mp4", metadata_path)
            saved, _ = load_phase_payload(metadata_path)

        self.assertEqual(int(arrays["person_candidate_counts"][0]), 2)
        np.testing.assert_array_equal(
            arrays["person_boxes_xyxy"][0, 0],
            np.array([10, 5, 50, 55], dtype=np.float32),
        )
        self.assertEqual(arrays["poses_2d_candidates"].shape, (1, 2, 17, 3))
        self.assertEqual(saved["pose_frame_count"], metadata["pose_frame_count"])


class PersonSelectionFeedbackTests(unittest.TestCase):
    def test_saved_review_keeps_role_and_selected_person_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "delivery_data.json"
            video_path = root / "delivery.mp4"
            metadata_path.write_text("{}", encoding="utf-8")
            video_path.write_bytes(b"test video")

            save_review(
                player_id="player_002",
                delivery_id="delivery_001",
                metadata_path=metadata_path,
                video_path=video_path,
                frame_index=3,
                image_width=100,
                image_height=80,
                keypoint_rows=_keypoint_rows(),
                mechanics_labels=(),
                coach_rating=None,
                drill_completed=False,
                drill_helpful=None,
                notes="",
                metrics={},
                consent_confirmed=True,
                database_path=root / "reviews.sqlite3",
                player_role="bowler",
                person_detection_index=1,
                person_box=[10, 5, 50, 75],
            )

            review = list_reviews(database_path=root / "reviews.sqlite3")[0]

        self.assertEqual(review["player_role"], "bowler")
        self.assertEqual(review["person_detection_index"], 1)
        self.assertEqual(review["person_box"], [10.0, 5.0, 50.0, 75.0])


if __name__ == "__main__":
    unittest.main()
