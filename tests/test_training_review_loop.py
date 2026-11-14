import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from cricket_coach.phase6_app import dashboard
from cricket_coach.phase6_app.dashboard import _history_rows
from cricket_coach.data_contract import save_phase_payload
from cricket_coach.training.mechanics_model import (
    SEQUENCE_LENGTH,
    _features_from_pose,
    _load_samples,
    mechanics_training_readiness,
    predict_active_mechanics,
    train_mechanics_model,
)
from cricket_coach.training.pose_dataset import (
    MIN_POSE_REVIEWS,
    _yolo_pose_row,
    export_reviewed_pose_dataset,
    split_reviews_by_player,
)
from cricket_coach.training import train_pose
from cricket_coach.training.review_store import (
    KEYPOINT_NAMES,
    initialize_store,
    list_reviews,
    review_stats,
    save_review,
)


def _keypoint_rows(width: int = 100, height: int = 80) -> list[dict]:
    return [
        {
            "index": index,
            "keypoint": name,
            "x_px": float(10 + index),
            "y_px": float(20 + index),
            "visible": True,
            "model confidence": 0.9,
        }
        for index, name in enumerate(KEYPOINT_NAMES)
    ]


class ReviewStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "learning.sqlite3"
        self.metadata_path = self.root / "delivery_data.json"
        self.video_path = self.root / "delivery.mp4"
        self.metadata_path.write_text("{}", encoding="utf-8")
        self.video_path.write_bytes(b"local test video")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _save(self, **overrides) -> str:
        values = {
            "player_id": "player_001",
            "delivery_id": "delivery_001",
            "metadata_path": self.metadata_path,
            "video_path": self.video_path,
            "frame_index": 0,
            "image_width": 100,
            "image_height": 80,
            "keypoint_rows": _keypoint_rows(),
            "mechanics_labels": ["trunk_alignment"],
            "coach_rating": 4,
            "drill_completed": True,
            "drill_helpful": True,
            "notes": "Reviewed after practice.",
            "metrics": {"front_knee_angle_deg": {"min": 155.0}},
            "consent_confirmed": True,
            "database_path": self.database_path,
        }
        values.update(overrides)
        return save_review(**values)

    def test_saves_and_lists_consent_based_review(self) -> None:
        review_id = self._save()

        reviews = list_reviews(database_path=self.database_path)
        stats = review_stats(database_path=self.database_path)

        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["review_id"], review_id)
        self.assertEqual(reviews[0]["player_id"], "player_001")
        self.assertEqual(reviews[0]["mechanics_labels"], ["trunk_alignment"])
        self.assertTrue(reviews[0]["keypoints"][0]["visible"])
        self.assertEqual(stats, {"reviews": 1, "players": 1, "tagged_reviews": 1})

    def test_repeated_frame_review_does_not_inflate_training_counts(self) -> None:
        self._save()
        self._save(notes="Corrected coordinates.")

        stats = review_stats(database_path=self.database_path)

        self.assertEqual(stats["reviews"], 1)
        self.assertEqual(stats["tagged_reviews"], 1)

    def test_rejects_unconsented_or_identifiable_player_ids(self) -> None:
        with self.assertRaises(PermissionError):
            self._save(consent_confirmed=False)
        with self.assertRaisesRegex(ValueError, "anonymous player ID"):
            self._save(player_id="Jordan Smith")

    def test_rejects_out_of_frame_keypoint_corrections(self) -> None:
        rows = _keypoint_rows()
        rows[0]["x_px"] = 101.0
        with self.assertRaisesRegex(ValueError, "within the 100x80 frame"):
            self._save(keypoint_rows=rows)

    def test_schema_v1_migrates_drill_text_and_row_version(self) -> None:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """
                CREATE TABLE delivery_reviews (
                    review_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    player_id TEXT NOT NULL,
                    delivery_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO delivery_reviews VALUES (?, ?, ?, ?, ?)",
                ("legacy-review", 1, "player_001", "delivery_001", "2024-01-01T00:00:00+00:00"),
            )
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
        finally:
            connection.close()

        initialize_store(self.database_path)

        connection = sqlite3.connect(self.database_path)
        try:
            columns = {
                row[1]: row
                for row in connection.execute("PRAGMA table_info(delivery_reviews)")
            }
            row = connection.execute(
                "SELECT schema_version, drill_text, player_role "
                "FROM delivery_reviews WHERE review_id = ?",
                ("legacy-review",),
            ).fetchone()
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        finally:
            connection.close()
        self.assertIn("drill_text", columns)
        self.assertIn("player_role", columns)
        self.assertEqual(row, (3, "", "unknown"))
        self.assertEqual(version, 3)


class TrainingPreparationTests(unittest.TestCase):
    def test_player_splits_do_not_leak_between_partitions(self) -> None:
        reviews = [
            {"player_id": f"player_{player}", "review_id": f"review_{player}_{index}"}
            for player in range(5)
            for index in range(2)
        ]

        splits = split_reviews_by_player(reviews, seed=3)
        players_by_split = {
            name: {review["player_id"] for review in rows}
            for name, rows in splits.items()
        }

        self.assertTrue(players_by_split["train"])
        self.assertTrue(players_by_split["val"])
        self.assertTrue(players_by_split["test"])
        self.assertFalse(players_by_split["train"] & players_by_split["val"])
        self.assertFalse(players_by_split["train"] & players_by_split["test"])
        self.assertFalse(players_by_split["val"] & players_by_split["test"])

    def test_pose_export_refuses_insufficient_reviewed_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "reviews.sqlite3"
            with self.assertRaisesRegex(ValueError, str(MIN_POSE_REVIEWS)):
                export_reviewed_pose_dataset(
                    Path(directory) / "dataset",
                    database_path=database_path,
                )

    def test_pose_export_creates_player_grouped_yolo_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "reviews.sqlite3"
            metadata_path = root / "delivery_data.json"
            video_path = root / "delivery.mp4"
            metadata_path.write_text("{}", encoding="utf-8")
            video_path.write_bytes(b"placeholder")
            for player in range(5):
                for delivery in range(4):
                    save_review(
                        player_id=f"player_{player}",
                        delivery_id=f"delivery_{player}_{delivery}",
                        metadata_path=metadata_path,
                        video_path=video_path,
                        frame_index=delivery,
                        image_width=100,
                        image_height=80,
                        keypoint_rows=_keypoint_rows(),
                        mechanics_labels=[],
                        coach_rating=None,
                        drill_completed=False,
                        drill_helpful=None,
                        notes="",
                        metrics={},
                        consent_confirmed=True,
                        database_path=database_path,
                    )

            with patch(
                "cricket_coach.training.pose_dataset._read_video_frame",
                return_value=np.zeros((80, 100, 3), dtype=np.uint8),
            ):
                exported = export_reviewed_pose_dataset(
                    root / "pose_dataset",
                    database_path=database_path,
                    seed=2,
                )

            manifest = json.loads(Path(exported["manifest_path"]).read_text(encoding="utf-8"))
            player_splits = {}
            for entry in manifest:
                player_splits.setdefault(entry["player_id"], set()).add(entry["split"])
            self.assertEqual(exported["review_count"], 20)
            self.assertEqual(len(player_splits), 5)
            self.assertTrue(all(len(splits) == 1 for splits in player_splits.values()))
            self.assertTrue(Path(exported["dataset_yaml"]).is_file())

    def test_yolo_pose_label_contains_box_and_all_keypoints(self) -> None:
        review = {
            "review_id": "test",
            "image_width": 100,
            "image_height": 80,
            "keypoints": [
                {
                    "index": index,
                    "keypoint": name,
                    "x_px": float(10 + index),
                    "y_px": float(20 + index),
                    "visible": index < 5,
                }
                for index, name in enumerate(KEYPOINT_NAMES)
            ],
        }

        values = _yolo_pose_row(review).split()

        self.assertEqual(len(values), 5 + len(KEYPOINT_NAMES) * 3)
        self.assertEqual(values[0], "0")
        self.assertEqual(values[-1], "0")

    def test_temporal_features_are_fixed_size_and_finite(self) -> None:
        poses = np.ones((8, 17, 3), dtype=np.float32)
        poses[..., 0] *= 640
        poses[..., 1] *= 360
        poses[..., 2] = 0.8
        poses[0, 0] = np.nan

        features = _features_from_pose(poses, 1280, 720)

        self.assertEqual(features.shape, (SEQUENCE_LENGTH, 17 * 3))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(float(features[0, 2]), 0.0)

    def test_mechanics_training_refuses_empty_label_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "reviews.sqlite3"
            with self.assertRaisesRegex(ValueError, "different player IDs"):
                _load_samples(database_path, seed=17)

    def test_mechanics_readiness_uses_reviewed_delivery_and_player_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "reviews.sqlite3"
            metadata_path = root / "delivery_data.json"
            video_path = root / "delivery.mp4"
            metadata_path.write_text("{}", encoding="utf-8")
            video_path.write_bytes(b"placeholder")
            for player in range(5):
                for delivery in range(4):
                    save_review(
                        player_id=f"player_{player}",
                        delivery_id=f"delivery_{player}_{delivery}",
                        metadata_path=metadata_path,
                        video_path=video_path,
                        frame_index=delivery,
                        image_width=100,
                        image_height=80,
                        keypoint_rows=_keypoint_rows(),
                        mechanics_labels=["trunk_alignment"],
                        coach_rating=4,
                        drill_completed=False,
                        drill_helpful=None,
                        notes="",
                        metrics={},
                        consent_confirmed=True,
                        database_path=database_path,
                    )

            readiness = mechanics_training_readiness(database_path)

            self.assertTrue(readiness["ready"])
            self.assertEqual(readiness["counts"]["trunk_alignment"], 20)
            self.assertEqual(readiness["player_counts"]["trunk_alignment"], 5)

    def test_absent_promoted_mechanics_model_does_not_invent_predictions(self) -> None:
        predictions = predict_active_mechanics(np.zeros((2, 17, 3)), 1280, 720)
        self.assertIsNone(predictions)

    def test_history_rows_include_player_progress_fields(self) -> None:
        rows = _history_rows(
            [
                {
                    "created_at": "2026-01-02T03:04:05+00:00",
                    "delivery_id": "delivery_001",
                    "frame_index": 7,
                    "mechanics_labels": ["trunk_alignment"],
                    "coach_rating": 4,
                    "drill_completed": True,
                    "drill_helpful": True,
                    "metrics": {
                        "front_knee_angle_deg": {"min": 150.0},
                        "lateral_trunk_tilt_deg": {"max": 22.0},
                    },
                }
            ]
        )

        self.assertEqual(rows[0]["frame"], 8)
        self.assertEqual(rows[0]["mechanics tags"], "trunk_alignment")
        self.assertEqual(rows[0]["front knee min (deg)"], 150.0)
        self.assertEqual(rows[0]["trunk tilt max (deg)"], 22.0)

    def test_dashboard_disables_training_until_review_data_is_ready(self) -> None:
        st = MagicMock()
        st.button.return_value = False
        st.text_input.return_value = ""
        with patch.object(
            dashboard,
            "review_stats",
            return_value={"reviews": 0, "players": 0, "tagged_reviews": 0},
        ), patch.object(
            dashboard,
            "mechanics_training_readiness",
            return_value={
                "ready": False,
                "ready_labels": [],
                "counts": {},
                "player_counts": {},
                "minimum_examples_per_label": 20,
                "minimum_players_per_label": 5,
            },
        ):
            dashboard._render_training_review(
                st,
                Path("delivery_data.json"),
                {"delivery_id": "delivery_001"},
                {},
                Path("missing.mp4"),
                0,
            )

        self.assertEqual(len(st.button.call_args_list), 2)
        self.assertTrue(all(call.kwargs["disabled"] for call in st.button.call_args_list))

    def test_pose_trainer_promotes_only_when_both_gates_pass(self) -> None:
        class FakeYolo:
            instances = []

            def __init__(self, model_path: str) -> None:
                self.model_path = model_path
                self.validation_scores = (
                    [0.50, 0.55]
                    if not model_path.endswith("best.pt")
                    else [0.60, 0.45]
                )
                self.instances.append(self)

            def val(self, **kwargs):
                score = self.validation_scores.pop(0)
                return SimpleNamespace(pose=SimpleNamespace(map=score))

            def train(self, **kwargs):
                run = Path(kwargs["project"]) / kwargs["name"] / "weights"
                run.mkdir(parents=True)
                (run / "best.pt").write_bytes(b"candidate")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_model = root / "base.pt"
            data_yaml = root / "data.yaml"
            base_model.write_bytes(b"baseline")
            data_yaml.write_text("names:\n  0: bowler\n", encoding="utf-8")
            active_root = root / "models"
            (active_root / "pose").mkdir(parents=True)
            (active_root / "pose" / "active.pt").write_bytes(b"old-active")
            fake_ultralytics = SimpleNamespace(YOLO=FakeYolo)
            with patch.dict("sys.modules", {"ultralytics": fake_ultralytics}), patch.object(
                train_pose, "PROJECT_ROOT", root
            ):
                report = train_pose.train_pose_model(
                    dataset_yaml=data_yaml,
                    base_model=base_model,
                    project_dir=root / "runs",
                    run_name="candidate",
                    epochs=1,
                    promote=True,
                )

            self.assertTrue(report["promoted"])
            self.assertFalse(report["test_not_regressed"])
            self.assertTrue(Path(report["active_model"]).is_file())
            self.assertEqual(Path(report["active_model"]).read_bytes(), b"candidate")
            self.assertTrue(Path(report["archived_model"]).is_file())

    def test_mechanics_gru_trains_on_reviewed_grouped_sequences(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is an optional dependency.")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "reviews.sqlite3"
            metadata_path = root / "delivery_data.json"
            video_path = root / "delivery.mp4"
            video_path.write_bytes(b"placeholder")
            poses = np.zeros((8, 17, 3), dtype=np.float32)
            poses[..., 0] = 50
            poses[..., 1] = 40
            poses[..., 2] = 0.9
            save_phase_payload(metadata_path, {}, {"poses_2d": poses})
            for player in range(5):
                for delivery in range(4):
                    save_review(
                        player_id=f"player_{player}",
                        delivery_id=f"delivery_{player}_{delivery}",
                        metadata_path=metadata_path,
                        video_path=video_path,
                        frame_index=delivery,
                        image_width=100,
                        image_height=80,
                        keypoint_rows=_keypoint_rows(),
                        mechanics_labels=["trunk_alignment"],
                        coach_rating=4,
                        drill_completed=False,
                        drill_helpful=None,
                        notes="",
                        metrics={},
                        consent_confirmed=True,
                        database_path=database_path,
                    )

            with patch(
                "cricket_coach.training.mechanics_model.ACTIVE_MODEL_PATH",
                root / "active" / "mechanics.pt",
            ):
                report = train_mechanics_model(
                    database_path=database_path,
                    output_dir=root / "candidates",
                    epochs=1,
                    seed=5,
                    promote=False,
                )
                active_path = root / "active" / "mechanics.pt"
                active_path.parent.mkdir(parents=True, exist_ok=True)
                active_path.write_bytes(Path(report["candidate_model"]).read_bytes())
                prediction = predict_active_mechanics(poses, 100, 80)

            self.assertIn("candidate_validation_macro_f1", report["metrics"])
            self.assertFalse(report["promoted"])
            self.assertEqual(prediction["status"], "complete")
            self.assertIn("trunk_alignment", prediction["probabilities"])


if __name__ == "__main__":
    unittest.main()
