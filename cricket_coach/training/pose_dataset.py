"""Export reviewed pixel keypoints to leakage-safe Ultralytics pose datasets."""

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cricket_coach.training.review_store import list_reviews

MIN_POSE_REVIEWS = 20
MIN_PLAYER_GROUPS = 5
POSE_FLIP_INDEX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


def split_reviews_by_player(
    reviews: list[dict[str, Any]],
    seed: int = 17,
) -> dict[str, list[dict[str, Any]]]:
    """Split entire player groups into train, validation, and test partitions."""
    player_ids = sorted({review["player_id"] for review in reviews})
    if len(player_ids) < MIN_PLAYER_GROUPS:
        raise ValueError(
            f"Need reviewed examples from at least {MIN_PLAYER_GROUPS} different player IDs "
            f"before making train/validation/test splits; found {len(player_ids)}."
        )
    random.Random(seed).shuffle(player_ids)
    test_count = max(1, round(len(player_ids) * 0.15))
    validation_count = max(1, round(len(player_ids) * 0.15))
    if test_count + validation_count >= len(player_ids):
        test_count = 1
        validation_count = 1
    assignments = {
        "test": set(player_ids[:test_count]),
        "val": set(player_ids[test_count : test_count + validation_count]),
        "train": set(player_ids[test_count + validation_count :]),
    }
    return {
        split: [
            review
            for review in reviews
            if review["player_id"] in player_ids_for_split
        ]
        for split, player_ids_for_split in assignments.items()
    }


def _read_video_frame(video_path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open reviewed video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Unable to decode frame {frame_index} from {video_path}.")
        return frame
    finally:
        capture.release()


def _yolo_pose_row(review: dict[str, Any]) -> str:
    points = review["keypoints"]
    width = int(review["image_width"])
    height = int(review["image_height"])
    visible = [point for point in points if point["visible"]]
    if len(visible) < 3:
        raise ValueError(f"Review {review['review_id']} has fewer than three visible keypoints.")

    xs = np.asarray([point["x_px"] / width for point in visible], dtype=np.float64)
    ys = np.asarray([point["y_px"] / height for point in visible], dtype=np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_padding = max((x_max - x_min) * 0.05, 0.01)
    y_padding = max((y_max - y_min) * 0.05, 0.01)
    x_min, x_max = max(0.0, x_min - x_padding), min(1.0, x_max + x_padding)
    y_min, y_max = max(0.0, y_min - y_padding), min(1.0, y_max + y_padding)
    box = (
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        max(x_max - x_min, 1.0 / width),
        max(y_max - y_min, 1.0 / height),
    )

    values = ["0", *(f"{value:.6f}" for value in box)]
    for point in points:
        if point["visible"]:
            values.extend(
                (
                    f"{point['x_px'] / width:.6f}",
                    f"{point['y_px'] / height:.6f}",
                    "2",
                )
            )
        else:
            values.extend(("0", "0", "0"))
    return " ".join(values)


def export_reviewed_pose_dataset(
    output_dir: str | Path,
    database_path: str | Path | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Write consented reviewed frames to YOLO pose format, grouped by player."""
    latest_frames: dict[tuple[str, str, int], dict[str, Any]] = {}
    for review in list_reviews(database_path=database_path, limit=100000):
        key = (review["player_id"], review["video_path"], review["frame_index"])
        if key not in latest_frames or review["created_at"] > latest_frames[key]["created_at"]:
            latest_frames[key] = review
    reviews = list(latest_frames.values())
    if len(reviews) < MIN_POSE_REVIEWS:
        raise ValueError(
            f"Need at least {MIN_POSE_REVIEWS} reviewed frames before training; found {len(reviews)}."
        )
    splits = split_reviews_by_player(reviews, seed=seed)
    for split_name in ("train", "val", "test"):
        if not splits[split_name]:
            raise ValueError(f"The player-grouped {split_name} split is empty.")

    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Dataset output is not empty: {root}. Choose a new versioned output directory."
        )
    root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for split_name, split_reviews in splits.items():
        image_dir = root / "images" / split_name
        label_dir = root / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for review in split_reviews:
            frame = _read_video_frame(review["video_path"], review["frame_index"])
            actual_height, actual_width = frame.shape[:2]
            if (actual_width, actual_height) != (
                int(review["image_width"]),
                int(review["image_height"]),
            ):
                raise ValueError(
                    f"Reviewed frame dimensions changed for {review['review_id']}; "
                    "re-review this frame before exporting."
                )
            image_path = image_dir / f"{review['review_id']}.jpg"
            label_path = label_dir / f"{review['review_id']}.txt"
            if not cv2.imwrite(str(image_path), frame):
                raise OSError(f"Could not write training image: {image_path}")
            label_path.write_text(_yolo_pose_row(review) + "\n", encoding="utf-8")
            manifest.append(
                {
                    "review_id": review["review_id"],
                    "player_id": review["player_id"],
                    "delivery_id": review["delivery_id"],
                    "frame_index": review["frame_index"],
                    "split": split_name,
                    "image": str(image_path.relative_to(root)),
                    "label": str(label_path.relative_to(root)),
                }
            )

    root_for_yaml = str(root).replace("\\", "/")
    yaml_lines = [
        f'path: "{root_for_yaml}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "kpt_shape: [17, 3]",
        f"flip_idx: [{', '.join(str(index) for index in POSE_FLIP_INDEX)}]",
        "names:",
        "  0: bowler",
    ]
    (root / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "dataset_yaml": str(root / "dataset.yaml"),
        "review_count": len(manifest),
        "player_count": len({review["player_id"] for review in reviews}),
        "split_counts": {name: len(split) for name, split in splits.items()},
        "manifest_path": str(root / "manifest.json"),
    }


if __name__ == "__main__":
    from cricket_coach.training.review_store import default_database_path

    result = export_reviewed_pose_dataset(
        Path("data") / "training" / "pose_dataset_v1",
        default_database_path(),
    )
    print(json.dumps(result, indent=2))
