"""Human-supervised temporal classifier for reviewed bowling-mechanics tags."""

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cricket_coach.config import POSE_CONFIDENCE_THRESHOLD
from cricket_coach.data_contract import load_phase_payload
from cricket_coach.training.pose_dataset import MIN_PLAYER_GROUPS, split_reviews_by_player
from cricket_coach.training.review_store import MECHANICS_LABELS, default_database_path, list_reviews

SEQUENCE_LENGTH = 60
GRU_HIDDEN_SIZE = 64
MIN_EXAMPLES_PER_LABEL = 20
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "mechanics" / "active.pt"


def _features_from_pose(
    poses_2d: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    if poses_2d.ndim != 3 or poses_2d.shape[1] != 17 or poses_2d.shape[2] < 2:
        raise ValueError("Mechanics training requires (frames, 17, 2+) YOLO pose arrays.")
    if len(poses_2d) < 2 or image_width <= 0 or image_height <= 0:
        raise ValueError("Mechanics training requires at least two pose frames and valid image dimensions.")
    indices = np.rint(np.linspace(0, len(poses_2d) - 1, SEQUENCE_LENGTH)).astype(np.int64)
    sample = poses_2d[indices, :, :3].copy() if poses_2d.shape[2] >= 3 else poses_2d[indices, :, :2].copy()
    xy = sample[..., :2].astype(np.float32, copy=False)
    valid_xy = np.isfinite(xy).all(axis=-1)
    if sample.shape[-1] >= 3:
        confidence = sample[..., 2].astype(np.float32, copy=False)
        confidence[~np.isfinite(confidence)] = 0.0
        confidence = np.clip(confidence, 0.0, 1.0)
        valid_xy &= confidence >= POSE_CONFIDENCE_THRESHOLD
    else:
        confidence = valid_xy.astype(np.float32)
    xy[~valid_xy] = 0.0
    xy[..., 0] /= float(image_width)
    xy[..., 1] /= float(image_height)
    shoulder_indices = (5, 6)
    hip_indices = (11, 12)
    for frame_index in range(SEQUENCE_LENGTH):
        if valid_xy[frame_index, shoulder_indices].all() and valid_xy[frame_index, hip_indices].all():
            shoulder_center = xy[frame_index, shoulder_indices].mean(axis=0)
            hip_center = xy[frame_index, hip_indices].mean(axis=0)
            torso_scale = float(np.linalg.norm(shoulder_center - hip_center))
            if np.isfinite(torso_scale) and torso_scale > np.finfo(np.float32).eps:
                body_center = (shoulder_center + hip_center) / 2
                xy[frame_index] = np.clip(
                    (xy[frame_index] - body_center) / torso_scale,
                    -4.0,
                    4.0,
                )
    xy[~valid_xy] = 0.0
    confidence[~valid_xy] = 0.0
    features = np.concatenate((xy, confidence[..., None]), axis=-1)
    return features.reshape(SEQUENCE_LENGTH, 17 * 3)


def _unique_delivery_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for review in reviews:
        if not review["mechanics_labels"]:
            continue
        key = (review["player_id"], review["delivery_id"])
        if key not in latest or review["created_at"] > latest[key]["created_at"]:
            latest[key] = review
    return list(latest.values())


def mechanics_training_readiness(
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Report which reviewed tags meet the minimum pilot-training coverage."""
    reviews = _unique_delivery_reviews(list_reviews(database_path=database_path, limit=100000))
    counts = {label: 0 for label in MECHANICS_LABELS}
    players: dict[str, set[str]] = {label: set() for label in counts}
    for review in reviews:
        for label in review["mechanics_labels"]:
            if label in counts:
                counts[label] += 1
                players[label].add(review["player_id"])
    ready_labels = [
        label
        for label in counts
        if label != "other"
        and counts[label] >= MIN_EXAMPLES_PER_LABEL
        and len(players[label]) >= MIN_PLAYER_GROUPS
    ]
    return {
        "ready": bool(ready_labels),
        "ready_labels": ready_labels,
        "counts": counts,
        "player_counts": {label: len(ids) for label, ids in players.items()},
        "minimum_examples_per_label": MIN_EXAMPLES_PER_LABEL,
        "minimum_players_per_label": MIN_PLAYER_GROUPS,
    }


def _f1_macro(targets: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for index in range(targets.shape[1]):
        truth = targets[:, index].astype(bool)
        predicted = predictions[:, index].astype(bool)
        true_positive = int(np.logical_and(truth, predicted).sum())
        false_positive = int(np.logical_and(~truth, predicted).sum())
        false_negative = int(np.logical_and(truth, ~predicted).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _torch_model(torch: Any, label_count: int) -> Any:
    class PoseSequenceClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = torch.nn.GRU(
                input_size=17 * 3,
                hidden_size=GRU_HIDDEN_SIZE,
                batch_first=True,
            )
            self.output = torch.nn.Linear(GRU_HIDDEN_SIZE, label_count)

        def forward(self, sequence: Any) -> Any:
            _, hidden = self.recurrent(sequence)
            return self.output(hidden[-1])

    return PoseSequenceClassifier()


def _load_samples(
    database_path: str | Path | None,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], dict[str, Any]]:
    reviews = _unique_delivery_reviews(list_reviews(database_path=database_path, limit=100000))
    if len({review["player_id"] for review in reviews}) < MIN_PLAYER_GROUPS:
        raise ValueError(
            f"Need reviewed mechanics examples from at least {MIN_PLAYER_GROUPS} different player IDs."
        )

    readiness = mechanics_training_readiness(database_path)
    labels = readiness["ready_labels"]
    if not labels:
        summary = ", ".join(f"{label}={count}" for label, count in readiness["counts"].items())
        raise ValueError(
            f"No mechanics label has at least {MIN_EXAMPLES_PER_LABEL} examples from "
            f"{MIN_PLAYER_GROUPS} players. Current label counts: {summary}."
        )

    eligible_reviews = [
        review for review in reviews if set(review["mechanics_labels"]).intersection(labels)
    ]
    splits = split_reviews_by_player(eligible_reviews, seed=seed)
    examples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for split_name, split_reviews in splits.items():
        for review in split_reviews:
            review_labels = set(review["mechanics_labels"]).intersection(labels)
            if not review_labels:
                continue
            _, arrays = load_phase_payload(review["metadata_path"])
            poses = arrays.get("poses_2d")
            if poses is None:
                raise ValueError(
                    f"Reviewed delivery {review['delivery_id']} has no saved 2D poses."
                )
            features = _features_from_pose(
                poses,
                int(review["image_width"]),
                int(review["image_height"]),
            )
            target = np.asarray([label in review_labels for label in labels], dtype=np.float32)
            examples[split_name].append((features, target))

    for split_name, split_examples in examples.items():
        if not split_examples:
            raise ValueError(f"No labeled deliveries are available in the {split_name} player split.")
        targets = np.stack([example[1] for example in split_examples])
        missing = [
            label
            for index, label in enumerate(labels)
            if not bool(targets[:, index].any())
        ]
        if missing:
            raise ValueError(
                f"The {split_name} split has no positive examples for: {', '.join(missing)}. "
                "Collect examples across more players before training."
            )

    x_by_split = {
        split_name: np.stack([example[0] for example in split_examples]).astype(np.float32)
        for split_name, split_examples in examples.items()
    }
    y_by_split = {
        split_name: np.stack([example[1] for example in split_examples]).astype(np.float32)
        for split_name, split_examples in examples.items()
    }
    return x_by_split, y_by_split, labels, {
        "split_counts": {name: len(data) for name, data in examples.items()},
        "player_counts": {
            name: len({review["player_id"] for review in splits[name]})
            for name in splits
        },
        "eligible_label_counts": {label: readiness["counts"][label] for label in labels},
        "ignored_label_counts": {
            label: count for label, count in readiness["counts"].items() if label not in labels
        },
    }


def train_mechanics_model(
    *,
    database_path: str | Path | None = None,
    output_dir: str | Path = "models/mechanics/candidates",
    epochs: int = 100,
    learning_rate: float = 0.001,
    seed: int = 17,
    promote: bool = False,
) -> dict[str, Any]:
    """Train a small GRU classifier and require player-held-out improvement to promote."""
    if epochs < 1:
        raise ValueError("Training epochs must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be a positive finite number.")
    try:
        import torch
    except ImportError as error:
        raise ImportError("Install PyTorch to train the mechanics sequence model.") from error

    x, y, labels, dataset_summary = _load_samples(database_path, seed)
    torch.manual_seed(seed)
    model = _torch_model(torch, len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss()
    x_train = torch.from_numpy(x["train"])
    y_train = torch.from_numpy(y["train"])
    x_val = torch.from_numpy(x["val"])
    y_val = torch.from_numpy(y["val"])
    x_test = torch.from_numpy(x["test"])
    y_test = torch.from_numpy(y["test"])

    prevalence = y["train"].mean(axis=0)
    baseline_prediction = np.broadcast_to(prevalence >= 0.5, y["val"].shape)
    baseline_val_f1 = _f1_macro(y["val"], baseline_prediction)
    best_val_f1 = -1.0
    best_state: dict[str, Any] | None = None
    patience = max(10, min(20, epochs // 4))
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_train))
        for start in range(0, len(permutation), 16):
            batch = permutation[start : start + 16]
            optimizer.zero_grad()
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_function(model(x_val), y_val).item())
            val_predictions = (torch.sigmoid(model(x_val)).numpy() >= 0.5).astype(np.float32)
        val_f1 = _f1_macro(y["val"], val_predictions)
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and best_state is None):
            best_val_f1 = val_f1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Mechanics model training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_predictions = (torch.sigmoid(model(x_test)).numpy() >= 0.5).astype(np.float32)
    baseline_test_prediction = np.broadcast_to(prevalence >= 0.5, y["test"].shape)
    candidate_test_f1 = _f1_macro(y["test"], test_predictions)
    baseline_test_f1 = _f1_macro(y["test"], baseline_test_prediction)
    validation_improved = best_val_f1 > baseline_val_f1
    test_not_regressed = candidate_test_f1 >= baseline_test_f1
    eligible_for_promotion = validation_improved

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_path = root / f"mechanics_{version}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_count": 17 * 3,
            "hidden_size": GRU_HIDDEN_SIZE,
            "version": version,
        },
        candidate_path,
    )

    active_path = ACTIVE_MODEL_PATH.resolve()
    registry_path = active_path.parent / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else []
    if not isinstance(registry, list):
        raise ValueError(f"Mechanics model registry must contain a JSON list: {registry_path}")
    promoted = bool(promote and eligible_for_promotion)
    archived_path = None
    if promoted:
        active_path.parent.mkdir(parents=True, exist_ok=True)
        if active_path.is_file():
            archive_dir = active_path.parent / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archived = archive_dir / f"{version}_previous.pt"
            shutil.copy2(active_path, archived)
            archived_path = str(archived)
        staged_path = active_path.with_suffix(".pt.tmp")
        shutil.copy2(candidate_path, staged_path)
        staged_path.replace(active_path)

    report: dict[str, Any] = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_model": str(candidate_path),
        "active_model": str(active_path) if promoted else None,
        "archived_model": archived_path,
        "labels": labels,
        "dataset": dataset_summary,
        "metrics": {
            "baseline_validation_macro_f1": baseline_val_f1,
            "candidate_validation_macro_f1": best_val_f1,
            "baseline_test_macro_f1": baseline_test_f1,
            "candidate_test_macro_f1": candidate_test_f1,
        },
        "validation_improved": validation_improved,
        "test_not_regressed": test_not_regressed,
        "promotion_eligible": eligible_for_promotion,
        "promotion_gate": "validation macro-F1 must improve; test metrics are reported but not used for model selection",
        "promoted": promoted,
    }
    report_path = root / f"mechanics_{version}_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry.append(report)
    registry_path.write_text(json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8")
    return report


def predict_active_mechanics(poses_2d: np.ndarray, image_width: int, image_height: int) -> dict[str, Any] | None:
    """Return learned tag probabilities when an explicitly promoted model exists."""
    active_path = ACTIVE_MODEL_PATH.resolve()
    if not active_path.is_file():
        return None
    try:
        import torch
    except ImportError as error:
        raise ImportError("Install PyTorch to load the mechanics sequence model.") from error
    checkpoint = torch.load(active_path, map_location="cpu", weights_only=True)
    labels = checkpoint.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Invalid mechanics model checkpoint: {active_path}")
    model = _torch_model(torch, len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    features = _features_from_pose(poses_2d, image_width, image_height)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(features[None, ...])))[0].numpy()
    return {
        "status": "complete",
        "model_version": checkpoint.get("version", "unknown"),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(labels, probabilities)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a reviewed, player-held-out mechanics sequence classifier."
    )
    parser.add_argument("--database", default=str(default_database_path()))
    parser.add_argument("--output-dir", default="models/mechanics/candidates")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    report = train_mechanics_model(
        database_path=args.database,
        output_dir=args.output_dir,
        epochs=args.epochs,
        promote=args.promote,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
