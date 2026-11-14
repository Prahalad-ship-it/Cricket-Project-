"""Batch-train and explicitly gate a cricket-tuned YOLO pose model."""

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cricket_coach.config import MODEL_POSE_PATH
from cricket_coach.training.pose_dataset import export_reviewed_pose_dataset
from cricket_coach.training.review_store import default_database_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _pose_map(metrics: Any) -> float:
    pose_metrics = getattr(metrics, "pose", None)
    score = getattr(pose_metrics, "map", None)
    if score is None or not math.isfinite(float(score)):
        raise ValueError("Validation did not produce a finite pose mAP score.")
    return float(score)


def _promote_model(candidate_path: Path, active_path: Path, version: str) -> str | None:
    active_path.parent.mkdir(parents=True, exist_ok=True)
    archived_path = None
    if active_path.is_file():
        archive_dir = active_path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / f"{version}_previous.pt"
        shutil.copy2(active_path, archived)
        archived_path = str(archived)
    temporary_path = active_path.with_suffix(".pt.tmp")
    shutil.copy2(candidate_path, temporary_path)
    temporary_path.replace(active_path)
    return archived_path


def train_pose_model(
    *,
    dataset_yaml: str | Path,
    base_model: str | Path = MODEL_POSE_PATH,
    project_dir: str | Path = "runs/pose",
    run_name: str | None = None,
    epochs: int = 30,
    image_size: int = 640,
    promote: bool = False,
) -> dict[str, Any]:
    """Train from a reviewed dataset, compare against baseline, and optionally promote."""
    data_path = Path(dataset_yaml).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Pose dataset YAML was not found: {data_path}")
    model_path = Path(base_model).expanduser()
    if not model_path.is_file() and str(base_model) == "yolov8n-pose.pt":
        model_reference = str(base_model)
    elif model_path.is_file():
        model_reference = str(model_path.resolve())
    else:
        raise FileNotFoundError(f"Base pose model was not found: {model_path}")
    if epochs < 1:
        raise ValueError("Training epochs must be positive.")
    if image_size < 64:
        raise ValueError("Training image size must be at least 64 pixels.")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Install ultralytics to train the pose model.") from error

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or f"cricket_pose_{version}"
    project = Path(project_dir).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)

    baseline_model = YOLO(model_reference)
    baseline_val = _pose_map(
        baseline_model.val(data=str(data_path), split="val", verbose=False)
    )
    baseline_test = _pose_map(
        baseline_model.val(data=str(data_path), split="test", verbose=False)
    )

    trainer = YOLO(model_reference)
    trainer.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=image_size,
        project=str(project),
        name=name,
        exist_ok=False,
        patience=max(5, min(15, epochs // 3)),
    )
    run_dir = project / name
    candidate_path = run_dir / "weights" / "best.pt"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Training completed without a best checkpoint: {candidate_path}")

    candidate_model = YOLO(str(candidate_path))
    candidate_val = _pose_map(
        candidate_model.val(data=str(data_path), split="val", verbose=False)
    )
    candidate_test = _pose_map(
        candidate_model.val(data=str(data_path), split="test", verbose=False)
    )
    validation_improved = candidate_val > baseline_val
    test_not_regressed = candidate_test >= baseline_test
    eligible_for_promotion = validation_improved
    promoted = bool(promote and eligible_for_promotion)

    active_path = PROJECT_ROOT / "models" / "pose" / "active.pt"
    registry_path = active_path.parent / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else []
    if not isinstance(registry, list):
        raise ValueError(f"Model registry must contain a JSON list: {registry_path}")
    archived_path = None
    if promoted:
        archived_path = _promote_model(candidate_path, active_path, version)

    report: dict[str, Any] = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": str(data_path),
        "base_model": model_reference,
        "candidate_model": str(candidate_path.resolve()),
        "active_model": str(active_path) if promoted else None,
        "archived_model": archived_path,
        "epochs_requested": epochs,
        "image_size": image_size,
        "metrics": {
            "baseline_validation_pose_map": baseline_val,
            "candidate_validation_pose_map": candidate_val,
            "baseline_test_pose_map": baseline_test,
            "candidate_test_pose_map": candidate_test,
        },
        "validation_improved": validation_improved,
        "test_not_regressed": test_not_regressed,
        "promotion_eligible": eligible_for_promotion,
        "promotion_gate": "validation pose mAP must improve; test metrics are reported but not used for model selection",
        "promoted": promoted,
    }
    report_path = run_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry.append(report)
    registry_path.write_text(json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export reviewed annotations, train a pose candidate, and optionally promote it."
    )
    parser.add_argument("--database", default=str(default_database_path()))
    parser.add_argument("--dataset-output", default="data/training/pose_dataset_v1")
    parser.add_argument("--base-model", default=MODEL_POSE_PATH)
    parser.add_argument("--project", default="runs/pose")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote only if validation mAP improves; test metrics remain report-only.",
    )
    args = parser.parse_args()

    dataset = export_reviewed_pose_dataset(
        args.dataset_output,
        database_path=args.database,
    )
    report = train_pose_model(
        dataset_yaml=dataset["dataset_yaml"],
        base_model=args.base_model,
        project_dir=args.project,
        epochs=args.epochs,
        image_size=args.image_size,
        promote=args.promote,
    )
    print(json.dumps({"dataset": dataset, "training": report}, indent=2))


if __name__ == "__main__":
    main()
