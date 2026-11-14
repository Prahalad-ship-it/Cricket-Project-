"""Local SQLite storage for consented, human-reviewed delivery examples."""

import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

MECHANICS_LABELS = (
    "front_leg_stability",
    "trunk_alignment",
    "bowling_arm_timing",
    "release_point_consistency",
    "run_up_rhythm",
    "other",
)

DATABASE_SCHEMA_VERSION = 3
MIN_VISIBLE_KEYPOINTS = 3
MAX_NOTES_LENGTH = 2000
PLAYER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
PLAYER_ROLES = ("bowler", "batter", "wicketkeeper", "other")


def default_database_path() -> Path:
    """Return the ignored, project-local database path."""
    return Path(__file__).resolve().parents[2] / "data" / "training" / "learning.sqlite3"


@contextmanager
def _connect(database_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(database_path) if database_path is not None else default_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_store(database_path: str | Path | None = None) -> None:
    """Create the local review tables, rejecting unknown future schemas."""
    with _connect(database_path) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Review database schema {version} is newer than supported "
                f"schema {DATABASE_SCHEMA_VERSION}."
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_reviews (
                review_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                delivery_id TEXT NOT NULL,
                metadata_path TEXT NOT NULL,
                video_path TEXT NOT NULL,
                frame_index INTEGER NOT NULL CHECK (frame_index >= 0),
                image_width INTEGER NOT NULL CHECK (image_width > 0),
                image_height INTEGER NOT NULL CHECK (image_height > 0),
                consent_confirmed INTEGER NOT NULL CHECK (consent_confirmed = 1),
                keypoints_json TEXT NOT NULL,
                mechanics_labels_json TEXT NOT NULL,
                coach_rating INTEGER CHECK (coach_rating BETWEEN 1 AND 5),
                drill_completed INTEGER NOT NULL CHECK (drill_completed IN (0, 1)),
                drill_helpful INTEGER CHECK (drill_helpful IN (0, 1)),
                drill_text TEXT NOT NULL DEFAULT '',
                player_role TEXT NOT NULL DEFAULT 'unknown',
                person_detection_index INTEGER,
                person_box_json TEXT,
                notes TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_delivery_reviews_player_created "
            "ON delivery_reviews(player_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_delivery_reviews_delivery "
            "ON delivery_reviews(delivery_id)"
        )
        if version < 2:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(delivery_reviews)")
            }
            if "drill_text" not in columns:
                connection.execute(
                    "ALTER TABLE delivery_reviews ADD COLUMN drill_text TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "UPDATE delivery_reviews SET schema_version = 2 WHERE schema_version < 2"
            )
        if version < 3:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(delivery_reviews)")
            }
            if "player_role" not in columns:
                connection.execute(
                    "ALTER TABLE delivery_reviews ADD COLUMN player_role "
                    "TEXT NOT NULL DEFAULT 'unknown'"
                )
            if "person_detection_index" not in columns:
                connection.execute(
                    "ALTER TABLE delivery_reviews ADD COLUMN person_detection_index INTEGER"
                )
            if "person_box_json" not in columns:
                connection.execute(
                    "ALTER TABLE delivery_reviews ADD COLUMN person_box_json TEXT"
                )
            connection.execute(
                "UPDATE delivery_reviews SET schema_version = 3 WHERE schema_version < 3"
            )
        if version < DATABASE_SCHEMA_VERSION:
            connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _records(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        result = rows.to_dict(orient="records")
    else:
        result = list(rows)
    if not all(isinstance(row, Mapping) for row in result):
        raise ValueError("Keypoint corrections must be rows with named fields.")
    return [dict(row) for row in result]


def _normalize_keypoints(
    rows: Any,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    records = _records(rows)
    if len(records) != len(KEYPOINT_NAMES):
        raise ValueError(f"Exactly {len(KEYPOINT_NAMES)} keypoint rows are required.")

    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in records:
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Each keypoint row must have a valid index.") from error
        if not 0 <= index < len(KEYPOINT_NAMES) or index in seen:
            raise ValueError("Keypoint indices must be unique values from 0 through 16.")
        expected_name = KEYPOINT_NAMES[index]
        if row.get("keypoint") != expected_name:
            raise ValueError(f"Keypoint {index} must be named {expected_name!r}.")
        seen.add(index)

        visible_value = row.get("visible")
        if visible_value not in (True, False, 0, 1):
            raise ValueError(f"Visibility for {expected_name} must be checked or unchecked.")
        visible = bool(visible_value)
        if visible:
            try:
                x = float(row["x_px"])
                y = float(row["y_px"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Visible keypoint {expected_name} needs pixel coordinates.") from error
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"Coordinates for {expected_name} must be finite.")
            if not (0 <= x <= image_width and 0 <= y <= image_height):
                raise ValueError(
                    f"Coordinates for {expected_name} must be within the {image_width}x{image_height} frame."
                )
        else:
            x = None
            y = None
        normalized.append(
            {
                "index": index,
                "keypoint": expected_name,
                "x_px": x,
                "y_px": y,
                "visible": visible,
            }
        )
    normalized.sort(key=lambda point: point["index"])
    if sum(point["visible"] for point in normalized) < MIN_VISIBLE_KEYPOINTS:
        raise ValueError(
            f"At least {MIN_VISIBLE_KEYPOINTS} visible keypoints are needed for a pose training example."
        )
    return normalized


def save_review(
    *,
    player_id: str,
    delivery_id: str,
    metadata_path: str | Path,
    video_path: str | Path,
    frame_index: int,
    image_width: int,
    image_height: int,
    keypoint_rows: Any,
    mechanics_labels: Iterable[str],
    coach_rating: int | None,
    drill_completed: bool,
    drill_helpful: bool | None,
    notes: str,
    metrics: Mapping[str, Any],
    consent_confirmed: bool,
    database_path: str | Path | None = None,
    drill_text: str = "",
    player_role: str = "unknown",
    person_detection_index: int | None = None,
    person_box: Iterable[float] | None = None,
) -> str:
    """Validate and append one consented review; return its stable review ID."""
    normalized_player_id = player_id.strip()
    if not PLAYER_ID_PATTERN.fullmatch(normalized_player_id):
        raise ValueError("Use an anonymous player ID with letters, numbers, '_' or '-'; do not enter a name.")
    if not delivery_id.strip():
        raise ValueError("A delivery ID is required.")
    normalized_role = player_role.strip().lower()
    if normalized_role != "unknown" and normalized_role not in PLAYER_ROLES:
        raise ValueError(f"Player role must be one of: {', '.join(PLAYER_ROLES)}.")
    if person_detection_index is not None and (
        isinstance(person_detection_index, bool)
        or not isinstance(person_detection_index, int)
        or person_detection_index < 0
    ):
        raise ValueError("Person detection index must be a non-negative integer.")
    if not consent_confirmed:
        raise PermissionError("Consent must be confirmed before saving video-derived labels.")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
    ):
        raise ValueError("Frame index must be a non-negative integer.")
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, int)
        or image_width <= 0
        or isinstance(image_height, bool)
        or not isinstance(image_height, int)
        or image_height <= 0
    ):
        raise ValueError("Frame width and height must be positive integers.")
    if coach_rating is not None and (
        isinstance(coach_rating, bool)
        or not isinstance(coach_rating, int)
        or not 1 <= coach_rating <= 5
    ):
        raise ValueError("Coach rating must be between 1 and 5.")
    if not isinstance(drill_completed, bool):
        raise ValueError("Drill completion must be a yes/no value.")
    if drill_helpful is not None and (not isinstance(drill_helpful, bool) or not drill_completed):
        raise ValueError("A drill can only be rated helpful after it was completed.")
    notes = notes.strip()
    if len(notes) > MAX_NOTES_LENGTH:
        raise ValueError(f"Notes must be no longer than {MAX_NOTES_LENGTH} characters.")
    drill_text = drill_text.strip()
    if len(drill_text) > MAX_NOTES_LENGTH:
        raise ValueError(f"Prescribed drill text must be no longer than {MAX_NOTES_LENGTH} characters.")

    selected_labels = sorted(set(mechanics_labels))
    invalid_labels = set(selected_labels).difference(MECHANICS_LABELS)
    if invalid_labels:
        raise ValueError(f"Unsupported mechanics label(s): {', '.join(sorted(invalid_labels))}.")

    source_metadata = Path(metadata_path).expanduser().resolve()
    source_video = Path(video_path).expanduser().resolve()
    if not source_metadata.is_file():
        raise FileNotFoundError(f"Delivery metadata was not found: {source_metadata}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Delivery video was not found: {source_video}")

    keypoints = _normalize_keypoints(keypoint_rows, int(image_width), int(image_height))
    normalized_box = None
    if person_box is not None:
        try:
            normalized_box = [float(coordinate) for coordinate in person_box]
        except (TypeError, ValueError) as error:
            raise ValueError("Person box must contain four finite pixel coordinates.") from error
        if (
            len(normalized_box) != 4
            or not all(math.isfinite(coordinate) for coordinate in normalized_box)
            or not (
                0 <= normalized_box[0] <= normalized_box[2] <= image_width
                and 0 <= normalized_box[1] <= normalized_box[3] <= image_height
            )
        ):
            raise ValueError("Person box must be ordered within the video frame.")
    keypoints_json = json.dumps(keypoints, allow_nan=False)
    labels_json = json.dumps(selected_labels, allow_nan=False)
    metrics_json = json.dumps(dict(metrics), allow_nan=False, default=_json_default)
    person_box_json = (
        json.dumps(normalized_box, allow_nan=False)
        if normalized_box is not None
        else None
    )
    review_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    initialize_store(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO delivery_reviews (
                review_id, schema_version, player_id, delivery_id, metadata_path,
                video_path, frame_index, image_width, image_height, consent_confirmed,
                keypoints_json, mechanics_labels_json, coach_rating, drill_completed,
                drill_helpful, drill_text, player_role, person_detection_index,
                person_box_json, notes, metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                DATABASE_SCHEMA_VERSION,
                normalized_player_id,
                delivery_id.strip(),
                str(source_metadata),
                str(source_video),
                int(frame_index),
                int(image_width),
                int(image_height),
                keypoints_json,
                labels_json,
                coach_rating,
                int(drill_completed),
                None if drill_helpful is None else int(drill_helpful),
                drill_text,
                normalized_role,
                person_detection_index,
                person_box_json,
                notes,
                metrics_json,
                created_at,
            ),
        )
    return review_id


def list_reviews(
    player_id: str | None = None,
    limit: int = 2000,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return recent reviewed examples, optionally filtered to one player ID."""
    if isinstance(limit, bool) or limit < 1:
        raise ValueError("Review limit must be a positive integer.")
    initialize_store(database_path)
    with _connect(database_path) as connection:
        if player_id is None:
            rows = connection.execute(
                "SELECT * FROM delivery_reviews ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM delivery_reviews WHERE player_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (player_id.strip(), int(limit)),
            ).fetchall()
    result = []
    for row in rows:
        review = dict(row)
        review["keypoints"] = json.loads(review.pop("keypoints_json"))
        review["mechanics_labels"] = json.loads(review.pop("mechanics_labels_json"))
        review["metrics"] = json.loads(review.pop("metrics_json"))
        box_json = review.pop("person_box_json", None)
        review["person_box"] = json.loads(box_json) if box_json else None
        review["consent_confirmed"] = bool(review["consent_confirmed"])
        review["drill_completed"] = bool(review["drill_completed"])
        if review["drill_helpful"] is not None:
            review["drill_helpful"] = bool(review["drill_helpful"])
        result.append(review)
    return result


def review_stats(database_path: str | Path | None = None) -> dict[str, int]:
    """Summarize locally reviewed examples and training label coverage."""
    reviews = list_reviews(database_path=database_path, limit=100000)
    latest_frames: dict[tuple[str, str, int], dict[str, Any]] = {}
    latest_tagged_deliveries: dict[tuple[str, str], dict[str, Any]] = {}
    for review in reviews:
        frame_key = (review["player_id"], review["video_path"], review["frame_index"])
        if (
            frame_key not in latest_frames
            or review["created_at"] > latest_frames[frame_key]["created_at"]
        ):
            latest_frames[frame_key] = review
        if review["mechanics_labels"]:
            delivery_key = (review["player_id"], review["delivery_id"])
            if (
                delivery_key not in latest_tagged_deliveries
                or review["created_at"] > latest_tagged_deliveries[delivery_key]["created_at"]
            ):
                latest_tagged_deliveries[delivery_key] = review
    return {
        "reviews": len(latest_frames),
        "players": len({review["player_id"] for review in latest_frames.values()}),
        "tagged_reviews": len(latest_tagged_deliveries),
    }


if __name__ == "__main__":
    initialize_store()
    print(f"Review database ready: {default_database_path()}")
