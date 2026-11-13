"""Shared JSON and NumPy data contract for pipeline phases."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_phase_payload(
    json_path: str | Path,
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[Path, Path | None]:
    """Write JSON metadata and an optional adjacent compressed NumPy archive."""
    destination = Path(json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array_path = destination.with_name(f"{destination.stem}_arrays.npz") if arrays else None
    if arrays and array_path:
        np.savez_compressed(array_path, **arrays)
        payload["arrays_path"] = str(array_path.resolve())
    with destination.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
    return destination, array_path


def load_phase_payload(json_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load JSON metadata and its optional NumPy archive."""
    source = Path(json_path)
    with source.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    arrays_path = payload.get("arrays_path")
    arrays: dict[str, np.ndarray] = {}
    if arrays_path:
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    return payload, arrays


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        metadata_path, _ = save_phase_payload(
            Path(directory) / "delivery_001_data.json",
            {"delivery_id": "delivery_001"},
            {"pose": np.zeros((2, 17, 4), dtype=np.float32)},
        )
        metadata, arrays = load_phase_payload(metadata_path)
        assert metadata["delivery_id"] == "delivery_001"
        assert arrays["pose"].shape == (2, 17, 4)
    print("Data-contract checks passed.")
