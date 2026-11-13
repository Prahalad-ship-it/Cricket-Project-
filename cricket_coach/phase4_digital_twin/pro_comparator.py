"""Normalize pose trajectories and compare against supplied reference sequences."""

from pathlib import Path
from typing import Any

import numpy as np

from cricket_coach.config import MIN_VALID_BONE_LENGTHS


class ProComparator:
    """Compare sequences geometrically; reference provenance must be user supplied."""

    BONES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))

    def normalize_bone_lengths(self, poses: np.ndarray) -> tuple[np.ndarray, float]:
        """Scale each frame by median observed bone length; retain NaNs."""
        if poses.ndim != 3 or poses.shape[1] < 17 or poses.shape[2] < 3:
            raise ValueError("Expected (frames, 17, at least 3) pose array.")
        lengths = []
        for first, second in self.BONES:
            length = np.linalg.norm(poses[:, first, :3] - poses[:, second, :3], axis=1)
            lengths.extend(length[np.isfinite(length) & (length > 0)].tolist())
        if len(lengths) < MIN_VALID_BONE_LENGTHS:
            raise ValueError("Not enough valid bone lengths to normalize this pose.")
        scale = float(np.median(lengths))
        return poses[..., :3] / scale, scale

    @staticmethod
    def sequence_distance(first: np.ndarray, second: np.ndarray, use_dtaidistance: bool = True) -> float:
        """Compute multivariate DTW distance, with an explicit NumPy fallback."""
        a = ProComparator._frame_features(first)
        b = ProComparator._frame_features(second)
        if not len(a) or not len(b):
            raise ValueError("Both pose sequences need valid frames.")
        if use_dtaidistance:
            try:
                from dtaidistance import dtw_ndim

                return float(dtw_ndim.distance_fast(a.astype(np.double), b.astype(np.double), use_pruning=True))
            except ImportError:
                pass
        # Dynamic programming fallback keeps the module usable without the optional package.
        previous = np.full(len(b) + 1, np.inf, dtype=np.float64)
        previous[0] = 0.0
        for row in a:
            current = np.full(len(b) + 1, np.inf, dtype=np.float64)
            for index, other in enumerate(b, start=1):
                current[index] = np.linalg.norm(row - other) + min(
                    current[index - 1], previous[index], previous[index - 1]
                )
            previous = current
        return float(previous[-1])

    @staticmethod
    def _frame_features(poses: np.ndarray) -> np.ndarray:
        sequence = np.asarray(poses, dtype=np.float64)
        if sequence.ndim == 3:
            sequence = sequence.reshape((len(sequence), -1))
        if sequence.ndim != 2:
            raise ValueError("Expected a sequence of pose frames.")
        valid = np.isfinite(sequence).all(axis=1)
        return sequence[valid]

    def compare(self, bowler: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
        """Return normalized DTW score and reference-comparison caveat."""
        normalized_bowler, scale = self.normalize_bone_lengths(bowler)
        normalized_reference, _ = self.normalize_bone_lengths(reference)
        return {
            "dtw_distance": self.sequence_distance(normalized_bowler, normalized_reference),
            "bowler_median_bone_scale": scale,
            "reference_frames": int(len(reference)),
            "bowler_frames": int(len(bowler)),
            "note": "Similarity is not a quality score; verify reference identity, camera view, and pose model.",
        }

    @staticmethod
    def load_reference(path: str | Path) -> np.ndarray:
        """Load a numeric pose sequence from .npy or .npz."""
        source = Path(path)
        if source.suffix.lower() == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                if "poses_3d" not in archive:
                    raise ValueError("Reference archive must contain 'poses_3d'.")
                return archive["poses_3d"]
        return np.load(source, allow_pickle=False)


if __name__ == "__main__":
    sequence = np.ones((3, 17, 3), dtype=np.float32)
    for first, second in ProComparator.BONES:
        sequence[:, second, 0] += 1
    distance = ProComparator.sequence_distance(sequence, sequence, use_dtaidistance=False)
    assert distance == 0
    print("Pro-comparison checks passed.")
