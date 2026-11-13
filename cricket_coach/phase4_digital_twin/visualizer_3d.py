"""Optional interactive Plotly skeleton visualization and HTML export."""

from pathlib import Path

import numpy as np


class SkeletonVisualizer:
    """Render a pose frame or pose sequence to an interactive Plotly HTML page."""

    BONES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))

    def create_figure(self, poses: np.ndarray, frame_index: int = 0) -> object:
        """Create a Plotly 3D skeleton figure; unavailable depth raises clearly."""
        if poses.ndim != 3 or poses.shape[1] < 17 or poses.shape[2] < 3:
            raise ValueError("Expected pose array shaped (frames, 17, at least 3).")
        if not 0 <= frame_index < len(poses):
            raise IndexError("Requested pose frame is out of range.")
        frame = poses[frame_index, :, :3]
        if not np.isfinite(frame).all():
            raise ValueError("This frame lacks complete 3D coordinates.")
        try:
            import plotly.graph_objects as go
        except ImportError as error:
            raise ImportError("Install plotly to export interactive 3D visualizations.") from error
        traces = [
            go.Scatter3d(
                x=[frame[a, 0], frame[b, 0]],
                y=[frame[a, 1], frame[b, 1]],
                z=[frame[a, 2], frame[b, 2]],
                mode="lines+markers",
                line={"width": 6},
                name=f"bone_{a}_{b}",
                showlegend=False,
            )
            for a, b in self.BONES
        ]
        return go.Figure(data=traces, layout=go.Layout(scene={"aspectmode": "data"}, title="Bowling action pose"))

    def export_html(self, poses: np.ndarray, output_path: str | Path, frame_index: int = 0) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.create_figure(poses, frame_index).write_html(str(destination), include_plotlyjs="cdn")
        return destination


if __name__ == "__main__":
    try:
        import plotly  # noqa: F401
    except ImportError:
        print("Plotly is optional; install it to render skeletons.")
    else:
        print("Visualizer dependency check passed.")
