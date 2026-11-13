# Cricket Bowling Analyzer

This package provides a staged, local-first analysis prototype:

1. **Capture:** four-point pitch calibration and pose-triggered delivery clips.
2. **Tracking:** YOLOv8-pose 2D keypoints, optional 3D lifter, ball heatmap adapter, and event candidates.
3. **Biomechanics:** joint-angle and action-screening measurements.
4. **Digital twin:** normalized DTW reference comparisons and optional Plotly skeleton export.
5. **Coach:** rule-based observations, optional retrieval/LLM callbacks, and optional offline TTS.
6. **App:** FastAPI upload API and Streamlit dashboard.

## Setup and run

Use Python 3.10 or later. Install the core packages with `pip install -r requirements.txt`.
Optional integrations are listed in `requirements-optional.txt`.
The pose slicer requires Ultralytics and the YOLO pose weights; the model downloads or loads
weights according to the Ultralytics installation. Calibration is automatic on first use. The calibrator looks for white crease lines and falls
back to an identity matrix with a warning when it cannot confidently detect the pitch box.
The identity fallback permits the rest of the analysis to run but is not a metric calibration.

```powershell
python run_pipeline.py --video_path data\raw_sessions\session.mp4 --output_dir data\sliced_clips
```

Provide a previously saved calibration matrix to skip automatic calibration:

```powershell
python run_pipeline.py --video_path data\raw_sessions\session.mp4 --output_dir data\sliced_clips --matrix_path data\pitch_homography.npy --skip_interactive_calibration
```

For adaptive slicing, per-frame wrist speed is normalized by torso height and sampled in
torso-lengths per second. SciPy peak detection uses the source FPS and configured cooldown.
Clips shorter than eight seconds produce one active-bowler fallback clip; longer videos with
no detected release peak also produce one fallback clip when a person was detected.
`--interactive_calibration` remains available as an explicit manual override.

Run the optional web interfaces with `uvicorn cricket_coach.phase6_app.api:app` or
`streamlit run cricket_coach/phase6_app/dashboard.py`.

## Data contract

Each extracted `delivery_NNN.mp4` has a companion `delivery_NNN_data.json` and
`delivery_NNN_arrays.npz`. The JSON contains identifiers, source frame timing, calibration
path, per-phase status, and summaries. The NumPy archive contains numeric per-frame data;
3D pose depth is NaN when no trained 3D lifter was supplied. Phases update the same pair
of files so downstream consumers can inspect both metadata and arrays.

## Model and interpretation limits

- A ground-plane homography maps points on the pitch surface only; it cannot recover airborne
  ball depth or metric release speed.
- `Pose3DProcessor` accepts a caller-provided trained lifter. Without one it stores 2D points
  and marks depth unavailable instead of inventing 3D coordinates.
- `BallTracker` accepts a TrackNet-compatible heatmap callable. No weights or custom TrackNet
  implementation are bundled; without a model, ball positions remain unavailable.
- Event timing and biomechanics are camera/model-sensitive screening estimates, not official
  legality decisions, medical diagnoses, or injury predictions.
- Optional integrations (`dtaidistance`, Plotly, Streamlit, ChromaDB, an LLM provider, and
  `pyttsx3`) are deliberately injected or loaded only when used. Supply vetted pro-reference
  pose files; no professional athlete data is included.

Every implementation module has a lightweight executable check using `python -m
cricket_coach.<module>`. Model inference, GUI calibration, network services, and optional
provider integrations should be tested separately with the actual deployment environment.
