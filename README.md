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
To enable NVIDIA coaching locally, put `NVIDIA_API_KEY=your_key` in the repository-root
`.env` file. The file is ignored by Git; environment variables already set in the shell take
precedence.
The MediaPipe compatibility range preserves the `solutions.pose` API used by the default
3D pose lifter; the optional Streamlit range is capped to remain compatible with its
protobuf requirement.
The pose slicer requires Ultralytics and the YOLO pose weights; the model downloads or loads
weights according to the Ultralytics installation. Calibration is automatic on first use. The calibrator looks for white crease lines and falls
back to an identity matrix with a warning when it cannot confidently detect the pitch box.
The identity fallback permits the rest of the analysis to run but is not a metric calibration.

```powershell
python run_pipeline.py --video_path data\raw_sessions\session.mp4 --output_dir data\sliced_clips
```

Put your source video in `data\raw_sessions\`. MP4 is recommended. The command-line
pipeline uses OpenCV, so other formats work only if your installed OpenCV video codecs
can read them. The optional FastAPI `/analyze` endpoint accepts `.mp4`, `.mov`, `.avi`,
`.mkv`, and `.m4v` files up to 500 MB. Streamlit itself does not upload raw videos; it
opens delivery clips already saved by the pipeline.

### Video and data privacy

The pipeline, review database, and training run locally on your computer. The project
does not upload source videos, saved pose arrays, or the review database for training.
New raw recordings, generated outputs, review data, and model weights are excluded by
`.gitignore` so they are not added to Git by default. Files already tracked by Git remain
tracked.

If `NVIDIA_API_KEY` is configured, the optional NVIDIA coaching request sends the
structured biomechanics measurements and event observations for that delivery; it does
not send the video or review database. To keep coaching local, leave `NVIDIA_API_KEY`
unset. Without it, the coach uses local rules. On first use, Ultralytics may download
pretrained pose-model weights to your computer; that is a model download, not a video
upload.

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
the default MediaPipe pose lifter estimates 3D world landmarks, with unavailable detections
left as NaN. Phases update the same pair of files so downstream consumers can inspect both
metadata and arrays.

## Model and interpretation limits

- A ground-plane homography maps points on the pitch surface only; it cannot recover airborne
  ball depth or metric release speed.
- `Pose3DProcessor` uses MediaPipe's world-landmark pose model by default and accepts a
  caller-provided lifter. Its 3D coordinates are model estimates, not motion-capture data.
- Biomechanics uses exact event frames only when explicitly marked as exact; otherwise it
  summarizes a configurable window around peak wrist speed. Metrics are screening estimates
  and remain unavailable when required landmarks are missing.
- `BallTracker` accepts a TrackNet-compatible heatmap callable. No weights or custom TrackNet
  implementation are bundled; without a model, ball positions remain unavailable.
- Event timing and biomechanics are camera/model-sensitive screening estimates, not official
  legality decisions, medical diagnoses, or injury predictions.
- Set `NVIDIA_API_KEY` in the environment to enable the optional NVIDIA NIM coaching report.
  Without a key or when the provider is unavailable, the coach returns rule-based feedback.
- Optional integrations (`dtaidistance`, Plotly, Streamlit, ChromaDB, an LLM provider, and
  `pyttsx3`) are deliberately injected or loaded only when used. Supply vetted pro-reference
  pose files; no professional athlete data is included.

Every implementation module has a lightweight executable check using `python -m
cricket_coach.<module>`. Model inference, GUI calibration, network services, and optional
provider integrations should be tested separately with the actual deployment environment.

## Training from reviewed examples

### First: analyze a video (this does not train a model)

Run your video through the pipeline, then open its saved delivery in Streamlit:

```powershell
python run_pipeline.py --video_path "data\raw_sessions\my_bowling.mp4" --output_dir "data\sliced_clips"
streamlit run cricket_coach\phase6_app\dashboard.py
```

In the browser, open a delivery and expand **Training data review & player progress**.
You can view your analysis without labeling anything; the existing pretrained pose model
does the analysis.

### Next: save a reviewed training example

For each example you want the system to learn from:

1. Pick a frame, choose the numbered person box around the bowler, and verify that the
   keypoint overlay follows that person. Detection numbers are ordered by box size and may
   change from frame to frame; choose the person again on every frame.
2. Enter that bowler's stable anonymous ID (for example, `player_001`) and role. Use the same
   ID for that person every time; do not make up extra IDs for one person.
3. Confirm you have permission to use the video. Correct visible body keypoints in the table;
   uncheck points that are not visible or cannot be placed accurately.
4. Optionally select mechanics tags only when you have reviewed the technique. Rate the
   feedback or drill if useful, then click **Save reviewed example**.

Repeat with more frames/deliveries. The examples are saved locally in
`data/training/learning.sqlite3`, including the selected person's role and detected box as
feedback for future person-selection training. These labels do not automatically retrain a
model or change the current delivery's biomechanics. The current pipeline still analyzes the
largest detected person, which can switch between players; manually choosing a box affects
the reviewed training example only. Older analyzed clips need to be rerun to include saved
per-person detections. Keep videos local unless participants have explicitly agreed to
another use. Do not enter names, medical details, or other identifying notes.

### Finally: train and evaluate (only when the buttons unlock)

The dashboard shows your review counts and enables training when these minimums are met:

- **Pose model:** 20 reviewed frames total from at least 5 different bowlers.
- **Mechanics model:** at least 20 reviewed deliveries for a mechanics tag, from at least
  5 different bowlers. Repeat this for every tag you want it to learn.

These are real different people, not five IDs for yourself. If you are currently the only
bowler, you can still analyze your bowling and save reviews, but the training buttons should
remain disabled. You do not need to train just to analyze a video.

When enabled, click **Train/evaluate pose model** or **Train/evaluate mechanics model**.
This creates a candidate model and tests it on players held out from training. Promotion is
optional: leave the **Promote** checkbox off for a first run. A model is not activated unless
you opt in and its validation score improves. The held-out test score is reported but is not
used to select the model. A promoted pose model is used on later pipeline runs; a promoted
mechanics model adds experimental tag probabilities to later delivery reports. Training is
not automatic and it does not learn directly from coach reports.

Install PyTorch if it is not already available before training: `python -m pip install "torch>=2.0"`.

The current three raw clips and one analyzed delivery are not training data until they are
reviewed and labeled. These are minimum software gates for a small pilot, not evidence of
reliable generalization; gather substantially more diverse players, camera angles, lighting,
and bowling styles before relying on model predictions.

Ultralytics' COCO-pretrained pose model is a starting point for generic human keypoints, not
cricket-technique supervision. A brief scan did not verify a clearly licensed, labeled public
cricket-bowling pose dataset. Check the dataset's labels, source rights, and license before
using public material; if none qualifies, label consented local clips rather than training on
unverified downloads.

From the repository root, the same offline workflows are available from the terminal:

```powershell
python -m cricket_coach.training.train_pose --epochs 30
python -m cricket_coach.training.mechanics_model --epochs 100
```

Add `--promote` only when you want an evaluated candidate to become active if it passes the
gates. Pose training exports reviewed keypoints to a new versioned dataset directory and
refuses to overwrite a non-empty dataset. See the [Ultralytics pose training guide](https://docs.ultralytics.com/tasks/pose/)
for the pretrained model and dataset format.
