"""FastAPI upload and analysis endpoints for local use."""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from cricket_coach.config import ALLOWED_VIDEO_SUFFIXES, MAX_UPLOAD_BYTES

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.concurrency import run_in_threadpool
except ImportError as error:
    raise ImportError("Install fastapi and python-multipart to serve the API.") from error

app = FastAPI(title="Cricket Bowling Analyzer", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic readiness endpoint."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)) -> dict[str, Any]:
    """Receive a bounded video upload and run the local pipeline."""
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported video extension.")
    matrix_path = os.getenv("CRICKET_CALIBRATION_MATRIX")
    if not matrix_path or not Path(matrix_path).is_file():
        raise HTTPException(
            status_code=503,
            detail="Configure CRICKET_CALIBRATION_MATRIX with an existing pitch calibration for headless analysis.",
        )
    contents = await video.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Video exceeds the upload size limit.")
    output_root = Path(os.getenv("CRICKET_OUTPUT_DIR", "data/api_runs"))
    run_dir = output_root / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="cricket-analysis-") as directory:
        input_path = Path(directory) / f"session{suffix}"
        input_path.write_bytes(contents)
        output_path = run_dir / "output"
        try:
            from cricket_coach.run_pipeline import run_pipeline

            result = await run_in_threadpool(run_pipeline, input_path, output_path, matrix_path, True)
        except (ValueError, RuntimeError, ImportError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result


if __name__ == "__main__":
    import uvicorn

    assert app.title == "Cricket Bowling Analyzer"
    uvicorn.run(app, host=os.getenv("CRICKET_HOST", "127.0.0.1"), port=int(os.getenv("CRICKET_PORT", "8000")))
