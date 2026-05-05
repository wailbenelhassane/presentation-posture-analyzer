"""
REST API for the Presentation Body Language Analyzer.

Endpoints:
    GET  /health               Health check
    POST /analyze              Synchronous analysis (blocks until done)
    POST /analyze/async        Start async analysis job, returns job_id immediately
    GET  /jobs/{job_id}        Poll async job status and retrieve results
    DELETE /jobs/{job_id}      Delete a completed or errored job

Run with:
    uvicorn presentation_analyzer.api:app --host 0.0.0.0 --port 8000
Or:
    python run_api.py
"""

import os
import uuid
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .analyzer import PresentationAnalyzer, report_to_dict


app = FastAPI(
    title="Voxly — Presentation Body Language Analyzer",
    description=(
        "Analyze body language in presentation videos. "
        "Upload a video and receive a structured JSON report with score, "
        "gesture events, timeline, and coaching recommendations."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory async job store ────────────────────────────────────────────────

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SEC = 3600  # jobs expire after 1 hour


def _cleanup_expired_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SEC
    with _jobs_lock:
        stale = [
            jid for jid, j in _jobs.items()
            if j["status"] in ("done", "error") and j["_created_ts"] < cutoff
        ]
        for jid in stale:
            del _jobs[jid]


def _new_job(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "status": "pending",       # pending | processing | done | error
        "progress": 0.0,           # 0.0 – 1.0
        "analysis": None,          # filled when done
        "error": None,             # filled when error
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "_created_ts": time.time(),
    }


def _public(job: dict) -> dict:
    """Return job dict without internal underscore keys."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


# ── Core analysis helper ─────────────────────────────────────────────────────

def _run_analysis(
    video_path: str,
    *,
    fast: bool,
    max_duration: Optional[float],
    max_persons: int,
    progress_cb=None,
) -> dict:
    """Run the full analysis pipeline and return a serialisable dict."""
    config = dict(
        use_hands=not fast,
        model_complexity=0 if fast else 1,
        process_every_n=4 if fast else 2,
        max_persons=max_persons,
    )
    with PresentationAnalyzer(**config) as analyzer:
        report = analyzer.analyze_video(
            video_path,
            max_duration_sec=max_duration,
            progress_callback=progress_cb,
        )

    result = report_to_dict(report)
    result["percentage"] = (
        round(report.final_score / report.max_score * 100, 1)
        if report.max_score > 0
        else 100.0
    )
    return result


def _video_suffix(filename: Optional[str]) -> str:
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in ("mp4", "mov", "avi", "mkv", "webm", "m4v"):
            return f".{ext}"
    return ".mp4"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    """Returns API status and version. Use as a liveness probe."""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/analyze", summary="Synchronous video analysis")
async def analyze_sync(
    video: UploadFile = File(..., description="Video file (mp4, mov, avi, mkv, webm)"),
    fast: bool = Form(False, description="Fast mode: lighter model, skips hand detection (~3× faster)"),
    max_duration: Optional[float] = Form(None, description="Limit analysis to first N seconds"),
    max_persons: int = Form(4, description="Maximum persons to track (1–4)"),
):
    """
    Analyze a video and return the full JSON report synchronously.

    Suitable for videos up to ~3–5 minutes. For longer videos use
    **POST /analyze/async** to avoid timeout issues.

    ### Response schema
    ```json
    {
      "status": "ok",
      "analysis": {
        "final_score": 82.5,
        "max_score": 100.0,
        "percentage": 82.5,
        "grade": "B+",
        "num_persons": 1,
        "video_duration_sec": 95.3,
        "total_frames_analyzed": 2859,
        "gesture_summaries": [...],
        "timeline": [...],
        "recommendations": [...],
        "penalty_breakdown": {...}
      },
      "metadata": {
        "filename": "demo.mp4",
        "fast_mode": false,
        "analyzed_at": "2026-05-05T12:00:00+00:00"
      }
    }
    ```
    """
    max_persons = max(1, min(4, max_persons))
    suffix = _video_suffix(video.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        analysis = _run_analysis(
            tmp_path,
            fast=fast,
            max_duration=max_duration,
            max_persons=max_persons,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content={
        "status": "ok",
        "analysis": analysis,
        "metadata": {
            "filename": video.filename,
            "fast_mode": fast,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        },
    })


@app.post("/analyze/async", status_code=202, summary="Start async video analysis")
async def analyze_async(
    video: UploadFile = File(..., description="Video file (mp4, mov, avi, mkv, webm)"),
    fast: bool = Form(False, description="Fast mode"),
    max_duration: Optional[float] = Form(None, description="Limit analysis to first N seconds"),
    max_persons: int = Form(4, description="Maximum persons to track (1–4)"),
):
    """
    Start an analysis job in the background and return immediately.

    Poll **GET /jobs/{job_id}** to check progress and retrieve results when
    `status` is `"done"`. The job and its results are kept for 1 hour.

    ### Response (202 Accepted)
    ```json
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "pending",
      "poll_url": "/jobs/550e8400-e29b-41d4-a716-446655440000"
    }
    ```
    """
    max_persons = max(1, min(4, max_persons))
    suffix = _video_suffix(video.filename)

    # Write temp file before spawning thread (UploadFile is not thread-safe)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await video.read())
    tmp.close()
    tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = _new_job(job_id)

    def _worker():
        with _jobs_lock:
            _jobs[job_id]["status"] = "processing"

        def _progress(p: float):
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["progress"] = round(p, 3)

        try:
            analysis = _run_analysis(
                tmp_path,
                fast=fast,
                max_duration=max_duration,
                max_persons=max_persons,
                progress_cb=_progress,
            )
            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "done",
                    "progress": 1.0,
                    "analysis": analysis,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "error",
                    "error": str(exc),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
        finally:
            os.unlink(tmp_path)

    threading.Thread(target=_worker, daemon=True).start()
    _cleanup_expired_jobs()

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "pending",
            "poll_url": f"/jobs/{job_id}",
        },
    )


@app.get("/jobs/{job_id}", summary="Get async job status and results")
async def get_job(job_id: str):
    """
    Returns the current state of an async analysis job.

    ### Job states
    - **pending** — queued, not started yet
    - **processing** — analysis running (`progress` 0.0 → 1.0)
    - **done** — finished; `analysis` contains the full report
    - **error** — failed; `error` contains the message

    ### Response when done
    ```json
    {
      "job_id": "...",
      "status": "done",
      "progress": 1.0,
      "analysis": { ... },
      "error": null,
      "created_at": "...",
      "completed_at": "..."
    }
    ```
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or already expired")

    return JSONResponse(content=_public(job))


@app.delete("/jobs/{job_id}", status_code=204, summary="Delete a completed job")
async def delete_job(job_id: str):
    """
    Delete a completed or errored job to free memory.
    Returns 409 if the job is still processing.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] == "processing":
            raise HTTPException(status_code=409, detail="Job is still processing")
        del _jobs[job_id]
