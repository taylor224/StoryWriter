"""Background job queue. One GPU means one worker thread."""

import queue
import threading
import traceback
from pathlib import Path
from typing import Any

from . import audio, db, diarize, pipeline

_queue: "queue.Queue[tuple[int, Path, dict[str, Any]]]" = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_current_job_id: int | None = None


def start() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="stt-worker", daemon=True)
            _worker.start()


def submit(job_id: int, source: Path, params: dict[str, Any]) -> None:
    db.update_job(job_id, status="queued", stage="Queued", progress=0)
    _queue.put((job_id, Path(source), params))
    start()


def pending() -> int:
    return _queue.qsize()


def current_job_id() -> int | None:
    return _current_job_id


def friendly_error(exc: BaseException) -> str:
    """Show the cause and the fix instead of a stack trace."""
    if isinstance(exc, (diarize.DiarizationSetupError, audio.FFmpegMissing)):
        return str(exc)
    text = str(exc)
    lowered = text.lower()
    if "out of memory" in lowered:
        return (
            "Out of GPU memory. Lower BATCH_SIZE to 4 in .env, or set "
            "UNLOAD_BETWEEN_STAGES=true, then try again.\n\n"
            f"Original error: {text}"
        )
    if "cudnn" in lowered or "cublas" in lowered:
        return (
            "Could not load the CUDA/cuDNN libraries. Reinstall torch 2.4+ from the "
            "CUDA wheel index:\n"
            "  pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu128\n\n"
            f"Original error: {text}"
        )
    if "1314" in text or "symlink" in lowered or "symbolic link" in lowered:
        return (
            "Blocked by Windows symlink permissions while building the model cache.\n"
            "Try these in order.\n\n"
            "  1) Delete the whole models folder and run again.\n"
            "     Recent versions avoid symlinks entirely, so this usually ends it.\n"
            "  2) If it persists, move the project off the Desktop to a path with no\n"
            "     OneDrive sync, such as C:\\StoryWriter.\n"
            "  3) Or turn on Developer Mode under Windows Settings >\n"
            "     Privacy & security > For developers.\n\n"
            f"Original error: {text}"
        )
    if "401" in text or "403" in text or "gated" in lowered:
        return diarize.GATE_HELP + f"\n\nOriginal error: {text}"
    return f"{type(exc).__name__}: {text}"


def _loop() -> None:
    global _current_job_id
    while True:
        job_id, source, params = _queue.get()
        _current_job_id = job_id
        try:
            _run(job_id, source, params)
        except Exception as exc:  # noqa: BLE001 - the worker must never die
            traceback.print_exc()
            db.update_job(
                job_id,
                status="error",
                stage="Failed",
                error=friendly_error(exc),
                finished_at=db.now(),
            )
        finally:
            _current_job_id = None
            _queue.task_done()


def _run(job_id: int, source: Path, params: dict[str, Any]) -> None:
    db.update_job(job_id, status="running", stage="Starting", progress=1, error="")

    def report(stage: str, percent: float) -> None:
        db.update_job(job_id, stage=stage, progress=float(percent))

    job = db.get_job(job_id)
    name = job["name"] if job else params.get("name", "")

    pipeline.run(
        source,
        name,
        language=params.get("language") or None,
        initial_prompt=params.get("initial_prompt"),
        min_speakers=params.get("min_speakers"),
        max_speakers=params.get("max_speakers"),
        progress=report,
    )

    db.update_job(
        job_id, status="done", stage="Done", progress=100, finished_at=db.now()
    )
