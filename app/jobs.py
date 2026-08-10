"""백그라운드 작업 큐. GPU 가 하나뿐이라 워커 스레드도 하나만 돌린다."""

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
    db.update_job(job_id, status="queued", stage="대기 중", progress=0)
    _queue.put((job_id, Path(source), params))
    start()


def pending() -> int:
    return _queue.qsize()


def current_job_id() -> int | None:
    return _current_job_id


def friendly_error(exc: BaseException) -> str:
    """스택트레이스 대신 원인과 해결책을 보여준다."""
    if isinstance(exc, (diarize.DiarizationSetupError, audio.FFmpegMissing)):
        return str(exc)
    text = str(exc)
    lowered = text.lower()
    if "out of memory" in lowered:
        return (
            "GPU 메모리가 부족합니다. .env 의 BATCH_SIZE 를 4 로 낮추거나 "
            "UNLOAD_BETWEEN_STAGES=true 로 설정한 뒤 다시 시도하세요.\n\n"
            f"원본 오류: {text}"
        )
    if "cudnn" in lowered or "cublas" in lowered:
        return (
            "CUDA/cuDNN 라이브러리를 불러오지 못했습니다. torch 를 2.4 이상 CUDA 휠로 "
            "재설치하세요:\n"
            "  pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu128\n\n"
            f"원본 오류: {text}"
        )
    if "401" in text or "403" in text or "gated" in lowered:
        return diarize.GATE_HELP + f"\n\n원본 오류: {text}"
    return f"{type(exc).__name__}: {text}"


def _loop() -> None:
    global _current_job_id
    while True:
        job_id, source, params = _queue.get()
        _current_job_id = job_id
        try:
            _run(job_id, source, params)
        except Exception as exc:  # noqa: BLE001 - 워커는 절대 죽으면 안 됨
            traceback.print_exc()
            db.update_job(
                job_id,
                status="error",
                stage="실패",
                error=friendly_error(exc),
                finished_at=db.now(),
            )
        finally:
            _current_job_id = None
            _queue.task_done()


def _run(job_id: int, source: Path, params: dict[str, Any]) -> None:
    db.update_job(job_id, status="running", stage="시작", progress=1, error="")

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
        job_id, status="done", stage="완료", progress=100, finished_at=db.now()
    )
