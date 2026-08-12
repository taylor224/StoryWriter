"""FastAPI app: uploads, results and speaker management."""

import json
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import audio, cache, config, db, diarize, jobs, matching, pipeline, render

BASE = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    db.reset_stale_jobs()  # clean up rows left running when the server died mid-job
    jobs.start()
    yield


app = FastAPI(title="Speaker-Aware Speech To Text for ChatGPT/Claude", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))


# ── Pages ─────────────────────────────────────────────────────────────
@app.get("/")
def page_index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_name": render.default_name(),
            "glossary": db.get_setting(pipeline.GLOSSARY_KEY, ""),
        },
    )


@app.get("/result/{name}")
def page_result(request: Request, name: str):
    payload = render.load(name)
    if payload is None:
        raise HTTPException(404, f"Result not found: {name}")
    return templates.TemplateResponse(
        request, "result.html", {"name": payload["name"]}
    )


@app.get("/speakers")
def page_speakers(request: Request):
    return templates.TemplateResponse(request, "speakers.html")


# ── Diagnostics ───────────────────────────────────────────────────────
@app.get("/api/health")
def api_health():
    info: dict[str, Any] = {
        "whisper_model": config.WHISPER_MODEL,
        "diarize_model": config.DIARIZE_MODEL,
        "hf_token": bool(config.HF_TOKEN),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "device": config.DEVICE,
        "resolved_device": "unknown",
        "asr_device": "unknown",
        "compute_type": "",
        "gpu": "",
        "match_threshold": config.MATCH_THRESHOLD,
    }
    try:
        import torch

        info["resolved_device"] = config.resolve_device()
        info["asr_device"] = config.asr_device()
        info["compute_type"] = config.resolve_compute_type(info["asr_device"])
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        info["resolved_device"] = f"failed to load torch: {exc}"
    return info


# ── Jobs ──────────────────────────────────────────────────────────────
@app.post("/api/jobs")
async def api_create_job(
    file: UploadFile,
    name: str = Form(""),
    language: str = Form(""),
    initial_prompt: str | None = Form(None),
    use_prompt: str = Form("1"),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXT:
        raise HTTPException(
            400,
            f"Unsupported format: {suffix or '(no extension)'} — "
            f"allowed: {', '.join(sorted(config.ALLOWED_EXT))}",
        )

    result_name = _reserve_name(name)
    staged = config.UPLOAD_DIR / f"{result_name}{suffix}"
    try:
        with staged.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    finally:
        await file.close()

    if staged.stat().st_size == 0:
        staged.unlink(missing_ok=True)
        raise HTTPException(400, "The file is empty.")

    if initial_prompt is not None:
        db.set_setting(pipeline.GLOSSARY_KEY, initial_prompt.strip())

    # Turning this off keeps participant names and the glossary away from Whisper
    # entirely. Batched inference applies the prompt to every window, so it can
    # bleed into the transcript.
    wants_prompt = use_prompt.strip().lower() not in ("0", "false", "off", "")
    params = {
        "language": language.strip(),
        "initial_prompt": (
            pipeline.build_initial_prompt(initial_prompt) if wants_prompt else ""
        ),
        "min_speakers": _opt_int(min_speakers),
        "max_speakers": _opt_int(max_speakers),
        "source": str(staged),  # kept so a retry can find the original again
    }
    job_id = db.create_job(result_name, file.filename or staged.name, params)
    jobs.submit(job_id, staged, params)
    return {"job_id": job_id, "name": result_name, "queued": jobs.pending()}


@app.post("/api/jobs/{job_id}/retry")
def api_retry_job(job_id: int):
    """Resume a failed job. Stages whose inputs are unchanged reuse their cache."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "That job is already queued or running.")

    try:
        params = json.loads(job["params"] or "{}")
    except json.JSONDecodeError:
        params = {}

    source = Path(params.get("source") or "")
    if not source.exists():
        raise HTTPException(
            400, "The original audio is gone. Please upload the file again."
        )

    done = cache.stages(job["name"])
    db.update_job(
        job_id, status="queued", stage="Queued", progress=0, error="", finished_at=""
    )
    jobs.submit(job_id, source, params)
    return {"ok": True, "job_id": job_id, "resumed_from": done}


@app.get("/api/jobs")
def api_list_jobs(limit: int = 30):
    rows = db.list_jobs(limit)
    for row in rows:
        # So the UI can show how far a failed job got
        row["cached_stages"] = (
            cache.stages(row["name"]) if row["status"] == "error" else []
        )
    return {"jobs": rows, "running": jobs.current_job_id()}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: int):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    job["cached_stages"] = cache.stages(job["name"])
    return job


# ── Results ───────────────────────────────────────────────────────────
@app.get("/api/results")
def api_list_results():
    return {"results": render.list_results()}


@app.get("/api/results/{name}")
def api_get_result(name: str):
    payload = render.load(name)
    if payload is None:
        raise HTTPException(404, "Result not found.")
    # The UI uses neither embeddings nor word arrays, and they are big (hundreds of KB per hour)
    slim = dict(payload)
    slim["speakers"] = _strip_embeddings(payload.get("speakers", {}))
    slim["segments"] = [
        {k: v for k, v in seg.items() if k != "words"}
        for seg in payload.get("segments", [])
    ]
    slim["registered"] = db.list_speakers()
    return slim


@app.get("/api/results/{name}/clip")
def api_clip(name: str, start: float = 0.0, end: float = 0.0):
    """The audio for one transcript line, played when that line is clicked.

    We do not ship the whole wav: a 10-hour recording is 1.1GB, and nothing would
    play until the browser had all of it.
    """
    payload = render.load(name)
    if payload is None:
        raise HTTPException(404, "Result not found.")

    # We wrote audio_file ourselves, but the result json is a file the user can
    # edit. Take only the filename so nothing can escape UPLOAD_DIR.
    wav_path = config.UPLOAD_DIR / Path(payload.get("audio_file") or "").name
    if not wav_path.is_file():
        raise HTTPException(404, "Source audio is missing (it may have been deleted from uploads).")

    # Widen very slightly on both sides so the first sound is not clipped
    data = audio.clip_wav(wav_path, max(0.0, start - 0.15), end + 0.25)
    return Response(
        content=data,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store", "Content-Length": str(len(data))},
    )


@app.get("/api/results/{name}/download")
def api_download(name: str, fmt: str = "txt"):
    path = render.txt_path(name) if fmt == "txt" else render.json_path(name)
    if not path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.post("/api/results/{name}/speakers")
def api_label_speakers(name: str, body: dict[str, Any] = Body(...)):
    """Confirm speaker names -> enroll voiceprints -> rebuild the txt (no re-transcription)."""
    payload = render.load(name)
    if payload is None:
        raise HTTPException(404, "Result not found.")

    assignments = body.get("assignments") or []
    speakers = payload.get("speakers") or {}
    enrolled: list[str] = []

    for item in assignments:
        label = item.get("label")
        if label not in speakers:
            continue
        entry = speakers[label]
        source = f"result:{payload['name']}:{label}"

        speaker_id = item.get("speaker_id")
        raw_name = (item.get("name") or "").strip()
        if speaker_id:
            existing = db.get_speaker(int(speaker_id))
            if existing is None:
                raise HTTPException(404, f"Speaker id {speaker_id} not found.")
            raw_name = existing["name"]

        if not raw_name:
            # Unassigned — also cancel the enrollment
            db.delete_voiceprints_from_source(source)
            entry.update(speaker_id=None, matched=False, manual=False, reason="unassigned by user")
            continue

        sid = db.upsert_speaker(raw_name)
        entry.update(speaker_id=sid, matched=True, manual=True, display=raw_name,
                     reason="set by user")
        if entry.get("embedding"):
            matching.enroll(
                sid, entry["embedding"], source=source,
                speech_sec=entry.get("total_speech", 0.0),
            )
            enrolled.append(raw_name)

    # Keep Speaker A/B/C for unnamed speakers (naming one must not shift the others)
    ordered = list(speakers.keys())
    matches = {
        label: {"matched": bool(info.get("speaker_id")),
                "name": info.get("display") if info.get("speaker_id") else None}
        for label, info in speakers.items()
    }
    anon = {
        label: info["anon_label"]
        for label, info in speakers.items()
        if info.get("anon_label")
    }
    displays, anon = render.assign_displays(ordered, matches, anon)
    for label, display in displays.items():
        speakers[label]["display"] = display
        speakers[label]["anon_label"] = anon.get(label)

    render.save(payload)
    return {
        "ok": True,
        "enrolled": enrolled,
        "speakers": _strip_embeddings(speakers),
        "lines": payload.get("lines", []),
    }


@app.delete("/api/results/{name}")
def api_delete_result(name: str):
    if not render.delete(name):
        raise HTTPException(404, "Result not found.")
    return {"ok": True}


# ── Speakers ──────────────────────────────────────────────────────────
@app.get("/api/speakers")
def api_list_speakers():
    people = db.list_speakers()
    for person in people:
        person["voiceprints"] = db.voiceprints_of(person["id"])
    return {"speakers": people}


@app.post("/api/speakers")
def api_create_speaker(body: dict[str, Any] = Body(...)):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Please enter a name.")
    return {"id": db.upsert_speaker(name, body.get("note", ""))}


@app.patch("/api/speakers/{speaker_id}")
def api_rename_speaker(speaker_id: int, body: dict[str, Any] = Body(...)):
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(400, "Please enter a name.")
    current = db.get_speaker(speaker_id)
    if current is None:
        raise HTTPException(404, "Speaker not found.")
    clash = db.get_speaker_by_name(new_name)
    if clash and int(clash["id"]) != speaker_id:
        raise HTTPException(409, f"A speaker named '{new_name}' already exists.")

    db.rename_speaker(speaker_id, new_name)
    updated = _propagate_rename(speaker_id, new_name) if body.get("update_results") else 0
    return {"ok": True, "updated_results": updated}


@app.delete("/api/speakers/{speaker_id}")
def api_delete_speaker(speaker_id: int):
    if db.get_speaker(speaker_id) is None:
        raise HTTPException(404, "Speaker not found.")
    db.delete_speaker(speaker_id)
    return {"ok": True}


@app.post("/api/speakers/{speaker_id}/samples")
async def api_add_sample(speaker_id: int, file: UploadFile):
    person = db.get_speaker(speaker_id)
    if person is None:
        raise HTTPException(404, "Speaker not found.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format: {suffix or '(no extension)'}")

    safe = render.sanitize_name(f"{person['name']}-{db.now().replace(':', '')}")
    raw_path = config.SAMPLE_DIR / f"{safe}{suffix}"
    try:
        with raw_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    finally:
        await file.close()

    wav_path = config.SAMPLE_DIR / f"{safe}.wav"
    try:
        # Apply the same filter transcription uses. A voiceprint taken from
        # differently-processed audio makes the same voice look different later.
        audio.to_wav16k(raw_path, wav_path, config.AUDIO_FILTER)
        length = audio.duration_sec(wav_path)
        if length < 3:
            raise HTTPException(400, "The sample is too short. At least 10 seconds is recommended.")
        vector, speech = diarize.embed_single_speaker(wav_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, jobs.friendly_error(exc)) from exc

    db.add_voiceprint(
        speaker_id, vector, source=f"sample:{wav_path.name}", speech_sec=speech or length
    )
    return {"ok": True, "speech_sec": round(speech or length, 2)}


# ── Settings ──────────────────────────────────────────────────────────
@app.get("/api/settings")
def api_get_settings():
    return {"glossary": db.get_setting(pipeline.GLOSSARY_KEY, "")}


@app.post("/api/settings")
def api_set_settings(body: dict[str, Any] = Body(...)):
    if "glossary" in body:
        db.set_setting(pipeline.GLOSSARY_KEY, (body.get("glossary") or "").strip())
    return {"ok": True}


# ── Helpers ───────────────────────────────────────────────────────────
_name_lock = threading.Lock()


def _reserve_name(requested: str) -> str:
    """Claim a result name.

    render.unique_name alone can collide with a queued or running job that has no
    result file yet, so this checks the queued names too.
    """
    with _name_lock:
        base = render.sanitize_name(requested or render.default_name())
        candidate, index = base, 1
        while (config.RESULT_DIR / f"{candidate}.json").exists() or db.name_taken(candidate):
            index += 1
            candidate = f"{base}-{index}"
        return candidate


def _strip_embeddings(speakers: dict[str, Any]) -> dict[str, Any]:
    return {
        label: {k: v for k, v in info.items() if k != "embedding"}
        for label, info in speakers.items()
    }


def _opt_int(value: str) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _propagate_rename(speaker_id: int, new_name: str) -> int:
    """Refresh display names and txt files for existing results assigned to this speaker."""
    count = 0
    for item in render.list_results():
        payload = render.load(item["name"])
        if payload is None:
            continue
        changed = False
        for info in (payload.get("speakers") or {}).values():
            if info.get("speaker_id") == speaker_id and info.get("display") != new_name:
                info["display"] = new_name
                changed = True
        if changed:
            render.save(payload)
            count += 1
    return count


@app.exception_handler(HTTPException)
def _http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def serve() -> None:
    """Entry point for run.bat. Uses HOST/PORT from .env as-is."""
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{config.HOST}:{config.PORT}"
    print(f"\n  Opening at {url}\n")
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    serve()
