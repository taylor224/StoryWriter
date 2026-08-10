"""FastAPI 앱: 업로드 / 결과 / 화자 관리."""

import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import audio, config, db, diarize, jobs, matching, pipeline, render

BASE = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    db.reset_stale_jobs()  # 서버가 작업 도중 죽었을 때 남은 running 정리
    jobs.start()
    yield


app = FastAPI(title="화자 구분 음성 기록기", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))


# ── 페이지 ────────────────────────────────────────────────────────────
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
        raise HTTPException(404, f"결과를 찾을 수 없습니다: {name}")
    return templates.TemplateResponse(
        request, "result.html", {"name": payload["name"]}
    )


@app.get("/speakers")
def page_speakers(request: Request):
    return templates.TemplateResponse(request, "speakers.html")


# ── 진단 ──────────────────────────────────────────────────────────────
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
        info["resolved_device"] = f"torch 로드 실패: {exc}"
    return info


# ── 작업 ──────────────────────────────────────────────────────────────
@app.post("/api/jobs")
async def api_create_job(
    file: UploadFile,
    name: str = Form(""),
    language: str = Form(""),
    initial_prompt: str | None = Form(None),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXT:
        raise HTTPException(
            400,
            f"지원하지 않는 형식입니다: {suffix or '(확장자 없음)'} — "
            f"허용: {', '.join(sorted(config.ALLOWED_EXT))}",
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
        raise HTTPException(400, "빈 파일입니다.")

    if initial_prompt is not None:
        db.set_setting(pipeline.GLOSSARY_KEY, initial_prompt.strip())

    params = {
        "language": language.strip(),
        "initial_prompt": pipeline.build_initial_prompt(initial_prompt),
        "min_speakers": _opt_int(min_speakers),
        "max_speakers": _opt_int(max_speakers),
    }
    job_id = db.create_job(result_name, file.filename or staged.name, params)
    jobs.submit(job_id, staged, params)
    return {"job_id": job_id, "name": result_name, "queued": jobs.pending()}


@app.get("/api/jobs")
def api_list_jobs(limit: int = 30):
    return {"jobs": db.list_jobs(limit), "running": jobs.current_job_id()}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: int):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return job


# ── 결과 ──────────────────────────────────────────────────────────────
@app.get("/api/results")
def api_list_results():
    return {"results": render.list_results()}


@app.get("/api/results/{name}")
def api_get_result(name: str):
    payload = render.load(name)
    if payload is None:
        raise HTTPException(404, "결과를 찾을 수 없습니다.")
    # 임베딩과 단어 배열은 화면에서 안 쓰고 용량만 크다 (1시간 회의면 수백 KB)
    slim = dict(payload)
    slim["speakers"] = _strip_embeddings(payload.get("speakers", {}))
    slim["segments"] = [
        {k: v for k, v in seg.items() if k != "words"}
        for seg in payload.get("segments", [])
    ]
    slim["registered"] = db.list_speakers()
    return slim


@app.get("/api/results/{name}/download")
def api_download(name: str, fmt: str = "txt"):
    path = render.txt_path(name) if fmt == "txt" else render.json_path(name)
    if not path.exists():
        raise HTTPException(404, "파일이 없습니다.")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.post("/api/results/{name}/speakers")
def api_label_speakers(name: str, body: dict[str, Any] = Body(...)):
    """화자 이름 확정 → 보이스프린트 등록 → txt 재생성 (재전사 없음)."""
    payload = render.load(name)
    if payload is None:
        raise HTTPException(404, "결과를 찾을 수 없습니다.")

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
                raise HTTPException(404, f"화자 id {speaker_id} 를 찾을 수 없습니다.")
            raw_name = existing["name"]

        if not raw_name:
            # 지정 해제 — 등록도 취소
            db.delete_voiceprints_from_source(source)
            entry.update(speaker_id=None, matched=False, manual=False, reason="사용자가 지정 해제")
            continue

        sid = db.upsert_speaker(raw_name)
        entry.update(speaker_id=sid, matched=True, manual=True, display=raw_name,
                     reason="사용자 지정")
        if entry.get("embedding"):
            matching.enroll(
                sid, entry["embedding"], source=source,
                speech_sec=entry.get("total_speech", 0.0),
            )
            enrolled.append(raw_name)

    # 미지정 화자의 화자A/B/C 는 그대로 유지한다 (한 명 이름 지었다고 번호가 밀리면 안 됨)
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
        raise HTTPException(404, "결과를 찾을 수 없습니다.")
    return {"ok": True}


# ── 화자 ──────────────────────────────────────────────────────────────
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
        raise HTTPException(400, "이름을 입력하세요.")
    return {"id": db.upsert_speaker(name, body.get("note", ""))}


@app.patch("/api/speakers/{speaker_id}")
def api_rename_speaker(speaker_id: int, body: dict[str, Any] = Body(...)):
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(400, "이름을 입력하세요.")
    current = db.get_speaker(speaker_id)
    if current is None:
        raise HTTPException(404, "화자를 찾을 수 없습니다.")
    clash = db.get_speaker_by_name(new_name)
    if clash and int(clash["id"]) != speaker_id:
        raise HTTPException(409, f"'{new_name}' 은(는) 이미 있는 화자입니다.")

    db.rename_speaker(speaker_id, new_name)
    updated = _propagate_rename(speaker_id, new_name) if body.get("update_results") else 0
    return {"ok": True, "updated_results": updated}


@app.delete("/api/speakers/{speaker_id}")
def api_delete_speaker(speaker_id: int):
    if db.get_speaker(speaker_id) is None:
        raise HTTPException(404, "화자를 찾을 수 없습니다.")
    db.delete_speaker(speaker_id)
    return {"ok": True}


@app.post("/api/speakers/{speaker_id}/samples")
async def api_add_sample(speaker_id: int, file: UploadFile):
    person = db.get_speaker(speaker_id)
    if person is None:
        raise HTTPException(404, "화자를 찾을 수 없습니다.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXT:
        raise HTTPException(400, f"지원하지 않는 형식입니다: {suffix or '(확장자 없음)'}")

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
        audio.to_wav16k(raw_path, wav_path)
        length = audio.duration_sec(wav_path)
        if length < 3:
            raise HTTPException(400, "샘플이 너무 짧습니다. 10초 이상을 권장합니다.")
        vector, speech = diarize.embed_single_speaker(wav_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, jobs.friendly_error(exc)) from exc

    db.add_voiceprint(
        speaker_id, vector, source=f"sample:{wav_path.name}", speech_sec=speech or length
    )
    return {"ok": True, "speech_sec": round(speech or length, 2)}


# ── 설정 ──────────────────────────────────────────────────────────────
@app.get("/api/settings")
def api_get_settings():
    return {"glossary": db.get_setting(pipeline.GLOSSARY_KEY, "")}


@app.post("/api/settings")
def api_set_settings(body: dict[str, Any] = Body(...)):
    if "glossary" in body:
        db.set_setting(pipeline.GLOSSARY_KEY, (body.get("glossary") or "").strip())
    return {"ok": True}


# ── 헬퍼 ──────────────────────────────────────────────────────────────
_name_lock = threading.Lock()


def _reserve_name(requested: str) -> str:
    """결과 이름을 선점한다.

    render.unique_name 만 쓰면 결과 파일이 아직 없는 대기/진행 중 작업과
    이름이 겹칠 수 있다. 큐에 올라간 이름까지 확인한다.
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
    """이 화자로 지정된 기존 결과들의 표시 이름과 txt 를 갱신한다."""
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
    """run.bat 진입점. .env 의 HOST/PORT 를 그대로 쓴다."""
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{config.HOST}:{config.PORT}"
    print(f"\n  {url}  에서 열립니다.\n")
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    serve()
