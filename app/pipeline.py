"""전사 → 정렬 → 화자 분리 → 화자 자동 인식 → 저장 전체 흐름.

CLI 로도 쓸 수 있다:
    python -m app.pipeline 회의.mp3 --name 2026-08-10 --max-speakers 4
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import asr, audio, config, db, diarize, matching, render

ProgressFn = Callable[[str, float], None]

GLOSSARY_KEY = "glossary"


def _noop(stage: str, percent: float) -> None:  # pragma: no cover - 기본 콜백
    pass


def build_initial_prompt(extra: str | None = None) -> str:
    """등록된 화자 이름 + 용어사전을 Whisper 의 initial_prompt 로 넘긴다.

    고유명사(사람 이름, 회사 용어)를 미리 알려주면 인식 정확도가 올라간다.
    """
    parts: list[str] = []
    names = list(db.speaker_names())
    if names:
        parts.append("참석자: " + ", ".join(names) + ".")
    glossary = (extra if extra is not None else db.get_setting(GLOSSARY_KEY, "")).strip()
    if glossary:
        parts.append(glossary)
    return " ".join(parts).strip()


def run(
    source: Path,
    name: str,
    language: str | None = None,
    initial_prompt: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress: ProgressFn = _noop,
) -> dict[str, Any]:
    source = Path(source)
    name = render.sanitize_name(name)
    warnings: list[str] = []

    # 1) 오디오 정규화 (16kHz mono wav)
    # 확장자가 .wav 여도 44.1kHz 스테레오일 수 있으므로 항상 변환한다.
    progress("오디오 변환", 5)
    wav_path = config.UPLOAD_DIR / f"{name}.16k.wav"
    audio.to_wav16k(source, wav_path)
    duration = audio.duration_sec(wav_path)

    # 2) 전사
    progress("음성 인식 중", 12)
    waveform = asr.load_audio(wav_path)
    prompt = initial_prompt if initial_prompt is not None else build_initial_prompt()
    result = asr.transcribe(waveform, language=language or None, initial_prompt=prompt)
    detected = result.get("language") or language or "unknown"
    segments = result.get("segments") or []

    # 3) 단어 단위 정렬
    progress("단어 타임스탬프 정렬 중", 58)
    segments, align_warning = asr.align(segments, detected, waveform)
    if align_warning:
        warnings.append(align_warning)
    if config.UNLOAD_BETWEEN_STAGES:
        asr.unload_model()
        asr.unload_align_models()

    # 4) 화자 분리 + 임베딩
    progress("화자 분리 중", 72)
    turns, embeddings, speech_sec = diarize.diarize(
        wav_path, min_speakers=min_speakers, max_speakers=max_speakers
    )
    if config.UNLOAD_BETWEEN_STAGES:
        diarize.unload_pipeline()
    if not turns:
        warnings.append("화자 구간을 찾지 못했습니다. 전체를 한 명으로 처리합니다.")
    if not embeddings:
        warnings.append(
            "화자 임베딩을 얻지 못해 이번 결과는 자동 화자 인식을 건너뜁니다."
        )

    # 5) 세그먼트에 화자 붙이기 (화자 바뀌는 지점에서 분할)
    progress("화자 배정 중", 88)
    segments = diarize.attach_speakers(segments, turns)

    # 6) 등록된 화자와 대조
    progress("등록 화자 대조 중", 93)
    matches = matching.match(embeddings, speech_sec)
    ordered = render.order_labels(segments, speech_sec)
    displays, anon = render.assign_displays(ordered, matches)

    speakers: dict[str, Any] = {}
    for label in ordered:
        info = matches.get(label) or {}
        vector = embeddings.get(label)
        speakers[label] = {
            "display": displays.get(label, label),
            "anon_label": anon.get(label),
            "speaker_id": info.get("speaker_id"),
            "matched": bool(info.get("matched")),
            "score": info.get("score", 0.0),
            "runner_up": info.get("runner_up", 0.0),
            "reason": info.get("reason", ""),
            "total_speech": round(speech_sec.get(label, 0.0), 2),
            "embedding": db.normalize(vector).tolist() if vector is not None else None,
        }

    # 7) 저장
    progress("저장 중", 97)
    payload: dict[str, Any] = {
        "name": name,
        "source_file": source.name,
        "audio_file": wav_path.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration": round(duration, 2),
        "language": detected,
        "initial_prompt": prompt,
        "warnings": warnings,
        "speakers": speakers,
        "segments": [
            {
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "speaker": seg.get("speaker"),
                "text": seg["text"],
                "words": [
                    {
                        "word": word.get("word", ""),
                        "start": round(float(word["start"]), 3),
                        "end": round(float(word["end"]), 3),
                        "speaker": word.get("speaker"),
                    }
                    for word in (seg.get("words") or [])
                    if word.get("start") is not None and word.get("end") is not None
                ],
            }
            for seg in segments
            if (seg.get("text") or "").strip()
        ],
    }
    txt_file, json_file = render.save(payload)
    payload["txt_file"] = str(txt_file)
    payload["json_file"] = str(json_file)

    progress("완료", 100)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="음성 파일 전사 + 화자 분리")
    parser.add_argument("audio", type=Path, help="오디오/영상 파일 경로")
    parser.add_argument("--name", default=None, help="결과 이름 (기본: 오늘 날짜)")
    parser.add_argument("--language", default=None, help="ko / en 등. 생략하면 자동 감지")
    parser.add_argument("--prompt", default=None, help="initial_prompt 직접 지정")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"파일을 찾을 수 없습니다: {args.audio}", file=sys.stderr)
        return 1

    db.init()
    name = render.unique_name(args.name or render.default_name())

    staged = config.UPLOAD_DIR / f"{name}{args.audio.suffix.lower()}"
    if staged.resolve() != args.audio.resolve():
        shutil.copy2(args.audio, staged)

    def show(stage: str, percent: float) -> None:
        print(f"[{percent:5.1f}%] {stage}", flush=True)

    payload = run(
        staged,
        name,
        language=args.language,
        initial_prompt=args.prompt,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        progress=show,
    )

    print()
    for warning in payload.get("warnings", []):
        print(f"[경고] {warning}")
    print(f"언어: {payload['language']}   길이: {payload['duration']:.1f}초")
    print(f"저장: {payload['txt_file']}")
    print(f"      {payload['json_file']}")
    print("-" * 60)
    for line in payload.get("lines", [])[:20]:
        print(f"{line['name']} : {line['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
