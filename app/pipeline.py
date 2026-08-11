"""전사 → 정렬 → 화자 분리 → 화자 자동 인식 → 저장 전체 흐름.

각 단계의 출력은 data/cache 에 남는다. 실패한 작업을 다시 돌리면 입력이 같은
단계는 건너뛰고 실패한 지점부터 이어서 계산한다. 전사가 가장 비싼 단계라
화자 분리에서 죽었다고 전사를 다시 돌리는 일은 없어야 한다.

CLI 로도 쓸 수 있다:
    python -m app.pipeline 회의.mp3 --name 2026-08-10 --max-speakers 4
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import asr, audio, cache, config, db, diarize, matching, render

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
    resume: bool = True,
) -> dict[str, Any]:
    source = Path(source)
    name = render.sanitize_name(name)
    warnings: list[str] = []
    reused: list[str] = []

    def cached(stage: str, key: dict[str, Any]) -> dict[str, Any] | None:
        return cache.load(name, stage, key) if resume else None

    # ── 1) 오디오 정규화 (16kHz mono wav) ─────────────────────────────
    # 확장자가 .wav 여도 44.1kHz 스테레오일 수 있으므로 항상 변환한다.
    progress("오디오 변환", 5)
    wav_path = config.UPLOAD_DIR / f"{name}.16k.wav"
    audio_key = cache.audio_key(source)

    hit = cached("audio", audio_key)
    if hit and wav_path.exists():
        duration = hit["duration"]
        reused.append("오디오 변환")
    else:
        audio.to_wav16k(source, wav_path)
        duration = audio.duration_sec(wav_path)
        cache.save(name, "audio", audio_key, {"duration": duration})

    # 파형은 전사와 정렬에서만 쓴다. 둘 다 캐시에 맞으면 읽지 않는다.
    loaded: dict[str, Any] = {}

    def waveform():
        if "value" not in loaded:
            loaded["value"] = asr.load_audio(wav_path)
        return loaded["value"]

    prompt = initial_prompt if initial_prompt is not None else build_initial_prompt()

    # ── 2) 전사 ─────────────────────────────────────────────────────
    transcribe_key = {
        "audio": audio_key,
        "model": config.WHISPER_MODEL,
        "language": language or "",
        "prompt": prompt,
    }
    hit = cached("transcribe", transcribe_key)
    if hit:
        detected, segments = hit["language"], hit["segments"]
        detection = hit.get("detection") or {}
        reused.append("음성 인식")
        progress("음성 인식 결과 재사용", 58)
    else:
        progress("음성 인식 중", 12)
        result = asr.transcribe(
            waveform(), language=language or None, initial_prompt=prompt
        )
        detected = result.get("language") or language or "unknown"
        segments = result.get("segments") or []
        detection = result.get("detection") or {}
        cache.save(
            name, "transcribe", transcribe_key,
            {"language": detected, "segments": segments, "detection": detection},
        )

    # 언어를 잘못 잡으면 Whisper 는 하지도 않은 말을 그럴듯하게 지어낸다.
    # 조용히 넘어가면 원인을 못 찾으므로 판정 근거를 결과에 남긴다.
    if detection.get("auto"):
        votes = detection.get("votes") or {}
        confidence = detection.get("confidence", 0.0)
        if confidence < 0.9 or len(votes) > 1:
            tally = ", ".join(f"{k} {v}" for k, v in sorted(votes.items(), key=lambda x: -x[1]))
            warnings.append(
                f"언어를 '{detected}' 로 자동 감지했습니다 (확신도 {confidence}). "
                f"표본별 득표: {tally}. "
                "결과에 하지 않은 말이 섞여 있으면 언어를 고정하고 다시 돌리세요."
            )

    # ── 3) 단어 단위 정렬 ────────────────────────────────────────────
    align_key = {"transcribe": transcribe_key, "language": detected}
    hit = cached("align", align_key)
    if hit:
        segments = hit["segments"]
        if hit.get("warning"):
            warnings.append(hit["warning"])
        reused.append("단어 정렬")
        progress("단어 정렬 결과 재사용", 70)
    else:
        progress("단어 타임스탬프 정렬 중", 58)
        segments, align_warning = asr.align(segments, detected, waveform())
        if align_warning:
            warnings.append(align_warning)
        cache.save(
            name, "align", align_key,
            {"segments": segments, "warning": align_warning},
        )

    if config.UNLOAD_BETWEEN_STAGES:
        asr.unload_model()
        asr.unload_align_models()
    loaded.pop("value", None)  # 파형은 여기까지만 필요하다

    # ── 4) 화자 분리 + 임베딩 ────────────────────────────────────────
    diarize_key = {
        "audio": audio_key,
        "model": config.DIARIZE_MODEL,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }
    hit = cached("diarize", diarize_key)
    if hit:
        turns = hit["turns"]
        embeddings = {
            label: np.asarray(vector, dtype=np.float32)
            for label, vector in hit["embeddings"].items()
        }
        speech_sec = hit["speech_sec"]
        reused.append("화자 분리")
        progress("화자 분리 결과 재사용", 88)
    else:
        progress("화자 분리 중", 72)
        turns, embeddings, speech_sec = diarize.diarize(
            wav_path, min_speakers=min_speakers, max_speakers=max_speakers
        )
        cache.save(
            name, "diarize", diarize_key,
            {
                "turns": turns,
                "embeddings": {k: v.tolist() for k, v in embeddings.items()},
                "speech_sec": speech_sec,
            },
        )
        if config.UNLOAD_BETWEEN_STAGES:
            diarize.unload_pipeline()

    if not turns:
        warnings.append("화자 구간을 찾지 못했습니다. 전체를 한 명으로 처리합니다.")
    if not embeddings:
        warnings.append(
            "화자 임베딩을 얻지 못해 이번 결과는 자동 화자 인식을 건너뜁니다."
        )

    # ── 5) 세그먼트에 화자 붙이기 (화자 바뀌는 지점에서 분할) ─────────
    # 여기부터는 싼 단계라 캐시하지 않는다. 등록된 화자가 바뀌었을 수 있으므로
    # 재시도할 때마다 다시 대조하는 편이 오히려 맞다.
    progress("화자 배정 중", 92)
    segments = diarize.attach_speakers(segments, turns)

    # ── 6) 등록된 화자와 대조 ────────────────────────────────────────
    progress("등록 화자 대조 중", 95)
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

    # ── 7) 저장 ─────────────────────────────────────────────────────
    progress("저장 중", 98)
    payload: dict[str, Any] = {
        "name": name,
        "source_file": source.name,
        "audio_file": wav_path.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration": round(duration, 2),
        "language": detected,
        "language_detection": detection,
        "initial_prompt": prompt,
        "warnings": warnings,
        "reused_stages": reused,
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

    cache.clear(name)  # 성공했으니 중간 결과는 필요 없다
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
    parser.add_argument(
        "--no-resume", action="store_true", help="중간 결과 캐시를 무시하고 처음부터"
    )
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"파일을 찾을 수 없습니다: {args.audio}", file=sys.stderr)
        return 1

    db.init()
    name = render.sanitize_name(args.name or render.default_name())
    if not cache.stages(name) and render.json_path(name).exists():
        name = render.unique_name(name)

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
        resume=not args.no_resume,
    )

    print()
    for warning in payload.get("warnings", []):
        print(f"[경고] {warning}")
    if payload.get("reused_stages"):
        print(f"재사용한 단계: {', '.join(payload['reused_stages'])}")
    print(f"언어: {payload['language']}   길이: {payload['duration']:.1f}초")
    print(f"저장: {payload['txt_file']}")
    print(f"      {payload['json_file']}")
    print("-" * 60)
    for line in payload.get("lines", [])[:20]:
        print(f"{line['name']} : {line['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
