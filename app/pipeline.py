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

from . import (
    asr, audio, cache, cleanup, config, db, diarize, matching, render, stitch, vad,
)

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

    # ── 2) 무음 구간 계산 ─────────────────────────────────────────────
    # 긴 침묵을 어디서 들어낼지만 정하고, 파형을 실제로 자르는 건 쓰는 쪽에서
    # 한다. 전사·정렬·화자 분리가 전부 잘라낸 파형으로 돌아가고, 각자 결과를
    # 내놓기 직전에 timeline 으로 원본 시각을 되돌린다. 그래서 5)번 이후는
    # 잘린 적이 있다는 사실을 몰라도 된다.
    #
    # 남긴 구간 목록은 작아서 캐시에 담아 둔다. 재시도 때 파형을 다시 읽지
    # 않고도 화자 분리가 같은 기준으로 잘라 쓸 수 있어야 하기 때문.
    trim_params = vad.params() if config.TRIM_SILENCE else None
    trim_key = {"audio": audio_key, "params": trim_params}

    hit = cached("trim", trim_key) if config.TRIM_SILENCE else None
    if not config.TRIM_SILENCE:
        # ffprobe 가 없으면 duration 이 0 이므로 wav 헤더에서 직접 센다.
        # 여기가 0 이면 조각 범위가 빈 구간이 되어 전사가 통째로 날아간다.
        timeline = vad.Timeline.identity(audio.sample_count(wav_path))
    elif hit:
        timeline = vad.Timeline(hit["regions"], hit["total"])
        reused.append("무음 제거")
    else:
        progress("무음 구간 확인", 8)
        timeline, note = vad.plan(audio.read_wav(wav_path))
        if note:
            warnings.append(note)
        cache.save(
            name, "trim", trim_key,
            {"regions": timeline.regions, "total": timeline.total},
        )
    trim_info = timeline.stats()
    if timeline.trimmed:
        progress(f"무음 {trim_info['removed']:.0f}초 제거", 10)

    prompt = initial_prompt if initial_prompt is not None else build_initial_prompt()

    # ── 3) 조각 나누기 ────────────────────────────────────────────────
    # 긴 파일을 한 번에 돌리면 느린 데다, pyannote 가 같은 사람을 여러 명으로
    # 갈라놓기 시작한다 (10시간 동안 목소리도 마이크 상태도 변하니까).
    # 조각마다 전사 → 정렬 → 화자 분리를 끝내고 마지막에 목소리로 이어 붙인다.
    #
    # 조각 Timeline 은 원본 좌표를 그대로 들고 있다. 그래서 조각 안에서 나온
    # 타임스탬프는 restore 한 번이면 곧바로 원본 시각이 된다 — 조각 번호를
    # 들고 다니며 오프셋을 더할 일이 없다.
    pieces = vad.chunks(timeline, config.CHUNK_SEC)
    if len(pieces) > 1:
        progress(f"{len(pieces)}개 조각으로 나눠 처리", 11)

    def stage(base: str, index: int) -> str:
        return base if len(pieces) == 1 else f"{base}.{index}"

    all_segments: list[dict] = []
    parts: list[tuple[list[dict], dict[str, np.ndarray], dict[str, float]]] = []
    detected: str = language or ""
    detection: dict[str, Any] = {}
    span_low, span_high = 12.0, 88.0

    for index, piece in enumerate(pieces):
        head = f"[{index + 1}/{len(pieces)}] " if len(pieces) > 1 else ""
        base = span_low + (span_high - span_low) * index / len(pieces)
        step = (span_high - span_low) / len(pieces)
        origin = piece.span[0]
        chunk_id = list(piece.span) if len(pieces) > 1 else None
        held: dict[str, Any] = {}

        def waveform(piece=piece, origin=origin, held=held):
            """이 조각의 파형. 두 단계 다 캐시에 맞으면 파일을 읽지도 않는다.

            wav 전체가 아니라 조각 범위만 읽는다. 10시간을 통째로 올리면
            파형만 2.3GB 라 조각으로 나눈 의미가 없어진다.
            """
            if "value" not in held:
                window = audio.read_wav(wav_path, origin, piece.span[1])
                held["value"] = piece.apply(window, origin)
            return held["value"]

        # ── 3-1) 전사 ────────────────────────────────────────────────
        # 언어는 첫 조각에서만 감지하고 나머지는 그 결과를 강제한다. 조각마다
        # 따로 감지하면 조용한 조각 하나가 엉뚱한 언어로 새 버린다.
        forced = language or (detected if index else "")
        transcribe_key = {
            "audio": audio_key,
            "model": config.WHISPER_MODEL,
            "language": forced,
            "prompt": prompt,
            # 무음을 다르게 자르면 전사 입력 자체가 달라진다. 키에 없으면 설정을
            # 바꿔 재시도했는데 옛 결과가 그대로 살아남는다.
            "trim": trim_params,
            "chunk": chunk_id,
        }
        hit = cached(stage("transcribe", index), transcribe_key)
        if hit:
            chunk_language, segments = hit["language"], hit["segments"]
            chunk_detection = hit.get("detection") or {}
            reused.append(f"{head}음성 인식")
            progress(f"{head}음성 인식 결과 재사용", base + step * 0.6)
        else:
            progress(f"{head}음성 인식 중", base)
            result = asr.transcribe(
                waveform(), language=forced or None, initial_prompt=prompt
            )
            chunk_language = result.get("language") or forced or "unknown"
            segments = result.get("segments") or []
            chunk_detection = result.get("detection") or {}
            cache.save(
                name, stage("transcribe", index), transcribe_key,
                {
                    "language": chunk_language,
                    "segments": segments,
                    "detection": chunk_detection,
                },
            )
        if not index:
            detected, detection = chunk_language, chunk_detection

        # ── 3-2) 단어 단위 정렬 ──────────────────────────────────────
        # 정렬까지는 조각 안 기준 시각이다 (transcribe 캐시도 그 기준).
        # 정렬이 끝나는 여기서 원본 시각으로 되돌리고, align 캐시부터는 원본
        # 기준으로 저장한다. 두 캐시의 기준이 다르다는 점만 지키면 된다.
        align_key = {"transcribe": transcribe_key, "language": chunk_language}
        hit = cached(stage("align", index), align_key)
        if hit:
            segments = hit["segments"]
            if hit.get("warning"):
                warnings.append(hit["warning"])
            reused.append(f"{head}단어 정렬")
            progress(f"{head}단어 정렬 결과 재사용", base + step * 0.7)
        else:
            progress(f"{head}단어 타임스탬프 정렬 중", base + step * 0.55)
            segments, align_warning = asr.align(segments, chunk_language, waveform())
            segments = piece.restore(segments)
            if align_warning and align_warning not in warnings:
                warnings.append(align_warning)
            cache.save(
                name, stage("align", index), align_key,
                {"segments": segments, "warning": align_warning},
            )
        all_segments.extend(segments)

        if config.UNLOAD_BETWEEN_STAGES:
            asr.unload_model()
            asr.unload_align_models()
        held.clear()  # 화자 분리는 wav 를 다시 읽으므로 파형은 여기까지

        # ── 3-3) 화자 분리 + 임베딩 ─────────────────────────────────
        # pyannote 의 세그멘테이션은 슬라이딩 윈도로 주어진 오디오 전체를 훑기
        # 때문에 침묵도 그대로 비용이다. 여기에 조각 Timeline 을 넘기는 것이
        # 긴 녹음에서 가장 크게 줄어드는 지점. turns 는 diarize 안에서
        # 원본 시각으로 되돌아온다.
        diarize_key = {
            "audio": audio_key,
            "model": config.DIARIZE_MODEL,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "trim": trim_params,
            "chunk": chunk_id,
        }
        hit = cached(stage("diarize", index), diarize_key)
        if hit:
            turns = hit["turns"]
            embeddings = {
                label: np.asarray(vector, dtype=np.float32)
                for label, vector in hit["embeddings"].items()
            }
            speech_sec = hit["speech_sec"]
            reused.append(f"{head}화자 분리")
            progress(f"{head}화자 분리 결과 재사용", base + step * 0.95)
        else:
            progress(f"{head}화자 분리 중", base + step * 0.75)
            turns, embeddings, speech_sec = diarize.diarize(
                wav_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                timeline=piece,
            )
            cache.save(
                name, stage("diarize", index), diarize_key,
                {
                    "turns": turns,
                    "embeddings": {k: v.tolist() for k, v in embeddings.items()},
                    "speech_sec": speech_sec,
                },
            )
            if config.UNLOAD_BETWEEN_STAGES:
                diarize.unload_pipeline()
        parts.append((turns, embeddings, speech_sec))

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

    # ── 4) 조각 합치기 ────────────────────────────────────────────────
    all_segments.sort(key=lambda seg: (seg.get("start", 0.0), seg.get("end", 0.0)))
    if len(parts) == 1:
        turns, embeddings, speech_sec = parts[0]
    else:
        # 합치는 건 싸고, 임계값을 바꿔 다시 돌려보고 싶은 단계라 캐시하지 않는다
        progress("조각 이어 붙이는 중", 89)
        turns, embeddings, speech_sec, notes = stitch.merge(parts)
        warnings.append(stitch.summary(len(parts), len(speech_sec)))
        for note in notes:
            print(f"[화자 이어붙이기] {note}", flush=True)
    segments = all_segments

    # ── 5) 환각 걸러내기 ──────────────────────────────────────────────
    # Whisper 가 잡음 구간에서 지어낸 문장을 버린다. 화자 배정 전에 해야
    # 지어낸 말이 화자 통계까지 오염시키지 않는다.
    # 확실한 것만 버리고, 애매한 것은 남긴 채 표시만 한다.
    segments, dropped, suspect = cleanup.clean(segments)
    warnings.extend(cleanup.summary(dropped, suspect))

    if not turns:
        warnings.append("화자 구간을 찾지 못했습니다. 전체를 한 명으로 처리합니다.")
    if not embeddings:
        warnings.append(
            "화자 임베딩을 얻지 못해 이번 결과는 자동 화자 인식을 건너뜁니다."
        )

    # ── 6) 세그먼트에 화자 붙이기 (화자 바뀌는 지점에서 분할) ─────────
    # 여기부터는 싼 단계라 캐시하지 않는다. 등록된 화자가 바뀌었을 수 있으므로
    # 재시도할 때마다 다시 대조하는 편이 오히려 맞다.
    progress("화자 배정 중", 92)
    segments = diarize.attach_speakers(segments, turns)

    # ── 7) 등록된 화자와 대조 ────────────────────────────────────────
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

    # ── 8) 저장 ─────────────────────────────────────────────────────
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
        "trim": trim_info,
        "chunks": [
            {"start": round(p.span[0] / vad.SAMPLE_RATE, 2),
             "end": round(p.span[1] / vad.SAMPLE_RATE, 2)}
            for p in pieces
        ] if len(pieces) > 1 else [],
        "dropped": dropped,
        "suspect": suspect,
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
    parser.add_argument(
        "--no-trim", action="store_true", help="전사 전 무음 제거를 끈다"
    )
    args = parser.parse_args(argv)

    if args.no_trim:
        config.TRIM_SILENCE = False

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
    trim = payload.get("trim") or {}
    if trim.get("enabled"):
        print(
            f"무음 제거: {trim['removed']:.1f}초 잘라내고 {trim['kept']:.1f}초를 전사 "
            f"(구간 {trim['regions']}개). 타임스탬프는 원본 기준."
        )
    if len(payload.get("chunks") or []) > 1:
        print(f"조각 처리: {len(payload['chunks'])}개로 나눠 돌리고 목소리로 이어 붙임")
    for item in payload.get("dropped") or []:
        print(f"[제외] {render.timestamp(item['start'])} \"{item['text']}\" — {item['reason']}")
    for item in payload.get("suspect") or []:
        print(f"[의심] {render.timestamp(item['start'])} \"{item['text']}\" — {item['reason']}")
    print(f"언어: {payload['language']}   길이: {payload['duration']:.1f}초")
    print(f"저장: {payload['txt_file']}")
    print(f"      {payload['json_file']}")
    print("-" * 60)
    for line in payload.get("lines", [])[:20]:
        print(f"{line['name']} : {line['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
