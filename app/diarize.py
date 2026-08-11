"""pyannote.audio 화자 분리 + 화자 임베딩 추출.

whisperx 의 DiarizationPipeline 래퍼 대신 pyannote 를 직접 호출한다.
이유: 래퍼가 `return_embeddings=True` 를 노출하지 않는데, 이 임베딩이
"한 번 저장한 화자를 다음 파일에서 자동 인식"하는 기능의 전부이기 때문.
"""

import bisect
import threading
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from . import asr, audio, config, vad

_lock = threading.RLock()
_pipeline = None

GATE_HELP = (
    "화자 분리 모델을 불러오지 못했습니다. 다음을 확인하세요:\n"
    "  1) .env 의 HF_TOKEN 이 채워져 있는지\n"
    f"  2) https://huggingface.co/{config.DIARIZE_MODEL} 에서 약관에 동의(게이트 승인)했는지\n"
    "  3) 인터넷 연결 (첫 실행 시 모델 다운로드 필요)"
)


class DiarizationSetupError(RuntimeError):
    pass


def get_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = _build_pipeline()
        return _pipeline


def unload_pipeline() -> None:
    global _pipeline
    with _lock:
        _pipeline = None
    asr.free_vram()


def load_waveform(wav_path: Path, timeline: "vad.Timeline | None" = None) -> dict[str, Any]:
    """16kHz mono PCM wav 를 pyannote 가 받는 in-memory 형태로 읽는다.

    timeline 을 주면 무음을 들어낸 파형을 넘긴다. pyannote 의 세그멘테이션은
    슬라이딩 윈도로 파일 전체를 훑기 때문에 침묵도 그대로 비용이다. 10시간짜리
    회의 녹음처럼 빈 구간이 많으면 여기서 줄어드는 시간이 가장 크다.

    pyannote 4.x 는 파일 경로를 주면 torchcodec 으로 디코딩하는데, torchcodec 은
    FFmpeg 공유 라이브러리(avcodec/avformat DLL)를 요구한다. Windows 의 winget
    ffmpeg 은 정적 빌드라 .exe 만 설치되어 DLL 로딩이 실패한다.

    우리는 이미 ffmpeg CLI 로 16kHz mono 16bit wav 를 만들어 두므로, 표준
    라이브러리로 직접 읽어 넘기면 torchcodec 경로를 아예 타지 않는다.
    """
    import torch

    # 조각이면 그 범위만 읽는다. 10시간짜리를 통째로 올리지 않기 위한 것.
    origin, finish = timeline.span if timeline is not None else (0, None)
    samples = audio.read_wav(wav_path, origin, finish)
    if timeline is not None:
        samples = timeline.apply(samples, origin)

    waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": vad.SAMPLE_RATE}


def _build_pipeline():
    if not config.HF_TOKEN:
        raise DiarizationSetupError(GATE_HELP)

    import torch

    # torchcodec 경고는 config 에서 이미 걸러진다. 파형을 직접 넘기므로 무해하다.
    from pyannote.audio import Pipeline

    try:
        # pyannote 4.x 는 token=, 3.x 는 use_auth_token=
        pipeline = Pipeline.from_pretrained(config.DIARIZE_MODEL, token=config.HF_TOKEN)
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            config.DIARIZE_MODEL, use_auth_token=config.HF_TOKEN
        )
    except Exception as exc:
        raise DiarizationSetupError(f"{GATE_HELP}\n\n원인: {exc}") from exc

    if pipeline is None:
        # from_pretrained 는 게이트 미승인 시 예외 대신 None 을 돌려준다
        raise DiarizationSetupError(GATE_HELP)

    pipeline.to(torch.device(config.resolve_device()))
    return pipeline


class Diarization(NamedTuple):
    turns: list[dict]                      # [{'start','end','speaker'}] 시작 시각 오름차순
    embeddings: dict[str, np.ndarray]      # {'SPEAKER_00': (256,)} — 없는 화자도 있다
    speech_sec: dict[str, float]           # {'SPEAKER_00': 총 발화 초}
    overlaps: list[list[str]]              # 동시에 말한 화자 쌍 — 같은 사람일 수 없다


def _overlapping_pairs(annotation, minimum: float) -> list[list[str]]:
    """동시에 말한 화자 쌍을 뽑는다.

    두 라벨이 겹쳐서 말했다면 같은 사람일 수 없다. 나중에 화자를 자동으로
    다시 묶을 때(stitch.collapse) 이게 유일하게 확실한 반증이라 여기서 챙긴다.

    경계에서 몇십 ms 스치는 건 무시한다 — 그건 겹쳐 말한 게 아니라 분할 오차다.
    """
    spans = sorted(
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    )
    shared: dict[tuple[str, str], float] = {}
    active: list[tuple[float, str]] = []
    for start, end, label in spans:
        active = [item for item in active if item[0] > start]
        for other_end, other in active:
            if other == label:
                continue  # 한 화자가 여러 트랙에 걸친 것뿐이다
            pair = (label, other) if label < other else (other, label)
            shared[pair] = shared.get(pair, 0.0) + min(end, other_end) - start
        active.append((end, label))
    return [list(pair) for pair, seconds in shared.items() if seconds >= minimum]


def diarize(
    wav_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    timeline: "vad.Timeline | None" = None,
) -> Diarization:
    """화자 분리 결과.

    timeline 을 주면 무음을 들어낸 파형으로 돌리되 turns 는 원본 시각으로
    되돌려 내보낸다. 호출부는 항상 원본 기준 결과만 본다.
    """
    pipeline = get_pipeline()

    kwargs: dict[str, Any] = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)

    # 파일 경로 대신 파형을 직접 넘겨 torchcodec 디코딩 경로를 우회한다
    audio_input: Any = load_waveform(wav_path, timeline)
    full, exclusive, raw_embeddings = _apply(pipeline, audio_input, kwargs)

    # 단어에 화자를 붙일 때는 겹침이 제거된 쪽을 쓴다. 겹치는 구간에서는 어차피
    # 한 명만 배정할 수 있으므로, pyannote 가 전사 매칭용으로 만들어 둔 결과가 맞다.
    turns = [
        {"start": float(seg.start), "end": float(seg.end), "speaker": str(label)}
        for seg, _, label in (exclusive or full).itertracks(yield_label=True)
    ]
    if timeline is not None:
        turns = _restore_turns(turns, timeline)
    turns.sort(key=lambda t: (t["start"], t["end"]))

    # 발화 시간은 겹침을 포함한 원본 기준이어야 실제로 얼마나 말했는지에 가깝다.
    # 임베딩을 믿을 만한지 판단하는 데 쓰이므로 이쪽이 맞다.
    #
    # 무음을 들어낸 경우에도 이 합은 그대로 쓴다. 잘라낸 건 침묵뿐이고
    # Timeline.split 은 길이를 보존하므로, 원본으로 되돌려 더해도 값이 같다.
    speech: dict[str, float] = {}
    for seg, _, label in full.itertracks(yield_label=True):
        key = str(label)
        speech[key] = speech.get(key, 0.0) + float(seg.end - seg.start)

    embeddings: dict[str, np.ndarray] = {}
    if raw_embeddings is not None:
        arr = np.asarray(raw_embeddings)
        # 임베딩 순서는 speaker_diarization.labels() 순서다 (exclusive 가 아님)
        for idx, label in enumerate(full.labels()):
            if idx >= arr.shape[0]:
                break
            vec = np.asarray(arr[idx], dtype=np.float32).ravel()
            # 겹침만 있는 화자는 NaN 이 나오고, 화자 수가 모자라면 0 으로 패딩된다.
            # 둘 다 대조에 쓸 수 없는 값이라 버린다.
            if vec.size and np.isfinite(vec).all() and vec.any():
                embeddings[str(label)] = vec

    # 겹침이 살아 있는 full 에서 뽑아야 한다. exclusive 는 겹침을 지운 결과다.
    overlaps = _overlapping_pairs(full, config.MERGE_MIN_OVERLAP_SEC)
    return Diarization(turns, embeddings, speech, overlaps)


def _restore_turns(turns: list[dict], timeline: "vad.Timeline") -> list[dict]:
    """화자 구간을 원본 시각으로 되돌린다. 잘린 자리를 넘는 구간은 쪼갠다.

    쪼개지 않으면 그 화자가 없앤 침묵까지 차지하고, 침묵 건너편에 있는 다른
    화자의 단어를 겹침 계산에서 빼앗아 간다 (attach_speakers 참고).
    """
    restored: list[dict] = []
    for turn in turns:
        for start, end in timeline.split(turn["start"], turn["end"]):
            if end > start:
                restored.append({"start": start, "end": end, "speaker": turn["speaker"]})
    return restored


def _apply(pipeline, audio_input: Any, kwargs: dict[str, Any]):
    """pyannote 버전별 반환 형태를 (원본, 겹침제거, 임베딩) 으로 통일한다.

    4.x  : DiarizeOutput(speaker_diarization, exclusive_speaker_diarization,
           speaker_embeddings). 임베딩이 항상 들어 있고 return_embeddings 인자는
           없다. 넘기면 "Ignoring unexpected keyword arguments" 경고만 난다.
    3.x  : return_embeddings=True 를 주면 (Annotation, ndarray) 튜플.
    legacy: Annotation 하나만.
    """
    import inspect

    try:
        params = inspect.signature(pipeline.apply).parameters
    except (TypeError, ValueError):
        params = {}

    if "return_embeddings" in params:  # pyannote 3.x
        output = pipeline(audio_input, return_embeddings=True, **kwargs)
    else:  # pyannote 4.x 이상
        output = pipeline(audio_input, **kwargs)

    full = getattr(output, "speaker_diarization", None)
    if full is not None:  # DiarizeOutput
        return (
            full,
            getattr(output, "exclusive_speaker_diarization", None),
            getattr(output, "speaker_embeddings", None),
        )

    if isinstance(output, tuple):  # (Annotation, embeddings)
        return output[0], None, output[1]

    return output, None, None  # Annotation 하나만


# ── 단어/세그먼트에 화자 붙이기 ─────────────────────────────────────────
def _window(turns: list[dict], starts: list[float], max_dur: float, start: float, end: float):
    lo = bisect.bisect_left(starts, start - max_dur)
    hi = bisect.bisect_right(starts, end)
    return range(max(0, lo), min(len(turns), hi))


def _best_overlap(turns, starts, max_dur, start, end) -> str | None:
    best, best_ov = None, 0.0
    for j in _window(turns, starts, max_dur, start, end):
        turn = turns[j]
        ov = min(end, turn["end"]) - max(start, turn["start"])
        if ov > best_ov:
            best, best_ov = turn["speaker"], ov
    return best


def _nearest(turns, point: float) -> str | None:
    best, best_dist = None, float("inf")
    for turn in turns:
        if turn["start"] <= point <= turn["end"]:
            return turn["speaker"]
        dist = turn["start"] - point if point < turn["start"] else point - turn["end"]
        if dist < best_dist:
            best, best_dist = turn["speaker"], dist
    return best


def attach_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """세그먼트에 화자를 배정한다. 한 세그먼트 안에서 화자가 바뀌면 쪼갠다.

    Whisper 는 화자 경계를 모르고 문장을 자르기 때문에 한 세그먼트에 두 사람이
    섞이는 일이 흔하다. 단어 타임스탬프가 있으면 그 지점에서 분할한다.
    """
    if not turns:
        # 화자 구간을 못 찾은 경우 — 전체를 한 명으로 취급한다
        return [{**seg, "speaker": "SPEAKER_00"} for seg in segments]

    starts = [t["start"] for t in turns]
    max_dur = max((t["end"] - t["start"]) for t in turns)

    result: list[dict] = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", seg_start) or seg_start)
        seg_text = (seg.get("text") or "").strip()

        words = [
            w for w in (seg.get("words") or [])
            if w.get("start") is not None and w.get("end") is not None
        ]

        if not words:
            speaker = _best_overlap(turns, starts, max_dur, seg_start, seg_end) or _nearest(
                turns, (seg_start + seg_end) / 2
            )
            if seg_text:
                result.append(
                    {"start": seg_start, "end": seg_end, "text": seg_text,
                     "speaker": speaker, "words": []}
                )
            continue

        for word in words:
            ws, we = float(word["start"]), float(word["end"])
            word["speaker"] = (
                _best_overlap(turns, starts, max_dur, ws, we)
                or _nearest(turns, (ws + we) / 2)
            )

        # 연속 동일 화자끼리 묶기
        groups: list[list[dict]] = []
        for word in words:
            if groups and groups[-1][-1]["speaker"] == word["speaker"]:
                groups[-1].append(word)
            else:
                groups.append([word])

        if len(groups) == 1:
            # 쪼갤 필요 없음 — 원문 텍스트를 그대로 써서 공백/구두점을 보존
            if seg_text:
                result.append(
                    {"start": seg_start, "end": seg_end, "text": seg_text,
                     "speaker": groups[0][0]["speaker"], "words": words}
                )
            continue

        for group in groups:
            text = " ".join((w.get("word") or "").strip() for w in group).strip()
            if not text:
                continue
            result.append(
                {
                    "start": float(group[0]["start"]),
                    "end": float(group[-1]["end"]),
                    "text": text,
                    "speaker": group[0]["speaker"],
                    "words": group,
                }
            )

    result.sort(key=lambda s: s["start"])
    return result


def embed_single_speaker(wav_path: Path) -> tuple[np.ndarray, float]:
    """등록용 샘플 오디오에서 화자 임베딩 하나를 뽑는다 (1인 발화 가정)."""
    result = diarize(wav_path, num_speakers=1)
    embeddings, speech = result.embeddings, result.speech_sec
    if not embeddings:
        raise RuntimeError(
            "샘플에서 화자 임베딩을 추출하지 못했습니다. 10초 이상, "
            "한 사람만 말하는 깨끗한 녹음을 사용해 주세요."
        )
    label = max(speech, key=speech.get) if speech else next(iter(embeddings))
    if label not in embeddings:
        label = next(iter(embeddings))
    return embeddings[label], speech.get(label, 0.0)
