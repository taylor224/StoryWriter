"""pyannote.audio 화자 분리 + 화자 임베딩 추출.

whisperx 의 DiarizationPipeline 래퍼 대신 pyannote 를 직접 호출한다.
이유: 래퍼가 `return_embeddings=True` 를 노출하지 않는데, 이 임베딩이
"한 번 저장한 화자를 다음 파일에서 자동 인식"하는 기능의 전부이기 때문.
"""

import bisect
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import asr, config

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


def load_waveform(wav_path: Path) -> dict[str, Any]:
    """16kHz mono PCM wav 를 pyannote 가 받는 in-memory 형태로 읽는다.

    pyannote 4.x 는 파일 경로를 주면 torchcodec 으로 디코딩하는데, torchcodec 은
    FFmpeg 공유 라이브러리(avcodec/avformat DLL)를 요구한다. Windows 의 winget
    ffmpeg 은 정적 빌드라 .exe 만 설치되어 DLL 로딩이 실패한다.

    우리는 이미 ffmpeg CLI 로 16kHz mono 16bit wav 를 만들어 두므로, 표준
    라이브러리로 직접 읽어 넘기면 torchcodec 경로를 아예 타지 않는다.
    """
    import wave

    import torch

    try:
        with wave.open(str(wav_path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise RuntimeError(
            f"16bit PCM wav 로 읽을 수 없습니다 ({wav_path.name}): {exc}. "
            "audio.to_wav16k 를 거친 파일인지 확인하세요."
        ) from exc

    if width != 2:
        raise RuntimeError(
            f"16bit PCM wav 가 아닙니다 (sampwidth={width}). audio.to_wav16k 를 거쳤는지 확인하세요."
        )

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": rate}


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


def diarize(
    wav_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, float]]:
    """반환: (turns, embeddings, speech_sec)

    turns      : [{'start', 'end', 'speaker'}] 시작 시각 오름차순
    embeddings : {'SPEAKER_00': np.ndarray(256,)}  — 값이 없는 화자는 빠질 수 있음
    speech_sec : {'SPEAKER_00': 총 발화 초}
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
    audio_input: Any = load_waveform(wav_path)

    try:
        output = pipeline(audio_input, return_embeddings=True, **kwargs)
    except TypeError:
        # return_embeddings 를 모르는 구버전 — 임베딩 없이 진행 (자동 인식만 비활성)
        output = pipeline(audio_input, **kwargs)

    if isinstance(output, tuple):
        annotation, raw_embeddings = output[0], output[1]
    else:
        annotation, raw_embeddings = output, None

    turns = [
        {"start": float(seg.start), "end": float(seg.end), "speaker": str(label)}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t["start"], t["end"]))

    speech: dict[str, float] = {}
    for turn in turns:
        speech[turn["speaker"]] = speech.get(turn["speaker"], 0.0) + (
            turn["end"] - turn["start"]
        )

    embeddings: dict[str, np.ndarray] = {}
    if raw_embeddings is not None:
        labels = list(annotation.labels())
        arr = np.asarray(raw_embeddings)
        for idx, label in enumerate(labels):
            if idx >= arr.shape[0]:
                break
            vec = np.asarray(arr[idx], dtype=np.float32).ravel()
            # 겹침 구간만 있는 화자는 임베딩이 NaN 으로 나올 수 있다
            if vec.size and np.isfinite(vec).all():
                embeddings[str(label)] = vec

    return turns, embeddings, speech


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
    turns, embeddings, speech = diarize(wav_path, num_speakers=1)
    if not embeddings:
        raise RuntimeError(
            "샘플에서 화자 임베딩을 추출하지 못했습니다. 10초 이상, "
            "한 사람만 말하는 깨끗한 녹음을 사용해 주세요."
        )
    label = max(speech, key=speech.get) if speech else next(iter(embeddings))
    if label not in embeddings:
        label = next(iter(embeddings))
    return embeddings[label], speech.get(label, 0.0)
