"""WhisperX 전사 + wav2vec2 강제정렬.

모델은 첫 사용 시 로드해서 프로세스에 상주시킨다 (재로딩 20~30초 낭비 방지).
VRAM 이 모자라면 config.UNLOAD_BETWEEN_STAGES=true 로 단계마다 내린다.
"""

import gc
import threading
from typing import Any

from . import config

_lock = threading.RLock()
_model = None
_align_cache: dict[str, tuple[Any, Any]] = {}


def free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_audio(path):
    import whisperx

    return whisperx.load_audio(str(path))


# ── Whisper 본체 ──────────────────────────────────────────────────────
def get_model():
    global _model
    with _lock:
        if _model is None:
            import whisperx

            device = config.asr_device()  # CTranslate2 는 MPS 를 못 쓴다
            _model = whisperx.load_model(
                config.WHISPER_MODEL,
                device=device,
                compute_type=config.resolve_compute_type(device),
            )
        return _model


def unload_model() -> None:
    global _model
    with _lock:
        _model = None
    free_vram()


def _apply_initial_prompt(model, prompt: str | None) -> None:
    """잡마다 프롬프트가 바뀌므로 모델 재로딩 없이 옵션만 교체한다.

    whisperx 버전에 따라 model.options 가 NamedTuple 이거나 dataclass 라서 둘 다 시도.
    """
    options = getattr(model, "options", None)
    if options is None:
        return
    value = prompt.strip() if prompt and prompt.strip() else None
    replace = getattr(options, "_replace", None)
    if callable(replace):
        try:
            model.options = replace(initial_prompt=value)
            return
        except (TypeError, ValueError):
            pass
    try:
        options.initial_prompt = value
    except Exception:
        pass


SAMPLE_RATE = 16000
LANG_WINDOWS = 5
LANG_SILENCE_RMS = 0.005


def _detect_window(model, chunk) -> tuple[str, float]:
    from whisperx.audio import N_SAMPLES, log_mel_spectrogram

    n_mels = model.model.feat_kwargs.get("feature_size") or 80
    padding = 0 if chunk.shape[0] >= N_SAMPLES else N_SAMPLES - chunk.shape[0]
    segment = log_mel_spectrogram(chunk[:N_SAMPLES], n_mels=n_mels, padding=padding)
    encoded = model.model.encode(segment)
    token, probability = model.model.model.detect_language(encoded)[0][0]
    return token[2:-2], float(probability)


def detect_language(audio) -> tuple[str, float, dict[str, float]]:
    """오디오 여러 지점을 표본으로 언어를 정한다.

    whisperx 기본 감지는 첫 30초만 본다(asr.py 의 audio[: N_SAMPLES]). 녹음
    시작이 침묵이거나 짧은 영어 인사말이면 파일 전체가 엉뚱한 언어로 강제
    디코딩되고, Whisper 는 하지도 않은 말을 그럴듯하게 지어낸다.

    반환: (언어, 확신도 0~1, 언어별 득표)
    """
    import numpy as np
    from whisperx.audio import N_SAMPLES

    model = get_model()
    total = int(audio.shape[0])

    if total <= N_SAMPLES:
        starts = [0]
    else:
        # 앞뒤 끝을 피해 고르게 표본을 잡는다
        span = total - N_SAMPLES
        starts = [
            int(span * ratio)
            for ratio in np.linspace(0.05, 0.95, min(LANG_WINDOWS, max(1, span // N_SAMPLES + 1)))
        ]

    votes: dict[str, float] = {}
    counted = 0
    for start in starts:
        chunk = audio[start : start + N_SAMPLES]
        if chunk.shape[0] < SAMPLE_RATE:
            continue
        # 거의 무음인 구간은 감지가 제멋대로다. 이런 표본이 전체를 망친다.
        if float(np.sqrt(np.mean(np.square(chunk)))) < LANG_SILENCE_RMS:
            continue
        language, probability = _detect_window(model, chunk)
        votes[language] = votes.get(language, 0.0) + probability
        counted += 1

    if not votes:  # 전부 무음으로 걸러진 경우 — 원래 방식으로 한 번
        language, probability = _detect_window(model, audio)
        return language, probability, {language: probability}

    best = max(votes, key=votes.get)
    confidence = votes[best] / max(1, counted)
    return best, round(confidence, 3), {k: round(v, 3) for k, v in votes.items()}


def transcribe(audio, language: str | None = None, initial_prompt: str | None = None) -> dict:
    """{'segments': [...], 'language': 'ko', ...} 반환. language=None 이면 자동 감지."""
    model = get_model()
    _apply_initial_prompt(model, initial_prompt)

    detection: dict[str, Any] = {}
    if not language:
        language, confidence, votes = detect_language(audio)
        detection = {"auto": True, "confidence": confidence, "votes": votes}

    # 언어를 항상 명시해서 넘긴다. whisperx 는 토크나이저를 모델 인스턴스에
    # 캐시해 두고, language 를 안 주면 직전 파일의 언어를 그대로 재사용한다.
    # 모델을 상주시키는 구조에서는 두 번째 파일부터 조용히 틀어진다.
    kwargs: dict[str, Any] = {
        "batch_size": max(1, config.BATCH_SIZE),
        "language": language,
    }

    try:
        result = model.transcribe(audio, **kwargs)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or kwargs["batch_size"] <= 1:
            raise
        # OOM 이면 배치를 반으로 줄여 한 번 더
        free_vram()
        kwargs["batch_size"] = max(1, kwargs["batch_size"] // 2)
        result = model.transcribe(audio, **kwargs)

    result.setdefault("language", language)
    result["detection"] = detection
    return result


# ── 강제 정렬 (단어 단위 타임스탬프) ────────────────────────────────────
def get_align_model(language: str):
    with _lock:
        if language not in _align_cache:
            import whisperx

            _align_cache[language] = whisperx.load_align_model(
                language_code=language, device=config.resolve_device()
            )
        return _align_cache[language]


def unload_align_models() -> None:
    with _lock:
        _align_cache.clear()
    free_vram()


def align(segments: list[dict], language: str, audio) -> tuple[list[dict], str | None]:
    """정렬된 세그먼트를 반환. 해당 언어의 정렬 모델이 없으면 원본을 그대로 돌려준다.

    반환: (segments, warning_or_None)
    """
    if not segments:
        return segments, None
    import whisperx

    try:
        model, metadata = get_align_model(language)
    except Exception as exc:  # 지원하지 않는 언어 등
        return segments, (
            f"'{language}' 언어의 단어 정렬 모델을 불러오지 못해 문장 단위로만 "
            f"화자를 배정했습니다. ({exc})"
        )

    out = whisperx.align(
        segments,
        model,
        metadata,
        audio,
        config.resolve_device(),
        return_char_alignments=False,
    )
    return out.get("segments", segments), None
