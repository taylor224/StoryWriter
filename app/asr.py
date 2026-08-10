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


def transcribe(audio, language: str | None = None, initial_prompt: str | None = None) -> dict:
    """{'segments': [...], 'language': 'ko'} 반환. language=None 이면 자동 감지."""
    model = get_model()
    _apply_initial_prompt(model, initial_prompt)

    kwargs: dict[str, Any] = {"batch_size": max(1, config.BATCH_SIZE)}
    if language:
        kwargs["language"] = language

    try:
        return model.transcribe(audio, **kwargs)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or kwargs["batch_size"] <= 1:
            raise
        # OOM 이면 배치를 반으로 줄여 한 번 더
        free_vram()
        kwargs["batch_size"] = max(1, kwargs["batch_size"] // 2)
        return model.transcribe(audio, **kwargs)


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
