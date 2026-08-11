"""WhisperX transcription plus wav2vec2 forced alignment.

Models load on first use and stay resident in the process (reloading wastes
20-30 seconds every time). Set config.UNLOAD_BETWEEN_STAGES=true to release them
between stages when VRAM is tight.
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


# ── Whisper itself ────────────────────────────────────────────────────
def get_model():
    global _model
    with _lock:
        if _model is None:
            import whisperx

            device = config.asr_device()  # CTranslate2 cannot use MPS
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
    """Swap just the options instead of reloading the model, since the prompt changes per job.

    Depending on the whisperx version model.options is either a NamedTuple or a
    dataclass, so try both.
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
    """Decide the language by sampling several points in the audio.

    whisperx's own detection looks only at the first 30 seconds (audio[: N_SAMPLES]
    in its asr.py). If the recording opens with silence or a short English
    greeting, the entire file gets force-decoded in the wrong language and
    Whisper confidently invents things nobody said.

    Returns: (language, confidence 0-1, votes per language)
    """
    import numpy as np
    from whisperx.audio import N_SAMPLES

    model = get_model()
    total = int(audio.shape[0])

    if total <= N_SAMPLES:
        starts = [0]
    else:
        # Sample evenly, staying away from the very start and end
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
        # Detection is arbitrary on near-silent windows, and one such sample ruins the file.
        if float(np.sqrt(np.mean(np.square(chunk)))) < LANG_SILENCE_RMS:
            continue
        language, probability = _detect_window(model, chunk)
        votes[language] = votes.get(language, 0.0) + probability
        counted += 1

    if not votes:  # everything got filtered as silence — fall back to the default way
        language, probability = _detect_window(model, audio)
        return language, probability, {language: probability}

    best = max(votes, key=votes.get)
    confidence = votes[best] / max(1, counted)
    return best, round(confidence, 3), {k: round(v, 3) for k, v in votes.items()}


def transcribe(audio, language: str | None = None, initial_prompt: str | None = None) -> dict:
    """Returns {'segments': [...], 'language': 'ko', ...}. language=None auto-detects."""
    model = get_model()
    _apply_initial_prompt(model, initial_prompt)

    detection: dict[str, Any] = {}
    if not language:
        language, confidence, votes = detect_language(audio)
        detection = {"auto": True, "confidence": confidence, "votes": votes}

    # Always pass the language explicitly. whisperx caches the tokenizer on the
    # model instance and, without a language, reuses whatever the previous file
    # used. With a resident model that silently breaks from the second file on.
    kwargs: dict[str, Any] = {
        "batch_size": max(1, config.BATCH_SIZE),
        "language": language,
    }

    try:
        result = model.transcribe(audio, **kwargs)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or kwargs["batch_size"] <= 1:
            raise
        # On OOM, halve the batch and try once more
        free_vram()
        kwargs["batch_size"] = max(1, kwargs["batch_size"] // 2)
        result = model.transcribe(audio, **kwargs)

    result.setdefault("language", language)
    result["detection"] = detection
    return result


# ── Forced alignment (word-level timestamps) ──────────────────────────
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
    """Return aligned segments, or the originals when no aligner exists for the language.

    Returns: (segments, warning_or_None)
    """
    if not segments:
        return segments, None
    import whisperx

    try:
        model, metadata = get_align_model(language)
    except Exception as exc:  # unsupported language, etc.
        return segments, (
            f"Could not load a word-alignment model for '{language}', so speakers were "
            f"assigned per sentence only. ({exc})"
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
