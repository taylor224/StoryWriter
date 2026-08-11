"""Global settings. Must be imported before any other app module.

HF_HOME has to be set before torch/huggingface are imported for the model cache
to land inside the project (models/). That is why this module sets environment
variables as a side effect.
"""

import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _path(env_key: str, default: Path) -> Path:
    raw = os.getenv(env_key)
    return Path(raw).expanduser().resolve() if raw else default


def _float(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, default))
    except (TypeError, ValueError):
        return default


def _int(env_key: str, default: int) -> int:
    try:
        return int(os.getenv(env_key, default))
    except (TypeError, ValueError):
        return default


def _bool(env_key: str, default: bool) -> bool:
    return os.getenv(env_key, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = _path("DATA_DIR", BASE_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
SAMPLE_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "cache"  # per-stage intermediate results (for resuming a failed job)
DB_PATH = DATA_DIR / "app.db"
MODEL_CACHE = _path("MODEL_CACHE", BASE_DIR / "models")

for _d in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, SAMPLE_DIR, CACHE_DIR, MODEL_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# Must be set before torch/transformers are imported
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE / "torch"))
# Silence the symlink warning (it never stops on Windows outside developer mode)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows cannot create symlinks unless you are an admin or in developer mode.
# huggingface_hub links blobs -> snapshots in its cache with symlinks, and even
# when its pre-flight check passes the actual creation can fail with WinError
# 1314. That raises a plain OSError rather than PermissionError, so the library's
# copy fallback never kicks in and the whole thing blows up. On Windows we turn
# symlinks off entirely and move files instead. (A freshly downloaded blob is
# moved, not copied, so this does not double the disk usage.)
if sys.platform.startswith("win"):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# pyannote warns at import time when torchcodec fails to load. This project hands
# it a decoded waveform directly (diarize.load_waveform), so that path is never
# taken. whisperx also imports pyannote for its VAD, so the filter has to live
# here. The message starts with a newline, so the regex needs (?s) to match.
warnings.filterwarnings(
    "ignore", message=r"(?s).*torchcodec is not installed correctly.*"
)

# ── Models ────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
# large-v3-turbo is a distilled model whose decoder went from 32 layers to 4.
# It is 5-8x faster for about 0.4pp more word error on average. The shallow
# decoder does fall into repetition and hallucination more easily, so this
# default assumes silence removal (vad) and the hallucination filter (cleanup)
# are both on. Set WHISPER_MODEL=large-v3 when accuracy matters most.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo").strip()
DIARIZE_MODEL = os.getenv(
    "DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"
).strip()

# ── Hardware ──────────────────────────────────────────────────────────
# auto | cuda | mps | cpu
DEVICE = os.getenv("DEVICE", "auto").strip().lower()
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "auto").strip().lower()
BATCH_SIZE = _int("BATCH_SIZE", 8)
UNLOAD_BETWEEN_STAGES = _bool("UNLOAD_BETWEEN_STAGES", False)

# ── Audio filter ──────────────────────────────────────────────────────
# Clean the sound up with an ffmpeg filter before transcription.
# off | voice | denoise | declip, or an ffmpeg filter string (see audio.FILTERS).
#
# Why the default is off: Whisper was trained on noisy real-world audio and is
# already robust to ordinary noise. Aggressive denoising creates artifacts it
# never saw in training, and accuracy often gets worse. Always compare results
# with and without before leaving it on.
AUDIO_FILTER = os.getenv("AUDIO_FILTER", "off").strip()

# ── Silence removal (cutting before transcription) ────────────────────
# vad.Timeline undoes the time shift the cutting causes. Result timestamps are
# always on the original audio's clock, so changing these never desyncs playback.
TRIM_SILENCE = _bool("TRIM_SILENCE", True)
# Only silence at least this long is cut. Short breaths and pauses are speech.
TRIM_MIN_SILENCE_SEC = _float("TRIM_MIN_SILENCE_SEC", 0.8)
# Headroom kept around each utterance, so first and last sounds are not clipped.
TRIM_PAD_SEC = _float("TRIM_PAD_SEC", 0.25)
# Where the line sits between the noise floor and speech level (0-1). Higher cuts more.
TRIM_SENSITIVITY = _float("TRIM_SENSITIVITY", 0.30)
# Sounds shorter than this do not count as speech (clicks, coughs)
TRIM_MIN_SPEECH_SEC = _float("TRIM_MIN_SPEECH_SEC", 0.10)
# If noise and speech are closer than this, do not cut at all
TRIM_MIN_DYNAMIC_DB = _float("TRIM_MIN_DYNAMIC_DB", 12.0)
# Strip everything below this frequency before judging silence (Hz). 0 disables it.
# Air conditioning, fans and desk vibration sit below 100Hz and raise the noise
# floor enough to defeat silence detection. Used only for the decision — what
# goes to the models is the untouched waveform.
TRIM_HIGHPASS_HZ = _float("TRIM_HIGHPASS_HZ", 80.0)

# ── Chunked processing for long files ─────────────────────────────────
# Files longer than this are processed in chunks (0 disables it). This is about
# quality as much as speed — pyannote tends to split one person into several
# speakers on very long files. Chunk boundaries land in the middle of silence,
# and speakers are stitched back together across chunks by voice (stitch.py).
CHUNK_SEC = _float("CHUNK_SEC", 3600.0)
# Cosine threshold for "same person" across chunks. Same session and same mic
# make this easier than matching enrolled speakers (MATCH_THRESHOLD), so it sits
# a little higher.
STITCH_THRESHOLD = _float("STITCH_THRESHOLD", 0.65)

# ── Re-merging over-split speakers ────────────────────────────────────
# pyannote splits one person into several speakers often (tone shifts, mic
# distance, long files). At the end we compare the final speakers once more and
# merge them. 0 disables it.
#
# Do not set this above STITCH_THRESHOLD: pairs that stitching already missed
# would then be unmergeable forever. Keep them equal — safety comes from the
# "never merge speakers who talked over each other" disproof, not the threshold.
MERGE_THRESHOLD = _float("MERGE_THRESHOLD", 0.65)
# Two speakers who overlapped for at least this long are never merged.
# A few tens of ms at a boundary is segmentation slop, not simultaneous speech.
MERGE_MIN_OVERLAP_SEC = _float("MERGE_MIN_OVERLAP_SEC", 0.5)

# ── Hallucination filter ──────────────────────────────────────────────
# Filters out sentences Whisper invented over noise. Contentless repetition is
# removed outright; boilerplate is only removed when there is evidence the audio
# does not back up the text. Everything else questionable stays in the output
# and is merely flagged (see cleanup.py).
DROP_HALLUCINATION = _bool("DROP_HALLUCINATION", True)
# Mean word confidence from the aligner below this counts as "audio does not
# match the text". 0 drops just this piece of evidence (the speech-rate check stays).
HALLUCINATION_MIN_SCORE = _float("HALLUCINATION_MIN_SCORE", 0.30)
# true also removes everything that would only have been flagged as suspect.
# Real speech can be lost this way.
DROP_SUSPECT = _bool("DROP_SUSPECT", False)

# ── Speaker matching ──────────────────────────────────────────────────
MATCH_THRESHOLD = _float("MATCH_THRESHOLD", 0.60)
MATCH_MARGIN = _float("MATCH_MARGIN", 0.05)
MIN_SPEECH_SEC = _float("MIN_SPEECH_SEC", 5.0)
MAX_VOICEPRINTS = _int("MAX_VOICEPRINTS", 10)

# ── Server ────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "127.0.0.1").strip()
PORT = _int("PORT", 8000)

# Upload extensions we accept (whatever ffmpeg can handle)
ALLOWED_EXT = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
    ".wma", ".mp4", ".mkv", ".mov", ".webm", ".avi",
}


def resolve_device() -> str:
    """The torch device for pyannote (diarization) and wav2vec2 (alignment).

    auto never picks mps: pyannote's MPS kernels are incomplete and there are
    reports of timestamps drifting. Set DEVICE=mps explicitly to use it on
    Apple Silicon.
    """
    if DEVICE == "cpu":
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"

    if DEVICE in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if DEVICE == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def asr_device() -> str:
    """The device WhisperX (CTranslate2) will use.

    CTranslate2 does not support Metal/MPS — it is CUDA or CPU. On Apple Silicon
    that means CPU inference through the Accelerate framework.
    """
    device = resolve_device()
    return "cuda" if device == "cuda" else "cpu"


def resolve_compute_type(device: str) -> str:
    if COMPUTE_TYPE == "auto":
        return "float16" if device == "cuda" else "int8"
    # CTranslate2 refuses float16 on CPU
    if device != "cuda" and COMPUTE_TYPE in ("float16", "fp16"):
        return "int8"
    return COMPUTE_TYPE
