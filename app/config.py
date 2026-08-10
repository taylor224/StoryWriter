"""전역 설정. 다른 app 모듈보다 먼저 import 되어야 한다.

torch / huggingface 를 import 하기 전에 HF_HOME 을 설정해야 모델 캐시가
프로젝트 안(models/)에 떨어진다. 그래서 이 모듈은 부수효과로 환경변수를 건드린다.
"""

import os
import sys
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


# ── 경로 ──────────────────────────────────────────────────────────────
DATA_DIR = _path("DATA_DIR", BASE_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
SAMPLE_DIR = DATA_DIR / "samples"
DB_PATH = DATA_DIR / "app.db"
MODEL_CACHE = _path("MODEL_CACHE", BASE_DIR / "models")

for _d in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, SAMPLE_DIR, MODEL_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# torch/transformers import 전에 반드시 설정
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE / "torch"))
# 심볼릭 링크 경고 억제 (Windows 에서 개발자 모드 아니면 계속 뜸)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows 는 관리자이거나 개발자 모드가 아니면 심볼릭 링크를 만들 수 없다.
# huggingface_hub 는 캐시에서 blobs -> snapshots 를 심링크로 잇는데, 사전 감지가
# 통과해도 실제 생성에서 WinError 1314 가 날 수 있다. 이때 나는 예외가
# PermissionError 가 아니라 일반 OSError 라서 라이브러리의 복사 폴백이 동작하지
# 않고 그대로 터진다. Windows 에서는 아예 심링크를 끄고 파일을 옮겨 저장한다.
# (새로 받은 blob 은 복사가 아니라 이동이므로 용량이 두 배가 되지는 않는다.)
if sys.platform.startswith("win"):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# ── 모델 ──────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3").strip()
DIARIZE_MODEL = os.getenv(
    "DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"
).strip()

# ── 하드웨어 ───────────────────────────────────────────────────────────
# auto | cuda | mps | cpu
DEVICE = os.getenv("DEVICE", "auto").strip().lower()
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "auto").strip().lower()
BATCH_SIZE = _int("BATCH_SIZE", 8)
UNLOAD_BETWEEN_STAGES = _bool("UNLOAD_BETWEEN_STAGES", False)

# ── 화자 매칭 ──────────────────────────────────────────────────────────
MATCH_THRESHOLD = _float("MATCH_THRESHOLD", 0.60)
MATCH_MARGIN = _float("MATCH_MARGIN", 0.05)
MIN_SPEECH_SEC = _float("MIN_SPEECH_SEC", 5.0)
MAX_VOICEPRINTS = _int("MAX_VOICEPRINTS", 10)

# ── 서버 ──────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "127.0.0.1").strip()
PORT = _int("PORT", 8000)

# 업로드 허용 확장자 (ffmpeg 가 처리 가능한 것들)
ALLOWED_EXT = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
    ".wma", ".mp4", ".mkv", ".mov", ".webm", ".avi",
}


def resolve_device() -> str:
    """pyannote(화자분리)와 wav2vec2(정렬)가 쓸 torch 디바이스.

    auto 는 mps 를 고르지 않는다. pyannote 의 MPS 커널이 불완전해서 타임스탬프가
    어긋나는 사례가 보고돼 있기 때문. Apple Silicon 에서 쓰려면 DEVICE=mps 로 명시.
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
    """WhisperX(CTranslate2)가 쓸 디바이스.

    CTranslate2 는 Metal/MPS 를 지원하지 않는다 (CUDA 아니면 CPU). Apple Silicon
    에서는 Accelerate 프레임워크로 CPU 추론이 돌아간다.
    """
    device = resolve_device()
    return "cuda" if device == "cuda" else "cpu"


def resolve_compute_type(device: str) -> str:
    if COMPUTE_TYPE == "auto":
        return "float16" if device == "cuda" else "int8"
    # CPU 에서 float16 은 CTranslate2 가 거부한다
    if device != "cuda" and COMPUTE_TYPE in ("float16", "fp16"):
        return "int8"
    return COMPUTE_TYPE
