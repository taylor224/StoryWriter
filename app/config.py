"""전역 설정. 다른 app 모듈보다 먼저 import 되어야 한다.

torch / huggingface 를 import 하기 전에 HF_HOME 을 설정해야 모델 캐시가
프로젝트 안(models/)에 떨어진다. 그래서 이 모듈은 부수효과로 환경변수를 건드린다.
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


# ── 경로 ──────────────────────────────────────────────────────────────
DATA_DIR = _path("DATA_DIR", BASE_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
SAMPLE_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "cache"  # 단계별 중간 결과 (재시도 시 이어서 돌리기 위함)
DB_PATH = DATA_DIR / "app.db"
MODEL_CACHE = _path("MODEL_CACHE", BASE_DIR / "models")

for _d in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, SAMPLE_DIR, CACHE_DIR, MODEL_CACHE):
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

# pyannote 는 import 시점에 torchcodec 로딩 실패를 경고한다. 이 프로젝트는
# 디코딩된 파형을 직접 넘기므로(diarize.load_waveform) 해당 경로를 타지 않는다.
# whisperx 도 VAD 때문에 pyannote 를 import 하므로 필터는 여기 있어야 걸린다.
# 메시지가 개행으로 시작해서 (?s) 없이는 정규식이 매칭되지 않는다.
warnings.filterwarnings(
    "ignore", message=r"(?s).*torchcodec is not installed correctly.*"
)

# ── 모델 ──────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
# large-v3-turbo 는 디코더를 32층에서 4층으로 줄인 증류 모델이다. 5~8배 빠르고
# 단어오류율 차이는 평균 0.4%p 수준. 대신 디코더가 얕아 반복·환각에 더 잘 빠지므로
# 무음 제거(vad)와 환각 필터(cleanup)를 켠 채로 쓰는 것을 전제한다.
# 정확도가 최우선이면 WHISPER_MODEL=large-v3.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo").strip()
DIARIZE_MODEL = os.getenv(
    "DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"
).strip()

# ── 하드웨어 ───────────────────────────────────────────────────────────
# auto | cuda | mps | cpu
DEVICE = os.getenv("DEVICE", "auto").strip().lower()
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "auto").strip().lower()
BATCH_SIZE = _int("BATCH_SIZE", 8)
UNLOAD_BETWEEN_STAGES = _bool("UNLOAD_BETWEEN_STAGES", False)

# ── 무음 제거 (전사 전 잘라내기) ────────────────────────────────────────
# 잘라낸 만큼 시각이 당겨지는 문제는 vad.Timeline 이 되돌린다. 결과 타임스탬프는
# 항상 원본 오디오 기준이므로 여기 값을 바꿔도 재생 위치는 어긋나지 않는다.
TRIM_SILENCE = _bool("TRIM_SILENCE", True)
# 이 길이 이상 이어지는 침묵만 잘라낸다. 짧은 숨·뜸은 말의 일부다.
TRIM_MIN_SILENCE_SEC = _float("TRIM_MIN_SILENCE_SEC", 0.8)
# 발화 앞뒤로 남겨 둘 여유. 말의 첫소리/끝소리가 잘리는 걸 막는다.
TRIM_PAD_SEC = _float("TRIM_PAD_SEC", 0.25)
# 잡음 바닥과 발화 세기 사이 어디에 기준선을 둘지 (0~1). 올리면 더 많이 잘린다.
TRIM_SENSITIVITY = _float("TRIM_SENSITIVITY", 0.30)
# 이보다 짧은 소리는 발화로 치지 않는다 (클릭·기침 등)
TRIM_MIN_SPEECH_SEC = _float("TRIM_MIN_SPEECH_SEC", 0.10)
# 잡음과 발화의 세기 차가 이 값 미만이면 아예 자르지 않는다
TRIM_MIN_DYNAMIC_DB = _float("TRIM_MIN_DYNAMIC_DB", 12.0)

# ── 긴 파일 조각 처리 ──────────────────────────────────────────────────
# 이 길이를 넘으면 조각으로 나눠 돌린다 (0 이면 끄기). 속도뿐 아니라 품질
# 문제이기도 하다 — pyannote 는 아주 긴 파일에서 같은 사람을 여러 명으로
# 갈라놓는 경향이 있다. 조각 경계는 침묵 한가운데로 잡고, 조각 간 화자는
# 목소리 임베딩으로 다시 이어 붙인다 (stitch.py).
CHUNK_SEC = _float("CHUNK_SEC", 3600.0)
# 조각 간 같은 사람 판정 임계값 (코사인). 같은 녹음·같은 마이크라 등록 화자
# 대조(MATCH_THRESHOLD)보다 조건이 좋아 조금 높게 잡는다.
STITCH_THRESHOLD = _float("STITCH_THRESHOLD", 0.65)

# ── 환각 필터 ──────────────────────────────────────────────────────────
# Whisper 가 잡음 구간에서 지어낸 문장을 걸러낸다. 내용이 없는 반복은 지우고,
# 상투구는 소리가 글자를 뒷받침하지 않는다는 증거가 있을 때만 지운다.
# 나머지 의심 구간은 결과에 그대로 두고 표시만 한다 (cleanup.py 참고).
DROP_HALLUCINATION = _bool("DROP_HALLUCINATION", True)
# 정렬 모델이 매긴 단어 신뢰도 평균이 이 값 미만이면 소리와 글자가 안 맞는
# 것으로 본다. 0 이면 이 증거만 쓰지 않는다 (말 속도 검사는 남는다).
HALLUCINATION_MIN_SCORE = _float("HALLUCINATION_MIN_SCORE", 0.30)
# true 면 "의심"으로 표시만 하던 구간까지 전부 지운다. 진짜 발언을 잃을 수 있다.
DROP_SUSPECT = _bool("DROP_SUSPECT", False)

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
