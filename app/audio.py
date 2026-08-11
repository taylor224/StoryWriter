"""오디오 입력 정규화. 어떤 포맷이 들어와도 16kHz mono wav 로 통일한다.

Whisper 도 pyannote 도 16kHz mono 를 기대하므로 여기서 한 번만 변환하고
이후 단계는 전부 같은 wav 파일을 재사용한다.
"""

import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


# 파이프라인 전체가 이 샘플레이트를 전제한다 (Whisper 도 pyannote 도 16kHz).
SAMPLE_RATE = 16000


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegMissing(
            "ffmpeg 를 찾을 수 없습니다. Windows 에서는 `winget install Gyan.FFmpeg` 로 설치한 뒤 "
            "터미널을 새로 열어 주세요."
        )
    return exe


# 전사 전에 걸 수 있는 ffmpeg 필터 묶음.
#
# 기본이 off 인 이유: Whisper 는 68만 시간의 잡음 섞인 실제 오디오로 학습돼서
# 웬만한 잡음에는 이미 강하다. 노이즈 제거를 세게 걸면 학습 때 본 적 없는
# 인공적인 아티팩트가 생겨 되레 인식률이 떨어지는 사례가 흔하다.
# "뚜렷하게" 가 곧 "잘 인식됨" 이 아니다.
#
# 그래도 도움이 되는 상황이 있어서 골라 쓸 수 있게 둔다.
FILTERS = {
    # 아무것도 안 건다 (권장 기본값)
    "off": "",
    # 저역 웅웅거림을 걷고 멀리 앉은 사람 목소리를 끌어올린다.
    # 회의실에서 한 명만 마이크에서 멀 때 그 사람 발언이 통째로 누락되는 걸 막는다.
    # 스펙트럼을 건드리지 않아 Whisper 에 상대적으로 안전한 쪽.
    "voice": "highpass=f=80,lowpass=f=7500,speechnorm=e=3",
    # 위에 FFT 노이즈 제거를 더한다. 팬 소리·화이트노이즈가 계속 깔린 녹음용.
    # 인식률이 오히려 나빠질 수 있으니 켜기 전후를 반드시 비교할 것.
    "denoise": "highpass=f=80,afftdn=nf=-25,lowpass=f=7500,speechnorm=e=3",
    # 찢어지게 녹음된(클리핑) 파일 복구
    "declip": "adeclip,highpass=f=80,speechnorm=e=3",
}


def filter_chain(name: str) -> str:
    """설정값을 ffmpeg 필터 문자열로. 목록에 없으면 그대로 필터로 넘긴다."""
    key = (name or "off").strip()
    return FILTERS.get(key.lower(), key)


def to_wav16k(src: Path, dst: Path, audio_filter: str = "") -> Path:
    """src 를 16kHz mono 16bit wav 로 변환해 dst 에 쓴다."""
    exe = ffmpeg_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    chain = filter_chain(audio_filter)
    cmd = [
        exe, "-nostdin", "-y",
        "-i", str(src),
        "-vn",              # 비디오 트랙 무시 (mp4/mkv 대응)
        *(["-af", chain] if chain else []),
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not dst.exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        hint = (
            f"\n\nAUDIO_FILTER='{audio_filter}' 를 적용하다 실패했을 수 있습니다. "
            f"쓸 수 있는 값: {', '.join(FILTERS)} 또는 ffmpeg 필터 문자열."
            if chain else ""
        )
        raise RuntimeError(f"오디오 변환 실패 ({src.name}):\n{tail}{hint}")
    return dst


def read_wav(path: Path, start: int = 0, end: int | None = None) -> np.ndarray:
    """to_wav16k 로 만든 wav 의 [start, end) 샘플 구간을 float32 로 읽는다.

    필요한 구간만 읽는 게 핵심이다. 10시간 녹음을 통째로 올리면 파형만 2.3GB 라
    조각 처리의 의미가 없어진다. wave 모듈은 setpos 로 임의 지점부터 읽을 수 있다.

    ffmpeg 로 다시 디코딩하지 않는 이유: 이미 16kHz mono 16bit PCM 이라
    디코딩이라고 할 게 없다. 바이트를 그대로 읽으면 결과가 같고 훨씬 빠르다.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            total = handle.getnframes()
            if width != 2 or rate != SAMPLE_RATE:
                # 여기서 막지 않으면 타임스탬프가 샘플레이트 비율만큼 어긋난다
                raise RuntimeError(
                    f"{SAMPLE_RATE}Hz 16bit PCM wav 가 아닙니다 "
                    f"(sampwidth={width}, rate={rate}). to_wav16k 를 거친 파일인지 확인하세요."
                )
            first = max(0, min(int(start), total))
            last = total if end is None else max(first, min(int(end), total))
            handle.setpos(first)
            frames = handle.readframes(last - first)
    except wave.Error as exc:
        raise RuntimeError(
            f"16bit PCM wav 로 읽을 수 없습니다 ({path.name}): {exc}. "
            "to_wav16k 를 거친 파일인지 확인하세요."
        ) from exc

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def clip_wav(path: Path, start: float, end: float, max_sec: float = 300.0) -> bytes:
    """[start, end) 초 구간만 담은 wav 바이트. 브라우저에 그대로 내려보낸다.

    전사록에서 한 줄을 눌렀을 때 그 부분만 들려주기 위한 것. 10시간 녹음이면
    wav 가 1.1GB 라 통째로 내려보낼 수 없다. 이미 16kHz mono PCM 이므로
    ffmpeg 를 부를 것도 없이 해당 바이트만 떠서 헤더를 새로 씌우면 끝이다.

    max_sec 는 실수로 몇 시간짜리를 요청했을 때를 막는 상한.
    """
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        total = handle.getnframes()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        first = max(0, min(int(start * rate), total))
        last = min(total, first + int(max_sec * rate), max(first, int(end * rate)))
        handle.setpos(first)
        frames = handle.readframes(last - first)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)
    return buffer.getvalue()


def sample_count(path: Path) -> int:
    """wav 의 총 샘플 수. 길이를 초가 아니라 샘플로 정확히 알아야 할 때 쓴다."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


def duration_sec(path: Path) -> float:
    """ffprobe 로 길이(초)를 읽는다. 실패하면 0.0."""
    exe = shutil.which("ffprobe")
    if not exe:
        return 0.0
    proc = subprocess.run(
        [exe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, errors="replace",
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0
