"""오디오 입력 정규화. 어떤 포맷이 들어와도 16kHz mono wav 로 통일한다.

Whisper 도 pyannote 도 16kHz mono 를 기대하므로 여기서 한 번만 변환하고
이후 단계는 전부 같은 wav 파일을 재사용한다.
"""

import shutil
import subprocess
from pathlib import Path


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


def to_wav16k(src: Path, dst: Path) -> Path:
    """src 를 16kHz mono 16bit wav 로 변환해 dst 에 쓴다."""
    exe = ffmpeg_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe, "-nostdin", "-y",
        "-i", str(src),
        "-vn",              # 비디오 트랙 무시 (mp4/mkv 대응)
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not dst.exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"오디오 변환 실패 ({src.name}):\n{tail}")
    return dst


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
