"""Audio input normalization. Whatever comes in becomes 16kHz mono wav.

Both Whisper and pyannote expect 16kHz mono, so we convert once here and every
later stage reuses the exact same wav file.
"""

import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


# The whole pipeline assumes this sample rate (Whisper and pyannote both want 16kHz).
SAMPLE_RATE = 16000


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegMissing(
            "ffmpeg not found. On Windows install it with `winget install Gyan.FFmpeg`, "
            "then open a new terminal."
        )
    return exe


# Filter chains that can be applied before transcription.
#
# Why the default is off: Whisper was trained on 680k hours of noisy real-world
# audio, so it is already robust to ordinary noise. Aggressive denoising creates
# artifacts the model never saw during training, and accuracy often gets *worse*.
# "Cleaner to the ear" does not mean "easier to transcribe".
#
# Still, there are situations where these help, so they are here to opt into.
FILTERS = {
    # Touch nothing (recommended default)
    "off": "",
    # Strip low-end rumble and lift up whoever is sitting far from the mic.
    # Stops a distant participant from being dropped entirely.
    # Leaves the spectrum alone, so it is the safer one for Whisper.
    "voice": "highpass=f=80,lowpass=f=7500,speechnorm=e=3",
    # The above plus FFT denoising. For recordings with constant fan/white noise.
    # Accuracy can get worse — always compare results with and without.
    "denoise": "highpass=f=80,afftdn=nf=-25,lowpass=f=7500,speechnorm=e=3",
    # Repair clipped (distorted) recordings
    "declip": "adeclip,highpass=f=80,speechnorm=e=3",
}


def filter_chain(name: str) -> str:
    """Setting value -> ffmpeg filter string. Unknown names pass through as-is."""
    key = (name or "off").strip()
    return FILTERS.get(key.lower(), key)


def to_wav16k(src: Path, dst: Path, audio_filter: str = "") -> Path:
    """Convert src to 16kHz mono 16-bit wav and write it to dst."""
    exe = ffmpeg_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    chain = filter_chain(audio_filter)
    cmd = [
        exe, "-nostdin", "-y",
        "-i", str(src),
        "-vn",              # ignore the video track (handles mp4/mkv)
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
            f"\n\nThis may have failed while applying AUDIO_FILTER='{audio_filter}'. "
            f"Valid values: {', '.join(FILTERS)}, or an ffmpeg filter string."
            if chain else ""
        )
        raise RuntimeError(f"Audio conversion failed ({src.name}):\n{tail}{hint}")
    return dst


def read_wav(path: Path, start: int = 0, end: int | None = None) -> np.ndarray:
    """Read samples [start, end) of a to_wav16k file as float32.

    Reading only the range you need is the whole point. A 10-hour recording is
    2.3GB as a waveform, which would defeat chunked processing. The wave module
    can seek to an arbitrary position with setpos.

    Why not decode through ffmpeg again: the file is already 16kHz mono 16-bit
    PCM, so there is nothing to decode. Reading the bytes directly gives the
    same result and is far faster.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            total = handle.getnframes()
            if width != 2 or rate != SAMPLE_RATE:
                # Without this guard every timestamp drifts by the sample-rate ratio
                raise RuntimeError(
                    f"Not a {SAMPLE_RATE}Hz 16-bit PCM wav "
                    f"(sampwidth={width}, rate={rate}). Make sure it went through to_wav16k."
                )
            first = max(0, min(int(start), total))
            last = total if end is None else max(first, min(int(end), total))
            handle.setpos(first)
            frames = handle.readframes(last - first)
    except wave.Error as exc:
        raise RuntimeError(
            f"Cannot read as 16-bit PCM wav ({path.name}): {exc}. "
            "Make sure it went through to_wav16k."
        ) from exc

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def clip_wav(path: Path, start: float, end: float, max_sec: float = 300.0) -> bytes:
    """wav bytes containing only seconds [start, end). Served straight to the browser.

    This is what plays back a single transcript line. A 10-hour recording is a
    1.1GB wav, so it cannot be shipped whole. Since it is already 16kHz mono PCM
    we just lift the relevant bytes and put a fresh header on them — no ffmpeg.

    max_sec caps an accidental request for hours of audio.
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
    """Total samples in the wav. Used when the length must be exact, not rounded to seconds."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


def duration_sec(path: Path) -> float:
    """Read the length in seconds via ffprobe. Returns 0.0 on failure."""
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
