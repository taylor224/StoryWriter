"""Installation check.

    python scripts/smoke.py              # environment only
    python scripts/smoke.py sample.wav   # also run a real transcription + diarization
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402  (before torch, so HF_HOME is set)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 66)
    print("Environment check")
    print("=" * 66)

    ok = True

    try:
        import torch

        check(f"torch {torch.__version__}", True)
        device = config.resolve_device()
        asr_dev = config.asr_device()
        compute = config.resolve_compute_type(asr_dev)

        if torch.cuda.is_available():
            check("CUDA", True, torch.cuda.get_device_name(0))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            check("Apple Silicon (MPS)", True,
                  "detected — but CTranslate2 has no MPS support, so transcription runs on CPU")
        else:
            check("Accelerator", True, "none — everything on CPU")
            print("      -> If you have an NVIDIA GPU and still see this, reinstall from the CUDA index:")
            print("         pip install -U torch torchaudio "
                  "--index-url https://download.pytorch.org/whl/cu128")

        print(f"       transcription (WhisperX): {asr_dev} / {compute}   "
              f"diarization + alignment (torch): {device}")
        if asr_dev == "cpu":
            print("       CPU transcription is slow. WHISPER_MODEL=large-v3-turbo is recommended.")
    except ImportError as exc:
        ok &= check("torch", False, str(exc))

    from app import audio

    try:
        ok &= check("ffmpeg", True, audio.ffmpeg_path())
    except audio.FFmpegMissing as exc:
        ok &= check("ffmpeg", False, str(exc))

    ok &= check(
        "HF_TOKEN",
        bool(config.HF_TOKEN),
        "set" if config.HF_TOKEN else "add HF_TOKEN to .env",
    )

    import warnings

    for module in ("whisperx", "pyannote.audio", "fastapi", "scipy"):
        try:
            with warnings.catch_warnings():
                # The torchcodec warning is harmless — we hand over waveforms directly.
                warnings.filterwarnings("ignore", message=".*torchcodec.*")
                __import__(module)
            ok &= check(f"import {module}", True)
        except ImportError as exc:
            ok &= check(f"import {module}", False, str(exc))

    print(f"\nModel cache: {config.MODEL_CACHE}")
    print(f"Data       : {config.DATA_DIR}")

    if len(sys.argv) < 2:
        print("\nPass an audio file to also verify the real pipeline:")
        print("    python scripts/smoke.py sample.wav")
        return 0 if ok else 1

    source = Path(sys.argv[1])
    if not source.exists():
        print(f"\nFile not found: {source}")
        return 1

    print("\n" + "=" * 66)
    print(f"Running the pipeline on {source.name}  (the first run downloads models and is slow)")
    print("=" * 66)

    from app import asr, db, diarize

    db.init()
    wav = config.UPLOAD_DIR / "_smoke.wav"
    audio.to_wav16k(source, wav, config.AUDIO_FILTER)
    print(f"Audio converted — {audio.duration_sec(wav):.1f}s")

    waveform = asr.load_audio(wav)
    result = asr.transcribe(waveform)
    language = result.get("language")
    print(f"Transcribed — language {language}, {len(result.get('segments', []))} segments")

    segments, warning = asr.align(result["segments"], language, waveform)
    if warning:
        print(f"  warning: {warning}")

    turns, embeddings, speech, overlaps = diarize.diarize(wav)
    print(f"Diarized — {len(turns)} turns, {len(speech)} speakers, {len(overlaps)} overlapping pairs")
    for label, seconds in sorted(speech.items()):
        vec = embeddings.get(label)
        shape = f"embedding {vec.shape}" if vec is not None else "no embedding"
        print(f"  {label}: {seconds:.1f}s · {shape}")

    if not embeddings:
        print("\n[warning] Without embeddings, automatic speaker recognition cannot work. "
              "Check that pyannote.audio is 4.x.")

    merged = diarize.attach_speakers(segments, turns)
    print("\n--- preview (first 10 lines) ---")
    for seg in merged[:10]:
        print(f"{seg['speaker']} : {seg['text']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
