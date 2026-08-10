"""설치 점검 스크립트.

    python scripts/smoke.py              # 환경만 확인
    python scripts/smoke.py sample.wav   # 실제 전사 + 화자 분리까지 확인
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402  (HF_HOME 설정을 위해 torch 보다 먼저)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 66)
    print("환경 점검")
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
                  "감지됨 — 단, CTranslate2 는 MPS 미지원이라 전사는 CPU 로 돕니다")
        else:
            check("가속기", True, "없음 — 전부 CPU")
            print("      → NVIDIA GPU 가 있는데 이 메시지가 보이면 CUDA 휠로 재설치:")
            print("         pip install -U torch torchaudio "
                  "--index-url https://download.pytorch.org/whl/cu128")

        print(f"       전사(WhisperX): {asr_dev} / {compute}   "
              f"화자분리·정렬(torch): {device}")
        if asr_dev == "cpu":
            print("       CPU 전사는 느립니다. WHISPER_MODEL=large-v3-turbo 를 권장합니다.")
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
        "설정됨" if config.HF_TOKEN else ".env 에 HF_TOKEN 을 넣으세요",
    )

    for module in ("whisperx", "pyannote.audio", "fastapi", "scipy"):
        try:
            __import__(module)
            ok &= check(f"import {module}", True)
        except ImportError as exc:
            ok &= check(f"import {module}", False, str(exc))

    print(f"\n모델 캐시: {config.MODEL_CACHE}")
    print(f"데이터   : {config.DATA_DIR}")

    if len(sys.argv) < 2:
        print("\n오디오 파일을 인자로 주면 실제 파이프라인까지 검증합니다:")
        print("    python scripts/smoke.py sample.wav")
        return 0 if ok else 1

    source = Path(sys.argv[1])
    if not source.exists():
        print(f"\n파일 없음: {source}")
        return 1

    print("\n" + "=" * 66)
    print(f"파이프라인 실행: {source.name}  (첫 실행은 모델 다운로드로 오래 걸립니다)")
    print("=" * 66)

    from app import asr, db, diarize

    db.init()
    wav = config.UPLOAD_DIR / "_smoke.wav"
    audio.to_wav16k(source, wav)
    print(f"오디오 변환 완료 — {audio.duration_sec(wav):.1f}초")

    waveform = asr.load_audio(wav)
    result = asr.transcribe(waveform)
    language = result.get("language")
    print(f"전사 완료 — 언어 {language}, 세그먼트 {len(result.get('segments', []))}개")

    segments, warning = asr.align(result["segments"], language, waveform)
    if warning:
        print(f"  경고: {warning}")

    turns, embeddings, speech = diarize.diarize(wav)
    print(f"화자 분리 완료 — 구간 {len(turns)}개, 화자 {len(speech)}명")
    for label, seconds in sorted(speech.items()):
        vec = embeddings.get(label)
        shape = f"임베딩 {vec.shape}" if vec is not None else "임베딩 없음"
        print(f"  {label}: {seconds:.1f}초 · {shape}")

    if not embeddings:
        print("\n[경고] 임베딩이 없으면 화자 자동 인식이 동작하지 않습니다. "
              "pyannote.audio 4.x 인지 확인하세요.")

    merged = diarize.attach_speakers(segments, turns)
    print("\n--- 미리보기 (앞 10줄) ---")
    for seg in merged[:10]:
        print(f"{seg['speaker']} : {seg['text']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
