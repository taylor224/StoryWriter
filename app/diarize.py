"""pyannote.audio speaker diarization plus speaker embedding extraction.

We call pyannote directly instead of using whisperx's DiarizationPipeline
wrapper, because that wrapper does not expose `return_embeddings=True` — and
those embeddings are the entire basis of "enroll a speaker once, recognize them
in every later file".
"""

import bisect
import threading
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from . import asr, audio, config, vad

_lock = threading.RLock()
_pipeline = None

GATE_HELP = (
    "Could not load the diarization model. Check the following:\n"
    "  1) HF_TOKEN is filled in inside .env\n"
    f"  2) You accepted the terms (gate approval) at https://huggingface.co/{config.DIARIZE_MODEL}\n"
    "  3) Internet access (the model downloads on first run)"
)


class DiarizationSetupError(RuntimeError):
    pass


def get_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = _build_pipeline()
        return _pipeline


def unload_pipeline() -> None:
    global _pipeline
    with _lock:
        _pipeline = None
    asr.free_vram()


def load_waveform(wav_path: Path, timeline: "vad.Timeline | None" = None) -> dict[str, Any]:
    """Read a 16kHz mono PCM wav into the in-memory form pyannote accepts.

    Pass a timeline to hand it the silence-trimmed waveform. pyannote's
    segmentation sweeps the whole file with a sliding window, so silence costs
    exactly as much as speech. On something like a 10-hour meeting recording
    with lots of dead air, this is where the biggest time saving comes from.

    pyannote 4.x decodes file paths through torchcodec, which needs the FFmpeg
    shared libraries (avcodec/avformat DLLs). The winget build of ffmpeg on
    Windows is static and installs only the .exe, so that DLL load fails.

    Since we already produced a 16kHz mono 16-bit wav with the ffmpeg CLI,
    reading it with the standard library avoids the torchcodec path entirely.
    """
    import torch

    # For a chunk, read only its range — this is what keeps a 10-hour file out of memory.
    origin, finish = timeline.span if timeline is not None else (0, None)
    samples = audio.read_wav(wav_path, origin, finish)
    if timeline is not None:
        samples = timeline.apply(samples, origin)

    waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": vad.SAMPLE_RATE}


def _build_pipeline():
    if not config.HF_TOKEN:
        raise DiarizationSetupError(GATE_HELP)

    import torch

    # The torchcodec warning is already filtered in config. We pass waveforms, so it is harmless.
    from pyannote.audio import Pipeline

    try:
        # pyannote 4.x takes token=, 3.x takes use_auth_token=
        pipeline = Pipeline.from_pretrained(config.DIARIZE_MODEL, token=config.HF_TOKEN)
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            config.DIARIZE_MODEL, use_auth_token=config.HF_TOKEN
        )
    except Exception as exc:
        raise DiarizationSetupError(f"{GATE_HELP}\n\nCause: {exc}") from exc

    if pipeline is None:
        # from_pretrained returns None instead of raising when the gate is not approved
        raise DiarizationSetupError(GATE_HELP)

    pipeline.to(torch.device(config.resolve_device()))
    return pipeline


class Diarization(NamedTuple):
    turns: list[dict]                      # [{'start','end','speaker'}] sorted by start time
    embeddings: dict[str, np.ndarray]      # {'SPEAKER_00': (256,)} — some speakers are missing
    speech_sec: dict[str, float]           # {'SPEAKER_00': total seconds spoken}
    overlaps: list[list[str]]              # speaker pairs who talked at once — cannot be one person


def _overlapping_pairs(annotation, minimum: float) -> list[list[str]]:
    """Find pairs of speakers who talked at the same time.

    If two labels overlap in speech they cannot be the same person. That is the
    only hard disproof available when speakers get re-merged automatically later
    (stitch.collapse), so we collect it here.

    A few tens of ms brushing at a boundary is ignored — that is segmentation
    slop, not simultaneous speech.
    """
    spans = sorted(
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    )
    shared: dict[tuple[str, str], float] = {}
    active: list[tuple[float, str]] = []
    for start, end, label in spans:
        active = [item for item in active if item[0] > start]
        for other_end, other in active:
            if other == label:
                continue  # just one speaker spread across multiple tracks
            pair = (label, other) if label < other else (other, label)
            shared[pair] = shared.get(pair, 0.0) + min(end, other_end) - start
        active.append((end, label))
    return [list(pair) for pair, seconds in shared.items() if seconds >= minimum]


def diarize(
    wav_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    timeline: "vad.Timeline | None" = None,
) -> Diarization:
    """Speaker diarization result.

    Pass a timeline to run on the silence-trimmed waveform while still emitting
    turns on the original clock. Callers only ever see original-clock results.
    """
    pipeline = get_pipeline()

    kwargs: dict[str, Any] = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)

    # Hand over the waveform rather than a path, to bypass torchcodec decoding
    audio_input: Any = load_waveform(wav_path, timeline)
    full, exclusive, raw_embeddings = _apply(pipeline, audio_input, kwargs)

    # Assigning speakers to words uses the overlap-free annotation: only one
    # speaker can win an overlapping region anyway, and that is exactly what
    # pyannote produces this variant for.
    turns = [
        {"start": float(seg.start), "end": float(seg.end), "speaker": str(label)}
        for seg, _, label in (exclusive or full).itertracks(yield_label=True)
    ]
    if timeline is not None:
        turns = _restore_turns(turns, timeline)
    turns.sort(key=lambda t: (t["start"], t["end"]))

    # Speech time should come from the overlap-inclusive annotation to reflect
    # how much someone actually spoke. It decides whether an embedding is
    # trustworthy, so that is the right source.
    #
    # These sums hold even when silence was trimmed: only silence was removed and
    # Timeline.split preserves length, so mapping back gives the same totals.
    speech: dict[str, float] = {}
    for seg, _, label in full.itertracks(yield_label=True):
        key = str(label)
        speech[key] = speech.get(key, 0.0) + float(seg.end - seg.start)

    embeddings: dict[str, np.ndarray] = {}
    if raw_embeddings is not None:
        arr = np.asarray(raw_embeddings)
        # Embedding order follows speaker_diarization.labels(), not the exclusive one
        for idx, label in enumerate(full.labels()):
            if idx >= arr.shape[0]:
                break
            vec = np.asarray(arr[idx], dtype=np.float32).ravel()
            # Speakers that only appear in overlap come back as NaN, and short
            # speaker counts get zero-padded. Neither is usable, so both go.
            if vec.size and np.isfinite(vec).all() and vec.any():
                embeddings[str(label)] = vec

    # Must come from `full`, which still has overlaps — `exclusive` has them stripped.
    overlaps = _overlapping_pairs(full, config.MERGE_MIN_OVERLAP_SEC)
    return Diarization(turns, embeddings, speech, overlaps)


def _restore_turns(turns: list[dict], timeline: "vad.Timeline") -> list[dict]:
    """Map speaker turns back to the original clock, splitting any that cross a cut.

    Without the split a speaker would own the silence we removed and steal words
    from whoever is on the far side of it in the overlap calculation
    (see attach_speakers).
    """
    restored: list[dict] = []
    for turn in turns:
        for start, end in timeline.split(turn["start"], turn["end"]):
            if end > start:
                restored.append({"start": start, "end": end, "speaker": turn["speaker"]})
    return restored


def _apply(pipeline, audio_input: Any, kwargs: dict[str, Any]):
    """Normalize every pyannote version's return shape to (full, exclusive, embeddings).

    4.x   : DiarizeOutput(speaker_diarization, exclusive_speaker_diarization,
            speaker_embeddings). Embeddings are always included and there is no
            return_embeddings argument — passing one only logs "Ignoring
            unexpected keyword arguments".
    3.x   : (Annotation, ndarray) tuple when given return_embeddings=True.
    legacy: just an Annotation.
    """
    import inspect

    try:
        params = inspect.signature(pipeline.apply).parameters
    except (TypeError, ValueError):
        params = {}

    if "return_embeddings" in params:  # pyannote 3.x
        output = pipeline(audio_input, return_embeddings=True, **kwargs)
    else:  # pyannote 4.x and newer
        output = pipeline(audio_input, **kwargs)

    full = getattr(output, "speaker_diarization", None)
    if full is not None:  # DiarizeOutput
        return (
            full,
            getattr(output, "exclusive_speaker_diarization", None),
            getattr(output, "speaker_embeddings", None),
        )

    if isinstance(output, tuple):  # (Annotation, embeddings)
        return output[0], None, output[1]

    return output, None, None  # just an Annotation


# ── Attaching speakers to words and segments ──────────────────────────
def _window(turns: list[dict], starts: list[float], max_dur: float, start: float, end: float):
    lo = bisect.bisect_left(starts, start - max_dur)
    hi = bisect.bisect_right(starts, end)
    return range(max(0, lo), min(len(turns), hi))


def _best_overlap(turns, starts, max_dur, start, end) -> str | None:
    best, best_ov = None, 0.0
    for j in _window(turns, starts, max_dur, start, end):
        turn = turns[j]
        ov = min(end, turn["end"]) - max(start, turn["start"])
        if ov > best_ov:
            best, best_ov = turn["speaker"], ov
    return best


def _nearest(turns, point: float) -> str | None:
    best, best_dist = None, float("inf")
    for turn in turns:
        if turn["start"] <= point <= turn["end"]:
            return turn["speaker"]
        dist = turn["start"] - point if point < turn["start"] else point - turn["end"]
        if dist < best_dist:
            best, best_dist = turn["speaker"], dist
    return best


def attach_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Assign a speaker to each segment, splitting where the speaker changes mid-segment.

    Whisper cuts sentences without any idea where speaker boundaries are, so two
    people sharing one segment is common. When word timestamps exist we split
    right at the handover.
    """
    if not turns:
        # No speaker turns were found — treat the whole thing as one person
        return [{**seg, "speaker": "SPEAKER_00"} for seg in segments]

    starts = [t["start"] for t in turns]
    max_dur = max((t["end"] - t["start"]) for t in turns)

    result: list[dict] = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", seg_start) or seg_start)
        seg_text = (seg.get("text") or "").strip()

        words = [
            w for w in (seg.get("words") or [])
            if w.get("start") is not None and w.get("end") is not None
        ]

        if not words:
            speaker = _best_overlap(turns, starts, max_dur, seg_start, seg_end) or _nearest(
                turns, (seg_start + seg_end) / 2
            )
            if seg_text:
                result.append(
                    {"start": seg_start, "end": seg_end, "text": seg_text,
                     "speaker": speaker, "words": []}
                )
            continue

        for word in words:
            ws, we = float(word["start"]), float(word["end"])
            word["speaker"] = (
                _best_overlap(turns, starts, max_dur, ws, we)
                or _nearest(turns, (ws + we) / 2)
            )

        # Group consecutive words by speaker
        groups: list[list[dict]] = []
        for word in words:
            if groups and groups[-1][-1]["speaker"] == word["speaker"]:
                groups[-1].append(word)
            else:
                groups.append([word])

        if len(groups) == 1:
            # No split needed — reuse the original text so spacing and punctuation survive
            if seg_text:
                result.append(
                    {"start": seg_start, "end": seg_end, "text": seg_text,
                     "speaker": groups[0][0]["speaker"], "words": words}
                )
            continue

        for group in groups:
            text = " ".join((w.get("word") or "").strip() for w in group).strip()
            if not text:
                continue
            result.append(
                {
                    "start": float(group[0]["start"]),
                    "end": float(group[-1]["end"]),
                    "text": text,
                    "speaker": group[0]["speaker"],
                    "words": group,
                }
            )

    result.sort(key=lambda s: s["start"])
    return result


def embed_single_speaker(wav_path: Path) -> tuple[np.ndarray, float]:
    """Extract one speaker embedding from an enrollment sample (assumes a single speaker)."""
    result = diarize(wav_path, num_speakers=1)
    embeddings, speech = result.embeddings, result.speech_sec
    if not embeddings:
        raise RuntimeError(
            "Could not extract a speaker embedding from the sample. Please use a clean "
            "recording of at least 10 seconds with only one person speaking."
        )
    label = max(speech, key=speech.get) if speech else next(iter(embeddings))
    if label not in embeddings:
        label = next(iter(embeddings))
    return embeddings[label], speech.get(label, 0.0)
