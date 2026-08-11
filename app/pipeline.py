"""The whole flow: transcribe -> align -> diarize -> recognize speakers -> save.

Every stage writes its output to data/cache. Re-running a failed job skips any
stage whose inputs are unchanged and picks up where it broke. Transcription is
the most expensive stage, so dying in diarization must never cost you a re-transcribe.

Also usable from the CLI:
    python -m app.pipeline meeting.mp3 --name 2026-08-10 --max-speakers 4
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import (
    asr, audio, cache, cleanup, config, db, diarize, matching, render, stitch, vad,
)

ProgressFn = Callable[[str, float], None]

GLOSSARY_KEY = "glossary"


def _noop(stage: str, percent: float) -> None:  # pragma: no cover - default callback
    pass


def build_initial_prompt(extra: str | None = None) -> str:
    """Feed enrolled speaker names plus the glossary to Whisper as initial_prompt.

    Telling it the proper nouns up front (people's names, company jargon)
    improves recognition.

    No label word ("Participants:" or the like) is prepended on purpose — that
    would bias language detection toward whatever language the label is in.
    """
    parts: list[str] = []
    names = list(db.speaker_names())
    if names:
        parts.append(", ".join(names) + ".")
    glossary = (extra if extra is not None else db.get_setting(GLOSSARY_KEY, "")).strip()
    if glossary:
        parts.append(glossary)
    return " ".join(parts).strip()


def run(
    source: Path,
    name: str,
    language: str | None = None,
    initial_prompt: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress: ProgressFn = _noop,
    resume: bool = True,
) -> dict[str, Any]:
    source = Path(source)
    name = render.sanitize_name(name)
    warnings: list[str] = []
    reused: list[str] = []

    def cached(stage: str, key: dict[str, Any]) -> dict[str, Any] | None:
        return cache.load(name, stage, key) if resume else None

    # ── 1) Normalize the audio (16kHz mono wav) ──────────────────────
    # A .wav extension can still be 44.1kHz stereo, so always convert.
    progress("Converting audio", 5)
    wav_path = config.UPLOAD_DIR / f"{name}.16k.wav"
    # Changing the filter changes the wav itself. Every later key embeds this one,
    # so putting it here alone makes everything from transcription on recompute.
    audio_key = {**cache.audio_key(source), "filter": config.AUDIO_FILTER}

    hit = cached("audio", audio_key)
    if hit and wav_path.exists():
        duration = hit["duration"]
        reused.append("audio conversion")
    else:
        audio.to_wav16k(source, wav_path, config.AUDIO_FILTER)
        duration = audio.duration_sec(wav_path)
        cache.save(name, "audio", audio_key, {"duration": duration})

    # ── 2) Work out where the silence is ─────────────────────────────
    # This only decides what to cut; the actual cutting happens wherever the
    # waveform is used. Transcription, alignment and diarization all run on the
    # trimmed waveform and map their results back through the timeline right
    # before returning them. That is why stage 5 onward never needs to know a
    # cut happened.
    #
    # The list of kept regions is small, so it goes in the cache: on a retry,
    # diarization must be able to cut identically without re-reading the waveform.
    trim_params = vad.params() if config.TRIM_SILENCE else None
    trim_key = {"audio": audio_key, "params": trim_params}

    hit = cached("trim", trim_key) if config.TRIM_SILENCE else None
    if not config.TRIM_SILENCE:
        # Without ffprobe, duration is 0 — so count the samples from the wav
        # header instead. A 0 here makes every chunk range empty and the whole
        # transcription silently vanishes.
        timeline = vad.Timeline.identity(audio.sample_count(wav_path))
    elif hit:
        timeline = vad.Timeline(hit["regions"], hit["total"])
        reused.append("silence removal")
    else:
        progress("Checking for silence", 8)
        timeline, note = vad.plan(audio.read_wav(wav_path))
        if note:
            warnings.append(note)
        cache.save(
            name, "trim", trim_key,
            {"regions": timeline.regions, "total": timeline.total},
        )
    trim_info = timeline.stats()
    if timeline.trimmed:
        progress(f"Removed {trim_info['removed']:.0f}s of silence", 10)

    prompt = initial_prompt if initial_prompt is not None else build_initial_prompt()

    # ── 3) Split into chunks ─────────────────────────────────────────
    # Running a long file in one pass is slow, and pyannote starts splitting one
    # person into several speakers (over 10 hours both the voice and the mic
    # situation drift). Each chunk goes through transcribe -> align -> diarize,
    # and the speakers are stitched back together by voice at the end.
    #
    # A chunk Timeline carries original coordinates, so a timestamp from inside a
    # chunk is original-clock after one restore — no chunk index to carry around
    # and no offsets to add.
    pieces = vad.chunks(timeline, config.CHUNK_SEC)
    if len(pieces) > 1:
        progress(f"Processing in {len(pieces)} chunks", 11)

    def stage(base: str, index: int) -> str:
        return base if len(pieces) == 1 else f"{base}.{index}"

    all_segments: list[dict] = []
    parts: list[tuple[list[dict], dict[str, np.ndarray], dict[str, float]]] = []
    detected: str = language or ""
    detection: dict[str, Any] = {}
    span_low, span_high = 12.0, 88.0

    for index, piece in enumerate(pieces):
        head = f"[{index + 1}/{len(pieces)}] " if len(pieces) > 1 else ""
        base = span_low + (span_high - span_low) * index / len(pieces)
        step = (span_high - span_low) / len(pieces)
        origin = piece.span[0]
        chunk_id = list(piece.span) if len(pieces) > 1 else None
        held: dict[str, Any] = {}

        def waveform(piece=piece, origin=origin, held=held):
            """This chunk's waveform. Never even opens the file if both stages hit cache.

            Reads only this chunk's range, not the whole wav. Loading 10 hours at
            once costs 2.3GB for the waveform alone, which would defeat chunking.
            """
            if "value" not in held:
                window = audio.read_wav(wav_path, origin, piece.span[1])
                held["value"] = piece.apply(window, origin)
            return held["value"]

        # ── 3-1) Transcribe ──────────────────────────────────────────
        # Detect the language on the first chunk only and force it for the rest.
        # Detecting per chunk lets one quiet chunk wander off into another language.
        forced = language or (detected if index else "")
        transcribe_key = {
            "audio": audio_key,
            "model": config.WHISPER_MODEL,
            "language": forced,
            "prompt": prompt,
            # Trimming silence differently changes the transcription input itself.
            # Leave it out of the key and an old result survives a retry with new settings.
            "trim": trim_params,
            "chunk": chunk_id,
        }
        hit = cached(stage("transcribe", index), transcribe_key)
        if hit:
            chunk_language, segments = hit["language"], hit["segments"]
            chunk_detection = hit.get("detection") or {}
            reused.append(f"{head}transcription")
            progress(f"{head}reusing transcription", base + step * 0.6)
        else:
            progress(f"{head}transcribing", base)
            result = asr.transcribe(
                waveform(), language=forced or None, initial_prompt=prompt
            )
            chunk_language = result.get("language") or forced or "unknown"
            segments = result.get("segments") or []
            chunk_detection = result.get("detection") or {}
            cache.save(
                name, stage("transcribe", index), transcribe_key,
                {
                    "language": chunk_language,
                    "segments": segments,
                    "detection": chunk_detection,
                },
            )
        if not index:
            detected, detection = chunk_language, chunk_detection

        # ── 3-2) Word-level alignment ────────────────────────────────
        # Up to here everything is on the chunk's own clock (the transcribe cache
        # too). Alignment ends here, so this is where we map back to the original
        # clock; the align cache onward stores original-clock times. The only
        # rule to keep straight is that the two caches use different clocks.
        align_key = {"transcribe": transcribe_key, "language": chunk_language}
        hit = cached(stage("align", index), align_key)
        if hit:
            segments = hit["segments"]
            if hit.get("warning"):
                warnings.append(hit["warning"])
            reused.append(f"{head}word alignment")
            progress(f"{head}reusing word alignment", base + step * 0.7)
        else:
            progress(f"{head}aligning word timestamps", base + step * 0.55)
            segments, align_warning = asr.align(segments, chunk_language, waveform())
            segments = piece.restore(segments)
            if align_warning and align_warning not in warnings:
                warnings.append(align_warning)
            cache.save(
                name, stage("align", index), align_key,
                {"segments": segments, "warning": align_warning},
            )
        all_segments.extend(segments)

        if config.UNLOAD_BETWEEN_STAGES:
            asr.unload_model()
            asr.unload_align_models()
        held.clear()  # diarization re-reads the wav, so the waveform ends here

        # ── 3-3) Diarization + embeddings ───────────────────────────
        # pyannote's segmentation sweeps whatever audio it is given with a
        # sliding window, so silence costs as much as speech. Passing the chunk
        # Timeline here is the single biggest saving on a long recording. turns
        # come back on the original clock from inside diarize.
        diarize_key = {
            "audio": audio_key,
            "model": config.DIARIZE_MODEL,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "trim": trim_params,
            "chunk": chunk_id,
            "v": 2,  # output gained `overlaps` — new code must not read old caches
        }
        hit = cached(stage("diarize", index), diarize_key)
        if hit:
            turns = hit["turns"]
            embeddings = {
                label: np.asarray(vector, dtype=np.float32)
                for label, vector in hit["embeddings"].items()
            }
            speech_sec = hit["speech_sec"]
            overlaps = hit.get("overlaps") or []
            reused.append(f"{head}diarization")
            progress(f"{head}reusing diarization", base + step * 0.95)
        else:
            progress(f"{head}diarizing", base + step * 0.75)
            turns, embeddings, speech_sec, overlaps = diarize.diarize(
                wav_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                timeline=piece,
            )
            cache.save(
                name, stage("diarize", index), diarize_key,
                {
                    "turns": turns,
                    "embeddings": {k: v.tolist() for k, v in embeddings.items()},
                    "speech_sec": speech_sec,
                    "overlaps": overlaps,
                },
            )
            if config.UNLOAD_BETWEEN_STAGES:
                diarize.unload_pipeline()
        parts.append((turns, embeddings, speech_sec, overlaps))

    # Get the language wrong and Whisper confidently invents things nobody said.
    # Letting that slide silently makes it impossible to diagnose, so the
    # reasoning behind the decision goes into the result.
    if detection.get("auto"):
        votes = detection.get("votes") or {}
        confidence = detection.get("confidence", 0.0)
        if confidence < 0.9 or len(votes) > 1:
            tally = ", ".join(f"{k} {v}" for k, v in sorted(votes.items(), key=lambda x: -x[1]))
            warnings.append(
                f"Language was auto-detected as '{detected}' (confidence {confidence}). "
                f"Votes per sample: {tally}. "
                "If the result contains things nobody said, pin the language and run again."
            )

    # ── 4) Stitch chunks together, then re-merge over-split speakers ──
    # Both are cheap and both are stages you want to re-run with a different
    # threshold, so neither is cached.
    all_segments.sort(key=lambda seg: (seg.get("start", 0.0), seg.get("end", 0.0)))
    if len(parts) == 1:
        turns, embeddings, speech_sec, overlaps = parts[0]
    else:
        progress("Stitching chunks together", 89)
        turns, embeddings, speech_sec, overlaps, notes = stitch.merge(parts)
        warnings.append(stitch.summary(len(parts), len(speech_sec)))
        for note in notes:
            print(f"[stitch] {note}", flush=True)

    # pyannote splits one person into several speakers often. Chunked or not, we
    # compare once more at the end and merge. Pairs who talked over each other
    # are never merged.
    progress("Re-merging speakers", 90)
    turns, embeddings, speech_sec, merges = stitch.collapse(
        turns, embeddings, speech_sec, overlaps
    )
    if merges:
        warnings.append(stitch.collapse_summary(merges, len(speech_sec)))
        for note in merges:
            print(f"[merge] {note}", flush=True)
    segments = all_segments

    # ── 5) Filter out hallucinations ─────────────────────────────────
    # Drop sentences Whisper invented over noise. Doing this before speaker
    # assignment keeps invented text out of the speaker statistics.
    # Only the certain cases are dropped; the rest stay and are merely flagged.
    segments, dropped, suspect = cleanup.clean(segments)
    warnings.extend(cleanup.summary(dropped, suspect))

    if not turns:
        warnings.append("No speaker turns were found. Treating everything as one person.")
    if not embeddings:
        warnings.append(
            "No speaker embeddings were produced, so automatic speaker recognition "
            "was skipped for this result."
        )

    # ── 6) Attach speakers to segments (splitting where the speaker changes) ──
    # Everything from here is cheap, so it is not cached. The enrolled speakers
    # may have changed, so re-matching on every retry is actually the right thing.
    progress("Assigning speakers", 92)
    segments = diarize.attach_speakers(segments, turns)

    # ── 7) Match against enrolled speakers ───────────────────────────
    progress("Matching enrolled speakers", 95)
    matches = matching.match(embeddings, speech_sec)
    ordered = render.order_labels(segments, speech_sec)
    displays, anon = render.assign_displays(ordered, matches)

    speakers: dict[str, Any] = {}
    for label in ordered:
        info = matches.get(label) or {}
        vector = embeddings.get(label)
        speakers[label] = {
            "display": displays.get(label, label),
            "anon_label": anon.get(label),
            "speaker_id": info.get("speaker_id"),
            "matched": bool(info.get("matched")),
            "score": info.get("score", 0.0),
            "runner_up": info.get("runner_up", 0.0),
            "reason": info.get("reason", ""),
            "total_speech": round(speech_sec.get(label, 0.0), 2),
            "embedding": db.normalize(vector).tolist() if vector is not None else None,
        }

    # ── 8) Save ──────────────────────────────────────────────────────
    progress("Saving", 98)
    payload: dict[str, Any] = {
        "name": name,
        "source_file": source.name,
        "audio_file": wav_path.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration": round(duration, 2),
        "language": detected,
        "language_detection": detection,
        "initial_prompt": prompt,
        "audio_filter": config.AUDIO_FILTER,
        "trim": trim_info,
        "chunks": [
            {"start": round(p.span[0] / vad.SAMPLE_RATE, 2),
             "end": round(p.span[1] / vad.SAMPLE_RATE, 2)}
            for p in pieces
        ] if len(pieces) > 1 else [],
        "dropped": dropped,
        "suspect": suspect,
        "merged_speakers": merges,
        "warnings": warnings,
        "reused_stages": reused,
        "speakers": speakers,
        "segments": [
            {
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "speaker": seg.get("speaker"),
                "text": seg["text"],
                "words": [
                    {
                        "word": word.get("word", ""),
                        "start": round(float(word["start"]), 3),
                        "end": round(float(word["end"]), 3),
                        "speaker": word.get("speaker"),
                    }
                    for word in (seg.get("words") or [])
                    if word.get("start") is not None and word.get("end") is not None
                ],
            }
            for seg in segments
            if (seg.get("text") or "").strip()
        ],
    }
    txt_file, json_file = render.save(payload)
    payload["txt_file"] = str(txt_file)
    payload["json_file"] = str(json_file)

    cache.clear(name)  # it succeeded, so the intermediates are no longer needed
    progress("Done", 100)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file and separate speakers")
    parser.add_argument("audio", type=Path, help="path to the audio/video file")
    parser.add_argument("--name", default=None, help="result name (default: today's date)")
    parser.add_argument("--language", default=None, help="ko / en / etc. Auto-detected if omitted")
    parser.add_argument("--prompt", default=None, help="set initial_prompt directly")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--no-resume", action="store_true", help="ignore cached intermediates and start over"
    )
    parser.add_argument(
        "--no-trim", action="store_true", help="turn off silence removal before transcription"
    )
    args = parser.parse_args(argv)

    if args.no_trim:
        config.TRIM_SILENCE = False

    if not args.audio.exists():
        print(f"File not found: {args.audio}", file=sys.stderr)
        return 1

    db.init()
    name = render.sanitize_name(args.name or render.default_name())
    if not cache.stages(name) and render.json_path(name).exists():
        name = render.unique_name(name)

    staged = config.UPLOAD_DIR / f"{name}{args.audio.suffix.lower()}"
    if staged.resolve() != args.audio.resolve():
        shutil.copy2(args.audio, staged)

    def show(stage: str, percent: float) -> None:
        print(f"[{percent:5.1f}%] {stage}", flush=True)

    payload = run(
        staged,
        name,
        language=args.language,
        initial_prompt=args.prompt,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        progress=show,
        resume=not args.no_resume,
    )

    print()
    for warning in payload.get("warnings", []):
        print(f"[warning] {warning}")
    if payload.get("reused_stages"):
        print(f"Reused stages: {', '.join(payload['reused_stages'])}")
    trim = payload.get("trim") or {}
    if trim.get("enabled"):
        print(
            f"Silence removal: cut {trim['removed']:.1f}s, transcribed {trim['kept']:.1f}s "
            f"across {trim['regions']} regions. Timestamps stay on the original clock."
        )
    if len(payload.get("chunks") or []) > 1:
        print(f"Chunked: ran as {len(payload['chunks'])} pieces, speakers stitched by voice")
    for item in payload.get("dropped") or []:
        print(f"[removed] {render.timestamp(item['start'])} \"{item['text']}\" — {item['reason']}")
    for item in payload.get("suspect") or []:
        print(f"[suspect] {render.timestamp(item['start'])} \"{item['text']}\" — {item['reason']}")
    print(f"Language: {payload['language']}   Length: {payload['duration']:.1f}s")
    print(f"Saved: {payload['txt_file']}")
    print(f"      {payload['json_file']}")
    print("-" * 60)
    for line in payload.get("lines", [])[:20]:
        print(f"{line['name']} : {line['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
