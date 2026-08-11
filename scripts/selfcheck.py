"""Check everything that can run without a model.

    python scripts/selfcheck.py

Silence removal, chunking, cross-chunk speaker stitching, speaker re-merging and
the hallucination filter. It all runs on synthetic data, so no model and no GPU
are needed (numpy and scipy are enough).

Run this first whenever you touch vad.py / stitch.py / cleanup.py. The timestamp
side especially: if that breaks, every time in the output is silently wrong —
the kind of failure you cannot spot by eye.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import audio as audio_io, config, vad  # noqa: E402

SR = vad.SAMPLE_RATE
rng = np.random.default_rng(7)
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def speech(sec: float, amp: float = 0.25) -> np.ndarray:
    """Band-limited noise with a syllable rhythm. Closer to a voice than a pure sine."""
    count = int(sec * SR)
    kernel = np.hanning(51)
    band = np.convolve(rng.normal(0, 1, count), kernel / kernel.sum(), mode="same")
    band *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * np.arange(count) / SR)
    return (amp * band / np.abs(band).max()).astype(np.float32)


def room(sec: float, amp: float = 0.001) -> np.ndarray:
    """A quiet stretch of a real recording — faint noise, never a true zero."""
    return rng.normal(0, amp, int(sec * SR)).astype(np.float32)


# ── 57.5 seconds of speech mixed with silence ─────────────────────────
layout = [("talk", 3.0), ("", 12.0), ("talk", 4.0), ("", 1.5), ("talk", 2.0), ("", 30.0), ("talk", 5.0)]
parts, spoken, cursor = [], [], 0.0
for kind, seconds in layout:
    parts.append(speech(seconds) if kind else room(seconds))
    if kind:
        spoken.append((cursor, cursor + seconds))
    cursor += seconds
audio = np.concatenate(parts)

timeline, note = vad.plan(audio)
stats = timeline.stats()
print(f"\noriginal {stats['original']}s -> {stats['kept']}s to transcribe "
      f"({stats['removed']}s of silence removed across {stats['regions']} regions)")
if note:
    print(f"       {note}")
print()

check("long silence was cut", timeline.trimmed, f"{stats['removed']}s")
check("speech region count is right", stats["regions"] == 4, f"{stats['regions']}")

trimmed = timeline.apply(audio)
check(
    "trimmed waveform == kept regions concatenated",
    np.array_equal(trimmed, np.concatenate([audio[s:e] for s, e in timeline.regions])),
)

# ── The core question: does a trimmed-clock time map to the same sample? ──
worst = 0.0
for index in rng.integers(0, len(trimmed), 20000):
    back = int(round(timeline.to_original(index / SR) * SR))
    worst = max(worst, abs(float(audio[min(back, len(audio) - 1)] - trimmed[index])))
check("20000 random points map back sample-exactly", worst == 0.0, f"max error {worst:.1e}")

probe = [timeline.to_original(t) for t in np.linspace(-1.0, stats["kept"] + 5.0, 5000)]
check("mapping is monotonic (order never flips)",
      all(b >= a - 1e-9 for a, b in zip(probe, probe[1:])))
check("mapped times stay inside the original length",
      probe[0] >= 0.0 and probe[-1] <= stats["original"] + 1e-9)

# ── Did we chop off any speech? ───────────────────────────────────────
kept = np.zeros(len(audio), dtype=bool)
for start, end in timeline.regions:
    kept[start:end] = True
for start_sec, end_sec in spoken:
    covered = float(kept[int(start_sec * SR):int(end_sec * SR)].mean())
    check(f"speech {start_sec:5.1f}-{end_sec:5.1f}s preserved", covered > 0.97, f"{covered * 100:.1f}%")

# ── Spans crossing a cut get split (used for speaker turns) ───────────
# Without the split a speaker would own the silence we removed.
cut = vad.Timeline([(0, 2 * SR), (10 * SR, 14 * SR)], 14 * SR)
check("a span inside one region passes through unchanged", cut.split(1.0, 1.5) == [(1.0, 1.5)], str(cut.split(1.0, 1.5)))

crossing = cut.split(1.5, 3.0)
check("a span crossing a cut gets split", crossing == [(1.5, 2.0), (10.0, 11.0)], str(crossing))
check("splitting preserves total length",
      abs(sum(b - a for a, b in crossing) - 1.5) < 1e-9,
      f"{sum(b - a for a, b in crossing):.3f}s")
check("every split piece lies inside a kept region",
      all(any(r[0] / SR <= a and b <= r[1] / SR for r in cut.regions) for a, b in crossing))
check("splitting the whole span yields every kept region",
      cut.split(0.0, 6.0) == [(0.0, 2.0), (10.0, 14.0)], str(cut.split(0.0, 6.0)))
check("split is a no-op on the identity Timeline", vad.Timeline.identity(0).split(3.0, 9.0) == [(3.0, 9.0)])

# Restoring speaker turns — exactly what diarize._restore_turns does
from app import diarize  # noqa: E402  (imported here because it pulls in torch)

turns = diarize._restore_turns(
    [{"start": 0.5, "end": 3.0, "speaker": "SPEAKER_00"},
     {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"}],
    cut,
)
check("speaker turns do not swallow removed silence", len(turns) == 3, f"{len(turns)}")
check("total speech time is unchanged",
      abs(sum(t["end"] - t["start"] for t in turns) - 3.5) < 1e-9,
      f"{sum(t['end'] - t['start'] for t in turns):.2f}s")
check("speaker labels are preserved",
      [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"],
      str([t["speaker"] for t in turns]))

# ── Restoring segment and word times ──────────────────────────────────
segments = [
    {"start": 2.3, "end": 4.0, "words": [
        {"word": "hello", "start": 2.3, "end": 3.0},
        {"word": "125", "start": None, "end": None},  # a word the aligner could not place
    ]},
]
vad.Timeline([(0, 2 * SR), (10 * SR, 14 * SR)], 14 * SR).restore(segments)
check("segment times restored", abs(segments[0]["start"] - 10.3) < 1e-6 and
      abs(segments[0]["end"] - 12.0) < 1e-6, str(segments[0]["start"]))
check("word times restored", abs(segments[0]["words"][0]["start"] - 10.3) < 1e-6)
check("words without timestamps are left alone", segments[0]["words"][1]["start"] is None)

# ── Cases where nothing should be cut ─────────────────────────────────
identity = vad.Timeline.identity(len(audio))
check("identity Timeline does not copy the waveform", identity.apply(audio) is audio)
check("identity Timeline leaves times untouched", identity.to_original(3.25) == 3.25)
check("identity Timeline reports lengths correctly",
      identity.stats()["kept"] == identity.stats()["original"], str(identity.stats()))
check("pure noise is never cut",
      not vad.plan(rng.normal(0, 0.05, 10 * SR).astype(np.float32))[0].trimmed)
check("pure silence is never cut", not vad.plan(room(10))[0].trimmed)
check("files under a second are never cut", not vad.plan(speech(0.5))[0].trimmed)
check("short silence (0.4s) is kept as part of speech",
      not vad.plan(np.concatenate([speech(1), room(0.4), speech(1)]))[0].trimmed)
check("long silence (3s) is cut",
      vad.plan(np.concatenate([speech(1), room(3.0), speech(1)]))[0].trimmed)

# ── Do trimmed utterances end up butted together? ─────────────────────
# If they do, Whisper reads two separate utterances as one sentence.
original_pad, config.TRIM_PAD_SEC = config.TRIM_PAD_SEC, 0.0  # worst-case setting
try:
    gapped = np.concatenate([speech(1), room(3.0), speech(1)])
    joined = vad.plan(gapped)[0]
    output = joined.apply(gapped)
    level = vad._frame_db(output, int(vad.FRAME_SEC * SR))
    inner = [
        (end - start) * vad.FRAME_SEC
        for start, end in vad._runs(level < level.max() - 25)
        if start > 0 and end < len(level)
    ]
    check("silence remains between utterances even at TRIM_PAD_SEC=0",
          bool(inner) and max(inner) >= 0.09, f"{max(inner or [0]):.2f}s")
finally:
    config.TRIM_PAD_SEC = original_pad


# ── Does silence detection survive low-frequency rumble? ──────────────
# Air conditioning and fans sit below 100Hz. Leave them in and the noise floor
# rises until the "less than 12dB range" guard blocks all cutting.
print("\n── Low-frequency rumble ──")
moment = np.arange(len(audio)) / SR
hum = (0.035 * np.sin(2 * np.pi * 45 * moment)
       + 0.02 * np.sin(2 * np.pi * 90 * moment)).astype(np.float32)
rumbling = (audio + hum).astype(np.float32)

original_hz, config.TRIM_HIGHPASS_HZ = config.TRIM_HIGHPASS_HZ, 0.0
try:
    without = vad.plan(rumbling)[0]
finally:
    config.TRIM_HIGHPASS_HZ = original_hz
with_filter = vad.plan(rumbling)[0]
print(f"       no highpass: {without.stats()['removed']:.1f}s removed / "
      f"{config.TRIM_HIGHPASS_HZ:.0f}Hz applied: {with_filter.stats()['removed']:.1f}s removed")

check("with rumble present, nothing is cut without the highpass", not without.trimmed,
      f"{without.stats()['removed']:.1f}s")
check("the highpass restores normal cutting", with_filter.trimmed,
      f"{with_filter.stats()['removed']:.1f}s")
check("a clean recording is unaffected by the highpass",
      abs(with_filter.stats()["removed"] - stats["removed"]) < 1.5,
      f"{with_filter.stats()['removed']:.1f} vs {stats['removed']:.1f}")
check("the highpass is part of the cache key", "highpass" in vad.params())

# ── ffmpeg filter chains ──────────────────────────────────────────────
print("\n── ffmpeg filter chains ──")
import shutil  # noqa: E402
import subprocess  # noqa: E402

check("'off' is an empty string", audio_io.filter_chain("off") == "")
check("unknown names pass through as a filter string",
      audio_io.filter_chain("highpass=f=200") == "highpass=f=200")
check("case-insensitive", audio_io.filter_chain("VOICE") == audio_io.FILTERS["voice"])

if shutil.which("ffmpeg"):
    for preset, chain in audio_io.FILTERS.items():
        if not chain:
            continue
        done = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
             "-i", "sine=frequency=300:duration=1:sample_rate=16000",
             "-af", chain, "-f", "null", "-"],
            capture_output=True, text=True,
        )
        check(f"the '{preset}' filter actually runs in ffmpeg", done.returncode == 0,
              done.stderr.strip().splitlines()[-1] if done.returncode else chain)
else:
    print("       ffmpeg not installed — skipping the filter execution check")

# ── Chunking ──────────────────────────────────────────────────────────
print("\n── Chunking ──")
pieces = vad.chunks(timeline, 20.0)   # 57.5s in 20s pieces
print("chunks:", [(round(p.span[0] / SR, 2), round(p.span[1] / SR, 2)) for p in pieces])

check("the file splits into several chunks", len(pieces) > 1, f"{len(pieces)}")
check("chunks are ordered and never overlap",
      all(a.span[1] <= b.span[0] for a, b in zip(pieces, pieces[1:])))
check("all chunks together cover exactly the kept regions",
      [r for p in pieces for r in p.regions] == timeline.regions)
check("concatenated chunks equal trimming in one pass",
      np.array_equal(np.concatenate([p.apply(audio) for p in pieces]), timeline.apply(audio)))

# The path that reads only a chunk's range (so 10 hours never loads at once)
windowed = [p.apply(audio[p.span[0]:p.span[1]], p.span[0]) for p in pieces]
check("reading only a chunk's range gives the same result",
      all(np.array_equal(a, p.apply(audio)) for a, p in zip(windowed, pieces)))

# Is a time from inside a chunk already an original-clock time?
for order, piece in enumerate(pieces):
    length = piece.kept / SR
    lo, hi = piece.to_original(0.0), piece.to_original(length)
    inside = piece.span[0] / SR <= lo and hi <= piece.span[1] / SR
    check(f"chunk {order + 1} maps back inside its own range", inside,
          f"{lo:.2f}-{hi:.2f} (range {piece.span[0]/SR:.2f}-{piece.span[1]/SR:.2f})")

check("boundaries land in silence (never mid-speech)",
      all(any(abs(p.span[0] - s) < 2 for s, _ in timeline.regions) for p in pieces[1:]))
check("no split needed means exactly one chunk", len(vad.chunks(timeline, 3600.0)) == 1)
check("CHUNK_SEC=0 disables chunking", len(vad.chunks(timeline, 0.0)) == 1)

# ── Reading a wav by range ────────────────────────────────────────────
# The premise of chunking. Get this wrong and every chunk transcribes the
# wrong audio.
print("\n── Ranged wav reads ──")
import io  # noqa: E402
import tempfile  # noqa: E402
import wave as wave_mod  # noqa: E402


with tempfile.TemporaryDirectory() as folder:
    sample_path = Path(folder) / "sample.wav"
    stored = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave_mod.open(str(sample_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(stored.tobytes())

    whole = audio_io.read_wav(sample_path)
    check("full read returns the right length", len(whole) == len(audio), f"{len(whole)} vs {len(audio)}")
    check("sample count can be read separately", audio_io.sample_count(sample_path) == len(audio))
    # Must equal the stored integers / 32768 (same convention as whisperx.load_audio)
    check("decoding is exact", np.array_equal(whole, stored.astype(np.float32) / 32768.0))

    lo, hi = 7 * SR, 21 * SR
    check("a ranged read equals slicing the full read",
          np.array_equal(audio_io.read_wav(sample_path, lo, hi), whole[lo:hi]))
    check("reading past the end stops at the end",
          len(audio_io.read_wav(sample_path, len(audio) - SR, len(audio) + 99 * SR)) == SR)
    check("out of range returns an empty array", len(audio_io.read_wav(sample_path, len(audio) + 5, None)) == 0)

    # The clip served when a transcript line is clicked
    clip = audio_io.clip_wav(sample_path, 15.0, 19.0)
    with wave_mod.open(io.BytesIO(clip), "rb") as handle:
        check("the clip is a valid wav file",
              handle.getframerate() == SR and handle.getnchannels() == 1)
        check("clip length matches the request", abs(handle.getnframes() / SR - 4.0) < 0.01,
              f"{handle.getnframes() / SR:.2f}s")
        cut = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    check("clip content matches that range of the original",
          np.array_equal(cut, stored[15 * SR:19 * SR]))
    check("clips respect the max length",
          len(audio_io.clip_wav(sample_path, 0.0, 9999.0, max_sec=2.0)) <= 2 * SR * 2 + 100)
    check("a reversed range does not crash", audio_io.clip_wav(sample_path, 20.0, 5.0) is not None)

    bad_path = Path(folder) / "bad.wav"
    with wave_mod.open(str(bad_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 1000)
    try:
        audio_io.read_wav(bad_path)
        check("44.1kHz wav is rejected", False, "no exception was raised")
    except RuntimeError as exc:
        check("44.1kHz wav is rejected", "44100" in str(exc), str(exc)[:60])

# ── Cross-chunk speaker stitching ─────────────────────────────────────
print("\n── Cross-chunk speaker stitching ──")
from app import db, stitch  # noqa: E402

people = [db.normalize(rng.normal(0, 1, 192)) for _ in range(3)]


def voice(person: int, drift: float = 0.04):
    """The same person's embedding in a different chunk — slightly perturbed.

    These are normalized 192-dim vectors, so one component is about
    1/sqrt(192) ~= 0.072. Drift larger than that drowns the signal in noise and
    it stops being the same person. 0.04 lands near cosine 0.87, which is what
    different chunks of one recording actually look like.
    """
    return db.normalize(people[person] + rng.normal(0, drift, 192))


same = float(voice(0) @ voice(0))
other = float(voice(0) @ voice(1))
print(f"       same person cosine {same:.2f} / different person {other:.2f} "
      f"(threshold {config.STITCH_THRESHOLD:.2f})")
check("synthetic data lands on both sides of the threshold",
      other < config.STITCH_THRESHOLD <= same, f"{other:.2f} / {same:.2f}")


# chunk 1: person0=SPEAKER_00, person1=SPEAKER_01
# chunk 2: person1=SPEAKER_00, person0=SPEAKER_01   <- labels swapped
# chunk 3: only person2 appears (a new person)
parts = [
    ([{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
      {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"}],
     {"SPEAKER_00": voice(0), "SPEAKER_01": voice(1)},
     {"SPEAKER_00": 5.0, "SPEAKER_01": 4.0}),
    ([{"start": 20.0, "end": 26.0, "speaker": "SPEAKER_00"},
      {"start": 26.0, "end": 29.0, "speaker": "SPEAKER_01"}],
     {"SPEAKER_00": voice(1), "SPEAKER_01": voice(0)},
     {"SPEAKER_00": 6.0, "SPEAKER_01": 3.0}),
    ([{"start": 40.0, "end": 48.0, "speaker": "SPEAKER_00"}],
     {"SPEAKER_00": voice(2)},
     {"SPEAKER_00": 8.0}),
]
merged_turns, merged_emb, merged_speech, merged_overlaps, notes = stitch.merge(parts)
for note in notes:
    print(f"       {note}")

check("speaker count is right", len(merged_speech) == 3, f"{len(merged_speech)}")
check("a chunk with swapped labels gets corrected",
      merged_turns[0]["speaker"] == merged_turns[3]["speaker"],
      f"{merged_turns[0]['speaker']} vs {merged_turns[3]['speaker']}")
check("different people are not merged",
      merged_turns[0]["speaker"] != merged_turns[1]["speaker"])
check("speech time is summed",
      abs(sum(merged_speech.values()) - 26.0) < 1e-9, f"{sum(merged_speech.values())}s")
check("speech time groups correctly per person",
      sorted(round(v, 1) for v in merged_speech.values()) == [8.0, 8.0, 10.0],
      str(sorted(round(v, 1) for v in merged_speech.values())))
check("turns come out sorted by time",
      all(a["start"] <= b["start"] for a, b in zip(merged_turns, merged_turns[1:])))
check("merged embeddings are normalized",
      all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-5 for v in merged_emb.values()))

single = stitch.merge([parts[0]])
check("a single chunk keeps its labels",
      [t["speaker"] for t in single[0]] == ["SPEAKER_00", "SPEAKER_01"], str(single[0]))

# pyannote returns NaN for speakers that only appear in overlap, and diarize
# discards those — leaving a label with speech time but no known voice. One of
# those sitting in the speaker list used to crash the next comparison
# (max() on an empty sequence).
voiceless = [
    ([{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}],
     {},                                   # a chunk with no embeddings at all
     {"SPEAKER_00": 4.0}),
    ([{"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00"},
      {"start": 15.0, "end": 18.0, "speaker": "SPEAKER_01"}],
     {"SPEAKER_00": voice(0)},             # a chunk where only one has an embedding
     {"SPEAKER_00": 5.0, "SPEAKER_01": 3.0}),
    ([{"start": 20.0, "end": 26.0, "speaker": "SPEAKER_00"}],
     {"SPEAKER_00": voice(0)},             # the same person as above
     {"SPEAKER_00": 6.0}),
]
try:
    v_turns, v_emb, v_speech, _, _ = stitch.merge(voiceless)
    check("a voiceless speaker in the mix does not crash it", True)
    # The two voiceless ones stay separate (4.0, 3.0); only the voice(0) pair merges (5+6=11)
    check("voiceless speakers merge with nobody",
          sorted(round(v, 1) for v in v_speech.values()) == [3.0, 4.0, 11.0],
          f"{len(v_speech)} speakers {sorted(round(v, 1) for v in v_speech.values())}")
    check("the same person with a voice merges across chunks",
          v_turns[1]["speaker"] == v_turns[3]["speaker"],
          f"{v_turns[1]['speaker']} vs {v_turns[3]['speaker']}")
    check("voiceless speakers are absent from the result embeddings", len(v_emb) == 1, str(sorted(v_emb)))
    check("all speech time is preserved", abs(sum(v_speech.values()) - 18.0) < 1e-9,
          f"{sum(v_speech.values())}s")
except ValueError as exc:
    check("a voiceless speaker in the mix does not crash it", False, str(exc))

# A label that only appears in turns must still get a new name.
# Leaving the original label quietly fuses two different people into one.
stray = stitch.merge([
    ([{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
     {"SPEAKER_00": voice(0)}, {"SPEAKER_00": 5.0}),
    ([{"start": 9.0, "end": 12.0, "speaker": "SPEAKER_00"}],   # label missing from speech_sec
     {}, {}),
])
check("a label only seen in turns stays a separate person",
      stray[0][0]["speaker"] != stray[0][1]["speaker"],
      f"{stray[0][0]['speaker']} vs {stray[0][1]['speaker']}")

# ── Re-merging over-split speakers ────────────────────────────────────
print("\n── Re-merging over-split speakers ──")

# pyannote split person0 into SPEAKER_00 and SPEAKER_02. person1 is someone else.
split_turns = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"},
    {"start": 9.0, "end": 14.0, "speaker": "SPEAKER_02"},
]
split_emb = {"SPEAKER_00": voice(0), "SPEAKER_01": voice(1), "SPEAKER_02": voice(0)}
split_speech = {"SPEAKER_00": 5.0, "SPEAKER_01": 4.0, "SPEAKER_02": 5.0}

c_turns, c_emb, c_speech, c_notes = stitch.collapse(
    split_turns, split_emb, dict(split_speech), []
)
for note in c_notes:
    print(f"       {note}")
check("an over-split person is merged back", len(c_speech) == 2, f"{len(c_speech)}")
check("merged turns share one label",
      c_turns[0]["speaker"] == c_turns[2]["speaker"],
      f"{c_turns[0]['speaker']} vs {c_turns[2]['speaker']}")
check("the other person is left alone", c_turns[1]["speaker"] != c_turns[0]["speaker"])
check("speech time is summed", abs(max(c_speech.values()) - 10.0) < 1e-9, str(c_speech))
check("the total is preserved", abs(sum(c_speech.values()) - 14.0) < 1e-9, str(sum(c_speech.values())))

# Talking at the same time blocks the merge no matter how similar — they cannot be one person
o_turns, o_emb, o_speech, o_notes = stitch.collapse(
    split_turns, split_emb, dict(split_speech), [["SPEAKER_00", "SPEAKER_02"]]
)
check("a pair who talked at once is never merged, however similar", len(o_speech) == 3 and not o_notes,
      f"{len(o_speech)}")

# Disproofs must be inherited on merge (merge A and B, and whoever overlapped B is also not A)
chain_turns = [{"start": float(i), "end": i + 1.0, "speaker": f"SPEAKER_0{i}"} for i in range(3)]
chain = stitch.collapse(
    chain_turns,
    {"SPEAKER_00": voice(0), "SPEAKER_01": voice(0), "SPEAKER_02": voice(0)},
    {"SPEAKER_00": 5.0, "SPEAKER_01": 5.0, "SPEAKER_02": 5.0},
    [["SPEAKER_01", "SPEAKER_02"]],   # 01 and 02 talked at the same time
)
check("disproofs are inherited through a merge", len(chain[2]) == 2, f"{len(chain[2])} speakers {chain[2]}")

original_merge, config.MERGE_THRESHOLD = config.MERGE_THRESHOLD, 0.0
try:
    off = stitch.collapse(split_turns, split_emb, dict(split_speech), [])
    check("MERGE_THRESHOLD=0 merges nothing", len(off[2]) == 3 and not off[3])
finally:
    config.MERGE_THRESHOLD = original_merge

# With no voice there is no basis for merging, so they stay as they are
q_turns, q_emb, q_speech, _ = stitch.collapse(
    split_turns + [{"start": 20.0, "end": 23.0, "speaker": "SPEAKER_09"}],
    split_emb, {**split_speech, "SPEAKER_09": 3.0}, [],
)
check("speakers with no voice are left untouched", "SPEAKER_09" in q_speech, str(sorted(q_speech)))

# Does giving the same name join the lines? (the manual fix path)
from app import render  # noqa: E402

lines = render.merge_lines(
    [{"speaker": "SPEAKER_00", "text": "first half", "start": 0.0, "end": 1.0},
     {"speaker": "SPEAKER_02", "text": "second half", "start": 1.0, "end": 2.0}],
    {"SPEAKER_00": "Alex Kim", "SPEAKER_02": "Alex Kim"},
)
check("speakers given the same name join into one line",
      len(lines) == 1 and lines[0]["text"] == "first half second half", str(lines))

# ── Hallucination filter ──────────────────────────────────────────────
print("\n── Hallucination filter ──")
from app import cleanup  # noqa: E402

def seg(text, start=0.0, end=5.0, score=None):
    """A segment with or without an alignment score. score=None means no score info."""
    words = [] if score is None else [{"word": text, "start": start, "end": end, "score": score}]
    return {"text": text, "start": start, "end": end, "words": words}


def verdict(item):
    result = cleanup.inspect(item)
    return (result[0] if result else "keep"), (result[1] if result else "")


# Contentless repetition — deleted on the text alone (nothing lost even if real)
# The Korean strings below are deliberate: they exercise the Korean patterns in
# cleanup.BOILERPLATE, which have to stay Korean to match Korean transcripts.
for text in ["아 아 아 아 아 아 아 아", "아아아아아아아", "네 네 네 네 네 네 네"]:
    action, why = verdict(seg(text))
    check(f"repetition is dropped outright: {text[:20]}", action == "drop", why)

# Boilerplate — not deleted when the audio matches the words (high score)
for text in ["MBC 뉴스 김성현이었습니다", "시청해주셔서 감사합니다",
             "구독과 좋아요 부탁드립니다", "Thanks for watching!"]:
    action, why = verdict(seg(text, score=0.9))
    check(f"boilerplate is kept when the audio matches: {text[:22]}", action == "suspect", why)
    action, why = verdict(seg(text, score=0.05))
    check(f"boilerplate + low score is dropped: {text[:22]}", action == "drop", why)

# The classic hallucination: one sentence over 30s of silence — caught without any score
action, why = verdict(seg("시청해주셔서 감사합니다", 0.0, 30.0))
check("one sentence over 30s is dropped even with no score", action == "drop", why)

# Real speech is left alone
for text in ["감사합니다", "네 이거 좋아요", "다음 시간에 뵙겠습니다",
             "알림 설정 좀 바꿔주세요", "그래서 제가 어제 말씀드린 대로 진행하겠습니다",
             "아 그건 제가 확인해 보겠습니다", "네 네 네 알겠습니다 그러면 그렇게 진행할게요",
             "MBC 뉴스에서 그 얘기 나왔던 거 기억나세요", "구독자 수가 지난달보다 늘었어요"]:
    action, why = verdict(seg(text, score=0.85))
    check(f"real speech is kept: {text[:24]}", action == "keep", why)

# When transcribing the news, an anchor's sign-off is a real utterance
action, why = verdict(seg("MBC 뉴스 김성현이었습니다", 0.0, 2.4, score=0.88))
check("a news sign-off is not deleted when transcribing news", action == "suspect", why)

kept_segs, dropped_segs, suspect_segs = cleanup.clean([
    seg("uh uh uh uh uh uh uh uh"),                    # drop
    seg("시청해주셔서 감사합니다", score=0.02),          # drop (Korean boilerplate)
    seg("MBC 뉴스 김성현이었습니다", score=0.9),        # suspect (kept)
    seg("We'll proceed as I explained yesterday", score=0.9),  # keep
])
check("clean() splits into three buckets",
      (len(kept_segs), len(dropped_segs), len(suspect_segs)) == (2, 2, 1),
      f"kept {len(kept_segs)} / dropped {len(dropped_segs)} / suspect {len(suspect_segs)}")
check("suspect segments stay in the output",
      any("MBC" in (s["text"] or "") for s in kept_segs))
check("dropped and suspect entries carry a reason",
      all(item["reason"] for item in dropped_segs + suspect_segs))
check("the warning mentions both what was dropped and what was kept",
      len(cleanup.summary(dropped_segs, suspect_segs)) == 2)

original_suspect, config.DROP_SUSPECT = config.DROP_SUSPECT, True
try:
    _, hard_dropped, hard_suspect = cleanup.clean([seg("MBC 뉴스 김성현이었습니다", score=0.9)])
    check("DROP_SUSPECT=true drops suspect segments too",
          len(hard_dropped) == 1 and not hard_suspect)
finally:
    config.DROP_SUSPECT = original_suspect

original_drop, config.DROP_HALLUCINATION = config.DROP_HALLUCINATION, False
try:
    check("DROP_HALLUCINATION=false drops nothing",
          cleanup.clean([seg("아 아 아 아 아 아 아")])[1] == [])
finally:
    config.DROP_HALLUCINATION = original_drop

print()
if failures:
    print(f"{len(failures)} failure(s): {', '.join(failures)}")
    raise SystemExit(1)
print("All checks passed.")
