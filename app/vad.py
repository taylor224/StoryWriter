"""Cut silence out before transcription, then put the timestamps back.

Whisper likes to invent words during long silences, and pushing silence through
the models costs the same as pushing speech. Trimming it first reduces both.

The catch is that everything after a cut slides earlier by however much was
removed. So we build a Timeline alongside the trimmed waveform and, once
transcription and alignment are done, map every timestamp back to the original
clock. Diarization uses the untouched wav, so nothing downstream needs to know
a cut ever happened.

Silence detection is plain frame-RMS energy. No extra model to download, and
since the noise floor differs per file we place the threshold between that
file's own noise floor and its speech level rather than at an absolute dB.
"""

import bisect

import numpy as np

from . import audio, config

SAMPLE_RATE = audio.SAMPLE_RATE
FRAME_SEC = 0.02  # 20ms — short enough not to swallow a consonant, long enough to be steady
EPS = 1e-10


class Timeline:
    """Maps a time on the trimmed waveform back to a time on the original.

    regions holds the [start, end) sample ranges that were *kept* from the
    original. The trimmed waveform is those ranges concatenated in order, so any
    trimmed-clock time t falls inside exactly one region — there are no gaps.

    Empty regions means "nothing was cut" and every transform is the identity.
    That way callers never have to branch on whether trimming is on.
    """

    def __init__(self, regions, total: int, sample_rate: int = SAMPLE_RATE):
        self.regions: list[tuple[int, int]] = [(int(s), int(e)) for s, e in regions]
        self.total = int(total)
        self.sample_rate = int(sample_rate)

        # offsets[i] = where region i starts on the trimmed waveform
        self.offsets: list[int] = []
        cursor = 0
        for start, end in self.regions:
            self.offsets.append(cursor)
            cursor += end - start
        self.kept = cursor

    @classmethod
    def identity(cls, total: int, sample_rate: int = SAMPLE_RATE) -> "Timeline":
        """A table that cuts nothing. total is only used for reporting."""
        return cls([], total, sample_rate)

    @property
    def trimmed(self) -> bool:
        return bool(self.regions) and self.kept < self.total

    @property
    def span(self) -> tuple[int, int]:
        """The [start, end) range this table covers in the original. Used to read a chunk."""
        if not self.regions:
            return (0, self.total)
        return (self.regions[0][0], self.regions[-1][1])

    def apply(self, samples: np.ndarray, origin: int = 0) -> np.ndarray:
        """The kept regions concatenated.

        If samples is not the whole original but a window read from sample
        `origin` onward, pass origin (needed to avoid loading a 10-hour file
        into memory all at once).
        """
        if not self.regions:
            return samples
        # Tolerate a sample or two of drift between decoders instead of dying
        limit = origin + int(samples.shape[0])
        pieces = [
            samples[max(start, origin) - origin : min(end, limit) - origin]
            for start, end in self.regions
            if start < limit and end > origin
        ]
        if not pieces:
            return samples
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def to_original(self, seconds: float) -> float:
        """Trimmed-clock time -> original-clock time. Monotonically increasing."""
        if not self.trimmed:
            return float(seconds)
        return self._at(seconds * self.sample_rate) / self.sample_rate

    def _at(self, position: float) -> float:
        """Trimmed-clock sample position -> original sample position."""
        if position <= 0:
            return float(self.regions[0][0])
        index = bisect.bisect_right(self.offsets, position) - 1
        index = max(0, min(index, len(self.regions) - 1))
        start, end = self.regions[index]
        # min(..., end) catches times past the last region (alignment slop, etc.)
        return min(start + (position - self.offsets[index]), float(end))

    def split(self, start: float, end: float) -> list[tuple[float, float]]:
        """Break one trimmed-clock span into the original spans it covers.

        A span that crosses a cut is not contiguous in the original. Mapping
        only its endpoints would make it swallow the silence we removed — for a
        speaker turn that means the speaker owns silence they never spoke in. So
        we break it at every kept-region boundary.

        Total length is preserved: splitting never changes total speech time.
        """
        if not self.trimmed:
            return [(float(start), float(end))]

        low = max(0.0, start * self.sample_rate)
        high = min(float(self.kept), end * self.sample_rate)
        if high <= low:  # zero length or out of range — collapse to a point
            point = self._at(low) / self.sample_rate
            return [(point, point)]

        spans: list[tuple[float, float]] = []
        index = max(0, bisect.bisect_right(self.offsets, low) - 1)
        while index < len(self.regions) and self.offsets[index] < high:
            origin, finish = self.regions[index]
            piece_low = max(low, float(self.offsets[index]))
            piece_high = min(high, float(self.offsets[index] + finish - origin))
            if piece_high > piece_low:
                base = origin - self.offsets[index]
                spans.append(
                    ((base + piece_low) / self.sample_rate, (base + piece_high) / self.sample_rate)
                )
            index += 1
        return spans

    def restore(self, segments: list[dict]) -> list[dict]:
        """Map segment and word timestamps back to the original clock, in place."""
        if not self.trimmed:
            return segments
        for seg in segments:
            _shift(seg, self.to_original)
            for word in seg.get("words") or []:
                _shift(word, self.to_original)
        return segments

    def stats(self) -> dict:
        kept = self.kept if self.trimmed else self.total
        return {
            "enabled": self.trimmed,
            "original": round(self.total / self.sample_rate, 2),
            "kept": round(kept / self.sample_rate, 2),
            "removed": round((self.total - kept) / self.sample_rate, 2),
            "regions": len(self.regions),
        }


def _shift(item: dict, convert) -> None:
    # The aligner leaves start/end empty on words it could not pin down (numerals, symbols)
    for key in ("start", "end"):
        value = item.get(key)
        if value is not None:
            item[key] = convert(float(value))


def chunks(timeline: Timeline, max_span_sec: float) -> list[Timeline]:
    """Split a long recording into chunk Timelines. Returns [timeline] if no split is needed.

    Each chunk carries original coordinates, so timestamps coming out of a chunk
    are original-clock after one restore — no chunk index to thread around and
    no offsets to add.

    Boundaries land in the middle of silence wherever possible. Cutting mid-word
    gets that word half-recognized on both sides and makes diarization wobble at
    the seam. With silence trimming off we do not know where the silence is, so
    it falls back to even splits.
    """
    limit = int(max_span_sec * timeline.sample_rate)
    if limit <= 0 or timeline.total <= limit:
        return [timeline]

    if timeline.regions:
        groups: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        for region in timeline.regions:
            # Would adding this region make the chunk too long (measured from its start)?
            if current and region[1] - current[0][0] > limit:
                groups.append(current)
                current = []
            current.append(region)
        if current:
            groups.append(current)
    else:
        groups = [
            [(start, min(start + limit, timeline.total))]
            for start in range(0, timeline.total, limit)
        ]

    return [Timeline(group, timeline.total, timeline.sample_rate) for group in groups]


def params() -> dict:
    """Settings that go into the cache key. Change any of them and transcription reruns."""
    return {
        "min_silence": config.TRIM_MIN_SILENCE_SEC,
        "pad": config.TRIM_PAD_SEC,
        "sensitivity": config.TRIM_SENSITIVITY,
        "min_speech": config.TRIM_MIN_SPEECH_SEC,
        "min_dynamic_db": config.TRIM_MIN_DYNAMIC_DB,
        "highpass": config.TRIM_HIGHPASS_HZ,
    }


# ── Silence detection ─────────────────────────────────────────────────
def _power_db(block: np.ndarray, frame: int) -> np.ndarray:
    """Per-frame RMS in dBFS for a waveform folded into (n, frame).

    samples**2 would allocate another array the size of the input — 230MB for an
    hour — so einsum extracts just the sums of squares and skips that copy.
    """
    power = np.einsum("ij,ij->i", block, block) / frame
    return 10.0 * np.log10(power + EPS)


def _frame_db(samples: np.ndarray, frame: int, sample_rate: int = 0) -> np.ndarray:
    """Per-frame RMS in dBFS, with low frequencies removed first.

    Air conditioning, projector fans and desk vibration sit almost entirely below
    100Hz. The human voice bottoms out around 85Hz, so discarding what is under
    that costs no speech.

    Leave it in and the noise floor rises across the board, narrowing the gap
    between silence and speech until it trips the "less than 12dB" guard and
    nothing gets cut at all. That is usually why trimming does nothing on a
    quiet office recording.

    The filtered waveform is used here and thrown away. What goes to
    transcription and diarization is the untouched original — Whisper was
    trained on noisy audio and cleaning it up tends to make things worse.
    """
    count = samples.shape[0] // frame
    if count == 0:
        return np.empty(0, dtype=np.float32)
    usable = count * frame
    if config.TRIM_HIGHPASS_HZ <= 0 or not sample_rate:
        return _power_db(samples[:usable].reshape(count, frame), frame)

    from scipy.signal import butter, sosfilt, sosfilt_zi

    sos = butter(2, config.TRIM_HIGHPASS_HZ / (sample_rate / 2), btype="highpass", output="sos")
    state = sosfilt_zi(sos) * float(samples[0])
    # Stream it in blocks. Filtering 10 hours in one go would cost 2.3GB just for the copy.
    step = frame * 20_000
    parts = []
    for start in range(0, usable, step):
        block = samples[start : min(start + step, usable)]
        filtered, state = sosfilt(sos, block, zi=state)
        parts.append(_power_db(filtered.reshape(-1, frame), frame))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """The [start, end) ranges where mask stays True."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _merge(regions: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Join two regions when the distance between them is less than gap."""
    merged: list[list[int]] = [list(regions[0])]
    for start, end in regions[1:]:
        if start - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def plan(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[Timeline, str | None]:
    """Decide what to keep.

    Returns (Timeline, reason it was skipped or None). When the evidence for
    cutting is weak it returns the identity Timeline — leaving the original
    alone beats chopping off someone's words.
    """
    total = int(samples.shape[0])
    identity = Timeline.identity(total, sample_rate)
    frame = max(1, int(round(FRAME_SEC * sample_rate)))
    if total < sample_rate:  # under a second, nothing to cut
        return identity, None

    level = _frame_db(samples, frame, sample_rate)

    # Every file has a different noise floor. Cutting at an absolute dB would
    # turn a quiet recording entirely into "silence" and a loud one into nothing
    # at all. So measure this file's own quiet side and speaking side first, then
    # put the line between them.
    floor_db = float(np.percentile(level, 10))
    speech_db = float(np.percentile(level, 95))
    span = speech_db - floor_db
    if span < config.TRIM_MIN_DYNAMIC_DB:
        return identity, (
            f"Loudness barely changes from start to end ({span:.1f}dB range), so silence "
            "removal was skipped. Either the recording is noisy or nobody ever pauses."
        )

    threshold = floor_db + max(6.0, span * config.TRIM_SENSITIVITY)
    threshold = min(threshold, speech_db - 6.0)

    speech = _runs(level > threshold)
    if not speech:
        return identity, "No speech regions found, so silence removal was skipped."

    min_speech = int(round(config.TRIM_MIN_SPEECH_SEC * sample_rate))
    min_silence = int(round(config.TRIM_MIN_SILENCE_SEC * sample_rate))
    # pad widens each speech region, and it is also exactly the silence left
    # between two utterances after concatenation. At 0 they butt together and
    # Whisper reads two sentences as one, hence the floor.
    pad = max(int(round(config.TRIM_PAD_SEC * sample_rate)), int(round(0.05 * sample_rate)))

    regions = [(s * frame, min(e * frame, total)) for s, e in speech]
    # Holding on to coughs and mouse clicks would keep the silence around them
    regions = [r for r in regions if r[1] - r[0] >= min_speech]
    if not regions:
        return identity, "Speech regions were too short, so silence removal was skipped."

    regions = _merge(regions, min_silence)  # a short breath is part of speech, not silence
    regions = [(max(0, s - pad), min(total, e + pad)) for s, e in regions]
    regions = _merge(regions, 1)  # tidy up whatever pad made overlap

    timeline = Timeline(regions, total, sample_rate)
    if timeline.kept < sample_rate:
        return identity, "Less than a second was judged to be speech, so silence removal was skipped."
    if timeline.kept >= total * 0.98:
        return identity, None  # under 2% is not worth the copy
    return timeline, None
