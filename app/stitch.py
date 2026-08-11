"""Turn per-chunk speaker labels into one set for the whole file.

When a long file is processed in chunks, pyannote numbers speakers from
SPEAKER_00 inside each chunk. Nothing guarantees that chunk 2's SPEAKER_00 is
the same person as chunk 1's. So we match them up by cosine similarity of their
voice embeddings.

Same principle as matching.py, different target. matching compares against
people enrolled in the database; this compares chunks of the same recording
against each other. Same session and same microphone make that an easier
problem, so the threshold can sit a little higher.
"""

import numpy as np

from . import config, db


class _Speaker:
    """One speaker at whole-file scope. Collects one embedding per chunk."""

    def __init__(self, label: str):
        self.label = label
        self.vectors: list[np.ndarray] = []
        self.speech = 0.0

    def add(self, vector: np.ndarray, speech: float) -> None:
        self.vectors.append(vector)
        self.speech += speech

    @property
    def centroid(self) -> np.ndarray:
        return db.normalize(np.mean(self.vectors, axis=0))


def _similarity(vector: np.ndarray, speaker: _Speaker) -> float:
    """The mean blurs as chunks accumulate, so also take the best individual match.

    A speaker with no vectors cannot be compared at all. When pyannote returns
    NaN for a speaker that only ever appears in overlap, diarize discards the
    embedding but the speech time remains — so voiceless speakers end up in the
    list. Return -1 so they never clear any threshold.
    """
    if not speaker.vectors:
        return -1.0
    best = max(float(vector @ other) for other in speaker.vectors)
    return max(best, float(vector @ speaker.centroid))


def merge(parts: list) -> tuple:
    """Combine per-chunk diarization results into one whole-file result.

    parts is a list of (turns, embeddings, speech_sec[, overlaps]).
    Returns: (turns, embeddings, speech_sec, overlaps, notes)
    turns are already on the original clock, so only the labels get swapped.
    """
    speakers: list[_Speaker] = []
    turns: list[dict] = []
    notes: list[str] = []

    overlaps: list[list[str]] = []
    for index, part in enumerate(parts):
        part_turns, part_embeddings, part_speech = part[0], part[1], part[2]
        part_overlaps = part[3] if len(part) > 3 else []
        # Every label that appears in a turn must get mapped, including ones that
        # show up nowhere else. Leaving a label untranslated would make chunk 2's
        # SPEAKER_00 collide with the global SPEAKER_00 and quietly fuse two
        # different people into one.
        appearing = {turn["speaker"] for turn in part_turns}
        mapping = _assign(part_embeddings, part_speech, speakers, index, notes, appearing)
        for turn in part_turns:
            turns.append({**turn, "speaker": mapping[turn["speaker"]]})
        # Carry the simultaneous-speech evidence over to global labels too (collapse uses it)
        for one, two in part_overlaps:
            if one in mapping and two in mapping and mapping[one] != mapping[two]:
                overlaps.append([mapping[one], mapping[two]])

    turns.sort(key=lambda t: (t["start"], t["end"]))
    embeddings = {s.label: s.centroid for s in speakers if s.vectors}
    speech_sec = {s.label: s.speech for s in speakers}
    return turns, embeddings, speech_sec, overlaps, notes


def _assign(
    part_embeddings: dict[str, np.ndarray],
    part_speech: dict[str, float],
    speakers: list[_Speaker],
    index: int,
    notes: list[str],
    appearing: set[str] | None = None,
) -> dict[str, str]:
    """Map one chunk's labels onto global labels. {'SPEAKER_00': 'SPEAKER_02'}"""
    labels = sorted(set(part_speech) | set(part_embeddings) | set(appearing or ()))
    mapping: dict[str, str] = {}
    if not labels:
        return mapping

    # A label with no embedding cannot be compared to anything, so it has to
    # become a new speaker. (pyannote returns NaN for speakers that only appear
    # in overlap, and diarize throws those away.)
    known = [label for label in labels if part_embeddings.get(label) is not None]
    unknown = [label for label in labels if label not in set(known)]

    if speakers and known:
        vectors = {label: db.normalize(part_embeddings[label]) for label in known}
        scores = np.array(
            [[_similarity(vectors[label], speaker) for speaker in speakers] for label in known],
            dtype=np.float64,
        )
        # Two labels in one chunk must not fuse into the same person, hence 1:1
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-scores)
        taken = {
            known[row]: (speakers[col], float(scores[row, col]))
            for row, col in zip(rows, cols)
            if scores[row, col] >= config.STITCH_THRESHOLD
        }
    else:
        vectors = {label: db.normalize(part_embeddings[label]) for label in known}
        taken = {}

    for label in known:
        hit = taken.get(label)
        if hit is not None:
            speaker, score = hit
            speaker.add(vectors[label], part_speech.get(label, 0.0))
            mapping[label] = speaker.label
            notes.append(
                f"chunk {index + 1}: {label} looks like the same person as "
                f"{speaker.label} (similarity {score:.2f})"
            )
            continue
        speaker = _Speaker(f"SPEAKER_{len(speakers):02d}")
        speaker.add(vectors[label], part_speech.get(label, 0.0))
        speakers.append(speaker)
        mapping[label] = speaker.label

    for label in unknown:
        speaker = _Speaker(f"SPEAKER_{len(speakers):02d}")
        speaker.speech = part_speech.get(label, 0.0)
        speakers.append(speaker)
        mapping[label] = speaker.label

    return mapping


def collapse(
    turns: list[dict],
    embeddings: dict[str, np.ndarray],
    speech_sec: dict[str, float],
    overlaps: list[list[str]],
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, float], list[str]]:
    """Fuse speakers that look like the same person.

    pyannote splits one person into several speakers often enough: their tone
    shifts, they move away from the mic, or the file was chunked and stitching
    missed the link. This compares the final speakers once more and merges them.

    Unlike merge() there is no 1:1 constraint here. That is what catches splits
    pyannote made *within* one chunk as well as links stitching missed because
    of its 1:1 assignment. It applies to files that were never chunked too.

    Overruling pyannote needs a safety net, but raising the threshold is not it
    — that would make pairs stitching already missed unmergeable forever.
    Instead:

      1. A pair that ever spoke at the same time is never merged. If they talked
         over each other they cannot be one person — the only hard disproof there is.
      2. Disproofs are inherited on merge. Fold A into B and anyone who talked
         over B is also not A.

    Merging uses average linkage (centroid to centroid). Maximum linkage would
    chain A-B-C together when A~B and B~C are close but A~C is not.

    Returns: (turns, embeddings, speech_sec, what was merged)
    """
    labels = sorted(embeddings)
    if config.MERGE_THRESHOLD <= 0 or len(labels) < 2:
        return turns, embeddings, speech_sec, []

    groups: dict[str, list[str]] = {label: [label] for label in labels}
    vectors: dict[str, np.ndarray] = {label: db.normalize(embeddings[label]) for label in labels}
    speech_sec = dict(speech_sec)  # do not mutate the caller's dict
    forbidden: dict[str, set[str]] = {label: set() for label in labels}
    for one, two in overlaps:
        if one in forbidden and two in forbidden:
            forbidden[one].add(two)
            forbidden[two].add(one)

    notes: list[str] = []
    while True:
        best: tuple[float, str, str] | None = None
        alive = sorted(groups)
        for i, one in enumerate(alive):
            for two in alive[i + 1:]:
                if two in forbidden[one]:
                    continue
                score = float(vectors[one] @ vectors[two])
                if score >= config.MERGE_THRESHOLD and (best is None or score > best[0]):
                    best = (score, one, two)
        if best is None:
            break

        score, keeper, absorbed = best
        notes.append(
            f"merged {absorbed} into {keeper} as the same person (similarity {score:.2f})"
        )
        # Weight by speech time so the side that actually talked dominates the vector
        weights = (
            max(speech_sec.get(keeper, 0.0), 1e-6),
            max(speech_sec.get(absorbed, 0.0), 1e-6),
        )
        vectors[keeper] = db.normalize(
            vectors[keeper] * weights[0] + vectors[absorbed] * weights[1]
        )
        groups[keeper].extend(groups.pop(absorbed))
        speech_sec[keeper] = speech_sec.get(keeper, 0.0) + speech_sec.pop(absorbed, 0.0)
        # Inherit the disproofs: whoever talked over `absorbed` is also not `keeper`
        forbidden[keeper] |= forbidden.pop(absorbed)
        for others in forbidden.values():
            if absorbed in others:
                others.discard(absorbed)
                others.add(keeper)
        forbidden[keeper].discard(keeper)
        vectors.pop(absorbed)

    if not notes:
        return turns, embeddings, speech_sec, []

    # speech_sec was already combined above (absorbed labels are gone, their time
    # moved over). Voiceless speakers are still there — there is no basis for
    # merging them, so they stay untouched.
    rename = {member: keeper for keeper, members in groups.items() for member in members}
    merged_turns = [
        {**turn, "speaker": rename.get(turn["speaker"], turn["speaker"])} for turn in turns
    ]
    return merged_turns, {label: vectors[label] for label in groups}, speech_sec, notes


def collapse_summary(notes: list[str], speakers: int) -> str:
    return (
        f"Merged {len(notes)} pair(s) of speakers that looked like the same person, "
        f"leaving {speakers}. If two different people were merged, raise "
        f"MERGE_THRESHOLD in .env (currently {config.MERGE_THRESHOLD:.2f}). "
        "If one person is still split across several, lower it."
    )


def summary(count: int, speakers: int) -> str:
    return (
        f"Processed in {count} chunks and stitched the speakers back together by voice "
        f"({speakers} in the end). If one person came out as several, lower "
        f"STITCH_THRESHOLD in .env (currently {config.STITCH_THRESHOLD:.2f})."
    )
