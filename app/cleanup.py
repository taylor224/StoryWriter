"""Filter out what Whisper made up.

In stretches that are near-silent or pure noise, Whisper emits sentences that
were common in its training data: broadcast sign-offs, subtitle credits, the
same syllable repeated forever. Silence removal (vad) gets rid of many such
stretches, but noisy ones are never judged silent, so we screen once more on the
text side.

The rule: **never delete based on the text alone.** "This is Kim Sung-hyun, MBC
News" is a real utterance if you are transcribing the news, and "see you next
time" ends half the meetings ever recorded. Looking like boilerplate is not
evidence of hallucination.

So there are two verdicts:

  drop    : repetition ("uh uh uh uh uh") — no content, nothing lost even if real.
            boilerplate + evidence — when something else says the audio does not
            back the text up.
  suspect : everything else questionable — kept in the output, flagged for a human.

Evidence comes in two forms: the aligner's word confidence is on the floor, or
there are absurdly few characters for the duration (filling 30 seconds of
silence with one sentence is the classic hallucination).

Dropped or flagged, everything is recorded with a reason
(payload['dropped'], payload['suspect']).
"""

import re
from typing import Any

from . import config

# Sentences Whisper likes to emit over silence. Matching this list is NOT enough
# to delete something — _evidence has to agree. Every one of these is a real
# utterance in some recording (news transcripts, a YouTuber transcribing their
# own video, the end of a lecture).
#
# The Korean patterns stay in Korean on purpose: they have to match Korean
# transcripts. Add whatever phrases keep showing up in your own recordings.
BOILERPLATE = [
    # Korean broadcast / YouTube sign-offs
    r"(MBC|KBS|SBS|YTN|JTBC|MBN|채널\s*A|TV\s*조선)\s*뉴스\s*\S*\s*(입니다|이었습니다|였습니다|드립니다)",
    r"시청(해\s*주)?.{0,6}(감사|고맙)",          # "thanks for watching"
    r"구독\s*(과|와|,|그리고)?\s*좋아요",         # "subscribe and like"
    r"좋아요\s*(와|과|,|그리고)?\s*구독",
    r"(한글\s*)?자막\s*(제공|제작)",              # "subtitles provided by"
    r"(이|본)\s*(영상|방송)은.{0,20}(제작|후원|지원|협찬)",  # "this video was sponsored by"
    # English subtitle credits
    r"thanks?\s+for\s+watching",
    r"subtitles?\s+(by|provided)",
    r"amara\.org",
    r"please\s+subscribe",
    r"^\s*you\s*$",
    # Japanese subtitle credit (shows up in Korean recordings too)
    r"ご視聴ありがとう",
]
# Deliberately left out:
#   bare "thank you" / "like" / "subscribe" — the most common things said in a meeting.
#   "notification settings"                 — normal talk in a meeting about an app.
#   "see you next time"                     — ends most meetings and lectures.
# Whisper does hallucinate all of these, but they are far too likely to be real.

_PATTERNS = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE]

# Repetition with no spaces — "aaaaaa", "hahahahaha"
_GLUED = re.compile(r"(.{1,3}?)\1{4,}")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def _repetition_ratio(text: str) -> float:
    """Share of the text taken by its most frequent token. Repetition loops approach 1.0."""
    tokens = _tokens(text)
    if len(tokens) < 4:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return max(counts.values()) / len(tokens)


def _boilerplate_ratio(text: str) -> float:
    """How much of the segment length is covered by boilerplate."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    covered = 0
    for pattern in _PATTERNS:
        for match in pattern.finditer(stripped):
            covered += match.end() - match.start()
    return min(1.0, covered / len(stripped))


def _mean_score(seg: dict[str, Any]) -> float | None:
    """Mean word confidence from the aligner. None when there are no scores.

    When the audio does not match the text — i.e. the text was invented — this
    drops through the floor.
    """
    scores = [
        float(word["score"])
        for word in (seg.get("words") or [])
        if word.get("score") is not None
    ]
    return sum(scores) / len(scores) if scores else None


def _evidence(seg: dict[str, Any], text: str) -> str | None:
    """Evidence that the audio does not back up the text. None if there is none.

    The text alone cannot tell you it is a hallucination — there is always some
    recording where those exact words were really said. So we require a separate
    signal that comes from the audio.
    """
    if config.HALLUCINATION_MIN_SCORE > 0:
        score = _mean_score(seg)
        if score is not None and score < config.HALLUCINATION_MIN_SCORE:
            return f"alignment score {score:.2f}"

    # Filling 30 seconds of silence with a single sentence is the classic
    # hallucination. Nobody speaks that slowly (normal speech runs 4-8 chars/sec).
    duration = float(seg.get("end", 0.0) or 0.0) - float(seg.get("start", 0.0) or 0.0)
    if duration >= 6.0 and len(text) / duration < 1.5:
        return f"{len(text)} characters across {duration:.0f}s — nobody speaks that slowly"
    return None


def inspect(seg: dict[str, Any]) -> tuple[str, str] | None:
    """(verdict, reason). Verdict is 'drop' or 'suspect'. None when the segment is fine.

    drop    : delete it. Either it has no content (repetition), or it is
              boilerplate and the evidence agrees.
    suspect : keep it but flag it. Questionable, but it could be a real utterance.
    """
    text = (seg.get("text") or "").strip()
    if not text:
        return None  # empty segments are dropped at save time anyway

    # ── drop: repetition ─────────────────────────────────────────────
    # The text alone is enough here: there is no content, so nothing is lost
    # even if it was real. Only when it dominates the segment though — one
    # trailing "hahahaha" must not cost us the whole sentence.
    packed = text.replace(" ", "")
    glued = _GLUED.search(packed)
    if glued and len(glued.group(0)) / len(packed) >= 0.5:
        return "drop", f"same sound repeated ('{glued.group(1)}' 5+ times)"

    ratio = _repetition_ratio(text)
    if ratio >= 0.6:
        return "drop", f"same word repeated ({ratio * 100:.0f}% of the text)"

    # ── boilerplate: needs evidence before we delete ─────────────────
    covered = _boilerplate_ratio(text)
    proof = _evidence(seg, text)
    if covered >= 0.5:
        label = f"broadcast/subtitle sign-off is {covered * 100:.0f}% of the text"
        if proof:
            return "drop", f"{label} + {proof}"
        return "suspect", f"{label} — but the audio does match the words"

    if proof:
        return "suspect", f"audio does not match the words ({proof})"
    return None


def clean(segments: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (segments to keep, dropped, suspect).

    Suspect segments stay in the output. Deciding to delete them is a human's
    call. With DROP_SUSPECT=true they are dropped too.
    """
    if not config.DROP_HALLUCINATION:
        return segments, [], []

    kept: list[dict] = []
    dropped: list[dict] = []
    suspect: list[dict] = []
    for seg in segments:
        verdict = inspect(seg)
        if verdict is None:
            kept.append(seg)
            continue
        action, reason = verdict
        note = {
            "start": round(float(seg.get("start", 0.0) or 0.0), 3),
            "end": round(float(seg.get("end", 0.0) or 0.0), 3),
            "text": (seg.get("text") or "").strip(),
            "reason": reason,
        }
        if action == "drop" or config.DROP_SUSPECT:
            dropped.append(note)
        else:
            suspect.append(note)
            kept.append(seg)
    return kept, dropped, suspect


def summary(dropped: list[dict], suspect: list[dict]) -> list[str]:
    """Lines for the result page. Neither the deletions nor the keeps happen silently."""
    lines: list[str] = []
    if dropped:
        lines.append(
            f"Removed {len(dropped)} segment(s) judged to be hallucinations: "
            f"{_sample(dropped)}. If something real was deleted, set "
            "DROP_HALLUCINATION=false in .env and run again."
        )
    if suspect:
        lines.append(
            f"Kept {len(suspect)} segment(s) that might be hallucinations: "
            f"{_sample(suspect)}. Click a line to hear it and decide "
            "(set DROP_SUSPECT=true in .env to remove them all)."
        )
    return lines


def _sample(items: list[dict]) -> str:
    shown = ", ".join(f"\"{item['text'][:24]}\"" for item in items[:3])
    return shown + (f" and {len(items) - 3} more" if len(items) > 3 else "")
