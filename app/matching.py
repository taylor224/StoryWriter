"""Match new speaker embeddings against enrolled speakers and name them automatically.

How it works:
  - Cosine similarity (all vectors are stored L2-normalized, so a dot product is the cosine)
  - A speaker's score is the mean of their top 2 voiceprints (a single max is noise-prone)
  - Hungarian assignment makes it 1:1, so two people cannot collapse into one name
  - A name is only confirmed when it clears both the threshold and the first/second margin
"""

from typing import Any

import numpy as np

from . import config, db


def _score(vec: np.ndarray, profile_vectors: np.ndarray, top_k: int = 2) -> float:
    """Similarity between one normalized vector and a profile's set of vectors."""
    if profile_vectors.size == 0 or profile_vectors.shape[1] != vec.shape[0]:
        return -1.0
    sims = profile_vectors @ vec
    k = min(top_k, sims.shape[0])
    return float(np.sort(sims)[-k:].mean())


def match(
    embeddings: dict[str, Any],
    speech_sec: dict[str, float],
    profiles: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Returns {label: {matched, speaker_id, name, score, runner_up, reason}}.

    embeddings looks like {'SPEAKER_00': vector}. Normalization happens here.
    """
    if profiles is None:
        profiles = db.load_profiles()

    labels = sorted(embeddings.keys())
    out: dict[str, dict[str, Any]] = {
        label: {
            "matched": False,
            "speaker_id": None,
            "name": None,
            "score": 0.0,
            "runner_up": 0.0,
            "reason": "",
        }
        for label in labels
    }
    if not labels or not profiles:
        for label in labels:
            out[label]["reason"] = "no enrolled speakers"
        return out

    normed = {label: db.normalize(embeddings[label]) for label in labels}

    scores = np.full((len(labels), len(profiles)), -1.0, dtype=np.float64)
    for i, label in enumerate(labels):
        for j, profile in enumerate(profiles):
            scores[i, j] = _score(normed[label], profile["vectors"])

    # Too little speech makes the embedding unstable, so drop them from the running entirely
    too_short = {
        label for label in labels if speech_sec.get(label, 0.0) < config.MIN_SPEECH_SEC
    }
    for i, label in enumerate(labels):
        if label in too_short:
            scores[i, :] = -1.0
            out[label]["reason"] = (
                f"only {speech_sec.get(label, 0.0):.1f}s of speech — under "
                f"{config.MIN_SPEECH_SEC:.0f}s, so auto-recognition was skipped"
            )

    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-scores)

    for i, j in zip(rows, cols):
        label = labels[i]
        score = float(scores[i, j])
        others = np.delete(scores[i], j)
        runner_up = float(others.max()) if others.size else 0.0

        info = out[label]
        info["score"] = round(score, 4)
        info["runner_up"] = round(runner_up, 4)

        if label in too_short:
            continue
        if score < config.MATCH_THRESHOLD:
            info["reason"] = (
                f"best similarity {score:.3f} < threshold {config.MATCH_THRESHOLD:.2f}"
            )
            continue
        if runner_up > 0 and (score - runner_up) < config.MATCH_MARGIN:
            info["reason"] = (
                f"first {score:.3f} vs second {runner_up:.3f} — gap is under "
                f"{config.MATCH_MARGIN:.2f}, too close to call"
            )
            continue

        info["matched"] = True
        info["speaker_id"] = profiles[j]["id"]
        info["name"] = profiles[j]["name"]

    return out


def enroll(speaker_id: int, vector, source: str, speech_sec: float = 0.0) -> None:
    """Replace any existing vectors from the same source (prevents double enrollment)."""
    db.delete_voiceprints_from_source(source)
    db.add_voiceprint(speaker_id, vector, source=source, speech_sec=speech_sec)
