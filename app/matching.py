"""등록된 화자와 신규 화자 임베딩을 대조해 이름을 자동으로 붙인다.

핵심:
  - 코사인 유사도 (모든 벡터는 L2 정규화 저장 → 내적 = 코사인)
  - 화자별 점수는 상위 2개 보이스프린트의 평균 (단일 최댓값은 노이즈에 취약)
  - Hungarian 알고리즘으로 1:1 배정 → 두 사람이 같은 이름으로 뭉치는 사고 방지
  - 임계값 + 1등/2등 마진을 둘 다 통과해야 이름 확정
"""

from typing import Any

import numpy as np

from . import config, db


def _score(vec: np.ndarray, profile_vectors: np.ndarray, top_k: int = 2) -> float:
    """정규화된 벡터 하나와 프로필 벡터 묶음의 유사도."""
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
    """{라벨: {matched, speaker_id, name, score, runner_up, reason}} 반환.

    embeddings 는 {'SPEAKER_00': 벡터} 형태. 정규화는 여기서 한다.
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
            out[label]["reason"] = "등록된 화자 없음"
        return out

    normed = {label: db.normalize(embeddings[label]) for label in labels}

    scores = np.full((len(labels), len(profiles)), -1.0, dtype=np.float64)
    for i, label in enumerate(labels):
        for j, profile in enumerate(profiles):
            scores[i, j] = _score(normed[label], profile["vectors"])

    # 발화가 너무 짧으면 임베딩이 불안정하므로 아예 후보에서 제외
    too_short = {
        label for label in labels if speech_sec.get(label, 0.0) < config.MIN_SPEECH_SEC
    }
    for i, label in enumerate(labels):
        if label in too_short:
            scores[i, :] = -1.0
            out[label]["reason"] = (
                f"발화 {speech_sec.get(label, 0.0):.1f}초 — "
                f"{config.MIN_SPEECH_SEC:.0f}초 미만이라 자동 인식 생략"
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
                f"최고 유사도 {score:.3f} < 임계값 {config.MATCH_THRESHOLD:.2f}"
            )
            continue
        if runner_up > 0 and (score - runner_up) < config.MATCH_MARGIN:
            info["reason"] = (
                f"1등 {score:.3f} / 2등 {runner_up:.3f} — 차이가 "
                f"{config.MATCH_MARGIN:.2f} 미만이라 애매함"
            )
            continue

        info["matched"] = True
        info["speaker_id"] = profiles[j]["id"]
        info["name"] = profiles[j]["name"]

    return out


def enroll(speaker_id: int, vector, source: str, speech_sec: float = 0.0) -> None:
    """같은 출처의 기존 벡터를 지우고 새로 넣는다 (중복 등록 방지)."""
    db.delete_voiceprints_from_source(source)
    db.add_voiceprint(speaker_id, vector, source=source, speech_sec=speech_sec)
