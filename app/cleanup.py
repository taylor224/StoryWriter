"""Whisper 가 지어낸 말을 걸러낸다.

Whisper 는 소리가 거의 없거나 잡음뿐인 구간에서 학습 데이터에 흔했던 문장을
그대로 뱉는다. 방송 마감 인사, 자막 제작자 크레딧, 같은 음절의 무한 반복이
대표적이다. 무음 제거(vad)가 이런 구간을 상당수 없애 주지만, 잡음이 깔려
있으면 무음으로 판정되지 않으므로 텍스트 쪽에서 한 번 더 거른다.

원칙: **글자만 보고 지우지 않는다.** "MBC 뉴스 OOO입니다" 도 뉴스 녹음이면 진짜
발언이고, "다음 시간에 뵙겠습니다" 는 회의 마무리에서 늘 나오는 말이다. 문장이
상투구처럼 생겼다는 것만으로는 환각이라는 근거가 못 된다.

그래서 두 갈래로 나눈다.

  지움 : 반복 ("아 아 아 아 아") — 내용이 없다. 진짜여도 잃을 게 없다.
         상투구 + 증거 — 소리가 그 글자를 뒷받침하지 않는다는 증거가 따로 있을 때.
  표시 : 나머지 의심 구간 — 결과에 남겨 두고 사람이 보게 한다.

증거는 두 가지다. 정렬 모델이 매긴 단어 신뢰도가 바닥이거나, 길이에 견줘 글자가
터무니없이 적거나 (침묵 30초를 한 문장으로 때우는 게 환각의 전형이다).

지우든 표시하든 무엇을 왜 그랬는지 전부 남긴다 (payload['dropped'], ['suspect']).
"""

import re
from typing import Any

from . import config

# Whisper 가 침묵에서 잘 뱉는 문장들. 이 목록에 걸렸다는 것만으로는 지우지
# 않는다 — 증거(_evidence)가 같이 있어야 한다. 아래 어느 것이든 실제로 그렇게
# 말한 녹음이 존재하기 때문이다 (뉴스 전사, 유튜버 자기 영상 전사, 강의 마무리).
BOILERPLATE = [
    # 한국어 방송·유튜브 마감 상투구
    r"(MBC|KBS|SBS|YTN|JTBC|MBN|채널\s*A|TV\s*조선)\s*뉴스\s*\S*\s*(입니다|이었습니다|였습니다|드립니다)",
    r"시청(해\s*주)?.{0,6}(감사|고맙)",
    r"구독\s*(과|와|,|그리고)?\s*좋아요",
    r"좋아요\s*(와|과|,|그리고)?\s*구독",
    r"(한글\s*)?자막\s*(제공|제작)",
    r"(이|본)\s*(영상|방송)은.{0,20}(제작|후원|지원|협찬)",
    # 영어권 자막 크레딧
    r"thanks?\s+for\s+watching",
    r"subtitles?\s+(by|provided)",
    r"amara\.org",
    r"please\s+subscribe",
    r"^\s*you\s*$",
    # 일본어 자막 크레딧 (한국어 녹음에도 섞여 나온다)
    r"ご視聴ありがとう",
]
# 일부러 뺀 것:
#   "감사합니다" / "좋아요" / "구독" 단독 — 회의에서 실제로 가장 많이 나오는 말.
#   "알림 설정"                        — 앱 얘기하는 회의에서 정상 발언.
#   "다음 시간에 뵙겠습니다"            — 회의·강의 마무리에서 늘 나온다.
# 전부 Whisper 가 환각으로도 뱉는 말이지만, 진짜 발언일 확률이 너무 높다.

_PATTERNS = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE]

# 공백 없이 붙은 반복 — "아아아아아아", "ㅋㅋㅋㅋㅋㅋ"
_GLUED = re.compile(r"(.{1,3}?)\1{4,}")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def _repetition_ratio(text: str) -> float:
    """가장 많이 나온 토큰이 전체에서 차지하는 비율. 반복 환각은 1.0 에 가깝다."""
    tokens = _tokens(text)
    if len(tokens) < 4:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return max(counts.values()) / len(tokens)


def _boilerplate_ratio(text: str) -> float:
    """상투구가 세그먼트에서 차지하는 길이 비율."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    covered = 0
    for pattern in _PATTERNS:
        for match in pattern.finditer(stripped):
            covered += match.end() - match.start()
    return min(1.0, covered / len(stripped))


def _mean_score(seg: dict[str, Any]) -> float | None:
    """정렬 모델이 매긴 단어 신뢰도의 평균. 점수가 없으면 None.

    소리와 글자가 맞지 않으면(=지어낸 말이면) 이 값이 바닥으로 떨어진다.
    """
    scores = [
        float(word["score"])
        for word in (seg.get("words") or [])
        if word.get("score") is not None
    ]
    return sum(scores) / len(scores) if scores else None


def _evidence(seg: dict[str, Any], text: str) -> str | None:
    """소리가 그 글자를 뒷받침하지 않는다는 증거. 없으면 None.

    글자만 보고는 환각인지 알 수 없다. 실제로 그렇게 말한 녹음이 늘 있기
    때문이다. 그래서 오디오 쪽에서 나온 근거를 따로 요구한다.
    """
    if config.HALLUCINATION_MIN_SCORE > 0:
        score = _mean_score(seg)
        if score is not None and score < config.HALLUCINATION_MIN_SCORE:
            return f"정렬 점수 {score:.2f}"

    # 침묵 30초를 한 문장으로 때우는 게 환각의 전형이다. 사람이 그 속도로
    # 말하는 일은 없다 (한국어 정상 발화는 초당 4~8자).
    duration = float(seg.get("end", 0.0) or 0.0) - float(seg.get("start", 0.0) or 0.0)
    if duration >= 6.0 and len(text) / duration < 1.5:
        return f"{duration:.0f}초 동안 {len(text)}자 — 사람 말 속도가 아님"
    return None


def inspect(seg: dict[str, Any]) -> tuple[str, str] | None:
    """(처분, 이유). 처분은 'drop' 또는 'suspect'. 멀쩡하면 None.

    drop    : 지운다. 내용이 없거나(반복), 상투구인데 증거까지 있는 경우.
    suspect : 남기되 표시한다. 의심스럽지만 진짜 발언일 수 있는 경우.
    """
    text = (seg.get("text") or "").strip()
    if not text:
        return None  # 빈 세그먼트는 저장 단계에서 어차피 빠진다

    # ── 지운다: 반복 ──────────────────────────────────────────────────
    # 이건 글자만 봐도 된다. 내용이 없어서 진짜여도 잃을 게 없기 때문이다.
    # 단, 세그먼트를 차지할 만큼일 때만 — 진짜 발언 끝의 "ㅋㅋㅋㅋㅋ" 하나
    # 때문에 문장 전체를 버리면 안 된다.
    packed = text.replace(" ", "")
    glued = _GLUED.search(packed)
    if glued and len(glued.group(0)) / len(packed) >= 0.5:
        return "drop", f"같은 소리 반복 ('{glued.group(1)}' 5회 이상)"

    ratio = _repetition_ratio(text)
    if ratio >= 0.6:
        return "drop", f"같은 단어 반복 (전체의 {ratio * 100:.0f}%)"

    # ── 상투구: 증거가 있어야 지운다 ──────────────────────────────────
    covered = _boilerplate_ratio(text)
    proof = _evidence(seg, text)
    if covered >= 0.5:
        label = f"방송·자막 마감 문구가 {covered * 100:.0f}%"
        if proof:
            return "drop", f"{label} + {proof}"
        return "suspect", f"{label} — 다만 소리는 글자와 맞는다"

    if proof:
        return "suspect", f"소리와 글자가 맞지 않음 ({proof})"
    return None


def clean(segments: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(남길 세그먼트, 버린 것, 의심스러운 것) 반환.

    의심스러운 것은 결과에 그대로 남는다. 지우는 건 사람이 판단할 일이다.
    DROP_SUSPECT=true 면 의심 구간까지 버린다.
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
    """결과 페이지에 띄울 문장들. 지운 것도 남긴 것도 조용히 넘기지 않는다."""
    lines: list[str] = []
    if dropped:
        lines.append(
            f"환각으로 판정한 {len(dropped)}개 구간을 제외했습니다: "
            f"{_sample(dropped)}. 실제로 한 말이 지워졌다면 .env 에 "
            "DROP_HALLUCINATION=false 로 두고 다시 돌리세요."
        )
    if suspect:
        lines.append(
            f"환각일 수도 있는 {len(suspect)}개 구간을 그대로 두었습니다: "
            f"{_sample(suspect)}. 소리를 들어 보고 판단하세요 "
            "(전부 지우려면 .env 에 DROP_SUSPECT=true)."
        )
    return lines


def _sample(items: list[dict]) -> str:
    shown = ", ".join(f"\"{item['text'][:24]}\"" for item in items[:3])
    return shown + (f" 외 {len(items) - 3}건" if len(items) > 3 else "")
