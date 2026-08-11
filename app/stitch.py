"""조각별 화자 라벨을 파일 전체 기준 하나로 합친다.

긴 파일을 조각내서 돌리면 조각마다 pyannote 가 SPEAKER_00 부터 새로 매긴다.
2번 조각의 SPEAKER_00 이 1번 조각의 SPEAKER_00 과 같은 사람이라는 보장은 없다.
그래서 목소리 임베딩의 코사인 유사도로 같은 사람을 찾아 하나로 묶는다.

matching.py 와 같은 원리지만 대상이 다르다. matching 은 "DB 에 등록된 사람"과
대조하고, 여기는 "같은 파일의 다른 조각"끼리 대조한다. 같은 녹음·같은 마이크라
조건이 좋아서 임계값을 조금 더 높게 잡을 수 있다.
"""

import numpy as np

from . import config, db


class _Speaker:
    """파일 전체 기준 화자 한 명. 조각마다 임베딩이 하나씩 쌓인다."""

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
    """조각이 늘수록 평균만 보면 흐려진다. 개별 조각과의 최댓값도 같이 본다.

    벡터가 없는 화자와는 비교할 방법이 없다. pyannote 가 겹침만 있는 화자의
    임베딩을 NaN 으로 내놓으면 diarize 가 버리므로(발화 시간은 남는다), 목소리를
    모르는 화자가 목록에 섞인다. -1 을 돌려 어떤 임계값도 통과하지 못하게 한다.
    """
    if not speaker.vectors:
        return -1.0
    best = max(float(vector @ other) for other in speaker.vectors)
    return max(best, float(vector @ speaker.centroid))


def merge(
    parts: list[tuple[list[dict], dict[str, np.ndarray], dict[str, float]]],
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, float], list[str]]:
    """조각별 (turns, embeddings, speech_sec) 을 파일 전체 결과로 합친다.

    반환: (turns, embeddings, speech_sec, 메모)
    turns 는 이미 원본 시각이므로 라벨만 갈아 끼우면 된다.
    """
    speakers: list[_Speaker] = []
    turns: list[dict] = []
    notes: list[str] = []

    for index, (part_turns, part_embeddings, part_speech) in enumerate(parts):
        # turn 에만 나오는 라벨까지 반드시 대응시킨다. 빠뜨리고 원래 라벨을
        # 그대로 두면 조각 2 의 SPEAKER_00 이 전체 기준 SPEAKER_00 과 같은
        # 이름이 되어, 다른 사람이 조용히 한 명으로 합쳐진다.
        appearing = {turn["speaker"] for turn in part_turns}
        mapping = _assign(part_embeddings, part_speech, speakers, index, notes, appearing)
        for turn in part_turns:
            turns.append({**turn, "speaker": mapping[turn["speaker"]]})

    turns.sort(key=lambda t: (t["start"], t["end"]))
    embeddings = {s.label: s.centroid for s in speakers if s.vectors}
    speech_sec = {s.label: s.speech for s in speakers}
    return turns, embeddings, speech_sec, notes


def _assign(
    part_embeddings: dict[str, np.ndarray],
    part_speech: dict[str, float],
    speakers: list[_Speaker],
    index: int,
    notes: list[str],
    appearing: set[str] | None = None,
) -> dict[str, str]:
    """조각 하나의 라벨을 전체 라벨로 대응시킨다. {'SPEAKER_00': 'SPEAKER_02'}"""
    labels = sorted(set(part_speech) | set(part_embeddings) | set(appearing or ()))
    mapping: dict[str, str] = {}
    if not labels:
        return mapping

    # 임베딩이 없는 라벨은 대조할 방법이 없다. 새 화자로 두는 수밖에.
    # (pyannote 는 겹침만 있는 화자의 임베딩을 NaN 으로 내놓고, diarize 가 버린다)
    known = [label for label in labels if part_embeddings.get(label) is not None]
    unknown = [label for label in labels if label not in set(known)]

    if speakers and known:
        vectors = {label: db.normalize(part_embeddings[label]) for label in known}
        scores = np.array(
            [[_similarity(vectors[label], speaker) for speaker in speakers] for label in known],
            dtype=np.float64,
        )
        # 한 조각 안의 두 라벨이 같은 사람으로 뭉치면 안 되므로 1:1 로 배정한다
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
                f"{index + 1}번 조각의 {label} 를 {speaker.label} 와 같은 사람으로 "
                f"봤습니다 (유사도 {score:.2f})"
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


def summary(count: int, speakers: int) -> str:
    return (
        f"{count}개 조각으로 나눠 처리하고 목소리로 화자를 이어 붙였습니다 "
        f"(최종 {speakers}명). 같은 사람이 여러 명으로 갈라졌다면 .env 의 "
        f"STITCH_THRESHOLD 를 낮추세요 (현재 {config.STITCH_THRESHOLD:.2f})."
    )
