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


def merge(parts: list) -> tuple:
    """조각별 화자 분리 결과를 파일 전체 결과로 합친다.

    parts 는 (turns, embeddings, speech_sec[, overlaps]) 의 목록.
    반환: (turns, embeddings, speech_sec, overlaps, 메모)
    turns 는 이미 원본 시각이므로 라벨만 갈아 끼우면 된다.
    """
    speakers: list[_Speaker] = []
    turns: list[dict] = []
    notes: list[str] = []

    overlaps: list[list[str]] = []
    for index, part in enumerate(parts):
        part_turns, part_embeddings, part_speech = part[0], part[1], part[2]
        part_overlaps = part[3] if len(part) > 3 else []
        # turn 에만 나오는 라벨까지 반드시 대응시킨다. 빠뜨리고 원래 라벨을
        # 그대로 두면 조각 2 의 SPEAKER_00 이 전체 기준 SPEAKER_00 과 같은
        # 이름이 되어, 다른 사람이 조용히 한 명으로 합쳐진다.
        appearing = {turn["speaker"] for turn in part_turns}
        mapping = _assign(part_embeddings, part_speech, speakers, index, notes, appearing)
        for turn in part_turns:
            turns.append({**turn, "speaker": mapping[turn["speaker"]]})
        # 동시 발화 증거도 전체 라벨로 옮겨 둔다 (collapse 에서 쓴다)
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


def collapse(
    turns: list[dict],
    embeddings: dict[str, np.ndarray],
    speech_sec: dict[str, float],
    overlaps: list[list[str]],
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, float], list[str]]:
    """같은 사람으로 보이는 화자들을 하나로 묶는다.

    pyannote 는 한 사람을 여러 화자로 쪼개는 일이 잦다. 목소리 톤이 바뀌거나,
    마이크에서 멀어지거나, 조각을 나눠 돌렸는데 이어 붙이기가 실패했거나.
    여기서 최종 화자들끼리 한 번 더 대조해 묶는다.

    조각 잇기(merge)와 달리 1:1 제약이 없다. 그래서 한 조각 안에서 pyannote 가
    쪼개 놓은 것도, 조각 잇기가 1:1 때문에 놓친 것도 여기서 잡힌다. 조각을
    나누지 않은 파일에도 그대로 적용된다.

    pyannote 의 판단을 뒤집는 일이라 안전장치를 둔다. 임계값을 높이는 방식은
    쓰지 않는다 — 그러면 조각 잇기가 놓친 쌍을 영영 못 묶는다. 대신:

      1. 동시에 말한 적이 있는 쌍은 절대 묶지 않는다. 같이 말했다면 같은
         사람일 수 없다 — 이게 유일하게 확실한 반증이다.
      2. 합칠 때마다 반증을 물려받는다. A 를 B 에 합치면, B 와 겹쳐 말한
         사람은 A 와도 다른 사람이다.

    묶는 방식은 평균 연결(centroid 대 centroid)이다. 최대 연결로 하면 A~B,
    B~C 가 가까울 때 A~C 가 멀어도 셋이 줄줄이 엮인다.

    반환: (turns, embeddings, speech_sec, 합친 내역)
    """
    labels = sorted(embeddings)
    if config.MERGE_THRESHOLD <= 0 or len(labels) < 2:
        return turns, embeddings, speech_sec, []

    groups: dict[str, list[str]] = {label: [label] for label in labels}
    vectors: dict[str, np.ndarray] = {label: db.normalize(embeddings[label]) for label in labels}
    speech_sec = dict(speech_sec)  # 호출부 것을 건드리지 않는다
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
            f"{absorbed} 를 {keeper} 와 같은 사람으로 보고 합쳤습니다 (유사도 {score:.2f})"
        )
        # 발화가 많은 쪽 벡터에 무게가 실리도록 발화 시간으로 가중 평균한다
        weights = (
            max(speech_sec.get(keeper, 0.0), 1e-6),
            max(speech_sec.get(absorbed, 0.0), 1e-6),
        )
        vectors[keeper] = db.normalize(
            vectors[keeper] * weights[0] + vectors[absorbed] * weights[1]
        )
        groups[keeper].extend(groups.pop(absorbed))
        speech_sec[keeper] = speech_sec.get(keeper, 0.0) + speech_sec.pop(absorbed, 0.0)
        # 반증은 옮겨 받는다. absorbed 와 겹쳐 말한 사람은 keeper 와도 다른 사람이다
        forbidden[keeper] |= forbidden.pop(absorbed)
        for others in forbidden.values():
            if absorbed in others:
                others.discard(absorbed)
                others.add(keeper)
        forbidden[keeper].discard(keeper)
        vectors.pop(absorbed)

    if not notes:
        return turns, embeddings, speech_sec, []

    # speech_sec 은 위에서 이미 합쳐졌다 (흡수된 라벨은 빠지고 시간은 넘어갔다).
    # 목소리를 모르는 화자도 그대로 남아 있다 — 합칠 근거가 없으니 손대지 않는다.
    rename = {member: keeper for keeper, members in groups.items() for member in members}
    merged_turns = [
        {**turn, "speaker": rename.get(turn["speaker"], turn["speaker"])} for turn in turns
    ]
    return merged_turns, {label: vectors[label] for label in groups}, speech_sec, notes


def collapse_summary(notes: list[str], speakers: int) -> str:
    return (
        f"같은 사람으로 보이는 화자 {len(notes)}쌍을 합쳐 최종 {speakers}명이 "
        f"되었습니다. 서로 다른 사람이 합쳐졌다면 .env 의 MERGE_THRESHOLD 를 "
        f"올리세요 (현재 {config.MERGE_THRESHOLD:.2f}). 아직도 한 사람이 여럿으로 "
        "나뉘면 내리세요."
    )


def summary(count: int, speakers: int) -> str:
    return (
        f"{count}개 조각으로 나눠 처리하고 목소리로 화자를 이어 붙였습니다 "
        f"(최종 {speakers}명). 같은 사람이 여러 명으로 갈라졌다면 .env 의 "
        f"STITCH_THRESHOLD 를 낮추세요 (현재 {config.STITCH_THRESHOLD:.2f})."
    )
