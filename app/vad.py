"""전사 전에 무음을 들어내고, 끝나면 타임스탬프를 원본 시각으로 되돌린다.

Whisper 는 긴 침묵 구간에서 하지도 않은 말을 지어내는 버릇이 있고, 침묵을
통과시키는 계산도 그대로 비용이다. 전사 전에 무음을 잘라내면 둘 다 줄어든다.

문제는 잘라낸 만큼 뒤쪽 시각이 앞으로 당겨진다는 것이다. 그래서 잘라낸 파형과
함께 Timeline 을 만들어 두고, 전사·정렬이 끝나면 모든 타임스탬프를 원본 시각으로
되돌린다. 화자 분리는 원본 wav 를 그대로 쓰므로 그 뒤 단계는 잘린 적이 있다는
사실을 몰라도 된다.

무음 판정은 프레임 RMS 기준의 에너지 방식이다. 별도 모델을 받지 않고, 파일마다
잡음 바닥이 다르므로 절대 dB 가 아니라 그 파일의 잡음 바닥과 발화 세기 사이에서
기준선을 잡는다.
"""

import bisect

import numpy as np

from . import audio, config

SAMPLE_RATE = audio.SAMPLE_RATE
FRAME_SEC = 0.02  # 20ms — 자음 하나가 묻히지 않을 만큼 짧고 잡음에 덜 흔들린다
EPS = 1e-10


class Timeline:
    """잘라낸 파형의 시각을 원본 파형의 시각으로 되돌리는 표.

    regions 는 원본에서 "남긴" 구간 [start, end) 의 샘플 인덱스 목록이다.
    잘라낸 파형은 이 구간들을 순서대로 이어 붙인 것이므로, 잘린 쪽 시각 t 는
    항상 정확히 한 구간 안에 떨어진다 (빈틈이 생기지 않는다).

    regions 가 비어 있으면 "아무것도 자르지 않음" 이고 모든 변환이 항등이다.
    호출부가 켜짐/꺼짐을 분기하지 않아도 되게 하려는 것.
    """

    def __init__(self, regions, total: int, sample_rate: int = SAMPLE_RATE):
        self.regions: list[tuple[int, int]] = [(int(s), int(e)) for s, e in regions]
        self.total = int(total)
        self.sample_rate = int(sample_rate)

        # offsets[i] = 잘라낸 파형에서 i 번째 구간이 시작하는 샘플 위치
        self.offsets: list[int] = []
        cursor = 0
        for start, end in self.regions:
            self.offsets.append(cursor)
            cursor += end - start
        self.kept = cursor

    @classmethod
    def identity(cls, total: int, sample_rate: int = SAMPLE_RATE) -> "Timeline":
        """아무것도 자르지 않은 표. total 은 표시용으로만 쓰인다."""
        return cls([], total, sample_rate)

    @property
    def trimmed(self) -> bool:
        return bool(self.regions) and self.kept < self.total

    @property
    def span(self) -> tuple[int, int]:
        """원본에서 이 표가 걸쳐 있는 [start, end) 범위. 조각을 파일에서 읽을 때 쓴다."""
        if not self.regions:
            return (0, self.total)
        return (self.regions[0][0], self.regions[-1][1])

    def apply(self, samples: np.ndarray, origin: int = 0) -> np.ndarray:
        """남긴 구간만 이어 붙인 파형.

        samples 가 원본 전체가 아니라 origin 샘플부터 잘라 읽은 조각이면 origin
        을 준다 (10시간짜리를 통째로 올리지 않으려면 이 경로가 필요하다).
        """
        if not self.regions:
            return samples
        # 디코더가 달라 샘플 수가 한두 개 어긋나도 죽지 않게 끝을 잘라 맞춘다
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
        """잘라낸 파형의 시각 -> 원본 파형의 시각. 단조 증가한다."""
        if not self.trimmed:
            return float(seconds)
        return self._at(seconds * self.sample_rate) / self.sample_rate

    def _at(self, position: float) -> float:
        """잘라낸 쪽 샘플 위치 -> 원본 샘플 위치."""
        if position <= 0:
            return float(self.regions[0][0])
        index = bisect.bisect_right(self.offsets, position) - 1
        index = max(0, min(index, len(self.regions) - 1))
        start, end = self.regions[index]
        # min(..., end) 는 마지막 구간을 넘어선 시각(정렬 오차 등)을 잡아 준다
        return min(start + (position - self.offsets[index]), float(end))

    def split(self, start: float, end: float) -> list[tuple[float, float]]:
        """잘라낸 쪽 구간 하나를 원본 구간 여러 개로 쪼갠다.

        잘린 자리를 건너뛰는 구간은 원본에서 이어져 있지 않다. 양 끝만 되돌리면
        없앤 침묵까지 그 구간이 삼켜 버린다 — 화자 구간이라면 말하지도 않은
        침묵을 그 화자가 차지한다. 그래서 남긴 구간 경계마다 끊어서 돌려준다.

        길이의 합은 보존된다. 쪼개도 총 발화 시간은 그대로다.
        """
        if not self.trimmed:
            return [(float(start), float(end))]

        low = max(0.0, start * self.sample_rate)
        high = min(float(self.kept), end * self.sample_rate)
        if high <= low:  # 길이 0 이거나 범위 밖 — 한 점으로 되돌린다
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
        """세그먼트와 단어의 타임스탬프를 제자리에서 원본 시각으로 되돌린다."""
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
    # 정렬 모델은 숫자·기호처럼 발음을 못 붙인 단어에 start/end 를 비워 둔다
    for key in ("start", "end"):
        value = item.get(key)
        if value is not None:
            item[key] = convert(float(value))


def chunks(timeline: Timeline, max_span_sec: float) -> list[Timeline]:
    """긴 녹음을 조각 Timeline 여러 개로 나눈다. 나눌 필요가 없으면 [timeline].

    각 조각은 원본 좌표를 그대로 들고 있다. 조각을 돌려 나온 타임스탬프는 별도
    보정 없이 곧바로 원본 시각이다 — 조각 번호를 들고 다니며 더할 필요가 없다.

    경계는 되도록 침묵 한가운데에 둔다. 말하는 도중에 자르면 그 단어가 양쪽에서
    반씩 인식되고, 화자 분리도 경계에서 흔들린다. 무음 제거를 꺼 두면 어디가
    침묵인지 알 수 없어 균등 분할로 떨어진다.
    """
    limit = int(max_span_sec * timeline.sample_rate)
    if limit <= 0 or timeline.total <= limit:
        return [timeline]

    if timeline.regions:
        groups: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        for region in timeline.regions:
            # 이 구간까지 넣으면 조각이 너무 길어지는가 (조각 시작점 기준)
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
    """캐시 키에 넣을 설정값. 하나라도 바뀌면 전사를 다시 돌려야 한다."""
    return {
        "min_silence": config.TRIM_MIN_SILENCE_SEC,
        "pad": config.TRIM_PAD_SEC,
        "sensitivity": config.TRIM_SENSITIVITY,
        "min_speech": config.TRIM_MIN_SPEECH_SEC,
        "min_dynamic_db": config.TRIM_MIN_DYNAMIC_DB,
        "highpass": config.TRIM_HIGHPASS_HZ,
    }


# ── 무음 판정 ─────────────────────────────────────────────────────────
def _power_db(block: np.ndarray, frame: int) -> np.ndarray:
    """(n, frame) 로 접은 파형의 프레임별 RMS 를 dBFS 로.

    samples**2 는 원본 크기의 임시 배열을 하나 더 만든다. 1시간짜리면 230MB 라
    einsum 으로 제곱합만 뽑아 그 복사를 피한다.
    """
    power = np.einsum("ij,ij->i", block, block) / frame
    return 10.0 * np.log10(power + EPS)


def _frame_db(samples: np.ndarray, frame: int, sample_rate: int = 0) -> np.ndarray:
    """프레임별 RMS 를 dBFS 로. 판정 전에 저주파를 걷어낸다.

    에어컨·프로젝터 팬·책상 진동은 대부분 100Hz 아래에 몰려 있다. 사람 목소리의
    기본 주파수는 낮아도 85Hz 부터라 그 아래는 버려도 말이 상하지 않는다.

    안 걷으면 잡음 바닥이 통째로 올라가 무음과 발화의 세기 차가 좁아지고,
    "차이가 12dB 미만" 에 걸려 아예 자르지 못한다. 조용한 사무실 녹음인데
    무음 제거가 안 먹는 경우가 대개 이것이다.

    걸러낸 파형은 여기서만 쓰고 버린다. 전사·화자 분리에 넘기는 파형은 원본
    그대로다 — Whisper 는 잡음 섞인 오디오로 학습돼서 손대면 되레 나빠진다.
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
    # 블록으로 흘려 보낸다. 10시간짜리를 한 번에 필터링하면 사본만 2.3GB 다.
    step = frame * 20_000
    parts = []
    for start in range(0, usable, step):
        block = samples[start : min(start + step, usable)]
        filtered, state = sosfilt(sos, block, zi=state)
        parts.append(_power_db(filtered.reshape(-1, frame), frame))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 가 이어지는 구간 [start, end) 목록."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _merge(regions: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """앞 구간 끝과 뒤 구간 시작의 거리가 gap 미만이면 하나로 잇는다."""
    merged: list[list[int]] = [list(regions[0])]
    for start, end in regions[1:]:
        if start - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def plan(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[Timeline, str | None]:
    """어디를 남길지 정한다.

    반환: (Timeline, 건너뛴 이유 or None). 자를 근거가 약하면 항등 Timeline 을
    돌려준다. 애매할 때 원본을 그대로 두는 쪽이 말을 잘라먹는 것보다 낫다.
    """
    total = int(samples.shape[0])
    identity = Timeline.identity(total, sample_rate)
    frame = max(1, int(round(FRAME_SEC * sample_rate)))
    if total < sample_rate:  # 1초 미만이면 잘라낼 것도 없다
        return identity, None

    level = _frame_db(samples, frame, sample_rate)

    # 파일마다 잡음 바닥이 다르다. 절대 dB 로 자르면 조용한 녹음은 통째로
    # 무음이 되고, 시끄러운 녹음은 아무것도 안 잘린다. 그래서 이 파일 안에서
    # "조용한 쪽"과 "말하는 쪽"을 먼저 재고 그 사이에 기준선을 놓는다.
    floor_db = float(np.percentile(level, 10))
    speech_db = float(np.percentile(level, 95))
    span = speech_db - floor_db
    if span < config.TRIM_MIN_DYNAMIC_DB:
        return identity, (
            f"소리 크기가 처음부터 끝까지 거의 같아({span:.1f}dB 차이) 무음 제거를 "
            "건너뜁니다. 잡음이 큰 녹음이거나 쉬는 구간이 없는 녹음입니다."
        )

    threshold = floor_db + max(6.0, span * config.TRIM_SENSITIVITY)
    threshold = min(threshold, speech_db - 6.0)

    speech = _runs(level > threshold)
    if not speech:
        return identity, "발화 구간을 찾지 못해 무음 제거를 건너뜁니다."

    min_speech = int(round(config.TRIM_MIN_SPEECH_SEC * sample_rate))
    min_silence = int(round(config.TRIM_MIN_SILENCE_SEC * sample_rate))
    # pad 는 발화 앞뒤로 넓히는 여유이자, 이어 붙였을 때 두 발화 사이에 그대로
    # 남는 침묵이기도 하다. 0 이면 서로 다른 발화가 맞붙어 Whisper 가 두 문장을
    # 하나로 읽어 버리므로 최소치를 둔다.
    pad = max(int(round(config.TRIM_PAD_SEC * sample_rate)), int(round(0.05 * sample_rate)))

    regions = [(s * frame, min(e * frame, total)) for s, e in speech]
    # 기침·마우스 클릭 같은 순간 소음까지 붙들면 무음이 잘리지 않는다
    regions = [r for r in regions if r[1] - r[0] >= min_speech]
    if not regions:
        return identity, "발화 구간이 너무 짧아 무음 제거를 건너뜁니다."

    regions = _merge(regions, min_silence)  # 짧은 숨은 침묵이 아니라 말의 일부다
    regions = [(max(0, s - pad), min(total, e + pad)) for s, e in regions]
    regions = _merge(regions, 1)  # pad 때문에 겹친 것들을 정리

    timeline = Timeline(regions, total, sample_rate)
    if timeline.kept < sample_rate:
        return identity, "발화로 판정된 구간이 1초 미만이라 무음 제거를 건너뜁니다."
    if timeline.kept >= total * 0.98:
        return identity, None  # 2% 미만이면 복사 비용이 아깝다
    return timeline, None
