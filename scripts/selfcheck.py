"""모델 없이 돌릴 수 있는 부분을 전부 점검한다.

    python scripts/selfcheck.py

무음 제거 · 조각 나누기 · 조각 간 화자 이어붙이기 · 환각 필터. 합성 데이터로
돌리므로 모델도 GPU 도 필요 없다 (numpy/scipy 만 있으면 된다).

vad.py / stitch.py / cleanup.py 를 손대면 이걸 먼저 돌릴 것. 특히 타임스탬프
쪽이 깨지면 결과의 모든 시각이 조용히 어긋난다 — 눈으로는 알아채기 힘들다.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, vad  # noqa: E402

SR = vad.SAMPLE_RATE
rng = np.random.default_rng(7)
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def speech(sec: float, amp: float = 0.25) -> np.ndarray:
    """대역 제한 잡음 + 음절 리듬. 순수 사인파보다 사람 목소리에 가깝다."""
    count = int(sec * SR)
    kernel = np.hanning(51)
    band = np.convolve(rng.normal(0, 1, count), kernel / kernel.sum(), mode="same")
    band *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * np.arange(count) / SR)
    return (amp * band / np.abs(band).max()).astype(np.float32)


def room(sec: float, amp: float = 0.001) -> np.ndarray:
    """실제 녹음의 조용한 구간. 완전한 0 이 아니라 미세한 잡음이 깔린다."""
    return rng.normal(0, amp, int(sec * SR)).astype(np.float32)


# ── 말 / 침묵을 섞은 57.5초 ────────────────────────────────────────────
layout = [("말", 3.0), ("", 12.0), ("말", 4.0), ("", 1.5), ("말", 2.0), ("", 30.0), ("말", 5.0)]
parts, spoken, cursor = [], [], 0.0
for kind, seconds in layout:
    parts.append(speech(seconds) if kind else room(seconds))
    if kind:
        spoken.append((cursor, cursor + seconds))
    cursor += seconds
audio = np.concatenate(parts)

timeline, note = vad.plan(audio)
stats = timeline.stats()
print(f"\n원본 {stats['original']}초 -> 전사 대상 {stats['kept']}초 "
      f"(무음 {stats['removed']}초 제거, 구간 {stats['regions']}개)")
if note:
    print(f"       {note}")
print()

check("긴 침묵을 잘라냈다", timeline.trimmed, f"{stats['removed']}초")
check("발화 구간 수가 맞다", stats["regions"] == 4, f"{stats['regions']}개")

trimmed = timeline.apply(audio)
check(
    "잘라낸 파형 = 남긴 구간을 이어 붙인 것",
    np.array_equal(trimmed, np.concatenate([audio[s:e] for s, e in timeline.regions])),
)

# ── 핵심: 잘린 쪽 시각을 되돌리면 원본의 같은 샘플을 가리키는가 ────────
worst = 0.0
for index in rng.integers(0, len(trimmed), 20000):
    back = int(round(timeline.to_original(index / SR) * SR))
    worst = max(worst, abs(float(audio[min(back, len(audio) - 1)] - trimmed[index])))
check("무작위 20000개 지점의 시각 복원이 샘플 단위로 정확", worst == 0.0, f"오차 {worst:.1e}")

probe = [timeline.to_original(t) for t in np.linspace(-1.0, stats["kept"] + 5.0, 5000)]
check("복원이 단조 증가 (순서가 뒤집히지 않는다)",
      all(b >= a - 1e-9 for a, b in zip(probe, probe[1:])))
check("복원 결과가 원본 길이를 벗어나지 않는다",
      probe[0] >= 0.0 and probe[-1] <= stats["original"] + 1e-9)

# ── 말을 잘라먹지 않았는가 ────────────────────────────────────────────
kept = np.zeros(len(audio), dtype=bool)
for start, end in timeline.regions:
    kept[start:end] = True
for start_sec, end_sec in spoken:
    covered = float(kept[int(start_sec * SR):int(end_sec * SR)].mean())
    check(f"발화 {start_sec:5.1f}~{end_sec:5.1f}초 보존", covered > 0.97, f"{covered * 100:.1f}%")

# ── 잘린 자리를 넘는 구간은 쪼개진다 (화자 구간용) ────────────────────
# 쪼개지 않으면 그 화자가 없앤 침묵까지 차지한다.
cut = vad.Timeline([(0, 2 * SR), (10 * SR, 14 * SR)], 14 * SR)
check("구간이 한 조각 안에 있으면 그대로", cut.split(1.0, 1.5) == [(1.0, 1.5)], str(cut.split(1.0, 1.5)))

crossing = cut.split(1.5, 3.0)
check("잘린 자리를 넘는 구간은 쪼개진다", crossing == [(1.5, 2.0), (10.0, 11.0)], str(crossing))
check("쪼개도 길이 합은 보존된다",
      abs(sum(b - a for a, b in crossing) - 1.5) < 1e-9,
      f"{sum(b - a for a, b in crossing):.3f}초")
check("쪼갠 조각은 전부 남긴 구간 안에 있다",
      all(any(r[0] / SR <= a and b <= r[1] / SR for r in cut.regions) for a, b in crossing))
check("전체 구간을 쪼개면 남긴 구간 전부",
      cut.split(0.0, 6.0) == [(0.0, 2.0), (10.0, 14.0)], str(cut.split(0.0, 6.0)))
check("항등 Timeline 에서 split 은 그대로", vad.Timeline.identity(0).split(3.0, 9.0) == [(3.0, 9.0)])

# 화자 구간 복원 — diarize._restore_turns 가 하는 일 그대로
from app import diarize  # noqa: E402  (torch 를 끌고 오므로 여기서 import)

turns = diarize._restore_turns(
    [{"start": 0.5, "end": 3.0, "speaker": "SPEAKER_00"},
     {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"}],
    cut,
)
check("화자 구간이 침묵을 삼키지 않는다", len(turns) == 3, f"{len(turns)}개")
check("총 발화 시간은 그대로",
      abs(sum(t["end"] - t["start"] for t in turns) - 3.5) < 1e-9,
      f"{sum(t['end'] - t['start'] for t in turns):.2f}초")
check("화자 라벨이 유지된다",
      [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"],
      str([t["speaker"] for t in turns]))

# ── 세그먼트·단어 복원 ────────────────────────────────────────────────
segments = [
    {"start": 2.3, "end": 4.0, "words": [
        {"word": "가", "start": 2.3, "end": 3.0},
        {"word": "125", "start": None, "end": None},  # 정렬이 붙이지 못한 단어
    ]},
]
vad.Timeline([(0, 2 * SR), (10 * SR, 14 * SR)], 14 * SR).restore(segments)
check("세그먼트 시각 복원", abs(segments[0]["start"] - 10.3) < 1e-6 and
      abs(segments[0]["end"] - 12.0) < 1e-6, str(segments[0]["start"]))
check("단어 시각 복원", abs(segments[0]["words"][0]["start"] - 10.3) < 1e-6)
check("타임스탬프 없는 단어는 건드리지 않는다", segments[0]["words"][1]["start"] is None)

# ── 자르면 안 되는 경우들 ─────────────────────────────────────────────
identity = vad.Timeline.identity(len(audio))
check("항등 Timeline 은 파형을 복사하지 않는다", identity.apply(audio) is audio)
check("항등 Timeline 은 시각을 건드리지 않는다", identity.to_original(3.25) == 3.25)
check("항등 Timeline 도 길이를 제대로 보고한다",
      identity.stats()["kept"] == identity.stats()["original"], str(identity.stats()))
check("잡음만 있으면 자르지 않는다",
      not vad.plan(rng.normal(0, 0.05, 10 * SR).astype(np.float32))[0].trimmed)
check("무음만 있으면 자르지 않는다", not vad.plan(room(10))[0].trimmed)
check("1초 미만 파일은 자르지 않는다", not vad.plan(speech(0.5))[0].trimmed)
check("짧은 침묵(0.4초)은 말의 일부로 보고 남긴다",
      not vad.plan(np.concatenate([speech(1), room(0.4), speech(1)]))[0].trimmed)
check("긴 침묵(3초)은 잘라낸다",
      vad.plan(np.concatenate([speech(1), room(3.0), speech(1)]))[0].trimmed)

# ── 잘린 발화끼리 맞붙지 않는가 ───────────────────────────────────────
# 맞붙으면 Whisper 가 서로 다른 두 발화를 한 문장으로 읽는다.
original_pad, config.TRIM_PAD_SEC = config.TRIM_PAD_SEC, 0.0  # 최악의 설정으로
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
    check("TRIM_PAD_SEC=0 이어도 발화 사이에 침묵이 남는다",
          bool(inner) and max(inner) >= 0.09, f"{max(inner or [0]):.2f}초")
finally:
    config.TRIM_PAD_SEC = original_pad


# ── 조각 나누기 ───────────────────────────────────────────────────────
print("\n── 조각 나누기 ──")
pieces = vad.chunks(timeline, 20.0)   # 57.5초를 20초씩
print("조각:", [(round(p.span[0] / SR, 2), round(p.span[1] / SR, 2)) for p in pieces])

check("조각이 여러 개로 나뉜다", len(pieces) > 1, f"{len(pieces)}개")
check("조각 순서가 오름차순이고 겹치지 않는다",
      all(a.span[1] <= b.span[0] for a, b in zip(pieces, pieces[1:])))
check("조각을 모두 합치면 원래 남긴 구간 전부",
      [r for p in pieces for r in p.regions] == timeline.regions)
check("조각을 이어 붙이면 통째로 자른 것과 같다",
      np.array_equal(np.concatenate([p.apply(audio) for p in pieces]), timeline.apply(audio)))

# 조각 범위만 파일에서 읽어 자르는 경로 (10시간을 통째로 안 올리기 위한 것)
windowed = [p.apply(audio[p.span[0]:p.span[1]], p.span[0]) for p in pieces]
check("조각 범위만 읽어 잘라도 결과가 같다",
      all(np.array_equal(a, p.apply(audio)) for a, p in zip(windowed, pieces)))

# 조각 안에서 나온 시각이 곧바로 원본 시각인가
for order, piece in enumerate(pieces):
    length = piece.kept / SR
    lo, hi = piece.to_original(0.0), piece.to_original(length)
    inside = piece.span[0] / SR <= lo and hi <= piece.span[1] / SR
    check(f"{order + 1}번 조각의 시각이 제 범위 안으로 돌아온다", inside,
          f"{lo:.2f}~{hi:.2f} (범위 {piece.span[0]/SR:.2f}~{piece.span[1]/SR:.2f})")

check("경계가 침묵 안에 있다 (말 도중에 자르지 않는다)",
      all(any(abs(p.span[0] - s) < 2 for s, _ in timeline.regions) for p in pieces[1:]))
check("나눌 필요가 없으면 그대로 한 개", len(vad.chunks(timeline, 3600.0)) == 1)
check("CHUNK_SEC=0 이면 나누지 않는다", len(vad.chunks(timeline, 0.0)) == 1)

# ── wav 를 구간 단위로 읽기 ───────────────────────────────────────────
# 조각 처리의 전제. 여기가 어긋나면 조각마다 엉뚱한 오디오를 인식한다.
print("\n── wav 구간 읽기 ──")
import io  # noqa: E402
import tempfile  # noqa: E402
import wave as wave_mod  # noqa: E402

from app import audio as audio_io  # noqa: E402

with tempfile.TemporaryDirectory() as folder:
    sample_path = Path(folder) / "sample.wav"
    stored = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave_mod.open(str(sample_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(stored.tobytes())

    whole = audio_io.read_wav(sample_path)
    check("전체 읽기 길이가 맞다", len(whole) == len(audio), f"{len(whole)} vs {len(audio)}")
    check("샘플 수를 따로도 읽을 수 있다", audio_io.sample_count(sample_path) == len(audio))
    # 파일에 든 정수를 그대로 /32768 한 값이어야 한다 (whisperx.load_audio 와 같은 규약)
    check("디코딩이 정확하다", np.array_equal(whole, stored.astype(np.float32) / 32768.0))

    lo, hi = 7 * SR, 21 * SR
    check("구간만 읽어도 전체에서 자른 것과 같다",
          np.array_equal(audio_io.read_wav(sample_path, lo, hi), whole[lo:hi]))
    check("끝을 넘겨 요청하면 파일 끝까지",
          len(audio_io.read_wav(sample_path, len(audio) - SR, len(audio) + 99 * SR)) == SR)
    check("범위 밖이면 빈 배열", len(audio_io.read_wav(sample_path, len(audio) + 5, None)) == 0)

    # 전사록 한 줄을 눌렀을 때 내려보낼 조각
    clip = audio_io.clip_wav(sample_path, 15.0, 19.0)
    with wave_mod.open(io.BytesIO(clip), "rb") as handle:
        check("조각 wav 가 정상적인 wav 파일이다",
              handle.getframerate() == SR and handle.getnchannels() == 1)
        check("조각 길이가 요청한 만큼", abs(handle.getnframes() / SR - 4.0) < 0.01,
              f"{handle.getnframes() / SR:.2f}초")
        cut = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    check("조각 내용이 원본의 그 구간과 같다",
          np.array_equal(cut, stored[15 * SR:19 * SR]))
    check("조각은 상한을 넘지 않는다",
          len(audio_io.clip_wav(sample_path, 0.0, 9999.0, max_sec=2.0)) <= 2 * SR * 2 + 100)
    check("거꾸로 된 범위도 죽지 않는다", audio_io.clip_wav(sample_path, 20.0, 5.0) is not None)

    bad_path = Path(folder) / "bad.wav"
    with wave_mod.open(str(bad_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 1000)
    try:
        audio_io.read_wav(bad_path)
        check("44.1kHz wav 는 거부한다", False, "예외가 안 났다")
    except RuntimeError as exc:
        check("44.1kHz wav 는 거부한다", "44100" in str(exc), str(exc)[:60])

# ── 조각 간 화자 이어붙이기 ───────────────────────────────────────────
print("\n── 조각 간 화자 이어붙이기 ──")
from app import db, stitch  # noqa: E402

people = [db.normalize(rng.normal(0, 1, 192)) for _ in range(3)]


def voice(person: int, drift: float = 0.04):
    """같은 사람의 다른 조각 임베딩 — 조금 흔들리게.

    정규화된 192차원 벡터라 성분 하나의 크기는 1/sqrt(192)≈0.072 다. drift 를
    그보다 크게 주면 잡음이 신호를 덮어 같은 사람이 아니게 된다. 0.04 는
    코사인 0.87 근처 — 같은 녹음의 다른 조각에서 실제로 나오는 정도.
    """
    return db.normalize(people[person] + rng.normal(0, drift, 192))


same = float(voice(0) @ voice(0))
other = float(voice(0) @ voice(1))
print(f"       같은 사람 코사인 {same:.2f} / 다른 사람 {other:.2f} "
      f"(임계값 {config.STITCH_THRESHOLD:.2f})")
check("합성 데이터가 임계값 양쪽으로 갈린다",
      other < config.STITCH_THRESHOLD <= same, f"{other:.2f} / {same:.2f}")


# 조각1: 사람0=SPEAKER_00, 사람1=SPEAKER_01
# 조각2: 사람1=SPEAKER_00, 사람0=SPEAKER_01   ← 라벨이 뒤바뀐 상황
# 조각3: 사람2 만 등장 (새 사람)
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

check("사람 수를 정확히 셌다", len(merged_speech) == 3, f"{len(merged_speech)}명")
check("라벨이 뒤바뀐 조각을 바로잡았다",
      merged_turns[0]["speaker"] == merged_turns[3]["speaker"],
      f"{merged_turns[0]['speaker']} vs {merged_turns[3]['speaker']}")
check("다른 사람을 합치지 않았다",
      merged_turns[0]["speaker"] != merged_turns[1]["speaker"])
check("총 발화 시간이 합산된다",
      abs(sum(merged_speech.values()) - 26.0) < 1e-9, f"{sum(merged_speech.values())}초")
check("발화 시간이 사람별로 옳게 모였다",
      sorted(round(v, 1) for v in merged_speech.values()) == [8.0, 8.0, 10.0],
      str(sorted(round(v, 1) for v in merged_speech.values())))
check("turn 이 시각 순으로 정렬된다",
      all(a["start"] <= b["start"] for a, b in zip(merged_turns, merged_turns[1:])))
check("합친 임베딩이 정규화되어 있다",
      all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-5 for v in merged_emb.values()))

single = stitch.merge([parts[0]])
check("조각이 하나면 라벨을 그대로 둔다",
      [t["speaker"] for t in single[0]] == ["SPEAKER_00", "SPEAKER_01"], str(single[0]))

# pyannote 는 겹침만 있는 화자의 임베딩을 NaN 으로 내놓고 diarize 가 버린다.
# 그러면 "발화 시간은 있는데 목소리는 모르는" 라벨이 생긴다. 이게 조각 목록에
# 섞인 채 다음 조각과 대조하다가 터졌었다 (max() on empty).
voiceless = [
    ([{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}],
     {},                                   # 임베딩이 하나도 없는 조각
     {"SPEAKER_00": 4.0}),
    ([{"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00"},
      {"start": 15.0, "end": 18.0, "speaker": "SPEAKER_01"}],
     {"SPEAKER_00": voice(0)},             # 한쪽만 임베딩이 있는 조각
     {"SPEAKER_00": 5.0, "SPEAKER_01": 3.0}),
    ([{"start": 20.0, "end": 26.0, "speaker": "SPEAKER_00"}],
     {"SPEAKER_00": voice(0)},             # 위와 같은 사람
     {"SPEAKER_00": 6.0}),
]
try:
    v_turns, v_emb, v_speech, _, _ = stitch.merge(voiceless)
    check("임베딩 없는 화자가 섞여도 죽지 않는다", True)
    # 목소리 모르는 둘은 각각 따로 남고(4.0, 3.0), voice(0) 둘만 합쳐진다(5+6=11)
    check("임베딩 없는 화자는 아무와도 합쳐지지 않는다",
          sorted(round(v, 1) for v in v_speech.values()) == [3.0, 4.0, 11.0],
          f"{len(v_speech)}명 {sorted(round(v, 1) for v in v_speech.values())}")
    check("임베딩 있는 같은 사람은 조각을 넘어 합쳐진다",
          v_turns[1]["speaker"] == v_turns[3]["speaker"],
          f"{v_turns[1]['speaker']} vs {v_turns[3]['speaker']}")
    check("임베딩 없는 화자는 결과 임베딩에서 빠진다", len(v_emb) == 1, str(sorted(v_emb)))
    check("발화 시간은 전부 보존된다", abs(sum(v_speech.values()) - 18.0) < 1e-9,
          f"{sum(v_speech.values())}초")
except ValueError as exc:
    check("임베딩 없는 화자가 섞여도 죽지 않는다", False, str(exc))

# turn 에만 나오는 라벨도 반드시 새 이름을 받아야 한다.
# 원래 라벨을 그대로 두면 다른 사람이 조용히 한 명으로 합쳐진다.
stray = stitch.merge([
    ([{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
     {"SPEAKER_00": voice(0)}, {"SPEAKER_00": 5.0}),
    ([{"start": 9.0, "end": 12.0, "speaker": "SPEAKER_00"}],   # speech_sec 에 없는 라벨
     {}, {}),
])
check("turn 에만 있는 라벨도 딴 사람으로 분리된다",
      stray[0][0]["speaker"] != stray[0][1]["speaker"],
      f"{stray[0][0]['speaker']} vs {stray[0][1]['speaker']}")

# ── 과분할된 화자 다시 묶기 ───────────────────────────────────────────
print("\n── 과분할된 화자 다시 묶기 ──")

# pyannote 가 사람0 을 SPEAKER_00/SPEAKER_02 둘로 쪼갠 상황. 사람1 은 딴 사람.
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
check("쪼개진 같은 사람을 다시 묶는다", len(c_speech) == 2, f"{len(c_speech)}명")
check("합쳐진 쪽 turn 이 같은 라벨이 된다",
      c_turns[0]["speaker"] == c_turns[2]["speaker"],
      f"{c_turns[0]['speaker']} vs {c_turns[2]['speaker']}")
check("딴 사람은 그대로 둔다", c_turns[1]["speaker"] != c_turns[0]["speaker"])
check("발화 시간이 합산된다", abs(max(c_speech.values()) - 10.0) < 1e-9, str(c_speech))
check("합계는 보존된다", abs(sum(c_speech.values()) - 14.0) < 1e-9, str(sum(c_speech.values())))

# 동시에 말한 적이 있으면 아무리 닮아도 묶지 않는다 — 같은 사람일 수 없다
o_turns, o_emb, o_speech, o_notes = stitch.collapse(
    split_turns, split_emb, dict(split_speech), [["SPEAKER_00", "SPEAKER_02"]]
)
check("동시에 말한 쌍은 닮아도 안 묶는다", len(o_speech) == 3 and not o_notes,
      f"{len(o_speech)}명")

# 반증은 합쳐질 때 옮겨 받아야 한다 (A~B 를 묶으면 B 와 겹친 C 는 A 와도 다른 사람)
chain_turns = [{"start": float(i), "end": i + 1.0, "speaker": f"SPEAKER_0{i}"} for i in range(3)]
chain = stitch.collapse(
    chain_turns,
    {"SPEAKER_00": voice(0), "SPEAKER_01": voice(0), "SPEAKER_02": voice(0)},
    {"SPEAKER_00": 5.0, "SPEAKER_01": 5.0, "SPEAKER_02": 5.0},
    [["SPEAKER_01", "SPEAKER_02"]],   # 01 과 02 는 동시에 말했다
)
check("반증이 합병을 따라 옮겨간다", len(chain[2]) == 2, f"{len(chain[2])}명 {chain[2]}")

original_merge, config.MERGE_THRESHOLD = config.MERGE_THRESHOLD, 0.0
try:
    off = stitch.collapse(split_turns, split_emb, dict(split_speech), [])
    check("MERGE_THRESHOLD=0 이면 아무것도 안 묶는다", len(off[2]) == 3 and not off[3])
finally:
    config.MERGE_THRESHOLD = original_merge

# 목소리를 모르는 화자는 묶을 근거가 없으니 그대로 남는다
q_turns, q_emb, q_speech, _ = stitch.collapse(
    split_turns + [{"start": 20.0, "end": 23.0, "speaker": "SPEAKER_09"}],
    split_emb, {**split_speech, "SPEAKER_09": 3.0}, [],
)
check("목소리 모르는 화자는 건드리지 않는다", "SPEAKER_09" in q_speech, str(sorted(q_speech)))

# 같은 이름을 붙이면 한 줄로 합쳐지는가 (사용자가 직접 고치는 경로)
from app import render  # noqa: E402

lines = render.merge_lines(
    [{"speaker": "SPEAKER_00", "text": "앞부분", "start": 0.0, "end": 1.0},
     {"speaker": "SPEAKER_02", "text": "뒷부분", "start": 1.0, "end": 2.0}],
    {"SPEAKER_00": "안차돌", "SPEAKER_02": "안차돌"},
)
check("같은 이름을 준 화자는 한 줄로 합쳐진다",
      len(lines) == 1 and lines[0]["text"] == "앞부분 뒷부분", str(lines))

# ── 환각 필터 ─────────────────────────────────────────────────────────
print("\n── 환각 필터 ──")
from app import cleanup  # noqa: E402

def seg(text, start=0.0, end=5.0, score=None):
    """정렬 점수가 있는/없는 세그먼트. score=None 이면 점수 정보가 없는 상황."""
    words = [] if score is None else [{"word": text, "start": start, "end": end, "score": score}]
    return {"text": text, "start": start, "end": end, "words": words}


def verdict(item):
    result = cleanup.inspect(item)
    return (result[0] if result else "keep"), (result[1] if result else "")


# 내용 없는 반복 — 글자만 보고 지운다 (진짜여도 잃을 게 없다)
for text in ["아 아 아 아 아 아 아 아", "아아아아아아아", "네 네 네 네 네 네 네"]:
    action, why = verdict(seg(text))
    check(f"반복은 바로 버린다: {text[:20]}", action == "drop", why)

# 상투구 — 소리가 글자와 맞으면(점수 높음) 지우지 않는다
for text in ["MBC 뉴스 김성현이었습니다", "시청해주셔서 감사합니다",
             "구독과 좋아요 부탁드립니다", "Thanks for watching!"]:
    action, why = verdict(seg(text, score=0.9))
    check(f"상투구인데 소리가 맞으면 남긴다: {text[:22]}", action == "suspect", why)
    action, why = verdict(seg(text, score=0.05))
    check(f"상투구 + 낮은 점수면 버린다: {text[:22]}", action == "drop", why)

# 침묵 30초를 한 문장으로 때우는 전형적 환각 — 점수 정보가 없어도 잡힌다
action, why = verdict(seg("시청해주셔서 감사합니다", 0.0, 30.0))
check("30초를 한 문장으로 때우면 점수 없이도 버린다", action == "drop", why)

# 진짜 발언은 건드리지 않는다
for text in ["감사합니다", "네 이거 좋아요", "다음 시간에 뵙겠습니다",
             "알림 설정 좀 바꿔주세요", "그래서 제가 어제 말씀드린 대로 진행하겠습니다",
             "아 그건 제가 확인해 보겠습니다", "네 네 네 알겠습니다 그러면 그렇게 진행할게요",
             "MBC 뉴스에서 그 얘기 나왔던 거 기억나세요", "구독자 수가 지난달보다 늘었어요"]:
    action, why = verdict(seg(text, score=0.85))
    check(f"진짜 발언은 남긴다: {text[:24]}", action == "keep", why)

# 뉴스 녹음을 전사하는 경우 — 앵커 클로징은 진짜 발언이다
action, why = verdict(seg("MBC 뉴스 김성현이었습니다", 0.0, 2.4, score=0.88))
check("뉴스 전사에서 앵커 클로징은 지우지 않는다", action == "suspect", why)

kept_segs, dropped_segs, suspect_segs = cleanup.clean([
    seg("아 아 아 아 아 아 아 아"),                    # drop
    seg("시청해주셔서 감사합니다", score=0.02),          # drop
    seg("MBC 뉴스 김성현이었습니다", score=0.9),        # suspect (남김)
    seg("그래서 어제 말씀드린 대로 진행하겠습니다", score=0.9),  # keep
])
check("clean() 이 세 갈래로 나눈다",
      (len(kept_segs), len(dropped_segs), len(suspect_segs)) == (2, 2, 1),
      f"남김 {len(kept_segs)} / 버림 {len(dropped_segs)} / 의심 {len(suspect_segs)}")
check("의심 구간은 결과에 그대로 남는다",
      any("MBC" in (s["text"] or "") for s in kept_segs))
check("버린 것과 의심에 이유가 붙는다",
      all(item["reason"] for item in dropped_segs + suspect_segs))
check("경고 문구가 지운 것과 남긴 것을 둘 다 알린다",
      len(cleanup.summary(dropped_segs, suspect_segs)) == 2)

original_suspect, config.DROP_SUSPECT = config.DROP_SUSPECT, True
try:
    _, hard_dropped, hard_suspect = cleanup.clean([seg("MBC 뉴스 김성현이었습니다", score=0.9)])
    check("DROP_SUSPECT=true 면 의심 구간도 버린다",
          len(hard_dropped) == 1 and not hard_suspect)
finally:
    config.DROP_SUSPECT = original_suspect

original_drop, config.DROP_HALLUCINATION = config.DROP_HALLUCINATION, False
try:
    check("DROP_HALLUCINATION=false 면 아무것도 안 버린다",
          cleanup.clean([seg("아 아 아 아 아 아 아")])[1] == [])
finally:
    config.DROP_HALLUCINATION = original_drop

print()
if failures:
    print(f"실패 {len(failures)}건: {', '.join(failures)}")
    raise SystemExit(1)
print("전부 통과.")
