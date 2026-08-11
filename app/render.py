"""결과 저장·재생성.

data/results/<이름>.txt  — 요청 포맷 그대로 (`이름 : 내용`)
data/results/<이름>.json — 타임스탬프·임베딩·세그먼트 원본. 화자 이름만 바꿔
                           txt 를 다시 만들 수 있으므로 재전사가 필요 없다.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config

_INVALID_CHARS = set('\\/:*?"<>|')
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}


# ── 이름 처리 ─────────────────────────────────────────────────────────
def default_name() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def sanitize_name(name: str | None) -> str:
    raw = (name or "").strip()
    cleaned = "".join(
        "_" if (ch in _INVALID_CHARS or ord(ch) < 32) else ch for ch in raw
    ).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)[:100].strip()
    if not cleaned or cleaned.upper() in _RESERVED:
        return default_name()
    return cleaned


def unique_name(name: str) -> str:
    """이미 있는 이름이면 -2, -3 을 붙인다."""
    base = sanitize_name(name)
    candidate, index = base, 1
    while (config.RESULT_DIR / f"{candidate}.json").exists():
        index += 1
        candidate = f"{base}-{index}"
    return candidate


def txt_path(name: str) -> Path:
    return config.RESULT_DIR / f"{sanitize_name(name)}.txt"


def json_path(name: str) -> Path:
    return config.RESULT_DIR / f"{sanitize_name(name)}.json"


# ── 화자 표시 이름 ────────────────────────────────────────────────────
def anonymous_label(index: int) -> str:
    """0 -> 화자A, 25 -> 화자Z, 26 -> 화자AA"""
    letters = ""
    index += 1
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return f"화자{letters}"


def assign_displays(
    ordered_labels: list[str],
    matches: dict[str, dict[str, Any]],
    anon: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """매칭된 화자는 실명, 나머지는 등장 순서대로 화자A/B/C.

    anon 에 이미 부여된 익명 라벨을 넘기면 그대로 재사용한다. 한 명에게 이름을
    지정했다고 나머지 화자의 번호가 밀리면 안 되기 때문.

    반환: (표시이름, 갱신된 익명 라벨 맵)
    """
    anon = dict(anon or {})
    taken = {m["name"] for m in matches.values() if m.get("matched") and m.get("name")}
    taken |= set(anon.values())

    displays: dict[str, str] = {}
    counter = 0
    for label in ordered_labels:
        info = matches.get(label) or {}
        if info.get("matched") and info.get("name"):
            displays[label] = info["name"]
            continue
        if label in anon:
            displays[label] = anon[label]
            continue
        candidate = anonymous_label(counter)
        while candidate in taken:
            counter += 1
            candidate = anonymous_label(counter)
        anon[label] = candidate
        taken.add(candidate)
        displays[label] = candidate
        counter += 1
    return displays, anon


def order_labels(segments: list[dict], speech_sec: dict[str, float]) -> list[str]:
    """첫 등장 시각 순. 세그먼트에 안 나온 화자는 뒤에 붙인다."""
    seen: list[str] = []
    for seg in segments:
        label = seg.get("speaker")
        if label and label not in seen:
            seen.append(label)
    for label in speech_sec:
        if label not in seen:
            seen.append(label)
    return seen


# ── 텍스트 생성 ───────────────────────────────────────────────────────
def merge_lines(
    segments: list[dict], displays: dict[str, str], unknown: str = "화자?"
) -> list[dict[str, Any]]:
    """연속된 같은 화자의 세그먼트를 한 줄로 합친다.

    라벨이 아니라 표시 이름으로 비교한다. 한 사람이 여러 화자로 쪼개져 나왔을 때
    사용자가 둘에 같은 이름을 붙이면, 그것만으로 한 줄로 합쳐져야 하기 때문.
    """
    lines: list[dict[str, Any]] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        label = seg.get("speaker")
        name = displays.get(label, unknown) if label else unknown
        if lines and lines[-1]["name"] == name:
            lines[-1]["text"] = f"{lines[-1]['text']} {text}".strip()
            lines[-1]["end"] = float(seg.get("end", lines[-1]["end"]) or lines[-1]["end"])
            continue
        lines.append(
            {
                "speaker": label,
                "name": name,
                "text": text,
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
            }
        )
    return lines


def render_txt(lines: list[dict[str, Any]]) -> str:
    return "".join(f"{line['name']} : {line['text']}\n" for line in lines)


def timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


# ── 파일 입출력 ───────────────────────────────────────────────────────
def save(payload: dict[str, Any]) -> tuple[Path, Path]:
    name = payload["name"]
    displays = {label: info["display"] for label, info in payload["speakers"].items()}
    lines = merge_lines(payload["segments"], displays)
    payload["lines"] = lines

    jpath, tpath = json_path(name), txt_path(name)
    jpath.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tpath.write_text(render_txt(lines), encoding="utf-8")
    return tpath, jpath


def load(name: str) -> dict[str, Any] | None:
    path = json_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete(name: str) -> bool:
    removed = False
    for path in (json_path(name), txt_path(name)):
        if path.exists():
            path.unlink()
            removed = True
    return removed


def list_results() -> list[dict[str, Any]]:
    items = []
    for path in config.RESULT_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items.append(
            {
                "name": payload.get("name", path.stem),
                "created_at": payload.get("created_at", ""),
                "duration": payload.get("duration", 0.0),
                "language": payload.get("language", ""),
                "source_file": payload.get("source_file", ""),
                "speakers": [
                    info.get("display", "")
                    for info in (payload.get("speakers") or {}).values()
                ],
                "line_count": len(payload.get("lines") or []),
            }
        )
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items
