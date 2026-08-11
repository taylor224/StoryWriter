"""Per-stage intermediate result cache.

Transcription is by far the most expensive stage here. Redoing it because
diarization failed afterwards is pure waste. So every stage writes its output to
disk, and a retry skips any stage whose inputs are unchanged.

Each cache entry stores a "key" alongside the value. The key is whatever inputs
determine that stage's result. If any part of the key differs the cache is
ignored and the stage recomputes — that is what stops an old result from
surviving a retry with a different language or prompt.

If you ever change the *shape* of a stage's output, add a version value to that
stage's key. Otherwise new code will happily read caches in the old format.
"""

import json
from pathlib import Path
from typing import Any

from . import config


def _dir(name: str) -> Path:
    return config.CACHE_DIR / name


def _path(name: str, stage: str) -> Path:
    return _dir(name) / f"{stage}.json"


def audio_key(path: Path) -> dict[str, Any]:
    """Identifies the source audio. Changes if a different file is uploaded under the same name."""
    try:
        stat = path.stat()
    except OSError:
        return {"file": path.name, "size": 0, "mtime": 0}
    return {"file": path.name, "size": stat.st_size, "mtime": int(stat.st_mtime)}


def load(name: str, stage: str, key: dict[str, Any]) -> dict[str, Any] | None:
    """Returns the cached value only when the key matches. None if missing or stale."""
    path = _path(name, stage)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("key") != key:
        return None
    return payload.get("value")


def _jsonable(obj: Any) -> Any:
    """whisperx output contains numpy scalars, which do not serialize as-is."""
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize value of type {type(obj).__name__}")


def save(name: str, stage: str, key: dict[str, Any], value: Any) -> None:
    path = _path(name, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"key": key, "value": value}, ensure_ascii=False, default=_jsonable),
        encoding="utf-8",
    )
    tmp.replace(path)  # so a crash mid-write cannot leave half a cache behind


def stages(name: str) -> list[str]:
    """Which cache stages still exist for this job."""
    folder = _dir(name)
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.json"))


def clear(name: str) -> None:
    folder = _dir(name)
    if not folder.is_dir():
        return
    for item in folder.iterdir():
        item.unlink(missing_ok=True)
    folder.rmdir()
