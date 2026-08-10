"""단계별 중간 결과 캐시.

전사는 이 파이프라인에서 가장 비싼 단계다. 뒤쪽 화자 분리에서 실패했다고
전사를 다시 돌리는 건 낭비다. 각 단계의 출력을 디스크에 남겨 두고, 재시도할 때
입력이 그대로면 건너뛴다.

각 캐시는 "키"를 함께 저장한다. 키는 그 단계의 결과를 좌우하는 입력들이다.
키가 하나라도 다르면 캐시를 무시하고 다시 계산한다. 언어나 프롬프트를 바꿔
재시도했는데 옛 결과가 살아남는 사고를 막기 위해서다.
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
    """원본 오디오를 식별하는 값. 같은 이름으로 다른 파일을 올리면 달라진다."""
    try:
        stat = path.stat()
    except OSError:
        return {"file": path.name, "size": 0, "mtime": 0}
    return {"file": path.name, "size": stat.st_size, "mtime": int(stat.st_mtime)}


def load(name: str, stage: str, key: dict[str, Any]) -> dict[str, Any] | None:
    """키가 일치하는 캐시만 돌려준다. 없거나 어긋나면 None."""
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
    """whisperx 출력에는 numpy 스칼라가 섞여 있어 그대로는 직렬화되지 않는다."""
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"직렬화할 수 없는 값: {type(obj).__name__}")


def save(name: str, stage: str, key: dict[str, Any], value: Any) -> None:
    path = _path(name, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"key": key, "value": value}, ensure_ascii=False, default=_jsonable),
        encoding="utf-8",
    )
    tmp.replace(path)  # 쓰다 죽어도 반쪽짜리 캐시가 남지 않도록


def stages(name: str) -> list[str]:
    """이 작업에 남아 있는 캐시 단계 목록."""
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
