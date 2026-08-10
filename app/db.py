"""SQLite 저장소: 화자 / 보이스프린트 / 작업 / 설정.

임베딩은 float32 256차원을 L2 정규화한 뒤 raw bytes 로 BLOB 저장한다.
연결은 호출마다 새로 열고 닫는다 (워커 스레드 + FastAPI 스레드 동시 접근 대응).
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from . import config

_write_lock = threading.Lock()

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS speakers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    note       TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voiceprints (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    source     TEXT DEFAULT '',
    speech_sec REAL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vp_speaker ON voiceprints(speaker_id);

CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,
    stage       TEXT DEFAULT '',
    progress    REAL DEFAULT 0,
    params      TEXT DEFAULT '{}',
    error       TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    finished_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with _write_lock, connect() as conn:
        conn.executescript(SCHEMA)


# ── 벡터 직렬화 ────────────────────────────────────────────────────────
def normalize(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def pack(vec) -> bytes:
    return normalize(vec).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# ── 화자 ──────────────────────────────────────────────────────────────
def list_speakers() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.*,
                   COUNT(v.id)                     AS voiceprint_count,
                   COALESCE(SUM(v.speech_sec), 0)  AS total_speech
            FROM speakers s
            LEFT JOIN voiceprints v ON v.speaker_id = s.id
            GROUP BY s.id
            ORDER BY s.name COLLATE NOCASE
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_speaker(speaker_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM speakers WHERE id=?", (speaker_id,)).fetchone()
    return dict(row) if row else None


def get_speaker_by_name(name: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM speakers WHERE name=? COLLATE NOCASE", (name.strip(),)
        ).fetchone()
    return dict(row) if row else None


def upsert_speaker(name: str, note: str = "") -> int:
    """이름으로 화자를 찾고 없으면 만든다. 화자 id 반환."""
    name = name.strip()
    if not name:
        raise ValueError("화자 이름이 비어 있습니다.")
    existing = get_speaker_by_name(name)
    if existing:
        return int(existing["id"])
    with _write_lock, connect() as conn:
        cur = conn.execute(
            "INSERT INTO speakers (name, note, created_at, updated_at) VALUES (?,?,?,?)",
            (name, note, now(), now()),
        )
        return int(cur.lastrowid)


def rename_speaker(speaker_id: int, name: str) -> None:
    name = name.strip()
    if not name:
        raise ValueError("화자 이름이 비어 있습니다.")
    with _write_lock, connect() as conn:
        conn.execute(
            "UPDATE speakers SET name=?, updated_at=? WHERE id=?",
            (name, now(), speaker_id),
        )


def delete_speaker(speaker_id: int) -> None:
    with _write_lock, connect() as conn:
        conn.execute("DELETE FROM speakers WHERE id=?", (speaker_id,))


# ── 보이스프린트 ───────────────────────────────────────────────────────
def add_voiceprint(speaker_id: int, vector, source: str = "", speech_sec: float = 0.0) -> None:
    """벡터 추가 후 화자당 MAX_VOICEPRINTS 개만 남기고 오래된 것부터 삭제."""
    blob = pack(vector)
    with _write_lock, connect() as conn:
        conn.execute(
            """INSERT INTO voiceprints (speaker_id, vector, dim, source, speech_sec, created_at)
               VALUES (?,?,?,?,?,?)""",
            (speaker_id, blob, len(blob) // 4, source, float(speech_sec), now()),
        )
        conn.execute(
            """DELETE FROM voiceprints
               WHERE speaker_id = ?
                 AND id NOT IN (
                     SELECT id FROM voiceprints
                     WHERE speaker_id = ?
                     ORDER BY id DESC LIMIT ?
                 )""",
            (speaker_id, speaker_id, config.MAX_VOICEPRINTS),
        )
        conn.execute("UPDATE speakers SET updated_at=? WHERE id=?", (now(), speaker_id))


def load_profiles() -> list[dict[str, Any]]:
    """매칭용 프로필 목록: [{id, name, vectors: np.ndarray (n, dim)}]"""
    with connect() as conn:
        rows = conn.execute(
            """SELECT s.id, s.name, v.vector
               FROM speakers s JOIN voiceprints v ON v.speaker_id = s.id
               ORDER BY s.id"""
        ).fetchall()

    buckets: dict[int, dict[str, Any]] = {}
    for row in rows:
        entry = buckets.setdefault(
            int(row["id"]), {"id": int(row["id"]), "name": row["name"], "_vecs": []}
        )
        entry["_vecs"].append(unpack(row["vector"]))

    profiles = []
    for entry in buckets.values():
        vecs = entry.pop("_vecs")
        dims = {v.shape[0] for v in vecs}
        if len(dims) > 1:
            # 임베딩 모델이 바뀐 경우 — 최신 차원만 사용
            latest_dim = vecs[-1].shape[0]
            vecs = [v for v in vecs if v.shape[0] == latest_dim]
        entry["vectors"] = np.vstack(vecs)
        profiles.append(entry)
    return profiles


def voiceprints_of(speaker_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, source, speech_sec, created_at FROM voiceprints WHERE speaker_id=? ORDER BY id DESC",
            (speaker_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_voiceprints_from_source(source: str) -> None:
    """같은 결과에서 두 번 등록되는 것을 막기 위해 기존 항목 제거."""
    with _write_lock, connect() as conn:
        conn.execute("DELETE FROM voiceprints WHERE source=?", (source,))


# ── 작업 ──────────────────────────────────────────────────────────────
def create_job(name: str, filename: str, params: dict[str, Any]) -> int:
    with _write_lock, connect() as conn:
        cur = conn.execute(
            """INSERT INTO jobs (name, filename, status, stage, progress, params, created_at)
               VALUES (?,?,'queued','대기 중',0,?,?)""",
            (name, filename, json.dumps(params, ensure_ascii=False), now()),
        )
        return int(cur.lastrowid)


def update_job(job_id: int, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with _write_lock, connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def get_job(job_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def reset_stale_jobs() -> None:
    """서버가 작업 도중 죽었으면 running 상태가 남는다. 시작 시 정리."""
    with _write_lock, connect() as conn:
        conn.execute(
            """UPDATE jobs SET status='error', error='서버가 재시작되어 작업이 중단되었습니다.',
                               finished_at=? WHERE status IN ('queued','running')""",
            (now(),),
        )


def name_taken(name: str) -> bool:
    """아직 결과 파일이 없는 대기/진행 중 작업이 이 이름을 쓰고 있는지."""
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE name=? AND status IN ('queued','running') LIMIT 1",
            (name,),
        ).fetchone()
    return row is not None


# ── 설정 ──────────────────────────────────────────────────────────────
def get_setting(key: str, default: str = "") -> str:
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _write_lock, connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def speaker_names() -> Iterable[str]:
    return [s["name"] for s in list_speakers()]
