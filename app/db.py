"""SQLite store: speakers / voiceprints / jobs / settings.

Embeddings are L2-normalized float32 vectors stored as raw bytes in a BLOB.
Connections are opened and closed per call, since the worker thread and the
FastAPI threads both touch the database.
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


# ── Vector serialization ──────────────────────────────────────────────
def normalize(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def pack(vec) -> bytes:
    return normalize(vec).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# ── Speakers ──────────────────────────────────────────────────────────
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
    """Find a speaker by name, creating them if absent. Returns the speaker id."""
    name = name.strip()
    if not name:
        raise ValueError("Speaker name is empty.")
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
        raise ValueError("Speaker name is empty.")
    with _write_lock, connect() as conn:
        conn.execute(
            "UPDATE speakers SET name=?, updated_at=? WHERE id=?",
            (name, now(), speaker_id),
        )


def delete_speaker(speaker_id: int) -> None:
    with _write_lock, connect() as conn:
        conn.execute("DELETE FROM speakers WHERE id=?", (speaker_id,))


# ── Voiceprints ───────────────────────────────────────────────────────
def add_voiceprint(speaker_id: int, vector, source: str = "", speech_sec: float = 0.0) -> None:
    """Add a vector, then trim to MAX_VOICEPRINTS per speaker, oldest first."""
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
    """Profiles for matching: [{id, name, vectors: np.ndarray (n, dim)}]"""
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
            # The embedding model changed — keep only the newest dimensionality
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
    """Remove existing entries so the same result cannot enroll twice."""
    with _write_lock, connect() as conn:
        conn.execute("DELETE FROM voiceprints WHERE source=?", (source,))


# ── Jobs ──────────────────────────────────────────────────────────────
def create_job(name: str, filename: str, params: dict[str, Any]) -> int:
    with _write_lock, connect() as conn:
        cur = conn.execute(
            """INSERT INTO jobs (name, filename, status, stage, progress, params, created_at)
               VALUES (?,?,'queued','Queued',0,?,?)""",
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
    """A crash mid-job leaves rows stuck in running. Clean them up at startup."""
    with _write_lock, connect() as conn:
        conn.execute(
            """UPDATE jobs SET status='error', error='The server restarted, so this job was interrupted.',
                               finished_at=? WHERE status IN ('queued','running')""",
            (now(),),
        )


def name_taken(name: str) -> bool:
    """Is a queued or running job already using this name (no result file yet)?"""
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE name=? AND status IN ('queued','running') LIMIT 1",
            (name,),
        ).fetchone()
    return row is not None


# ── Settings ──────────────────────────────────────────────────────────
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
