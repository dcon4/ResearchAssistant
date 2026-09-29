import re
import sqlite3

import numpy as np

from research_assistant import logger
from research_assistant.config import INDEX_DIR


def connect() -> sqlite3.Connection:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDEX_DIR / "search.sqlite", timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            location TEXT NOT NULL,
            text TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, file_path, location, tokenize='porter'
        )"""
    )
    return conn


def _vec_load(conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec

        sqlite_vec.load(conn)
    except ImportError:
        logger.log("Store", "sqlite-vec not installed; vector search disabled")


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    _vec_load(conn)
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    if row and row[0]:
        match = re.search(r"float\[(\d+)\]", row[0])
        if match and int(match.group(1)) == dim:
            return
        conn.execute("DROP TABLE chunks_vec")
        logger.verbose("Store", f"Old vector table dropped ({row[0]})")
    conn.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{dim}])")
    logger.verbose("Store", f"Vector table ready, dim={dim}")


def set_embed_meta(conn: sqlite3.Connection, model: str, dim: int, mode: str) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embed_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    for key, value in (("model", model), ("dim", str(dim)), ("mode", mode)):
        conn.execute(
            "INSERT INTO embed_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_embed_meta(conn: sqlite3.Connection) -> dict:
    try:
        rows = conn.execute("SELECT key, value FROM embed_meta").fetchall()
    except sqlite3.OperationalError:
        return {}
    return dict(rows)


def delete_file_chunks(conn: sqlite3.Connection, file_path: str) -> list[int]:
    ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchall()
    ]
    if not ids:
        return []
    conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", ids)
    return ids


def insert_chunks(
    conn: sqlite3.Connection, file_path: str, items: list[tuple[str, str]]
) -> list[int]:
    ids: list[int] = []
    for location, text in items:
        cursor = conn.execute(
            "INSERT INTO chunks(file_path, location, text) VALUES (?, ?, ?)",
            (file_path, location, text),
        )
        chunk_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text, file_path, location) VALUES (?, ?, ?, ?)",
            (chunk_id, text, file_path, location),
        )
        ids.append(chunk_id)
    return ids


def _vec_table_exists(conn: sqlite3.Connection) -> bool:
    _vec_load(conn)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    return row is not None


def insert_vectors(
    conn: sqlite3.Connection, ids: list[int], vectors: np.ndarray
) -> None:
    if not ids or len(ids) != len(vectors):
        return
    if not _vec_table_exists(conn):
        logger.verbose("Store", "No vector table yet; vectors not stored")
        return
    rows = [
        (chunk_id, vector.astype(np.float32).tobytes())
        for chunk_id, vector in zip(ids, vectors, strict=True)
    ]
    conn.executemany("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", rows)


def delete_vectors(conn: sqlite3.Connection, ids: list[int]) -> None:
    if not ids or not _vec_table_exists(conn):
        return
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", ids)


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 500) -> list[int]:
    terms = [t for t in query.split() if t]
    if not terms:
        return []
    match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
    try:
        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.log("Store", f"FTS query failed: {exc}")
        return []
    return [row[0] for row in rows]


def fts_match(conn: sqlite3.Connection, match: str, limit: int = 200) -> list[int]:
    if not match:
        return []
    try:
        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.log("Store", f"FTS query failed: {exc}")
        return []
    return [row[0] for row in rows]


def vector_search(
    conn: sqlite3.Connection, query_vector: np.ndarray, limit: int = 500
) -> list[tuple[int, float]]:
    _vec_load(conn)
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    if not table:
        return []
    blob = query_vector.astype(np.float32).tobytes()
    rows = conn.execute(
        "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? "
        "ORDER BY distance LIMIT ?",
        (blob, limit),
    ).fetchall()
    return [(row[0], float(row[1])) for row in rows]


def fetch_chunks(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple]:
    if not ids:
        return {}
    result: dict[int, tuple] = {}
    step = 500
    for start in range(0, len(ids), step):
        batch = ids[start : start + step]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT id, file_path, location, text FROM chunks "
            f"WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows:
            result[row[0]] = (row[1], row[2], row[3])
    return result


def fetch_chunk_vectors(
    conn: sqlite3.Connection, ids: list[int]
) -> dict[int, np.ndarray]:
    if not ids:
        return {}
    _vec_load(conn)
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    if not table:
        return {}
    result: dict[int, np.ndarray] = {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT rowid, embedding FROM chunks_vec WHERE rowid IN ({placeholders})",
        ids,
    ).fetchall()
    for rowid, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32)
        result[rowid] = vector
    return result


def stats(conn: sqlite3.Connection) -> dict:
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    file_count = conn.execute(
        "SELECT COUNT(DISTINCT file_path) FROM chunks"
    ).fetchone()[0]
    vector_count = 0
    try:
        _vec_load(conn)
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        if table:
            vector_count = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    except sqlite3.OperationalError:
        pass
    return {"chunks": chunk_count, "files": file_count, "vectors": vector_count}
