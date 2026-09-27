import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from research_assistant import logger
from research_assistant.config import INDEX_DIR, INDEX_FOLDERS

SUPPORTED = {".pdf", ".epub", ".html", ".htm", ".txt", ".md", ".zip"}
SKIP: set[str] = set()


@dataclass
class FileRecord:
    path: str
    size: int
    mtime: float
    sha256: str
    kind: str
    status: str


def _connect() -> sqlite3.Connection:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDEX_DIR / "manifest.sqlite", timeout=60)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            sha256 TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new',
            error TEXT
        )"""
    )
    return conn


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def scan() -> dict:
    conn = _connect()
    found = 0
    unchanged = 0
    changed = 0
    skipped = 0
    errors = 0
    for folder in INDEX_FOLDERS:
        if not folder.is_dir():
            logger.log("Manifest", f"Folder missing, skipped: {folder}")
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in SKIP:
                skipped += 1
                continue
            if suffix not in SUPPORTED:
                skipped += 1
                continue
            try:
                stat = path.stat()
                row = conn.execute(
                    "SELECT size, mtime, status FROM files WHERE path = ?",
                    (str(path),),
                ).fetchone()
                if row and row[0] == stat.st_size and row[1] == stat.st_mtime:
                    unchanged += 1
                    continue
                sha = _hash_file(path)
                status = "new"
                if row:
                    status = "changed"
                    changed += 1
                else:
                    found += 1
                conn.execute(
                    """INSERT INTO files (path, size, mtime, sha256, kind, status)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(path) DO UPDATE SET
                         size=excluded.size, mtime=excluded.mtime,
                         sha256=excluded.sha256, kind=excluded.kind,
                         status=excluded.status, error=NULL""",
                    (
                        str(path),
                        stat.st_size,
                        stat.st_mtime,
                        sha,
                        suffix.lstrip("."),
                        status,
                    ),
                )
            except OSError as exc:
                errors += 1
                logger.log("Manifest", f"Cannot read {path}: {exc}")
    conn.commit()
    summary = {
        "new": found,
        "changed": changed,
        "unchanged": unchanged,
        "skipped": skipped,
        "errors": errors,
    }
    logger.log("Manifest", f"Scan done: {summary}")
    conn.close()
    return summary


def pending(statuses: tuple[str, ...] = ("new", "changed")) -> list[FileRecord]:
    conn = _connect()
    rows = conn.execute(
        f"SELECT path, size, mtime, sha256, kind, status FROM files "
        f"WHERE status IN ({','.join('?' * len(statuses))}) ORDER BY path",
        statuses,
    ).fetchall()
    conn.close()
    return [FileRecord(*row) for row in rows]


def mark(path: str, status: str, error: str | None = None) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE files SET status = ?, error = ? WHERE path = ?",
        (status, error, path),
    )
    conn.commit()
    conn.close()


def counts() -> dict:
    conn = _connect()
    rows = conn.execute("SELECT status, COUNT(*) FROM files GROUP BY status").fetchall()
    conn.close()
    return dict(rows)
