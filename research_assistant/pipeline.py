import fcntl
from contextlib import ExitStack, contextmanager
from pathlib import Path

from research_assistant import logger, manifest, store
from research_assistant.chunk import chunk_text
from research_assistant.config import INDEX_DIR, load_settings
from research_assistant.embed import backend_info, cached_embedder
from research_assistant.extract import extract


@contextmanager
def pipeline_lock():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = INDEX_DIR / "pipeline.lock"
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another scan or embed is already running") from None
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def index_file(conn, file_path: str) -> int:
    path = Path(file_path)
    try:
        pieces = extract(path)
    except Exception as exc:
        manifest.mark(file_path, "error", str(exc)[:500])
        logger.log("Index", f"Extract failed for {path.name}: {exc}")
        return 0
    if not pieces:
        manifest.mark(file_path, "error", "no extractable text")
        logger.log("Index", f"No text in {path.name}")
        return 0
    items: list[tuple[str, str]] = []
    for piece in pieces:
        for chunk in chunk_text(piece.text, piece.location):
            items.append((chunk.location, chunk.text))
    if not items:
        manifest.mark(file_path, "error", "no chunks produced")
        return 0
    old_ids = store.delete_file_chunks(conn, file_path)
    store.delete_vectors(conn, old_ids)
    ids = store.insert_chunks(conn, file_path, items)
    conn.commit()
    manifest.mark(file_path, "indexed")
    logger.log(
        "Index",
        f"{path.name}: {len(pieces)} sections -> {len(items)} chunks"
        + (f" (dropped {len(old_ids)} old)" if old_ids else ""),
    )
    return len(ids)


def embed_pending(conn) -> tuple[int, int]:
    info = backend_info()
    meta = store.get_embed_meta(conn)
    if meta and meta.get("mode") != info["mode"]:
        raise RuntimeError(
            f"Index was built with {meta.get('mode')} embeddings but settings "
            f"now say {info['mode']}. Run 'reembed' to rebuild vectors."
        )
    store.ensure_vec_table(conn, info["dim"])
    rows = conn.execute(
        "SELECT c.id, c.text FROM chunks c "
        "LEFT JOIN chunks_vec v ON v.rowid = c.id WHERE v.rowid IS NULL"
    ).fetchall()
    if not rows:
        logger.verbose("Embed", "No chunks need embeddings")
        return 0, 0
    embedder = cached_embedder(info["mode"])
    done = 0
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        try:
            vectors = embedder.embed([text for _id, text in batch])
        except Exception as exc:
            logger.log("Embed", f"Embedding failed after {done} chunks: {exc}")
            break
        store.insert_vectors(conn, [row[0] for row in batch], vectors)
        conn.commit()
        done += len(batch)
        logger.verbose("Embed", f"Embedded {done}/{len(rows)} chunks")
    if done and done == len(rows):
        store.set_embed_meta(conn, info["model"], info["dim"], info["mode"])
        conn.commit()
    elif done:
        logger.log(
            "Embed",
            f"Embedding incomplete ({done}/{len(rows)}); "
            "search will stay keyword-only until a full run finishes",
        )
    return done, len(rows)


def rescan() -> dict:
    settings = load_settings()
    stack = ExitStack()
    try:
        stack.enter_context(pipeline_lock())
    except RuntimeError as exc:
        logger.log("Pipeline", str(exc))
        return {"skipped": str(exc)}
    with stack:
        summary = manifest.scan()
        conn = store.connect()
        pending = manifest.pending()
        indexed = 0
        for record in pending:
            indexed += 1 if index_file(conn, record.path) else 0
        conn.commit()
        embedded, wanted = 0, 0
        if settings.get("rescan_timer", True):
            try:
                embedded, wanted = embed_pending(conn)
            except RuntimeError as exc:
                logger.log("Embed", str(exc))
        conn.close()
    result = {
        "scan": summary,
        "indexed": indexed,
        "embedded": embedded,
        "to_embed": wanted,
    }
    logger.log("Pipeline", f"Rescan complete: {result}")
    return result


def reembed() -> dict:
    stack = ExitStack()
    try:
        stack.enter_context(pipeline_lock())
    except RuntimeError as exc:
        logger.log("Pipeline", str(exc))
        return {"skipped": str(exc)}
    with stack:
        conn = store.connect()
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        if has_table:
            conn.execute("DROP TABLE chunks_vec")
            logger.verbose("Pipeline", "Old vector table dropped")
        conn.commit()
        embedded, wanted = embed_pending(conn)
        conn.close()
    result = {"embedded": embedded, "to_embed": wanted}
    logger.log("Pipeline", f"Full re-embed complete: {result}")
    return result
