from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research_assistant import logger, store
from research_assistant.embed import backend_info, cached_embedder

DOC_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf", ".epub"}


@dataclass
class Source:
    file_path: str
    location: str
    text: str
    score: float
    stage: str


def document_key(source: Source) -> str:
    location = source.location
    if "/" in location:
        tail = location.rsplit("/", 1)[-1]
        if Path(tail).suffix.lower() in DOC_SUFFIXES:
            return f"{source.file_path}::{location}"
    return source.file_path


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ query


def search(question: str, limit: int = 8) -> list[Source]:
    conn = store.connect()
    try:
        fts_ids = store.fts_search(conn, question, limit=500)
        info = backend_info()
        meta = store.get_embed_meta(conn)
        candidates: dict[int, float] = {}
        stage = "keyword"
        use_vectors = (
            meta.get("mode") == info["mode"] and meta.get("model") == info["model"]
        )
        if use_vectors and fts_ids:
            try:
                embedder = cached_embedder(info["mode"])
                query_vec = embedder.embed([question])[0]
                vectors = store.fetch_chunk_vectors(conn, fts_ids)
                if vectors:
                    ids = sorted(vectors)
                    matrix = np.stack([vectors[i] for i in ids])
                    scores = _cosine_scores(query_vec, matrix)
                    for chunk_id, score in zip(ids, scores, strict=True):
                        candidates[chunk_id] = float(score)
                    stage = "keyword+vector"
            except Exception as exc:
                logger.log("Search", f"Vector stage failed, keyword only: {exc}")
        if not candidates:
            for chunk_id in fts_ids:
                rank_score = 1.0 / (1.0 + len(candidates))
                candidates[chunk_id] = rank_score
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        top = ranked[:limit]
        texts = store.fetch_chunks(conn, [chunk_id for chunk_id, _ in top])
        sources = []
        for chunk_id, score in top:
            row = texts.get(chunk_id)
            if not row:
                continue
            file_path, location, text = row
            sources.append(
                Source(
                    file_path=file_path,
                    location=location,
                    text=text,
                    score=score,
                    stage=stage,
                )
            )
        logger.verbose(
            "Search",
            f"Query returned {len(sources)} sources "
            f"(fts candidates: {len(fts_ids)}, stage: {stage})",
        )
        return sources
    finally:
        conn.close()
