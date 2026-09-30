import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research_assistant import logger, store
from research_assistant.embed import backend_info, cached_embedder

DOC_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf", ".epub"}
HINT_WEIGHT = 0.15
LANE_SIZE = 60
LETTER_HINT = re.compile(
    r"\b(?:starts?|begins?|starting|beginning)\s+with\s+"
    r"(?:(?:the\s+)?(?:letter|character)\s+|['\"])"
    r"([a-z])\b",
    re.IGNORECASE,
)
MATCHING_WORD = r"\b{letter}[a-z]{{2,}}\b"
FILLER = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "this",
    "that",
    "these",
    "those",
    "there",
    "here",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "not",
    "no",
    "yes",
    "do",
    "does",
    "did",
    "done",
    "doing",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "shall",
    "may",
    "might",
    "must",
    "about",
    "with",
    "without",
    "for",
    "from",
    "into",
    "over",
    "under",
    "up",
    "down",
    "out",
    "off",
    "again",
    "more",
    "most",
    "some",
    "any",
    "all",
    "each",
    "every",
    "both",
    "few",
    "many",
    "much",
    "just",
    "only",
    "also",
    "too",
    "either",
    "really",
    "looking",
    "find",
    "found",
    "search",
    "searching",
    "want",
    "wants",
    "need",
    "needs",
    "trying",
    "try",
    "get",
    "got",
    "make",
    "made",
    "say",
    "said",
    "tell",
    "told",
    "one",
    "two",
    "thing",
    "things",
    "else",
    "starts",
    "starting",
    "begins",
    "beginning",
    "letter",
    "character",
}


def extract_letter_hint(text: str) -> str | None:
    match = LETTER_HINT.search(text)
    if not match:
        return None
    letter = match.group(1).lower()
    if letter in {"a", "i"} and not re.search(
        r"letter|character|['\"]", match.group(0), re.I
    ):
        return None
    return letter


def _content_terms(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9']+", text.lower())
        if word not in FILLER and len(word) > 2
    ]


def _rarity_bonus(
    conn: sqlite3.Connection, texts: dict[int, str], letter: str
) -> dict[int, float]:
    pattern = re.compile(MATCHING_WORD.format(letter=re.escape(letter)), re.I)
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    df_cache: dict[str, int] = {}

    def document_frequency(word: str) -> int:
        if word not in df_cache:
            try:
                df_cache[word] = conn.execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                    (word,),
                ).fetchone()[0]
            except sqlite3.OperationalError:
                df_cache[word] = total
        return df_cache[word]

    bonus: dict[int, float] = {}
    for chunk_id, text in texts.items():
        words = {word.lower() for word in pattern.findall(text)}
        if not words:
            bonus[chunk_id] = 0.0
            continue
        rarest = min(words, key=document_frequency)
        idf = math.log((total + 1) / (document_frequency(rarest) + 1))
        bonus[chunk_id] = HINT_WEIGHT * min(1.0, idf / 8.0)
    return bonus


@dataclass
class Source:
    file_path: str
    location: str
    text: str
    score: float
    stage: str
    lane: bool = False


def document_key(source: Source) -> str:
    location = source.location
    if "/" in location:
        tail = location.rsplit("/", 1)[-1]
        if Path(tail).suffix.lower() in DOC_SUFFIXES:
            return f"{source.file_path}::{location}"
    return source.file_path


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ query


def is_private_path(file_path: str, private_folder: str) -> bool:
    if not private_folder:
        return False
    path = Path(file_path)
    root = Path(private_folder)
    return path == root or root in path.parents


def search(
    question: str,
    limit: int = 8,
    scope: str = "public",
    private_folder: str = "",
) -> list[Source]:
    conn = store.connect()
    try:
        hint = extract_letter_hint(question)
        terms = _content_terms(question)
        fts_ids = store.fts_search(conn, question, limit=500)
        lane_ids = store.fts_search(conn, " ".join(terms), limit=LANE_SIZE)
        hint_ids: list[int] = []
        if hint and terms:
            match = f"{hint}* AND (" + " OR ".join(f'"{term}"' for term in terms) + ")"
            hint_ids = store.fts_match(conn, match, limit=LANE_SIZE)
        info = backend_info()
        meta = store.get_embed_meta(conn)
        wanted = list(dict.fromkeys([*fts_ids, *lane_ids, *hint_ids]))
        excluded = 0
        if wanted:
            paths = store.fetch_chunk_paths(conn, wanted)
            if scope == "private":
                allowed = {
                    chunk_id
                    for chunk_id, file_path in paths.items()
                    if is_private_path(file_path, private_folder)
                }
            else:
                allowed = {
                    chunk_id
                    for chunk_id, file_path in paths.items()
                    if not is_private_path(file_path, private_folder)
                }
            excluded = len(wanted) - len(allowed)
            fts_ids = [chunk_id for chunk_id in fts_ids if chunk_id in allowed]
            lane_ids = [chunk_id for chunk_id in lane_ids if chunk_id in allowed]
            hint_ids = [chunk_id for chunk_id in hint_ids if chunk_id in allowed]
            wanted = [chunk_id for chunk_id in wanted if chunk_id in allowed]
        candidates: dict[int, float] = {}
        stage = "keyword"
        use_vectors = (
            meta.get("mode") == info["mode"] and meta.get("model") == info["model"]
        )
        if use_vectors and wanted:
            try:
                embedder = cached_embedder(info["mode"])
                query_vec = embedder.embed([question])[0]
                vectors = store.fetch_chunk_vectors(conn, wanted)
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
            for chunk_id in wanted:
                rank_score = 1.0 / (1.0 + len(candidates))
                candidates[chunk_id] = rank_score
        if hint and hint_ids:
            texts = store.fetch_chunks(conn, hint_ids)
            bonus = _rarity_bonus(
                conn, {chunk_id: row[2] for chunk_id, row in texts.items()}, hint
            )
            for chunk_id, extra in bonus.items():
                candidates[chunk_id] = candidates.get(chunk_id, 0.0) + extra
            logger.verbose(
                "Search",
                f"Letter hint '{hint.upper()}' lane: {len(hint_ids)} passages, "
                f"rarity bonus on {sum(1 for value in bonus.values() if value)}",
            )
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        lane = set(lane_ids)
        hinted = set(hint_ids)
        priority_ids = lane | hinted
        selected = ranked[:limit]
        extras = [
            (chunk_id, score)
            for chunk_id, score in ranked[limit:]
            if chunk_id in priority_ids
        ]
        top = selected + extras
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
                    lane=chunk_id in priority_ids,
                )
            )
        logger.verbose(
            "Search",
            f"Query returned {len(sources)} sources "
            f"(scope: {scope}, excluded: {excluded}, "
            f"fts: {len(fts_ids)}, lane: {len(lane_ids)}, "
            f"hint lane: {len(hint_ids)}, extras kept: {len(extras)}, "
            f"stage: {stage})",
        )
        return sources
    finally:
        conn.close()
