from dataclasses import dataclass

CHUNK_CHARS = 3200
OVERLAP_CHARS = 400
MIN_CHUNK_CHARS = 80


@dataclass
class Chunk:
    text: str
    location: str


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_text(text: str, location: str) -> list[Chunk]:
    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > CHUNK_CHARS:
            if buffer:
                chunks.append(Chunk(text=buffer, location=location))
                buffer = ""
            start = 0
            while start < len(paragraph):
                end = start + CHUNK_CHARS
                piece = paragraph[start:end]
                chunks.append(Chunk(text=piece, location=location))
                start = end - OVERLAP_CHARS
            continue
        if buffer and len(buffer) + len(paragraph) + 2 > CHUNK_CHARS:
            chunks.append(Chunk(text=buffer, location=location))
            buffer = paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
    if buffer and len(buffer) >= MIN_CHUNK_CHARS:
        chunks.append(Chunk(text=buffer, location=location))
    elif buffer and chunks:
        previous = chunks[-1]
        chunks[-1] = Chunk(
            text=f"{previous.text}\n\n{buffer}", location=previous.location
        )
    elif buffer:
        chunks.append(Chunk(text=buffer, location=location))
    return [c for c in chunks if len(c.text.strip()) >= 20]
