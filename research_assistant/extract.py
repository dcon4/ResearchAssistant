import html
import io
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from research_assistant import logger

PDF_PAGE_BATCH = 200
MAX_HTML_ENTRY = 20 * 1024 * 1024
MAX_ZIP_MEMBER = 60 * 1024 * 1024
MAX_ZIP_TOTAL = 400 * 1024 * 1024
MAX_ZIP_MEMBERS = 500
ZIP_INNER = (".txt", ".md", ".html", ".htm", ".epub", ".pdf")
ZIP_JUNK = ("__MACOSX", ".DS_Store")
TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class Piece:
    text: str
    location: str


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\x00", " ")
    text = BLANK_RE.sub("\n\n", text)
    return text.strip()


def _strip_tags(markup: str) -> str:
    markup = SCRIPT_RE.sub(" ", markup)
    return TAG_RE.sub("\n", markup)


def _from_soup(markup: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(markup, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text("\n")
    except Exception:
        return _strip_tags(markup)


def _decode_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _text_from_bytes(data: bytes) -> str:
    return _clean(_decode_bytes(data))


def _html_from_bytes(raw: bytes) -> str:
    markup = _decode_bytes(raw)
    if len(raw) > MAX_HTML_ENTRY:
        return _clean(_strip_tags(markup))
    try:
        import html2text

        converter = html2text.HTML2Text()
        converter.ignore_images = True
        converter.body_width = 0
        return _clean(converter.handle(markup))
    except Exception:
        return _clean(_from_soup(markup))


def _title_of(markup: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", markup, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _clean(_strip_tags(match.group(1)))[:120]


def extract_txt(path: Path) -> list[Piece]:
    text = _text_from_bytes(path.read_bytes())
    if not text:
        return []
    return [Piece(text=text, location="full text")]


def extract_html(path: Path) -> list[Piece]:
    raw = path.read_bytes()
    markup = _decode_bytes(raw)
    try:
        import html2text

        converter = html2text.HTML2Text()
        converter.ignore_images = True
        converter.body_width = 0
        text = converter.handle(markup)
    except Exception:
        text = _from_soup(markup)
    text = _clean(text)
    if not text:
        return []
    return [Piece(text=text, location=_title_of(markup) or "full text")]


def _epub_spine(zfile: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        container = zfile.read("META-INF/container.xml").decode("utf-8", "replace")
        match = re.search(r'full-path="([^"]+)"', container)
        rootfile = match.group(1) if match else ""
        opf = zfile.read(rootfile).decode("utf-8", "replace")
        base = str(Path(rootfile).parent)
        if base == ".":
            base = ""
        manifest: dict[str, str] = {}
        for item in re.findall(r"<item[^>]+>", opf):
            id_match = re.search(r'id="([^"]+)"', item)
            href_match = re.search(r'href="([^"]+)"', item)
            if id_match and href_match:
                manifest[id_match.group(1)] = (
                    f"{base}/{href_match.group(1)}" if base else href_match.group(1)
                )
        entries: list[tuple[str, str]] = []
        for idref in re.findall(r'<itemref[^>]+idref="([^"]+)"', opf):
            href = manifest.get(idref)
            if href:
                entries.append((idref, href.replace("\\", "/")))
        return entries
    except KeyError:
        return [
            (name, name)
            for name in zfile.namelist()
            if name.endswith((".xhtml", ".html", ".htm"))
        ]


def _epub_pieces(zfile: zipfile.ZipFile) -> list[Piece]:
    pieces: list[Piece] = []
    for index, (_key, name) in enumerate(_epub_spine(zfile), start=1):
        try:
            raw = zfile.read(name)
        except KeyError:
            continue
        markup = _decode_bytes(raw)
        if len(raw) > MAX_HTML_ENTRY:
            text = _strip_tags(markup)
        else:
            text = _from_soup(markup)
        text = _clean(text)
        if not text:
            continue
        pieces.append(
            Piece(text=text, location=_title_of(markup) or f"chapter {index}")
        )
    return pieces


def extract_epub(path: Path) -> list[Piece]:
    with zipfile.ZipFile(path) as zfile:
        return _epub_pieces(zfile)


def extract_pdf(path: Path) -> list[Piece]:
    info = subprocess.run(
        ["pdfinfo", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    pages = 0
    for line in info.stdout.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split()[1])
            break
    if not pages:
        raise ValueError("pdfinfo returned no page count")
    pieces: list[Piece] = []
    empty_pages = 0
    for start in range(1, pages + 1, PDF_PAGE_BATCH):
        end = min(start + PDF_PAGE_BATCH - 1, pages)
        result = subprocess.run(
            ["pdftotext", "-f", str(start), "-l", str(end), "-layout", str(path), "-"],
            capture_output=True,
            timeout=600,
            check=False,
        )
        text = _clean(result.stdout.decode("utf-8", errors="replace"))
        if not text:
            empty_pages += end - start + 1
            continue
        pieces.append(Piece(text=text, location=f"pages {start}-{end}"))
    if not pieces:
        raise ValueError(f"needs_ocr: no extractable text in {pages} pages")
    if empty_pages:
        logger.verbose("Extract", f"{path.name}: {empty_pages} pages had no text")
    return pieces


def _pdf_pieces_from_bytes(name: str, raw: bytes) -> list[Piece]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(raw)
        temp_path = Path(handle.name)
    try:
        return [
            Piece(text=piece.text, location=f"{name}, {piece.location}")
            for piece in extract_pdf(temp_path)
        ]
    finally:
        temp_path.unlink(missing_ok=True)


def _zip_member_pieces(name: str, suffix: str, raw: bytes) -> list[Piece]:
    if suffix in (".txt", ".md", ".html", ".htm"):
        if suffix in (".txt", ".md"):
            text = _text_from_bytes(raw)
        else:
            text = _html_from_bytes(raw)
        if not text:
            return []
        return [Piece(text=text, location=name)]
    if suffix == ".epub":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                return [
                    Piece(text=piece.text, location=f"{name}, {piece.location}")
                    for piece in _epub_pieces(inner)
                ]
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{name}: not a readable epub ({exc})") from exc
    if suffix == ".pdf":
        return _pdf_pieces_from_bytes(name, raw)
    return []


def extract_zip(path: Path) -> list[Piece]:
    try:
        zfile = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a readable zip: {exc}") from exc
    pieces: list[Piece] = []
    total = 0
    kept = 0
    logged_nested = False
    with zfile:
        for info in zfile.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if any(part in ZIP_JUNK for part in Path(name).parts):
                continue
            suffix = Path(name).suffix.lower()
            if suffix == ".zip":
                if not logged_nested:
                    logger.log("Zip", f"{path.name}: nested zip skipped: {name}")
                    logged_nested = True
                continue
            if suffix not in ZIP_INNER:
                continue
            if kept >= MAX_ZIP_MEMBERS:
                logger.log("Zip", f"{path.name}: stopped at {MAX_ZIP_MEMBERS} files")
                break
            if info.file_size > MAX_ZIP_MEMBER:
                logger.log("Zip", f"{path.name}: {name} is too large, skipped")
                continue
            if total + info.file_size > MAX_ZIP_TOTAL:
                logger.log("Zip", f"{path.name}: size limit reached, skipping {name}")
                continue
            try:
                raw = zfile.read(info)
            except Exception as exc:
                logger.log("Zip", f"{path.name}: cannot read {name}: {exc}")
                continue
            total += len(raw)
            kept += 1
            try:
                pieces.extend(_zip_member_pieces(name, suffix, raw))
            except Exception as exc:
                logger.log("Zip", f"{path.name}: {name} failed: {exc}")
    if not pieces:
        raise ValueError("no extractable text in any zip file")
    logger.verbose("Zip", f"{path.name}: kept {kept} files, {len(pieces)} sections")
    return pieces


def extract(path: Path) -> list[Piece]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return extract_txt(path)
    if suffix == ".md":
        return extract_txt(path)
    if suffix in (".html", ".htm"):
        return extract_html(path)
    if suffix == ".epub":
        return extract_epub(path)
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".zip":
        return extract_zip(path)
    raise ValueError(f"unsupported type: {suffix}")
