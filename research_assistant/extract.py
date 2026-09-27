import html
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from research_assistant import logger

PDF_PAGE_BATCH = 200
MAX_HTML_ENTRY = 20 * 1024 * 1024
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


def extract_txt(path: Path) -> list[Piece]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    text = _clean(text)
    if not text:
        return []
    return [Piece(text=text, location="full text")]


def extract_html(path: Path) -> list[Piece]:
    data = path.read_bytes()
    try:
        markup = data.decode("utf-8")
    except UnicodeDecodeError:
        markup = data.decode("latin-1", errors="replace")
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
    title = ""
    match = re.search(r"<title[^>]*>(.*?)</title>", markup, re.IGNORECASE | re.DOTALL)
    if match:
        title = _clean(_strip_tags(match.group(1)))[:120]
    return [Piece(text=text, location=title or "full text")]


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


def extract_epub(path: Path) -> list[Piece]:
    pieces: list[Piece] = []
    with zipfile.ZipFile(path) as zfile:
        for index, (_key, name) in enumerate(_epub_spine(zfile), start=1):
            try:
                raw = zfile.read(name)
            except KeyError:
                continue
            markup = raw.decode("utf-8", errors="replace")
            if len(raw) > MAX_HTML_ENTRY:
                text = _strip_tags(markup)
            else:
                text = _from_soup(markup)
            text = _clean(text)
            if not text:
                continue
            title = ""
            match = re.search(
                r"<title[^>]*>(.*?)</title>", markup, re.IGNORECASE | re.DOTALL
            )
            if match:
                title = _clean(_strip_tags(match.group(1)))[:120]
            location = title or f"chapter {index}"
            pieces.append(Piece(text=text, location=location))
    return pieces


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
    raise ValueError(f"unsupported type: {suffix}")
