import os
import zipfile
from pathlib import Path

import pytest

from research_assistant import manifest, store
from research_assistant.chunk import chunk_text
from research_assistant.extract import extract


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    from research_assistant import config

    index_dir = tmp_path / "index"
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(store, "INDEX_DIR", index_dir)
    monkeypatch.setattr(manifest, "INDEX_DIR", index_dir)
    return tmp_path


def test_chunk_respects_size_limit():
    text = "\n\n".join(f"Paragraph number {i} with a little text." for i in range(400))
    chunks = chunk_text(text, "test location")
    assert len(chunks) > 1
    assert all(len(c.text) <= 3600 for c in chunks)
    assert all(c.location == "test location" for c in chunks)


def test_chunk_tiny_text_makes_single_chunk():
    chunks = chunk_text("Just one short paragraph about testing.", "here")
    assert len(chunks) == 1
    assert "testing" in chunks[0].text


def test_extract_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hello world.\n\nSecond paragraph.", encoding="utf-8")
    pieces = extract(path)
    assert len(pieces) == 1
    assert "Second paragraph" in pieces[0].text


def test_extract_md_as_text(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nBody of the note.", encoding="utf-8")
    pieces = extract(path)
    assert pieces and "Body of the note" in pieces[0].text


def test_extract_html(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>My Page</title></head>"
        "<body><p>Visible words.</p><script>hidden()</script></body></html>",
        encoding="utf-8",
    )
    pieces = extract(path)
    assert pieces
    assert "Visible words" in pieces[0].text
    assert "hidden()" not in pieces[0].text
    assert pieces[0].location == "My Page"


def test_extract_epub(tmp_path):
    path = tmp_path / "mini.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf"/>'
            "</rootfiles></container>",
        )
        archive.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package><manifest>'
            '<item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
            "</manifest>"
            '<spine><itemref idref="c1"/></spine></package>',
        )
        archive.writestr(
            "OEBPS/chapter1.xhtml",
            "<html><head><title>First Chapter</title></head>"
            "<body><p>A journey begins.</p></body></html>",
        )
    pieces = extract(path)
    assert len(pieces) == 1
    assert "journey begins" in pieces[0].text
    assert pieces[0].location == "First Chapter"


def test_scan_indexes_supported_files_only(tmp_path, monkeypatch, isolated_index):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("alpha content", encoding="utf-8")
    (folder / "b.7z").write_bytes(b"7z")
    (folder / "c.sh").write_text("#!/bin/sh", encoding="utf-8")
    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [folder])

    summary = manifest.scan()
    assert summary["new"] == 1
    assert summary["skipped"] == 2

    again = manifest.scan()
    assert again["new"] == 0
    assert again["unchanged"] == 1

    (folder / "a.txt").write_text("changed content", encoding="utf-8")
    third = manifest.scan()
    assert third["changed"] == 1


def test_manifest_counts(tmp_path, monkeypatch, isolated_index):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("one", encoding="utf-8")
    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [folder])
    manifest.scan()
    assert manifest.counts()["new"] == 1


def test_skip_files_are_recorded_and_never_pending(
    tmp_path, monkeypatch, isolated_index
):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "keep.txt").write_text("keep this", encoding="utf-8")
    (folder / "sample_copy_a.epub").write_text("copy one", encoding="utf-8")
    (folder / "sample_copy_b.epub").write_text("copy two", encoding="utf-8")
    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [folder])
    monkeypatch.setattr(
        manifest,
        "load_settings",
        lambda: {"skip_files": ["sample_copy_a.epub", "sample_copy_b.epub"]},
    )

    summary = manifest.scan()

    assert summary["skipped"] == 2
    assert manifest.counts()["skipped"] == 2
    assert manifest.skipped_files() == [
        str(folder / "sample_copy_a.epub"),
        str(folder / "sample_copy_b.epub"),
    ]
    assert [record.path for record in manifest.pending()] == [str(folder / "keep.txt")]

    again = manifest.scan()
    assert again["skipped"] == 2
    assert again["unchanged"] == 1


def test_rescan_drops_chunks_for_newly_skipped_file(
    tmp_path, monkeypatch, isolated_index
):
    from research_assistant import config, pipeline

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "keep.txt").write_text(
        "These are words worth finding in the search index. " * 20, encoding="utf-8"
    )
    (folder / "dupe.txt").write_text(
        "These are words worth dropping from the index. " * 20, encoding="utf-8"
    )
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"rescan_timer": false}', encoding="utf-8")

    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [folder])
    monkeypatch.setattr(config, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(pipeline, "INDEX_DIR", tmp_path / "index")

    def indexed_files() -> set[str]:
        conn = store.connect()
        paths = {
            row[0].split("/")[-1]
            for row in conn.execute("SELECT DISTINCT file_path FROM chunks")
        }
        conn.close()
        return paths

    assert pipeline.rescan()["indexed"] == 2
    assert indexed_files() == {"keep.txt", "dupe.txt"}

    monkeypatch.setattr(manifest, "SKIP_FILES", {"dupe.txt"})
    pipeline.rescan()

    assert manifest.counts()["skipped"] == 1
    assert indexed_files() == {"keep.txt"}


def test_rescan_drops_index_for_deleted_file(tmp_path, monkeypatch, isolated_index):
    from research_assistant import config, pipeline

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "keep.txt").write_text(
        "These are words worth finding in the search index. " * 20, encoding="utf-8"
    )
    (folder / "gone.txt").write_text(
        "These are words that will disappear from disk. " * 20, encoding="utf-8"
    )
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"rescan_timer": false}', encoding="utf-8")

    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [folder])
    monkeypatch.setattr(config, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(pipeline, "INDEX_DIR", tmp_path / "index")

    def indexed_files() -> set[str]:
        conn = store.connect()
        paths = {
            row[0].split("/")[-1]
            for row in conn.execute("SELECT DISTINCT file_path FROM chunks")
        }
        conn.close()
        return paths

    assert pipeline.rescan()["indexed"] == 2
    assert indexed_files() == {"keep.txt", "gone.txt"}

    (folder / "gone.txt").unlink()
    result = pipeline.rescan()

    assert result["scan"]["missing"] == 1
    assert indexed_files() == {"keep.txt"}
    assert manifest.missing_files() == []
    assert "missing" not in manifest.counts()


def test_scan_keeps_rows_when_folder_is_unavailable(
    tmp_path, monkeypatch, isolated_index
):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "keep.txt").write_text("still here", encoding="utf-8")
    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [folder])
    manifest.scan()
    assert manifest.counts()["new"] == 1

    (folder / "keep.txt").unlink()
    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [tmp_path / "unplugged"])
    summary = manifest.scan()

    assert summary["missing"] == 0
    assert manifest.counts()["new"] == 1


def test_missing_file_returns_to_the_index_when_restored(
    tmp_path, monkeypatch, isolated_index
):
    folder = tmp_path / "docs"
    folder.mkdir()
    target = folder / "keep.txt"
    target.write_text("content that comes back", encoding="utf-8")
    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [folder])
    manifest.scan()
    before = target.stat()

    target.unlink()
    assert manifest.scan()["missing"] == 1

    target.write_text("content that comes back", encoding="utf-8")
    os.utime(target, (before.st_atime, before.st_mtime))
    manifest.scan()

    assert manifest.missing_files() == []
    assert [record.path for record in manifest.pending()] == [str(target)]


def test_fts_roundtrip(isolated_index, tmp_path):
    conn = store.connect()
    ids = store.insert_chunks(
        conn,
        str(tmp_path / "doc.txt"),
        [
            ("chapter 1", "the quick brown fox jumps"),
            ("chapter 2", "a lazy dog sleeps"),
        ],
    )
    conn.commit()
    assert len(ids) == 2

    hits = store.fts_search(conn, "brown fox")
    assert hits == [ids[0]]

    hits = store.fts_search(conn, "sleeping dog")
    assert hits == [ids[1]]

    rows = store.fetch_chunks(conn, ids)
    assert rows[ids[0]][2] == "the quick brown fox jumps"

    removed = store.delete_file_chunks(conn, str(tmp_path / "doc.txt"))
    conn.commit()
    assert sorted(removed) == sorted(ids)
    assert store.fts_search(conn, "brown fox") == []
    conn.close()


def test_search_returns_sources_without_vectors(isolated_index, tmp_path):
    from research_assistant.search import search

    conn = store.connect()
    path = str(tmp_path / "doc.txt")
    store.insert_chunks(
        conn,
        path,
        [
            ("page 1", "Solar panels convert sunlight into electricity"),
            ("page 2", "Wind turbines harvest kinetic wind energy"),
        ],
    )
    conn.commit()
    conn.close()

    sources = search("how do solar panels work")
    assert sources
    assert sources[0].file_path == path
    assert "Solar panels" in sources[0].text


def test_private_failure_returns_passages(isolated_index, tmp_path, monkeypatch):
    from research_assistant import answer
    from research_assistant.answer import ask

    monkeypatch.setattr(answer, "api_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        answer,
        "load_settings",
        lambda: {
            "chat_model": "gemma3:1b",
            "chat_provider": "local",
            "private_folder": str(tmp_path),
        },
    )
    conn = store.connect()
    store.insert_chunks(
        conn,
        str(tmp_path / "doc.txt"),
        [("p1", "Mercury is the closest planet to the Sun")],
    )
    conn.commit()
    conn.close()

    result = ask("which planet is closest to the sun", scope="private")
    assert result["answer"] is None
    assert result["sources"]
    assert "local model" in result["error"]


def test_vec_table_dim_check_is_idempotent(isolated_index):
    conn = store.connect()
    store.ensure_vec_table(conn, 384)
    store.ensure_vec_table(conn, 384)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='chunks_vec'"
    ).fetchone()[0]
    assert "float[384]" in sql
    conn.close()


def test_vec_table_recreated_when_dim_changes(isolated_index):
    conn = store.connect()
    store.ensure_vec_table(conn, 384)
    store.ensure_vec_table(conn, 1024)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='chunks_vec'"
    ).fetchone()[0]
    assert "float[1024]" in sql
    conn.close()


def test_vector_stage_reranks_results(isolated_index, tmp_path, monkeypatch):
    import numpy as np

    from research_assistant.search import search

    conn = store.connect()
    path = str(tmp_path / "doc.txt")
    ids = store.insert_chunks(
        conn,
        path,
        [
            ("a", "cats are small domesticated felines"),
            ("b", "quantum physics describes subatomic particles"),
        ],
    )
    conn.commit()

    store.ensure_vec_table(conn, 4)
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], np.float32)
    store.insert_vectors(conn, ids, vectors)
    store.set_embed_meta(conn, "fake-model", 4, "local")
    conn.commit()
    conn.close()

    class FakeEmbedder:
        def embed(self, texts):
            return np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)

    monkeypatch.setattr(
        "research_assistant.search.cached_embedder",
        lambda mode=None: FakeEmbedder(),
    )
    monkeypatch.setattr(
        "research_assistant.search.backend_info",
        lambda mode=None: {"mode": "local", "model": "fake-model", "dim": 4},
    )

    sources = search("feline pets")
    assert sources
    assert sources[0].text == "cats are small domesticated felines"
    assert sources[0].stage == "keyword+vector"


def test_delete_vectors_on_fresh_connection(isolated_index):
    import numpy as np

    conn = store.connect()
    ids = store.insert_chunks(conn, "/x/doc.txt", [("a", "some text here")])
    conn.commit()

    store.ensure_vec_table(conn, 4)
    store.insert_vectors(conn, ids, np.array([[1.0, 0, 0, 0]], np.float32))
    conn.commit()
    conn.close()

    fresh = store.connect()
    store.delete_vectors(fresh, ids)
    fresh.commit()
    fresh.close()

    check = store.connect()
    store._vec_load(check)
    assert check.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0
    check.close()


def test_delete_vectors_without_vec_table_is_harmless(isolated_index):
    conn = store.connect()
    store.delete_vectors(conn, [1, 2, 3])
    conn.close()


def test_api_key_roundtrip_and_permissions(tmp_path, monkeypatch):
    from research_assistant import config, embed

    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(embed, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert embed.api_key() is None
    embed.set_api_key("sk-or-test-12345")
    assert embed.api_key() == "sk-or-test-12345"
    assert embed.has_api_key() is True

    mode = (tmp_path / ".env").stat().st_mode & 0o777
    assert mode == 0o600, f"expected 600, got {oct(mode)}"

    embed.set_api_key("sk-or-test-67890")
    assert embed.api_key() == "sk-or-test-67890"
    assert (tmp_path / ".env").read_text().count("OPENROUTER_API_KEY=") == 1


def test_api_key_rejects_empty_and_multiline(tmp_path, monkeypatch):
    from research_assistant import config, embed

    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(embed, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    for bad in ("", "   ", "one\ntwo", "one\rtwo"):
        try:
            embed.set_api_key(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad key input: {bad!r}")


def test_settings_page_never_echoes_saved_key(tmp_path, monkeypatch):
    from research_assistant import config, embed
    from research_assistant.webui import app

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(embed, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    embed.set_api_key("sk-or-super-secret-value")

    app.config["TESTING"] = True
    client = app.test_client()

    page = client.get("/settings")
    assert page.status_code == 200
    assert b"sk-or-super-secret-value" not in page.data
    assert b"A key is saved" in page.data
    assert b'name="provider_key"' in page.data
    assert b'type="password"' in page.data

    saved = client.post(
        "/settings",
        data={"verbose": "on", "embedding_mode": "local", "provider_key": ""},
        follow_redirects=True,
    )
    assert b"sk-or-super-secret-value" not in saved.data
    assert embed.api_key() == "sk-or-super-secret-value"


def test_saving_key_via_settings_writes_env(tmp_path, monkeypatch):
    from research_assistant import config, embed
    from research_assistant.webui import app

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(embed, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    app.config["TESTING"] = True
    client = app.test_client()
    client.post(
        "/settings",
        data={
            "verbose": "on",
            "embedding_mode": "local",
            "chat_provider": "openrouter",
            "provider_key": "sk-or-from-form-abc",
        },
        follow_redirects=True,
    )
    assert embed.api_key() == "sk-or-from-form-abc"
    assert b"sk-or-from-form-abc" not in client.get("/settings").data


def test_normalize_citations_fixes_model_formats():
    from research_assistant.answer import _normalize_citations

    assert _normalize_citations("see 【5†L1-L4】 for detail", 8) == "see [5] for detail"
    assert _normalize_citations("as shown [2, 3] here", 8) == "as shown [2][3] here"
    assert _normalize_citations("plain [1] and [8] stay", 8) == "plain [1] and [8] stay"
    assert _normalize_citations("[2, 15] partial", 8) == "[2] partial"
    assert _normalize_citations("out of range [12] gone", 8) == "out of range gone"
    assert (
        _normalize_citations("[note] is not a citation", 8)
        == "[note] is not a citation"
    )


def test_cited_numbers_never_exceed_source_count():
    from research_assistant.answer import _normalize_citations

    for text in ("【9†L1】", "[7, 9]", "[10]"):
        out = _normalize_citations(text, 3)
        for number in range(4, 11):
            assert f"[{number}]" not in out, f"{text} -> {out}"


def _mini_epub_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf"/>'
            "</rootfiles></container>",
        )
        archive.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package><manifest>'
            '<item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
            "</manifest>"
            '<spine><itemref idref="c1"/></spine></package>',
        )
        archive.writestr(
            "OEBPS/chapter1.xhtml",
            "<html><head><title>First Chapter</title></head>"
            "<body><p>A journey begins.</p></body></html>",
        )
    return buffer.getvalue()


def test_extract_zip_members(tmp_path):
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes/alpha.txt", "Alpha notes about testing.")
        archive.writestr("notes/beta.md", "# Beta\n\nBeta body text.")
        archive.writestr(
            "pages/gamma.html",
            "<html><head><title>Gamma</title></head>"
            "<body><p>Gamma words here.</p></body></html>",
        )
        archive.writestr("image.png", b"\x89PNG-not-text")
        archive.writestr("junk.7z", b"7z")
    pieces = extract(path)
    locations = sorted(piece.location for piece in pieces)
    assert locations == ["notes/alpha.txt", "notes/beta.md", "pages/gamma.html"]
    joined = " ".join(piece.text for piece in pieces)
    assert "Alpha notes" in joined
    assert "Beta body text" in joined
    assert "Gamma words" in joined


def test_extract_zip_skips_nested_zip_and_junk(tmp_path):
    path = tmp_path / "outer.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("__MACOSX/alpha.txt", "mac resource fork noise")
        archive.writestr(".DS_Store", "junk")
        archive.writestr("inner.zip", b"PK\x03\x04 not really a zip")
        archive.writestr("real.txt", "the one real file")
    pieces = extract(path)
    assert len(pieces) == 1
    assert pieces[0].location == "real.txt"
    assert "the one real file" in pieces[0].text


def test_extract_zip_epub_member_keeps_chapter_location(tmp_path):
    path = tmp_path / "books.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("library/mini.epub", _mini_epub_bytes())
    pieces = extract(path)
    assert len(pieces) == 1
    assert pieces[0].location == "library/mini.epub, First Chapter"
    assert "journey begins" in pieces[0].text


def test_extract_zip_pdf_member_uses_pdf_extractor(tmp_path, monkeypatch):
    import research_assistant.extract as extract_mod

    seen = {}

    def fake_pdf(path):
        seen["path"] = Path(path).suffix
        return [extract_mod.Piece(text="pdf body text", location="pages 1-2")]

    monkeypatch.setattr(extract_mod, "extract_pdf", fake_pdf)
    path = tmp_path / "docs.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("report.pdf", b"%PDF-1.4 fake")
    pieces = extract(path)
    assert seen["path"] == ".pdf"
    assert pieces[0].location == "report.pdf, pages 1-2"
    assert pieces[0].text == "pdf body text"


def test_extract_zip_rejects_unreadable_archive(tmp_path):
    path = tmp_path / "broken.zip"
    path.write_bytes(b"PK not a real zip file at all")
    with pytest.raises(ValueError, match="not a readable zip"):
        extract(path)


def test_extract_zip_skips_oversized_member(tmp_path, monkeypatch):
    import research_assistant.extract as extract_mod

    monkeypatch.setattr(extract_mod, "MAX_ZIP_MEMBER", 8)
    path = tmp_path / "guarded.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("huge.txt", "x" * 500)
        archive.writestr("tiny.txt", "tiny")
    pieces = extract(path)
    assert len(pieces) == 1
    assert pieces[0].text == "tiny"


def test_zip_files_are_scanned_not_skipped(tmp_path, monkeypatch, isolated_index):
    folder = tmp_path / "docs"
    folder.mkdir()
    with zipfile.ZipFile(folder / "bundle.zip", "w") as archive:
        archive.writestr("a.txt", "content")
    monkeypatch.setattr(manifest, "INDEX_FOLDERS", [folder])

    summary = manifest.scan()
    assert summary["new"] == 1
    assert summary["skipped"] == 0
    assert manifest.counts()["new"] == 1


def test_index_zip_produces_chunks_with_member_locations(
    tmp_path, monkeypatch, isolated_index
):
    from research_assistant import pipeline

    folder = tmp_path / "docs"
    folder.mkdir()
    zip_path = folder / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("chapter-one.txt", "First chapter text about the topic.")
        archive.writestr("chapter-two.txt", "Second chapter text about the topic.")
    conn = store.connect()
    count = pipeline.index_file(conn, str(zip_path))
    assert count > 0
    rows = conn.execute(
        "SELECT location FROM chunks WHERE file_path = ?", (str(zip_path),)
    ).fetchall()
    locations = sorted({row[0] for row in rows})
    assert locations == ["chapter-one.txt", "chapter-two.txt"]
    conn.close()


def _api_client(tmp_path, monkeypatch):
    from research_assistant import config, embed, store
    from research_assistant.webui import app

    index_dir = tmp_path / "index"
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(store, "INDEX_DIR", index_dir)
    monkeypatch.setattr(manifest, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(embed, "PROJECT_DIR", tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


def test_api_status_reports_index_state(tmp_path, monkeypatch):
    client = _api_client(tmp_path, monkeypatch)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert data["index_ready"] is False
    assert data["chunks"] == 0
    assert isinstance(data["model"], str)
    assert isinstance(data["folders"], list)
    assert "application/json" in response.content_type


def test_api_status_never_exposes_the_api_key(tmp_path, monkeypatch):
    from research_assistant import config, embed

    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(embed, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    embed.set_api_key("sk-or-super-secret-value")
    client = _api_client(tmp_path, monkeypatch)
    data = client.get("/api/status").get_json()
    assert "sk-or-super-secret-value" not in str(data)


def test_api_ask_rejects_an_empty_question(tmp_path, monkeypatch):
    client = _api_client(tmp_path, monkeypatch)
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400
    assert client.post("/api/ask").status_code == 400


def test_api_ask_passes_the_chosen_model(tmp_path, monkeypatch):
    from research_assistant import answer

    seen = {}

    def fake_ask(question, model=None, history=None, provider=None, scope="public"):
        seen["model"] = model
        return {
            "question": question,
            "answer": "Answer [1].",
            "model": model or "default-model",
            "provider": "openrouter",
            "history": [],
            "sources": [],
            "error": None,
        }

    monkeypatch.setattr(answer, "ask", fake_ask)
    client = _api_client(tmp_path, monkeypatch)

    data = client.post(
        "/api/ask",
        json={
            "question": "what is it",
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
        },
    ).get_json()
    assert seen["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert data["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert data["ok"] is True

    data = client.post("/api/ask", json={"question": "and without a model"}).get_json()
    assert seen["model"] is None
    assert data["model"] == "default-model"


def test_api_ask_passes_the_scope(tmp_path, monkeypatch):
    from research_assistant import answer

    seen = {}

    def fake_ask(question, model=None, history=None, provider=None, scope="public"):
        seen["scope"] = scope
        return {
            "question": question,
            "answer": "ok",
            "model": "m",
            "provider": "local",
            "scope": scope,
            "history": [],
            "sources": [],
            "error": None,
        }

    monkeypatch.setattr(answer, "ask", fake_ask)
    client = _api_client(tmp_path, monkeypatch)

    data = client.post(
        "/api/ask", json={"question": "q", "scope": "private"}
    ).get_json()
    assert seen["scope"] == "private"
    assert data["scope"] == "private"

    data = client.post("/api/ask", json={"question": "q"}).get_json()
    assert seen["scope"] == "public"
    assert data["scope"] == "public"

    data = client.post(
        "/api/ask", json={"question": "q", "scope": "nonsense"}
    ).get_json()
    assert seen["scope"] == "public"


def test_web_form_submits_the_scope(tmp_path, monkeypatch):
    from research_assistant import answer

    seen = {}

    def fake_ask(question, scope="public", **kwargs):
        seen["scope"] = scope
        return {
            "question": question,
            "answer": "ok",
            "model": "m",
            "provider": "local",
            "provider_label": "Local model (this computer)",
            "scope": scope,
            "history": [],
            "sources": [],
            "error": None,
        }

    monkeypatch.setattr(answer, "ask", fake_ask)
    client = _api_client(tmp_path, monkeypatch)

    page = client.post("/ask", data={"question": "q", "scope": "private"})
    assert page.status_code == 200
    assert seen["scope"] == "private"
    assert b"answered locally on this PC" in page.data

    page = client.post("/ask", data={"question": "q", "scope": "public"})
    assert page.status_code == 200
    assert seen["scope"] == "public"


def test_api_ask_serialises_sources_for_speech(tmp_path, monkeypatch):
    from research_assistant import answer
    from research_assistant.search import Source

    source = Source(
        file_path="/mnt/x/Ebooks/notes/summary.txt",
        location="notes/summary.txt",
        text="body",
        score=0.5,
        stage="bm25",
    )

    def fake_ask(question, model=None, history=None, provider=None, scope="public"):
        return {
            "question": question,
            "answer": "Answer [1].",
            "model": "m",
            "provider": "openrouter",
            "history": [],
            "sources": [source],
            "error": None,
        }

    monkeypatch.setattr(answer, "ask", fake_ask)
    from research_assistant import webui

    monkeypatch.setattr(webui, "INDEX_FOLDERS", [Path("/mnt/x/Ebooks")])
    client = _api_client(tmp_path, monkeypatch)
    data = client.post("/api/ask", json={"question": "q"}).get_json()
    assert data["sources"] == [
        {
            "file": "/mnt/x/Ebooks/notes/summary.txt",
            "folder": "Ebooks",
            "title": "summary",
            "location": "notes/summary.txt",
            "text": "body",
            "score": 0.5,
        }
    ]


def test_doc_title_reads_like_a_name_the_user_recognises():
    from research_assistant.webui import _doc_title

    assert (
        _doc_title("/mnt/x/Ebooks/Sample_Report_300.epub", "Example Journal")
        == "Sample Report 300"
    )
    assert (
        _doc_title("/mnt/x/Ebooks/Sample_Report_Filtered.pdf", "pages 401-600")
        == "Sample Report Filtered"
    )
    assert _doc_title("/mnt/x/reports/bundle.zip", "notes/summary.txt") == "summary"
    assert _doc_title("/mnt/x/Ebooks/book.epub", "Chapter One") == "book"


def test_answer_keeps_several_passages_per_document():
    from research_assistant import answer
    from research_assistant.search import Source

    def make(name, count):
        return [
            Source(
                f"/x/Ebooks/{name}",
                "full text",
                f"{name}-{i}",
                1.0 - i * 0.01,
                "keyword",
            )
            for i in range(count)
        ]

    kept = answer._select_sources(make("plan.txt", 6) + make("report.txt", 3))

    assert [source.text for source in kept[:4]] == [f"plan.txt-{i}" for i in range(4)]
    assert sum(1 for s in kept if s.file_path.endswith("plan.txt")) == 4
    assert sum(1 for s in kept if s.file_path.endswith("report.txt")) == 3


def test_selection_caps_documents_and_total():
    from research_assistant import answer
    from research_assistant.search import Source

    many_docs = [
        Source(f"/x/Ebooks/{i}.txt", "full text", "p", 1.0, "keyword")
        for i in range(20)
    ]
    assert len(answer._select_sources(many_docs)) == answer.MAX_DOCUMENTS

    deep_docs = [
        Source(f"/x/Ebooks/{i}.txt", "full text", f"p{i}-{j}", 1.0, "keyword")
        for i in range(10)
        for j in range(10)
    ]
    assert len(answer._select_sources(deep_docs)) == answer.MAX_PASSAGES


def test_extract_letter_hint():
    from research_assistant.search import extract_letter_hint

    assert extract_letter_hint("a drug that starts with the letter v") == "v"
    assert extract_letter_hint("it begins with the letter V") == "v"
    assert extract_letter_hint("starting with character b") == "b"
    assert extract_letter_hint("it starts with 'Q'") == "q"
    assert extract_letter_hint("starts with a drug") is None
    assert extract_letter_hint("begins with a") is None
    assert extract_letter_hint("what does the book say about it") is None


def test_rarity_bonus_prefers_a_rare_matching_word(isolated_index):
    from research_assistant.search import _rarity_bonus

    conn = store.connect()
    common_ids = store.insert_chunks(
        conn,
        "/x/common.txt",
        [(f"p{i}", "A very common idea about very ordinary things.") for i in range(6)],
    )
    rare_ids = store.insert_chunks(
        conn, "/x/rare.txt", [("r1", "The Vioxx recall hurt patients.")]
    )
    plain_ids = store.insert_chunks(
        conn, "/x/plain.txt", [("n1", "The fish swam past the boat.")]
    )
    conn.commit()
    all_ids = list(common_ids) + list(rare_ids) + list(plain_ids)
    rows = store.fetch_chunks(conn, all_ids)
    bonus = _rarity_bonus(
        conn, {chunk_id: row[2] for chunk_id, row in rows.items()}, "v"
    )

    assert bonus[rare_ids[0]] > bonus[common_ids[0]]
    assert bonus[plain_ids[0]] == 0.0
    conn.close()


def test_content_terms_drop_filler_and_hint_words():
    from research_assistant.search import _content_terms

    terms = _content_terms(
        "there was a drug that damaged hearts that starts with the letter v"
    )
    assert "drug" in terms
    assert "hearts" in terms
    assert "starts" not in terms
    assert "letter" not in terms
    assert "that" not in terms


def test_selection_gives_lane_passages_the_first_slots():
    from research_assistant import answer
    from research_assistant.search import Source

    by_score_first = [
        Source(f"/x/Ebooks/u{doc}.txt", f"page {i}", f"u{doc}-{i}", 1.0, "keyword")
        for doc in range(8)
        for i in range(4)
    ]
    lane_sources = [
        Source(f"/x/Ebooks/t{i}.txt", "full text", f"t{i}", 0.5, "keyword+vector", True)
        for i in range(4)
    ]
    kept = answer._select_sources(by_score_first + lane_sources)

    assert [source.text for source in kept[:4]] == ["t0", "t1", "t2", "t3"]
    assert len(kept) == answer.MAX_PASSAGES


def test_documents_inside_one_zip_count_separately():
    from research_assistant.search import Source, document_key

    one = Source("/x/bundle.zip", "notes/one.txt", "a", 1.0, "keyword")
    two = Source("/x/bundle.zip", "notes/two.txt", "b", 1.0, "keyword")
    again = Source("/x/bundle.zip", "notes/one.txt", "c", 1.0, "keyword")
    pages = Source("/x/bundle.zip", "pages 1-200", "d", 1.0, "keyword")
    assert document_key(one) == document_key(again)
    assert document_key(one) != document_key(two)
    assert document_key(pages) == "/x/bundle.zip"


def test_answer_fetches_extra_candidates_before_deduplicating(tmp_path, monkeypatch):
    from research_assistant import answer, config

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    seen = {}

    def fake_search(question, limit=8, **kwargs):
        seen["limit"] = limit
        return []

    monkeypatch.setattr(answer, "search", fake_search)
    result = answer.ask("anything at all")
    assert seen["limit"] == answer.SEARCH_CANDIDATES
    assert result["sources"] == []


def test_source_excerpt_matches_the_web_page(tmp_path, monkeypatch):
    from research_assistant import answer
    from research_assistant.search import Source

    def fake_ask(question, model=None, history=None, provider=None, scope="public"):
        return {
            "question": question,
            "answer": "a",
            "model": "m",
            "provider": "openrouter",
            "history": [],
            "sources": [
                Source("/x/Ebooks/long.epub", "Chapter One", "y" * 900, 1.0, "bm25"),
                Source("/x/Ebooks/short.epub", "Chapter Two", "z" * 50, 1.0, "bm25"),
            ],
            "error": None,
        }

    monkeypatch.setattr(answer, "ask", fake_ask)
    client = _api_client(tmp_path, monkeypatch)
    data = client.post("/api/ask", json={"question": "q"}).get_json()
    long_excerpt = data["sources"][0]["text"]
    assert long_excerpt == "y" * 400 + "..."
    assert data["sources"][1]["text"] == "z" * 50


def test_history_is_capped_at_three_turns():
    from research_assistant import answer

    assert answer.HISTORY_LIMIT == 3
    turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(6)]
    trimmed = answer._trim_history(turns)
    assert [turn["question"] for turn in trimmed] == ["q3", "q4", "q5"]

    junk = ["a string", None, {"question": "only", "answer": "  "}, "   "]
    assert answer._trim_history(junk) == []
    assert answer._trim_history(None) == []

    long = answer._trim_history([{"question": "q", "answer": "x" * 9000}])
    assert len(long[0]["answer"]) == answer.HISTORY_ANSWER_CHARS


def test_followup_search_includes_the_previous_question(tmp_path, monkeypatch):
    from research_assistant import answer, config

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    seen = {}

    def fake_search(question, limit=8, **kwargs):
        seen["query"] = question
        return []

    monkeypatch.setattr(answer, "search", fake_search)
    answer.ask(
        "what about her other books",
        history=[{"question": "who is Jane Doe", "answer": "a researcher"}],
    )
    assert seen["query"] == "who is Jane Doe what about her other books"

    answer.ask("plain question")
    assert seen["query"] == "plain question"


def test_prompt_carries_the_earlier_conversation():
    from research_assistant import answer
    from research_assistant.search import Source

    source = Source("/x/Ebooks/b.epub", "Chapter", "body", 1.0, "bm25")
    prompt = answer._build_user_prompt(
        "and then?", [source], [{"question": "first", "answer": "answered"}]
    )
    assert prompt.index("Earlier conversation:") < prompt.index("Question: and then?")
    assert "Q: first" in prompt
    assert "A: answered" in prompt
    assert "[1]" in prompt

    plain = answer._build_user_prompt("and then?", [source])
    assert "Earlier conversation:" not in plain


def test_api_accepts_history_and_returns_what_it_kept(tmp_path, monkeypatch):
    from research_assistant import answer

    captured = {}

    def fake_ask(question, model=None, history=None, provider=None, scope="public"):
        captured["history"] = history
        return {
            "question": question,
            "answer": "ok",
            "model": "m",
            "provider": "openrouter",
            "history": answer._trim_history(history),
            "sources": [],
            "error": None,
        }

    monkeypatch.setattr(answer, "ask", fake_ask)
    client = _api_client(tmp_path, monkeypatch)

    payload = {
        "question": "follow up",
        "history": [{"question": f"q{i}", "answer": f"a{i}"} for i in range(6)],
    }
    data = client.post("/api/ask", json=payload).get_json()
    assert len(captured["history"]) == 6
    assert [turn["question"] for turn in data["history"]] == ["q3", "q4", "q5"]

    data = client.post(
        "/api/ask", json={"question": "x", "history": "not a list"}
    ).get_json()
    assert data["history"] == []


def test_transient_model_errors_are_detected():
    from research_assistant.answer import _is_transient

    assert _is_transient({"code": 503, "message": "boom"})
    assert _is_transient(
        {"message": "Upstream error from Nvidia: Service temporarily overloaded"}
    )
    assert _is_transient("rate limit exceeded")
    assert not _is_transient(
        {
            "code": 429,
            "message": "Rate limit exceeded: free-models-per-day. "
            "Add 5 credits to unlock 1000 free model requests per day",
        }
    )
    assert not _is_transient({"message": "context window too large"})
    assert not _is_transient("model does not support this image type")


def test_ask_retries_when_the_model_service_is_busy(isolated_index, monkeypatch):
    from research_assistant import answer

    monkeypatch.setattr(answer, "api_key", lambda *args, **kwargs: "test-key")
    monkeypatch.setattr(answer, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(
        answer,
        "load_settings",
        lambda: {
            "chat_model": "space-bunny-free",
            "chat_provider": "zen",
            "web_search": False,
        },
    )
    conn = store.connect()
    store.insert_chunks(conn, "/x/doc.txt", [("p1", "Vioxx damaged hearts")])
    conn.commit()
    conn.close()

    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, payload):
            self.status_code = 200
            self.text = ""
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            return FakeResponse(
                {"error": {"code": 503, "message": "temporarily overloaded"}}
            )
        return FakeResponse(
            {"choices": [{"message": {"content": "Vioxx damaged hearts [1]"}}]}
        )

    monkeypatch.setattr("requests.post", fake_post)
    result = answer.ask("what drug damaged hearts")

    assert result["error"] is None
    assert "Vioxx" in result["answer"]
    assert calls["count"] == 3


def test_provider_registry_is_complete():
    from research_assistant.providers import PROVIDER_ORDER, PROVIDERS, get_provider

    assert set(PROVIDERS) == set(PROVIDER_ORDER)
    envs = []
    for provider_id in PROVIDER_ORDER:
        provider = PROVIDERS[provider_id]
        assert provider["url"].startswith("https://") or provider["url"].startswith(
            "http://127.0.0.1"
        )
        assert provider["label"]
        assert provider["site"]
        assert provider["env"].endswith("_API_KEY")
        assert provider["models"]
        model_ids = {model["id"] for model in provider["models"]}
        assert provider["default_model"] in model_ids
        envs.append(provider["env"])
    assert len(set(envs)) == len(envs)
    assert get_provider("does-not-exist")["id"] == "openrouter"
    assert get_provider(None)["id"] == "openrouter"


def test_every_model_suggestion_says_whether_it_is_free():
    from research_assistant.providers import all_models

    models = all_models()
    assert len(models) >= 15
    for model in models:
        assert model["id"]
        assert model["label"]
        assert isinstance(model["free"], bool)
    zen_free = [model for model in models if model["id"].endswith("-free")]
    assert zen_free
    assert all(model["free"] for model in zen_free)
    assert any(model["free"] for model in models)
    assert any(not model["free"] for model in models)


def test_settings_offers_provider_choice_and_free_filter(tmp_path, monkeypatch):
    from research_assistant import config
    from research_assistant.webui import app

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    app.config["TESTING"] = True
    page = app.test_client().get("/settings")

    assert page.status_code == 200
    assert b'name="chat_provider"' in page.data
    assert b'value="groq"' in page.data
    assert b'value="zen"' in page.data
    assert b'value="local"' in page.data
    assert b'id="free-only"' in page.data
    assert b'data-free="yes"' in page.data
    assert b'data-free="no"' in page.data
    assert b'name="provider_key"' in page.data
    assert b"Local search" in page.data
    assert b"Private search" in page.data


def test_switching_provider_keeps_its_own_model_name(tmp_path, monkeypatch):
    from research_assistant import config
    from research_assistant.webui import app

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    app.config["TESTING"] = True
    client = app.test_client()
    base = {"verbose": "on", "embedding_mode": "local"}

    client.post(
        "/settings",
        data={**base, "chat_provider": "groq", "chat_model": "llama-3.3-70b-versatile"},
    )
    settings = config.load_settings()
    assert settings["chat_provider"] == "groq"
    assert settings["chat_model"] == "llama-3.3-70b-versatile"

    client.post(
        "/settings",
        data={
            **base,
            "chat_provider": "openrouter",
            "chat_model": "llama-3.3-70b-versatile",
        },
    )
    settings = config.load_settings()
    assert settings["chat_provider"] == "openrouter"
    assert settings["chat_models"]["groq"] == "llama-3.3-70b-versatile"
    assert settings["chat_model"] != "llama-3.3-70b-versatile"


def test_keys_are_kept_separately_for_each_provider(tmp_path, monkeypatch):
    from research_assistant import config, embed

    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(embed, "PROJECT_DIR", tmp_path)
    for name in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "OPENCODE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    embed.set_api_key("openrouter-key-111")
    embed.set_api_key("groq-key-222", "groq")
    embed.set_api_key("zen-key-333", "zen")

    assert embed.api_key("openrouter") == "openrouter-key-111"
    assert embed.api_key("groq") == "groq-key-222"
    assert embed.api_key("zen") == "zen-key-333"
    assert embed.api_key("nim") is None
    assert embed.has_api_key("nim") is False

    text = (tmp_path / ".env").read_text()
    assert text.count("OPENROUTER_API_KEY=") == 1
    assert text.count("GROQ_API_KEY=") == 1
    assert text.count("OPENCODE_API_KEY=") == 1


def test_private_search_never_sends_documents_to_a_cloud_provider(
    isolated_index, monkeypatch
):
    from research_assistant import answer, store

    monkeypatch.setattr(answer, "api_key", lambda *args, **kwargs: "test-key")
    monkeypatch.setattr(
        answer,
        "load_settings",
        lambda: {
            "chat_model": "space-bunny-free",
            "chat_provider": "zen",
            "web_search": True,
            "private_folder": "/x/Keep",
        },
    )
    conn = store.connect()
    store.insert_chunks(conn, "/x/Keep/secret.txt", [("p1", "Vioxx damaged hearts")])
    conn.commit()
    conn.close()

    seen = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "Vioxx damaged hearts [1]"}}]}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["plugins"] = kwargs["json"].get("plugins")
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    result = answer.ask("what drug damaged hearts", scope="private")

    assert result["error"] is None
    assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert seen["plugins"] is None
    assert result["provider"] == "local"
    assert result["model"] == "gemma3:1b"
    assert result["scope"] == "private"


def test_public_search_excludes_the_private_folder(isolated_index, monkeypatch):
    from research_assistant import answer, store

    monkeypatch.setattr(answer, "api_key", lambda *args, **kwargs: "test-key")
    monkeypatch.setattr(
        answer,
        "load_settings",
        lambda: {
            "chat_model": "space-bunny-free",
            "chat_provider": "zen",
            "web_search": False,
            "private_folder": "/x/Keep",
        },
    )
    conn = store.connect()
    store.insert_chunks(conn, "/x/Keep/secret.txt", [("p1", "Vioxx damaged hearts")])
    store.insert_chunks(conn, "/x/Documents/open.txt", [("p1", "Vioxx damaged hearts")])
    conn.commit()
    conn.close()

    seen = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "Vioxx damaged hearts [1]"}}]}

    def fake_post(url, **kwargs):
        seen["url"] = url
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    result = answer.ask("what drug damaged hearts")

    assert result["error"] is None
    assert seen["url"] == "https://opencode.ai/zen/v1/chat/completions"
    assert result["provider"] == "zen"
    assert result["scope"] == "public"
    assert result["sources"]
    assert all(
        source.file_path == "/x/Documents/open.txt" for source in result["sources"]
    )


def test_private_search_only_returns_the_private_folder(isolated_index, monkeypatch):
    from research_assistant import answer, store

    monkeypatch.setattr(answer, "api_key", lambda *args, **kwargs: "test-key")
    monkeypatch.setattr(
        answer,
        "load_settings",
        lambda: {
            "chat_model": "gemma3:1b",
            "chat_provider": "local",
            "private_folder": "/x/Keep",
        },
    )
    conn = store.connect()
    store.insert_chunks(conn, "/x/Keep/secret.txt", [("p1", "Vioxx damaged hearts")])
    store.insert_chunks(conn, "/x/Documents/open.txt", [("p1", "Vioxx damaged hearts")])
    conn.commit()
    conn.close()

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "Vioxx damaged hearts [1]"}}]}

    monkeypatch.setattr("requests.post", lambda url, **kwargs: FakeResponse())
    result = answer.ask("what drug damaged hearts", scope="private")

    assert result["error"] is None
    assert result["sources"]
    assert all(source.file_path == "/x/Keep/secret.txt" for source in result["sources"])


def test_private_history_is_dropped_before_a_cloud_answer(
    isolated_index, tmp_path, monkeypatch
):
    from research_assistant import answer, store

    monkeypatch.setattr(answer, "api_key", lambda *args, **kwargs: "test-key")
    settings = {
        "chat_model": "space-bunny-free",
        "chat_provider": "zen",
        "web_search": False,
        "private_folder": str(tmp_path / "Keep"),
    }
    monkeypatch.setattr(answer, "load_settings", lambda: dict(settings))
    conn = store.connect()
    store.insert_chunks(
        conn, str(tmp_path / "Keep" / "secret.txt"), [("p1", "the code is 4242")]
    )
    store.insert_chunks(
        conn, str(tmp_path / "Documents" / "open.txt"), [("p1", "general notes")]
    )
    conn.commit()
    conn.close()

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "answer [1]"}}]}

    payloads = []

    def fake_post(url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    private_result = answer.ask("what is the secret code", scope="private")
    assert private_result["error"] is None
    private_answer = private_result["answer"]
    assert private_answer

    payloads.clear()
    followup = answer.ask(
        "and the notes",
        history=[
            {"question": "what is the secret code", "answer": private_answer},
            {"question": "notes please", "answer": "public answer"},
        ],
    )
    assert followup["error"] is None
    assert payloads
    prompt = payloads[0]["messages"][1]["content"]
    assert "4242" not in prompt
    assert "secret code" not in prompt
    assert "public answer" in prompt
    assert len(followup["history"]) == 1
    assert followup["history"][0]["question"] == "notes please"
