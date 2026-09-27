import zipfile

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
    (folder / "b.zip").write_bytes(b"PK")
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


def test_answer_without_api_key_returns_passages(isolated_index, tmp_path, monkeypatch):
    from research_assistant import answer
    from research_assistant.answer import ask

    monkeypatch.setattr(answer, "api_key", lambda: None)
    conn = store.connect()
    store.insert_chunks(
        conn,
        str(tmp_path / "doc.txt"),
        [("p1", "Mercury is the closest planet to the Sun")],
    )
    conn.commit()
    conn.close()

    result = ask("which planet is closest to the sun")
    assert result["answer"] is None
    assert result["sources"]
    assert "API key" in result["error"]


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
    assert b'name="openrouter_key"' in page.data
    assert b'type="password"' in page.data

    saved = client.post(
        "/settings",
        data={"verbose": "on", "embedding_mode": "local", "openrouter_key": ""},
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
            "openrouter_key": "sk-or-from-form-abc",
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
