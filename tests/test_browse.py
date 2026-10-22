import json

import pytest

from research_assistant import logger, manifest, store
from research_assistant.chunk import chunk_text, join_chunks
from research_assistant.webui import app

LONG_PARAGRAPH = "word " * 2000
MANY_PARAGRAPHS = "\n\n".join(
    f"Paragraph number {i} with a little text." for i in range(400)
)


@pytest.fixture(autouse=True)
def _fresh_logger(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "_writer", None)
    monkeypatch.setattr(logger, "_path", None)
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    yield


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    monkeypatch.setattr(store, "INDEX_DIR", index_dir)
    monkeypatch.setattr(manifest, "INDEX_DIR", index_dir)
    return tmp_path


def _client(tmp_path, monkeypatch, private_folder="", index_folders=None):
    from research_assistant import config, embed

    settings = {"verbose": True}
    if private_folder:
        settings["private_folder"] = private_folder
    if index_folders:
        settings["index_folders"] = index_folders
    (tmp_path / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(store, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(manifest, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(embed, "PROJECT_DIR", tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


def _insert(file_path, items):
    conn = store.connect()
    ids = store.insert_chunks(conn, file_path, items)
    conn.commit()
    conn.close()
    return ids


def test_join_chunks_reconstructs_an_oversized_paragraph():
    expected = LONG_PARAGRAPH.strip()
    chunks = chunk_text(LONG_PARAGRAPH, "chapter")
    assert len(chunks) > 1
    assert join_chunks([c.text for c in chunks]) == expected


def test_join_chunks_reconstructs_many_paragraphs():
    chunks = chunk_text(MANY_PARAGRAPHS, "chapter")
    assert len(chunks) > 1
    assert join_chunks([c.text for c in chunks]) == MANY_PARAGRAPHS


def test_join_chunks_handles_empty_and_single():
    assert join_chunks([]) == ""
    assert join_chunks(["only one"]) == "only one"


def test_join_chunks_trims_a_short_tail_overlap():
    previous = "a" * 3000 + "tail of the first chunk"
    following = "tail of the first chunk"
    assert join_chunks([previous, following]) == previous


def test_boolean_search_and_expressions(isolated_index):
    ids = _insert(
        "/x/doc.txt",
        [("a", "alpha beta gamma"), ("b", "beta delta epsilon")],
    )
    conn = store.connect()
    assert store.fts_boolean(conn, "alpha AND delta") == []
    assert store.fts_boolean(conn, "alpha AND beta") == [ids[0]]
    assert store.fts_boolean(conn, "delta AND beta") == [ids[1]]
    assert sorted(store.fts_boolean(conn, "gamma OR delta")) == sorted(ids)
    assert store.fts_boolean(conn, '"alpha beta"') == [ids[0]]
    with pytest.raises(ValueError, match="could not be understood"):
        store.fts_boolean(conn, "alpha AND (")
    conn.close()


def test_section_chunks_return_reading_order(isolated_index):
    ids = _insert(
        "/x/book.epub",
        [("Chapter One", "first"), ("Chapter One", "second"), ("Chapter One", "third")],
    )
    conn = store.connect()
    rows = store.section_chunks(conn, "/x/book.epub", "Chapter One")
    conn.close()
    assert [row[0] for row in rows] == ids
    assert [row[1] for row in rows] == ["first", "second", "third"]


def test_search_page_shows_plain_results_with_reader_links(
    isolated_index, tmp_path, monkeypatch
):
    client = _client(tmp_path, monkeypatch)
    ids = _insert(
        "/x/Ebooks/plan.epub",
        [("Chapter One", "Solar panels convert sunlight into electricity")],
    )
    page = client.get("/search", query_string={"q": "solar panels", "scope": "public"})
    assert page.status_code == 200
    assert b"convert sunlight" in page.data
    assert f"/source/{ids[0]}".encode() in page.data
    assert b"Jump" not in page.data


def test_search_page_boolean_mode_reports_bad_expressions(
    isolated_index, tmp_path, monkeypatch
):
    client = _client(tmp_path, monkeypatch)
    _insert("/x/Documents/note.txt", [("full text", "alpha beta gamma")])

    good = client.get("/search", query_string={"q": "alpha AND beta", "boolean": "on"})
    assert good.status_code == 200
    assert b"alpha beta gamma" in good.data

    bad = client.get("/search", query_string={"q": "alpha AND (", "boolean": "on"})
    assert bad.status_code == 200
    assert b"could not be understood" in bad.data


def test_search_page_respects_the_private_scope(isolated_index, tmp_path, monkeypatch):
    keep = tmp_path / "Keep"
    client = _client(tmp_path, monkeypatch, private_folder=str(keep))
    public_ids = _insert("/x/Documents/open.txt", [("p1", "Vioxx damaged hearts")])
    private_ids = _insert(str(keep / "secret.txt"), [("p1", "Vioxx damaged hearts")])

    public = client.get("/search", query_string={"q": "Vioxx", "scope": "public"})
    assert f"/source/{public_ids[0]}".encode() in public.data
    assert f"/source/{private_ids[0]}".encode() not in public.data
    assert str(keep).encode() not in public.data

    private = client.get("/search", query_string={"q": "Vioxx", "scope": "private"})
    assert f"/source/{private_ids[0]}".encode() in private.data
    assert f"/source/{public_ids[0]}".encode() not in private.data


def test_reader_shows_the_whole_chapter_and_offers_downloads(
    isolated_index, tmp_path, monkeypatch
):
    docs = tmp_path / "docs"
    docs.mkdir()
    original = docs / "plan.epub"
    original.write_bytes(b"fake epub bytes")
    client = _client(tmp_path, monkeypatch, index_folders=[str(docs)])

    chapter = "The plan begins here.\n\n" + "Body sentence about the plan. " * 200
    ids = _insert(str(original), [("Chapter One", chapter)])
    conn = store.connect()
    joined = join_chunks(
        [text for _id, text in store.section_chunks(conn, str(original), "Chapter One")]
    )
    conn.close()

    page = client.get(f"/source/{ids[0]}", query_string={"scope": "public"})
    assert page.status_code == 200
    assert b"Jump to the matched passage" in page.data
    assert b"Download this chapter as a text file" in page.data
    assert b"Download the original file" in page.data
    assert b"The plan begins here." in page.data

    chapter_dl = client.get(
        f"/source/{ids[0]}/download", query_string={"scope": "public"}
    )
    assert chapter_dl.status_code == 200
    assert chapter_dl.headers["Content-Disposition"].startswith("attachment")
    assert "plan.txt" in chapter_dl.headers["Content-Disposition"]
    assert chapter_dl.data.decode("utf-8") == joined

    original_dl = client.get(
        f"/source/{ids[0]}/original", query_string={"scope": "public"}
    )
    assert original_dl.status_code == 200
    assert original_dl.data == b"fake epub bytes"

    original.unlink()
    gone = client.get(f"/source/{ids[0]}", query_string={"scope": "public"})
    assert gone.status_code == 200
    assert b"Download the original file" not in gone.data
    assert b"no longer on this computer" in gone.data
    missing = client.get(f"/source/{ids[0]}/original", query_string={"scope": "public"})
    assert missing.status_code == 404


def test_reader_hides_private_chunks_from_public_urls(
    isolated_index, tmp_path, monkeypatch
):
    keep = tmp_path / "Keep"
    client = _client(tmp_path, monkeypatch, private_folder=str(keep))
    private_ids = _insert(str(keep / "secret.txt"), [("p1", "the code is 4242")])
    public_ids = _insert("/x/Documents/open.txt", [("p1", "general notes")])

    forbidden = client.get(
        f"/source/{private_ids[0]}", query_string={"scope": "public"}
    )
    assert forbidden.status_code == 404
    forbidden_dl = client.get(
        f"/source/{private_ids[0]}/download", query_string={"scope": "public"}
    )
    assert forbidden_dl.status_code == 404

    allowed = client.get(f"/source/{private_ids[0]}", query_string={"scope": "private"})
    assert allowed.status_code == 200
    assert b"the code is 4242" in allowed.data

    guess_missing = client.get("/source/999999")
    assert guess_missing.status_code == 404

    public_ok = client.get(f"/source/{public_ids[0]}", query_string={"scope": "public"})
    assert public_ok.status_code == 200


def test_original_download_refuses_paths_outside_the_index_folders(
    isolated_index, tmp_path, monkeypatch
):
    docs = tmp_path / "docs"
    docs.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    sneaky = outside / "file.txt"
    sneaky.write_text("should not be served", encoding="utf-8")
    client = _client(tmp_path, monkeypatch, index_folders=[str(docs)])

    ids = _insert(str(sneaky), [("full text", "should not be served")])
    page = client.get(f"/source/{ids[0]}", query_string={"scope": "public"})
    assert page.status_code == 200

    blocked = client.get(f"/source/{ids[0]}/original", query_string={"scope": "public"})
    assert blocked.status_code == 404
