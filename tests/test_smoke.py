import json

import pytest

from research_assistant import logger
from research_assistant.config import DEFAULTS, load_settings, save_settings


@pytest.fixture(autouse=True)
def _fresh_logger(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "_writer", None)
    monkeypatch.setattr(logger, "_path", None)
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    yield


def test_logger_creates_file_with_header():
    path = logger.open_log()
    assert path.exists()
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "Logger initialized" in first_line


def test_logger_flushes_every_line():
    path = logger.open_log()
    logger.log("Test", "immediate line")
    content = path.read_text(encoding="utf-8")
    assert "[Test] immediate line" in content


def test_verbose_flag_gates_lines(tmp_path):
    path = logger.open_log()
    logger.set_verbose(False)
    logger.verbose("Test", "hidden line")
    logger.log("Test", "always line")
    content = path.read_text(encoding="utf-8")
    assert "hidden line" not in content
    assert "always line" in content


def test_settings_roundtrip(tmp_path, monkeypatch):
    from research_assistant import config

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    loaded = load_settings()
    assert loaded == DEFAULTS
    loaded["verbose"] = False
    loaded["embedding_mode"] = "openrouter"
    save_settings(loaded)
    again = load_settings()
    assert again["verbose"] is False
    assert again["embedding_mode"] == "openrouter"


def test_settings_ignore_unknown_keys(tmp_path, monkeypatch):
    from research_assistant import config

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", settings_file)
    settings_file.write_text(json.dumps({"bogus": 1, "verbose": False}))
    loaded = load_settings()
    assert "bogus" not in loaded
    assert loaded["verbose"] is False


def test_settings_page_and_log_download(tmp_path, monkeypatch):
    from research_assistant import config
    from research_assistant.webui import app

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    app.config["TESTING"] = True
    client = app.test_client()

    home = client.get("/")
    assert home.status_code == 200
    assert b"<h1>" in home.data

    settings = client.get("/settings")
    assert settings.status_code == 200

    saved = client.post(
        "/settings",
        data={"verbose": "on", "embedding_mode": "local", "chat_model": "test/model"},
        follow_redirects=True,
    )
    assert saved.status_code == 200
    assert load_settings()["verbose"] is True

    logger.open_log()
    log_page = client.get("/download-log")
    assert log_page.status_code == 200
    assert log_page.headers["Content-Disposition"].startswith("attachment")
