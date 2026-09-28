import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_DIR / "settings.json"
LOG_DIR = PROJECT_DIR / "logs"
INDEX_DIR = PROJECT_DIR / "index"
MANIFEST_DB = INDEX_DIR / "manifest.sqlite"

INDEX_FOLDERS = [
    Path("/mnt/ls-share/opencode/Ebooks"),
    Path("/mnt/ls-share/opencode/Documents"),
    Path("/mnt/ls-share/opencode/reports"),
]

DEFAULTS = {
    "verbose": True,
    "embedding_mode": "local",
    "web_search": False,
    "rescan_timer": True,
    "chat_model": "nvidia/nemotron-3-super-120b-a12b:free",
    "chat_provider": "openrouter",
    "chat_models": {},
    "rerank": False,
    "host": "0.0.0.0",
    "port": 8642,
}


def load_settings() -> dict:
    settings = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for key, value in saved.items():
                if key in DEFAULTS:
                    settings[key] = value
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
