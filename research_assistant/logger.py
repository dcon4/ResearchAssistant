import sys
import threading
import time
import traceback
from pathlib import Path

from research_assistant import __version__
from research_assistant.config import LOG_DIR, load_settings

_lock = threading.RLock()
_writer = None
_path = None
_verbose_enabled = True


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _milliseconds() -> str:
    return f"{int(time.time() * 1000) % 1000:03d}"


def open_log() -> Path:
    """Open this launch's log file. Call once at startup, before anything else."""
    global _writer, _path, _verbose_enabled
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _verbose_enabled = bool(load_settings().get("verbose", True))
    name = time.strftime("research-assistant-debug.log.%Y-%m-%d-%H-%M.txt")
    _path = LOG_DIR / name
    _writer = _path.open("a", encoding="utf-8")
    log("Logger", f"Logger initialized (build {__version__})")
    verbose("Logger", f"Log file: {_path}")
    verbose("Logger", f"Verbose logging: {_verbose_enabled}")
    previous = sys.excepthook

    def _uncaught(exc_type, exc_value, exc_tb):
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log("CRASH", f"{exc_type.__name__}: {exc_value}")
        log("CRASH", message.strip())
        previous(exc_type, exc_value, exc_tb)

    sys.excepthook = _uncaught
    return _path


def set_verbose(enabled: bool) -> None:
    global _verbose_enabled
    _verbose_enabled = bool(enabled)


def verbose_enabled() -> bool:
    return _verbose_enabled


def log(tag: str, message: str) -> None:
    _write(tag, message, verbose_line=False)


def verbose(tag: str, message: str) -> None:
    if _verbose_enabled:
        _write(tag, message, verbose_line=True)


def _write(tag: str, message: str, verbose_line: bool) -> None:
    line = f"{_timestamp()}.{_milliseconds()} [{tag}]"
    if verbose_line:
        line += " [VERBOSE]"
    line += f" {message}\n"
    with _lock:
        if _writer is None:
            open_log()
        _writer.write(line)
        _writer.flush()


def current_log_path() -> Path | None:
    return _path
