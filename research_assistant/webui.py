import threading
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

from research_assistant import logger, manifest, store
from research_assistant.config import (
    INDEX_FOLDERS,
    PROJECT_DIR,
    load_settings,
    save_settings,
)

app = Flask(__name__, template_folder=str(PROJECT_DIR / "templates"))


class _PrefixMiddleware:
    def __init__(self, wsgi_app, prefix="/assistant"):
        self.wsgi_app = wsgi_app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == self.prefix or path.startswith(f"{self.prefix}/"):
            environ["SCRIPT_NAME"] = self.prefix
            environ["PATH_INFO"] = path[len(self.prefix) :] or "/"
        return self.wsgi_app(environ, start_response)


app.wsgi_app = _PrefixMiddleware(app.wsgi_app)

_scan_lock = threading.Lock()
_scan_status = {"running": False, "last": None}


def _run_scan() -> None:
    try:
        from research_assistant.pipeline import rescan

        result = rescan()
        if "skipped" in result:
            _scan_status["last"] = (
                "Scan skipped: another scan or embed is already running."
            )
        else:
            _scan_status["last"] = str(result)
    except Exception as exc:
        logger.log("WebUI", f"Scan failed: {exc}")
        _scan_status["last"] = f"Scan failed: {exc}"
    finally:
        _scan_status["running"] = False


@app.route("/scan", methods=["POST"])
def scan():
    if _scan_lock.locked():
        return render_template(
            "status.html",
            settings=load_settings(),
            folders=[{"path": str(p), "exists": p.is_dir()} for p in INDEX_FOLDERS],
            index_ready=True,
            stats=_index_stats(),
            file_counts=manifest.counts(),
            scan_result="A scan is already running.",
        )
    _scan_status["running"] = True
    thread = threading.Thread(target=_run_scan, daemon=True)
    thread.start()
    logger.log("WebUI", "Scan started from the status page")
    return render_template(
        "status.html",
        settings=load_settings(),
        folders=[{"path": str(p), "exists": p.is_dir()} for p in INDEX_FOLDERS],
        index_ready=True,
        stats=_index_stats(),
        file_counts=manifest.counts(),
        scan_result="Scan started. Refresh this page in a minute to see progress.",
    )


def _index_stats() -> dict:
    try:
        conn = store.connect()
        stats = store.stats(conn)
        conn.close()
        return stats
    except Exception as exc:
        logger.log("WebUI", f"Cannot read index stats: {exc}")
        return {"chunks": 0, "files": 0, "vectors": 0}


def _folder_name(path: str) -> str:
    target = Path(path)
    for root in INDEX_FOLDERS:
        try:
            if target.is_relative_to(root):
                return root.name
        except (OSError, ValueError):
            continue
    return target.parent.name


@app.route("/api/status")
def api_status():
    stats = _index_stats()
    settings = load_settings()
    return jsonify(
        {
            "ok": True,
            "index_ready": stats.get("chunks", 0) > 0,
            "chunks": stats.get("chunks", 0),
            "files": stats.get("files", 0),
            "vectors": stats.get("vectors", 0),
            "model": settings["chat_model"],
            "web_search": bool(settings.get("web_search")),
            "folders": [str(p) for p in INDEX_FOLDERS],
        }
    )


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "Please send a question."}), 400
    model = str(data.get("model") or "").strip() or None
    logger.log("WebUI", f"API question received: {question[:120]}")
    from research_assistant.answer import ask as answer_ask

    result = answer_ask(question, model=model)
    sources = [
        {
            "file": source.file_path,
            "folder": _folder_name(source.file_path),
            "location": source.location,
            "score": round(source.score, 4),
        }
        for source in result["sources"]
    ]
    return jsonify(
        {
            "ok": result["error"] is None,
            "question": result["question"],
            "answer": result["answer"],
            "model": result["model"],
            "sources": sources,
            "error": result["error"],
        }
    )


@app.route("/")
def home():
    settings = load_settings()
    return render_template("ask.html", settings=settings)


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.form.get("question") or "").strip()
    settings = load_settings()
    if not question:
        return render_template(
            "ask.html", settings=settings, error="Please type a question."
        )
    logger.log("WebUI", f"Question received: {question[:120]}")
    from research_assistant.answer import ask as answer_ask

    result = answer_ask(question)
    return render_template(
        "ask.html",
        settings=settings,
        question=question,
        answer=result["answer"],
        sources=result["sources"],
        error=result["error"],
    )


@app.route("/status")
def status():
    settings = load_settings()
    folders = [{"path": str(p), "exists": p.is_dir()} for p in INDEX_FOLDERS]
    stats = _index_stats()
    scan_result = _scan_status["last"]
    if _scan_status["running"]:
        scan_result = "A scan is running right now. Refresh in a minute."
    return render_template(
        "status.html",
        settings=settings,
        folders=folders,
        index_ready=stats.get("chunks", 0) > 0,
        stats=stats,
        file_counts=manifest.counts(),
        scan_result=scan_result,
    )


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    settings = load_settings()
    message = None
    if request.method == "POST":
        for key in ("verbose", "web_search", "rescan_timer", "rerank"):
            settings[key] = request.form.get(key) == "on"
        mode = request.form.get("embedding_mode")
        if mode in ("local", "openrouter"):
            if mode != settings["embedding_mode"]:
                logger.log(
                    "Settings",
                    f"Embedding mode changed {settings['embedding_mode']} -> {mode}"
                    " (full re-embed required)",
                )
                settings["_reembed_notice"] = True
            settings["embedding_mode"] = mode
        chat_model = (request.form.get("chat_model") or "").strip()
        if chat_model:
            settings["chat_model"] = chat_model
        save_settings(settings)
        logger.set_verbose(settings["verbose"])
        logger.log("Settings", "Settings saved")
        message = "Settings saved."
        key_value = (request.form.get("openrouter_key") or "").strip()
        if key_value:
            from research_assistant.embed import set_api_key

            try:
                set_api_key(key_value)
                message = "Settings and API key saved."
            except ValueError as exc:
                message = f"Settings saved, but the API key was not: {exc}"
    settings = load_settings()
    reembed = settings.pop("_reembed_notice", False)
    from research_assistant.embed import has_api_key

    return render_template(
        "settings.html",
        settings=settings,
        message=message,
        reembed=reembed,
        key_set=has_api_key(),
    )


@app.route("/download-log")
def download_log():
    path = logger.current_log_path()
    if path is None or not Path(path).exists():
        abort(404)
    logger.log("WebUI", "Debug log downloaded")
    return send_file(path, as_attachment=True)


def serve() -> None:
    settings = load_settings()
    host = settings["host"]
    port = int(settings["port"])
    logger.log("WebUI", f"Serving on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
