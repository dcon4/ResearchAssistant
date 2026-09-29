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
from research_assistant.search import DOC_SUFFIXES

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


EXCERPT_CHARS = 400


def _excerpt(text: str) -> str:
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS] + "..."


def _folder_name(path: str) -> str:
    target = Path(path)
    for root in INDEX_FOLDERS:
        try:
            if target.is_relative_to(root):
                return root.name
        except (OSError, ValueError):
            continue
    return target.parent.name


def _doc_title(file_path: str, location: str) -> str:
    if "/" in location:
        tail = location.rsplit("/", 1)[-1]
        if Path(tail).suffix.lower() in DOC_SUFFIXES:
            return Path(tail).stem.replace("_", " ").strip()
    return Path(file_path).stem.replace("_", " ").strip()


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
    provider = str(data.get("provider") or "").strip() or None
    raw_history = data.get("history")
    history = raw_history if isinstance(raw_history, list) else None
    logger.log("WebUI", f"API question received: {question[:120]}")
    from research_assistant.answer import ask as answer_ask

    result = answer_ask(question, model=model, history=history, provider=provider)
    sources = [
        {
            "file": source.file_path,
            "folder": _folder_name(source.file_path),
            "title": _doc_title(source.file_path, source.location),
            "location": source.location,
            "text": _excerpt(source.text),
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
            "provider": result["provider"],
            "history": result["history"],
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
    from research_assistant.embed import has_api_key, set_api_key
    from research_assistant.providers import (
        PROVIDER_ORDER,
        PROVIDERS,
        all_models,
        get_provider,
    )

    settings = load_settings()
    message = None
    if request.method == "POST":
        for key in (
            "verbose",
            "web_search",
            "rescan_timer",
            "rerank",
            "chat_fallback_enabled",
            "search_fallback_enabled",
        ):
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
        provider_id = request.form.get("chat_provider")
        provider = (
            get_provider(provider_id)
            if provider_id in PROVIDERS
            else get_provider(settings["chat_provider"])
        )
        submitted = (request.form.get("chat_model") or "").strip()
        cached = settings.get("chat_models") or {}
        provider_changed = provider["id"] != settings["chat_provider"]
        if provider_changed and submitted == settings["chat_model"]:
            model = cached.get(provider["id"]) or provider["default_model"]
        else:
            model = submitted or settings["chat_model"]
        settings["chat_provider"] = provider["id"]
        settings.setdefault("chat_models", {})[provider["id"]] = model
        settings["chat_model"] = model
        cloudflare_account_id = (
            request.form.get("cloudflare_account_id") or ""
        ).strip()
        if cloudflare_account_id:
            settings["cloudflare_account_id"] = cloudflare_account_id
        save_settings(settings)
        logger.set_verbose(settings["verbose"])
        logger.log(
            "Settings",
            f"Settings saved (provider {provider['label']}, model {model})",
        )
        message = "Settings saved."
        key_value = (request.form.get("provider_key") or "").strip()
        if key_value:
            try:
                set_api_key(key_value, provider["id"])
                message = f"Settings and {provider['label']} API key saved."
            except ValueError as exc:
                message = f"Settings saved, but the API key was not: {exc}"
        elif not has_api_key(provider["id"]):
            message = (
                f"Settings saved. No {provider['label']} key is saved yet - "
                "paste one into the key box above and save again."
            )
    settings = load_settings()
    reembed = settings.pop("_reembed_notice", False)
    provider = get_provider(settings["chat_provider"])
    return render_template(
        "settings.html",
        settings=settings,
        message=message,
        reembed=reembed,
        provider=provider,
        providers=[get_provider(pid) for pid in PROVIDER_ORDER],
        models=all_models(),
        key_flags={pid: has_api_key(pid) for pid in PROVIDER_ORDER},
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
    app.run(host=host, port=port, debug=False, threaded=True)
