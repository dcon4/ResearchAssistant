import io
import re
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
from research_assistant.chunk import join_chunks
from research_assistant.config import (
    INDEX_FOLDERS,
    PROJECT_DIR,
    load_settings,
    save_settings,
)
from research_assistant.search import DOC_SUFFIXES, allowed_chunk_ids, is_private_path

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
SEARCH_CANDIDATES = 100
SEARCH_SHOWN = 25
SNIPPET_LEAD = 120
READER_MAX_CHARS = 500_000


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


def _scope_arg() -> str:
    return "private" if request.values.get("scope") == "private" else "public"


def _search_excerpt(text: str, query: str) -> str:
    terms = [term for term in re.findall(r"[A-Za-z0-9']+", query) if len(term) > 1]
    lowered = text.lower()
    positions = [
        position
        for position in (lowered.find(term.lower()) for term in terms)
        if position >= 0
    ]
    start = max(0, (min(positions) if positions else 0) - SNIPPET_LEAD)
    snippet = text[start : start + EXCERPT_CHARS]
    prefix = "..." if start > 0 else ""
    suffix = "..." if start + EXCERPT_CHARS < len(text) else ""
    return prefix + snippet + suffix


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ._()-]", "_", name).strip(" ._")
    return cleaned[:80] or "chapter"


def _reader_data(chunk_id: int, scope: str):
    """Everything the reader and download pages need, or None when not allowed.

    A missing chunk and a private chunk requested in public scope both
    answer the same way, so guessing ids reveals nothing.
    """
    settings = load_settings()
    private_folder = (settings.get("private_folder") or "").strip()
    conn = store.connect()
    try:
        row = store.fetch_chunk(conn, chunk_id)
        if row is None:
            return None
        file_path, location, text = row
        if is_private_path(file_path, private_folder) and scope != "private":
            return None
        pieces = store.section_chunks(conn, file_path, location)
    finally:
        conn.close()
    return settings, file_path, location, text, pieces


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
    scope = str(data.get("scope") or "").strip().lower()
    scope = scope if scope == "private" else "public"
    raw_history = data.get("history")
    history = raw_history if isinstance(raw_history, list) else None
    logger.log("WebUI", f"API question received ({scope}): {question[:120]}")
    from research_assistant.answer import ask as answer_ask

    result = answer_ask(
        question, model=model, history=history, provider=provider, scope=scope
    )
    sources = [
        {
            "file": source.file_path,
            "folder": _folder_name(source.file_path),
            "title": _doc_title(source.file_path, source.location),
            "location": source.location,
            "text": _excerpt(source.text),
            "score": round(source.score, 4),
            "chunk_id": source.chunk_id,
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
            "scope": result.get("scope", scope),
            "history": result["history"],
            "sources": sources,
            "error": result["error"],
        }
    )


@app.route("/")
def home():
    settings = load_settings()
    return render_template("ask.html", settings=settings)


@app.route("/ask", methods=["GET", "POST"])
def ask():
    question = (request.values.get("question") or "").strip()
    scope = (request.values.get("scope") or "").strip().lower()
    scope = scope if scope == "private" else "public"
    settings = load_settings()
    if not question:
        return render_template(
            "ask.html", settings=settings, error="Please type a question."
        )
    logger.log("WebUI", f"Question received ({scope}): {question[:120]}")
    from research_assistant.answer import ask as answer_ask

    result = answer_ask(question, scope=scope)
    return render_template(
        "ask.html",
        settings=settings,
        question=question,
        scope=result.get("scope", scope),
        provider_label=result["provider_label"],
        answer=result["answer"],
        sources=result["sources"],
        error=result["error"],
    )


@app.route("/search")
def search_page():
    settings = load_settings()
    query = (request.values.get("q") or "").strip()
    boolean = request.values.get("boolean") == "on"
    scope = _scope_arg()
    private_folder = (settings.get("private_folder") or "").strip()
    results: list[dict] = []
    error = None
    searched = False
    if query:
        searched = True
        logger.log("WebUI", f"Search page query ({scope}): {query[:120]}")
        conn = store.connect()
        try:
            if boolean:
                try:
                    ids = store.fts_boolean(conn, query, limit=SEARCH_CANDIDATES)
                except ValueError as exc:
                    error = str(exc)
                    ids = []
            else:
                ids = store.fts_search(conn, query, limit=SEARCH_CANDIDATES)
            allowed = allowed_chunk_ids(conn, ids, scope, private_folder)
            rows = store.fetch_chunks(conn, [cid for cid in ids if cid in allowed])
            for chunk_id in ids:
                row = rows.get(chunk_id)
                if not row:
                    continue
                file_path, location, text = row
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "title": _doc_title(file_path, location) or file_path,
                        "file_path": file_path,
                        "location": location,
                        "excerpt": _search_excerpt(text, query),
                    }
                )
                if len(results) >= SEARCH_SHOWN:
                    break
        finally:
            conn.close()
        if not results and not error and scope == "private" and not private_folder:
            error = (
                "Private search has no folder set yet. Open Settings and "
                "choose the private folder first."
            )
    return render_template(
        "search.html",
        settings=settings,
        query=query,
        boolean=boolean,
        scope=scope,
        results=results,
        error=error,
        searched=searched,
    )


@app.route("/source/<int:chunk_id>")
def source_view(chunk_id):
    scope = _scope_arg()
    data = _reader_data(chunk_id, scope)
    if data is None:
        abort(404)
    settings, file_path, location, matched_text, pieces = data
    full_text = join_chunks([text for _id, text in pieces])
    truncated = False
    if len(full_text) > READER_MAX_CHARS:
        cut = full_text.rfind("\n\n", 0, READER_MAX_CHARS)
        full_text = full_text[: cut if cut > 0 else READER_MAX_CHARS]
        truncated = True
    blocks = [block.strip() for block in full_text.split("\n\n") if block.strip()]
    if not blocks and full_text.strip():
        blocks = [full_text.strip()]
    marker = matched_text.strip().split("\n", 1)[0][:120].strip()
    matched_index = None
    if marker:
        matched_index = next(
            (index for index, block in enumerate(blocks) if marker in block), None
        )
    try:
        original_exists = Path(file_path).is_file()
    except OSError:
        original_exists = False
    logger.log(
        "WebUI",
        f"Reader opened ({scope}): {Path(file_path).name} / {location[:80]}",
    )
    return render_template(
        "source.html",
        settings=settings,
        scope=scope,
        chunk_id=chunk_id,
        title=_doc_title(file_path, location) or Path(file_path).name,
        file_path=file_path,
        location=location,
        blocks=blocks,
        matched_index=matched_index,
        original_exists=original_exists,
        truncated=truncated,
    )


@app.route("/source/<int:chunk_id>/download")
def source_download(chunk_id):
    scope = _scope_arg()
    data = _reader_data(chunk_id, scope)
    if data is None:
        abort(404)
    _settings, file_path, location, _matched, pieces = data
    full_text = join_chunks([text for _id, text in pieces])
    name = _safe_name(_doc_title(file_path, location)) + ".txt"
    logger.log(
        "WebUI",
        f"Chapter download ({scope}): {Path(file_path).name} / {location[:80]}",
    )
    return send_file(
        io.BytesIO(full_text.encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=name,
    )


@app.route("/source/<int:chunk_id>/original")
def source_file_download(chunk_id):
    scope = _scope_arg()
    data = _reader_data(chunk_id, scope)
    if data is None:
        abort(404)
    settings, file_path, _location, _matched, _pieces = data
    private_folder = (settings.get("private_folder") or "").strip()
    roots = [Path(p) for p in settings.get("index_folders", [])]
    if private_folder:
        roots.append(Path(private_folder))
    path = Path(file_path)
    if not any(path == root or path.is_relative_to(root) for root in roots):
        abort(404)
    try:
        if not path.is_file():
            abort(404)
    except OSError:
        abort(404)
    logger.log("WebUI", f"Original file download ({scope}): {path.name}")
    return send_file(path, as_attachment=True, download_name=path.name)


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
