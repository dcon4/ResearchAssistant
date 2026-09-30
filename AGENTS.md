# research-assistant

Local research assistant: indexes documents (PDF, EPUB, TXT, HTML) and
answers questions about them with an OpenRouter model, citing sources.

## Rules for AI agents working here

This project inherits `~/.config/opencode/AGENTS.md`. Highlights that
matter for this codebase:

- Plain language in README and user-facing strings; no jargon without
  defining it in the same sentence.
- Required features that must never be removed:
  1. Debug log with a "Download debug log" button in the web UI.
  2. Verbose logging toggle, persisted in settings, on by default.
- Log file naming: `research-assistant-debug.log.YYYY-MM-DD-HH-MM.txt`
  in `logs/`, first line is a `Logger initialized` line with build info.
- Log line format: `YYYY-MM-DD HH:MM:SS.mmm [Tag] Message`, verbose
  lines add `[VERBOSE]` after the tag.
- The logger must open in `Application` startup (here: `serve`/CLI
  entry), append mode, flush every line, and install an uncaught
  exception handler that writes a final `CRASH` line.

## Commands

- Install: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt`
- Lint: `.venv/bin/ruff check .`
- Format check: `.venv/bin/ruff format --check .`
- Tests: `.venv/bin/pytest`
- Run: `.venv/bin/python -m research_assistant.cli serve`
- Index: `.venv/bin/python -m research_assistant.cli index`
- Ask from terminal: `.venv/bin/python -m research_assistant.cli ask "question"`
- Rebuild vectors after switching embedding mode: `.venv/bin/python -m research_assistant.cli reembed`

## Conventions

- Index database lives in `index/` on the local disk (never on the NAS).
- Source documents are read in place from the NAS; never moved or copied.
- Index folders are the NAS folders listed in `settings.json`
  (`index_folders`): Ebooks, Documents, reports, and Keep.
- Keep (`/mnt/ls-share/opencode/Keep`) is the private folder
  (`private_folder` in `settings.json`). Searches run with
  `scope="private"` look only there and answer with the local model
  only; `scope="public"` (the default) searches the other three
  folders and must never return a Keep path.
- Settings persist as JSON in `settings.json` (gitignored).
- Never commit `.env`, real API keys, `logs/`, `index/`, or `.venv/`.
