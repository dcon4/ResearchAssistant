import re

from research_assistant import logger
from research_assistant.config import load_settings
from research_assistant.embed import api_key
from research_assistant.search import Source, document_key, search

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SEARCH_CANDIDATES = 24
HISTORY_LIMIT = 3
HISTORY_ANSWER_CHARS = 2000
SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer only from the supplied "
    "passages. A short earlier conversation may be given above for "
    "context: use it to understand what is being asked, but cite only "
    "the passages listed below. If the passages do not contain the "
    "answer, say so plainly. "
    "Write for a non-expert: plain language, no jargon. "
    "Cite with square brackets and a bare number only, like [1] or [2]. "
    "Never use any other citation style: no curly brackets, no daggers, "
    "no line numbers, no author names. Each citation number must be one "
    "of the passage numbers given above."
)

_CURLY_CITE = re.compile(r"[\[【]\s*(\d{1,3})\s*(?:[†‡*][^\]】]*)?[\]】]")
_COMBINED_CITE = re.compile(r"\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})+)\s*\]")


def _normalize_citations(text: str, count: int) -> str:
    def single(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= count:
            return f"[{number}]"
        return ""

    text = _CURLY_CITE.sub(single, text)
    text = _COMBINED_CITE.sub(
        lambda m: "".join(
            f"[{int(part)}]"
            for part in m.group(1).split(",")
            if 1 <= int(part) <= count
        ),
        text,
    )
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _trim_history(history: list | None) -> list[dict]:
    turns: list[dict] = []
    for item in (history or [])[-HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question and answer:
            turns.append(
                {
                    "question": question,
                    "answer": answer[:HISTORY_ANSWER_CHARS],
                }
            )
    return turns


def _search_text(question: str, turns: list[dict]) -> str:
    if not turns:
        return question
    return f"{turns[-1]['question']} {question}"


def _build_user_prompt(
    question: str, sources: list[Source], turns: list[dict] | None = None
) -> str:
    lines: list[str] = []
    if turns:
        lines.append("Earlier conversation:")
        for turn in turns:
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer']}")
            lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Passages:")
    for index, source in enumerate(sources, start=1):
        lines.append(f"[{index}] {source.file_path} ({source.location})")
        lines.append(source.text[:2500])
        lines.append("")
    return "\n".join(lines)


def _distinct_documents(sources: list[Source], limit: int) -> list[Source]:
    seen: set[str] = set()
    kept: list[Source] = []
    for source in sources:
        key = document_key(source)
        if key in seen:
            continue
        seen.add(key)
        kept.append(source)
        if len(kept) >= limit:
            break
    return kept


def ask(question: str, model: str | None = None, history: list | None = None) -> dict:
    settings = load_settings()
    model_id = (model or "").strip() or settings["chat_model"]
    turns = _trim_history(history)
    search_text = _search_text(question, turns)
    if turns:
        logger.log(
            "Answer",
            f"Follow-up with {len(turns)} earlier turn(s); "
            f"searching on: {search_text[:160]}",
        )
    sources = _distinct_documents(search(search_text, limit=SEARCH_CANDIDATES), limit=8)
    result: dict = {
        "question": question,
        "answer": None,
        "model": model_id,
        "history": turns,
        "sources": sources,
        "error": None,
    }
    if not sources:
        result["error"] = (
            "No matching passages were found in the index. "
            "Check the status page, or re-run a scan."
        )
        logger.log("Answer", "No sources found for question")
        return result
    key = api_key()
    if not key:
        result["error"] = (
            "Passages were found, but no OpenRouter API key is set, so a "
            "written answer cannot be produced yet. Set OPENROUTER_API_KEY "
            "in the .env file and restart."
        )
        logger.log("Answer", "No API key; returning passages only")
        return result
    try:
        import requests

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(question, sources, turns),
                },
            ],
            "temperature": 0.2,
        }
        if settings.get("web_search"):
            payload["plugins"] = [{"id": "web", "max_results": 5}]
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"model returned {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        if "error" in data:
            raise RuntimeError(str(data["error"])[:300])
        result["answer"] = _normalize_citations(
            data["choices"][0]["message"]["content"].strip(),
            len(sources),
        )
        logger.log(
            "Answer",
            f"Answer produced ({len(result['answer'])} chars, model {model_id})",
        )
    except Exception as exc:
        result["error"] = f"The model could not answer: {exc}"
        logger.log("Answer", f"Chat failed: {exc}")
    return result
