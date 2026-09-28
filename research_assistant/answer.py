import re
import time

from research_assistant import logger
from research_assistant.config import load_settings
from research_assistant.embed import api_key
from research_assistant.providers import get_provider
from research_assistant.search import Source, document_key, search

SEARCH_CANDIDATES = 60
MAX_DOCUMENTS = 8
PASSAGES_PER_DOCUMENT = 4
MAX_PASSAGES = 16
LANE_PRIORITY = 8
PASSAGE_CHARS = 4000
RETRY_DELAYS = (3, 8, 15)
TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_WORDS = (
    "overloaded",
    "rate limit",
    "temporarily",
    "unavailable",
    "timeout",
    "timed out",
    "connection",
    "try again",
)
NON_TRANSIENT_WORDS = (
    "per-day",
    "add credits",
    "insufficient credit",
    "not supported",
    "invalid api key",
)
HISTORY_LIMIT = 3
HISTORY_ANSWER_CHARS = 2000
SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer only from the supplied "
    "passages. A short earlier conversation may be given above for "
    "context: use it to understand what is being asked, but cite only "
    "the passages listed below. If the passages do not contain the "
    "answer, say so plainly. "
    "If the question gives a partial clue such as a starting letter, "
    "scan the passages for words that begin with that letter before "
    "concluding the answer is missing. Prefer a rare name over a common "
    "word when both start with the same letter. "
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


def _is_transient(error: object) -> bool:
    code = None
    if isinstance(error, dict):
        code = error.get("code")
        error = error.get("message") or error
    text = str(error).lower()
    if any(word in text for word in NON_TRANSIENT_WORDS):
        return False
    if isinstance(code, int) and code in TRANSIENT_CODES:
        return True
    return any(word in text for word in TRANSIENT_WORDS)


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
        lines.append(source.text[:PASSAGE_CHARS])
        lines.append("")
    return "\n".join(lines)


def _select_sources(sources: list[Source]) -> list[Source]:
    counts: dict[str, int] = {}
    kept: list[Source] = []

    def take(source: Source) -> bool:
        key = document_key(source)
        if counts.get(key, 0) >= PASSAGES_PER_DOCUMENT:
            return True
        if key not in counts and len(counts) >= MAX_DOCUMENTS:
            return True
        counts[key] = counts.get(key, 0) + 1
        kept.append(source)
        return len(kept) < MAX_PASSAGES

    first_pass: list[Source] = []
    second_pass: list[Source] = []
    for source in sources:
        if source.lane and len(first_pass) < LANE_PRIORITY:
            first_pass.append(source)
        else:
            second_pass.append(source)
    for source in first_pass:
        if not take(source):
            return kept
    for source in second_pass:
        if not take(source):
            return kept
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
    sources = _select_sources(search(search_text, limit=SEARCH_CANDIDATES))
    logger.verbose(
        "Answer",
        f"Selected {len(sources)} passages from "
        f"{len({document_key(source) for source in sources})} documents",
    )
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
    provider = get_provider(settings["chat_provider"])
    key = api_key(provider["id"])
    if not key:
        result["error"] = (
            f"Passages were found, but no {provider['label']} API key is set, "
            "so a written answer cannot be produced yet. Open Settings and "
            "paste your key into the key box."
        )
        logger.log(
            "Answer",
            f"No {provider['label']} API key; returning passages only",
        )
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
        if settings.get("web_search") and provider["supports_web_search"]:
            payload["plugins"] = [{"id": "web", "max_results": 5}]
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        attempts = 1 + len(RETRY_DELAYS)
        last_error: Exception = RuntimeError("model did not answer")
        answer_text: str | None = None
        for attempt in range(attempts):
            if attempt:
                time.sleep(RETRY_DELAYS[attempt - 1])
            try:
                response = requests.post(
                    provider["url"], headers=headers, json=payload, timeout=180
                )
            except requests.exceptions.RequestException as exc:
                last_error = exc
                logger.log(
                    "Answer",
                    f"Network problem, attempt {attempt + 1}/{attempts}: {exc}",
                )
                continue
            if response.status_code in TRANSIENT_CODES:
                last_error = RuntimeError(
                    f"model returned {response.status_code}: {response.text[:300]}"
                )
                logger.log(
                    "Answer",
                    f"Model busy ({response.status_code}), "
                    f"attempt {attempt + 1}/{attempts}",
                )
                continue
            if response.status_code != 200:
                raise RuntimeError(
                    f"model returned {response.status_code}: {response.text[:300]}"
                )
            data = response.json()
            if "error" in data:
                if _is_transient(data["error"]):
                    last_error = RuntimeError(str(data["error"])[:300])
                    logger.log(
                        "Answer",
                        f"Model busy, attempt {attempt + 1}/{attempts}: "
                        f"{str(data['error'])[:120]}",
                    )
                    continue
                raise RuntimeError(str(data["error"])[:300])
            answer_text = data["choices"][0]["message"]["content"].strip()
            break
        if answer_text is None:
            raise last_error
        result["answer"] = _normalize_citations(answer_text, len(sources))
        logger.log(
            "Answer",
            f"Answer produced ({len(result['answer'])} chars, "
            f"model {model_id}, provider {provider['label']})",
        )
    except Exception as exc:
        result["error"] = f"The model could not answer: {exc}"
        logger.log("Answer", f"Chat failed: {exc}")
    return result
