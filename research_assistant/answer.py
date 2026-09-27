import re

from research_assistant import logger
from research_assistant.config import load_settings
from research_assistant.embed import api_key
from research_assistant.search import Source, search

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer only from the supplied "
    "passages. If the passages do not contain the answer, say so plainly. "
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


def _build_user_prompt(question: str, sources: list[Source]) -> str:
    lines = [f"Question: {question}", "", "Passages:"]
    for index, source in enumerate(sources, start=1):
        lines.append(f"[{index}] {source.file_path} ({source.location})")
        lines.append(source.text[:2500])
        lines.append("")
    return "\n".join(lines)


def ask(question: str, model: str | None = None) -> dict:
    settings = load_settings()
    model_id = (model or "").strip() or settings["chat_model"]
    sources = search(question, limit=8)
    result: dict = {
        "question": question,
        "answer": None,
        "model": model_id,
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
                {"role": "user", "content": _build_user_prompt(question, sources)},
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
