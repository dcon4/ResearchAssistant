import re
import time

import requests

from research_assistant import logger
from research_assistant.config import load_settings
from research_assistant.embed import api_key
from research_assistant.providers import PROVIDER_ORDER, PROVIDERS, get_provider
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


class CircuitBreaker:
    """Tracks provider failures and prevents repeated calls to failing providers."""

    def __init__(self, cooldown_seconds: int = 300):
        self.cooldown_seconds = cooldown_seconds
        self._failure_times: dict[str, float] = {}
        self._success_counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}

    def record_success(self, provider_id: str) -> None:
        self._success_counts[provider_id] = self._success_counts.get(provider_id, 0) + 1
        # Reset failure count on success
        self._failure_counts[provider_id] = 0
        self._failure_times.pop(provider_id, None)

    def record_failure(self, provider_id: str) -> None:
        import time

        self._failure_times[provider_id] = time.time()
        self._failure_counts[provider_id] = self._failure_counts.get(provider_id, 0) + 1

    def is_available(self, provider_id: str) -> bool:
        import time

        if provider_id not in self._failure_times:
            return True
        if self._failure_counts.get(provider_id, 0) < 3:
            return True
        elapsed = time.time() - self._failure_times[provider_id]
        return elapsed >= self.cooldown_seconds

    def get_stats(self, provider_id: str) -> dict:
        return {
            "successes": self._success_counts.get(provider_id, 0),
            "failures": self._failure_counts.get(provider_id, 0),
            "is_available": self.is_available(provider_id),
        }


_circuit_breaker = CircuitBreaker()


def get_fallback_order(settings: dict, current_provider: str) -> list[str]:
    """Get the ordered list of providers to try as fallbacks."""
    custom_order = settings.get("fallback_provider_order", [])
    if custom_order:
        # Filter to only valid providers
        return [p for p in custom_order if p in PROVIDERS]
    # Default: all providers except current, in PROVIDER_ORDER
    return [
        p
        for p in PROVIDER_ORDER
        if p != PROVIDERS.get(current_provider, {}).get("id", "")
    ]


def has_valid_key(provider_id: str) -> bool:
    """Check if a provider has a valid API key configured."""
    return api_key(provider_id) is not None


def get_fallback_order_with_keys(settings: dict, current_provider: str) -> list[str]:
    """Get the ordered list of providers that have valid API keys."""
    fallback_order = get_fallback_order(settings, current_provider)
    return [pid for pid in fallback_order if has_valid_key(pid)]


def build_provider_url(provider: dict, model_id: str, settings: dict) -> str:
    """Build the provider URL, handling template variables like Cloudflare's account_id."""
    url = provider["url"]
    if "{account_id}" in url:
        account_id = settings.get("cloudflare_account_id", "").strip()
        if not account_id:
            raise RuntimeError(
                "Cloudflare Workers AI requires an account ID. Set it in Settings."
            )
        url = url.replace("{account_id}", account_id)
    if "{model}" in url:
        url = url.replace("{model}", model_id)
    return url


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


def ask(
    question: str,
    model: str | None = None,
    history: list | None = None,
    provider: str | None = None,
) -> dict:
    settings = load_settings()
    model_id = (model or "").strip() or settings["chat_model"]
    # The app sends only a model name. Work out which provider owns that
    # model so a choice made in A.R.Y.A lands on the right service; when
    # the model is unknown or absent, keep the provider set here.
    provider_id = settings["chat_provider"]
    if model_id:
        for candidate_id in PROVIDER_ORDER:
            if any(m["id"] == model_id for m in PROVIDERS[candidate_id]["models"]):
                provider_id = candidate_id
                break
    explicit_provider = (provider or "").strip()
    if explicit_provider in PROVIDERS:
        provider_id = explicit_provider
    provider = get_provider(provider_id)
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
        "provider": provider_id,
        "provider_label": provider["label"],
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
    key = api_key(provider["id"])
    if not key:
        logger.log(
            "Answer",
            f"No {provider['label']} API key; will try fallback providers",
        )

    def try_provider(provider: dict, model_id: str, settings: dict) -> str | None:
        """Try to get an answer from a single provider. Returns answer text or None if failed."""
        key = api_key(provider["id"])
        if not key:
            return None
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
        if settings.get("web_search") and provider.get("supports_web_search"):
            payload["plugins"] = [{"id": "web", "max_results": 5}]
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        url = build_provider_url(provider, model_id, settings)
        attempts = 1 + len(RETRY_DELAYS)
        for attempt in range(attempts):
            if attempt:
                time.sleep(RETRY_DELAYS[attempt - 1])
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=180
                )
            except requests.exceptions.RequestException as exc:
                logger.log(
                    "Answer",
                    f"Network problem with {provider['label']}, attempt {attempt + 1}/{attempts}: {exc}",
                )
                if any(
                    marker in str(exc)
                    for marker in (
                        "Name or service not known",
                        "Failed to resolve",
                        "NameResolutionError",
                        "getaddrinfo failed",
                    )
                ):
                    return None
                continue
            if response.status_code in TRANSIENT_CODES:
                logger.log(
                    "Answer",
                    f"Model busy ({response.status_code}), attempt {attempt + 1}/{attempts}",
                )
                continue
            if response.status_code != 200:
                logger.log(
                    "Answer",
                    f"Model returned {response.status_code}: {response.text[:300]}",
                )
                return None
            data = response.json()
            if "error" in data:
                if _is_transient(data["error"]):
                    logger.log(
                        "Answer",
                        f"Model busy, attempt {attempt + 1}/{attempts}: {str(data['error'])[:120]}",
                    )
                    continue
                logger.log(
                    "Answer",
                    f"Model error: {str(data['error'])[:300]}",
                )
                return None
            answer_text = data["choices"][0]["message"]["content"].strip()
            return answer_text
        return None

    # Determine if fallback is enabled for this context
    is_search_context = (
        bool(turns) or False
    )  # If there are turns, it's a follow-up (search context)
    fallback_enabled = settings.get(
        "search_fallback_enabled" if is_search_context else "chat_fallback_enabled",
        True,
    )

    # Get fallback order
    current_provider_id = provider["id"]
    fallback_order = (
        get_fallback_order_with_keys(settings, current_provider_id)
        if fallback_enabled
        else []
    )
    providers_to_try = [provider] + [
        get_provider(pid)
        for pid in fallback_order
        if _circuit_breaker.is_available(pid)
    ]

    answer_text: str | None = None
    attempted = False
    breaker_skipped = 0

    for candidate in providers_to_try:
        if not _circuit_breaker.is_available(candidate["id"]):
            logger.log(
                "Answer", f"Skipping {candidate['label']} (circuit breaker open)"
            )
            breaker_skipped += 1
            continue
        if not api_key(candidate["id"]):
            logger.log("Answer", f"Skipping {candidate['label']} (no API key)")
            continue
        if candidate["id"] == current_provider_id:
            candidate_model = model_id
        else:
            candidate_model = (
                settings.get("chat_models", {}).get(candidate["id"])
                or candidate["default_model"]
            )
        attempted = True
        logger.log(
            "Answer",
            f"Trying provider: {candidate['label']} (model {candidate_model})",
        )
        answer_text = try_provider(candidate, candidate_model, settings)
        if answer_text is not None:
            _circuit_breaker.record_success(candidate["id"])
            result["provider"] = candidate["id"]
            result["provider_label"] = candidate["label"]
            result["model"] = candidate_model
            break
        _circuit_breaker.record_failure(candidate["id"])

    if answer_text is None:
        if breaker_skipped and not attempted:
            result["error"] = (
                "The model services are busy or failed recently. Wait a "
                "few minutes and try again."
            )
            logger.log("Answer", "All providers skipped (circuit breaker)")
        elif not attempted:
            result["error"] = (
                "Passages were found, but no provider has an API key set, "
                "so a written answer cannot be produced yet. Open Settings "
                "and paste your key into the key box."
            )
            logger.log("Answer", "No keyed provider available; passages only")
        else:
            result["error"] = (
                "The answer could not be produced just now. Every available "
                "model service failed or was busy. Try again in a minute."
            )
            logger.log("Answer", "All providers failed")
        return result

    result["answer"] = _normalize_citations(answer_text, len(sources))
    logger.log(
        "Answer",
        f"Answer produced ({len(result['answer'])} chars, model {model_id})",
    )
    return result
