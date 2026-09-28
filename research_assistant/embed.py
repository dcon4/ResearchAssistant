import os

import numpy as np

from research_assistant import logger
from research_assistant.config import PROJECT_DIR, load_settings

OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBED_MODEL = "baai/bge-m3"
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 64


def _env_name(provider: str | None) -> str:
    from research_assistant.providers import get_provider

    return get_provider(provider)["env"]


def api_key(provider: str | None = None) -> str | None:
    name = _env_name(provider)
    key = os.environ.get(name, "").strip()
    if key:
        return key
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        prefix = name + "="
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def has_api_key(provider: str | None = None) -> bool:
    return api_key(provider) is not None


def set_api_key(value: str, provider: str | None = None) -> None:
    name = _env_name(provider)
    label = name
    value = value.strip()
    if not value:
        raise ValueError("The key is empty.")
    if "\n" in value or "\r" in value:
        raise ValueError("The key must be a single line.")
    env_file = PROJECT_DIR / ".env"
    lines: list[str] = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
    placeholder = f"{label}="
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(label + "="):
            lines[index] = placeholder + value
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(placeholder + value)
    text = "\n".join(lines) + "\n"
    descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    logger.log("Settings", f"{label} key saved to .env")


def backend_info(mode: str | None = None) -> dict:
    settings = load_settings()
    mode = mode or settings.get("embedding_mode", "local")
    if mode == "openrouter":
        return {"mode": mode, "model": OPENROUTER_EMBED_MODEL, "dim": 1024}
    return {"mode": "local", "model": LOCAL_MODEL, "dim": 384}


class LocalEmbedder:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(LOCAL_MODEL)
        self.dim = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = list(self.model.embed(texts, batch_size=EMBED_BATCH))
        array = np.array(vectors, dtype=np.float32)
        return _normalize(array)


class OpenRouterEmbedder:
    def __init__(self) -> None:
        import requests

        key = api_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}"})
        self.dim = 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            response = self.session.post(
                OPENROUTER_EMBED_URL,
                json={"model": OPENROUTER_EMBED_MODEL, "input": batch},
                timeout=120,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Embedding API error {response.status_code}: {response.text[:200]}"
                )
            data = response.json()["data"]
            data.sort(key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in data)
        return _normalize(np.array(vectors, dtype=np.float32))


def _normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def get_embedder(mode: str | None = None):
    info = backend_info(mode)
    if info["mode"] == "openrouter":
        logger.verbose("Embed", f"Using OpenRouter embeddings ({info['model']})")
        return OpenRouterEmbedder()
    logger.verbose("Embed", f"Using local embeddings ({info['model']})")
    return LocalEmbedder()


_embedder = None
_embedder_mode = None


def cached_embedder(mode: str | None = None):
    global _embedder, _embedder_mode
    info = backend_info(mode)
    if _embedder is None or _embedder_mode != info["mode"]:
        _embedder = get_embedder(info["mode"])
        _embedder_mode = info["mode"]
    return _embedder
