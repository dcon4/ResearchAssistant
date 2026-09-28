PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "id": "openrouter",
        "label": "OpenRouter",
        "env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "site": "openrouter.ai",
        "supports_web_search": True,
        "default_model": "nvidia/nemotron-3-super-120b-a12b:free",
        "models": [
            {
                "id": "nvidia/nemotron-3-super-120b-a12b:free",
                "label": "OpenRouter: fast, short answers, cites sources",
                "free": True,
            },
            {
                "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
                "label": "OpenRouter: slower, longer, more thorough",
                "free": True,
            },
            {
                "id": "nvidia/nemotron-3.5-lightning:free",
                "label": "OpenRouter: very long context, thinks out loud",
                "free": True,
            },
        ],
    },
    "groq": {
        "id": "groq",
        "label": "Groq",
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "site": "console.groq.com/keys",
        "supports_web_search": False,
        "default_model": "llama-3.3-70b-versatile",
        "models": [
            {
                "id": "llama-3.3-70b-versatile",
                "label": "Groq: good all-round answers",
                "free": False,
            },
            {
                "id": "llama-3.1-8b-instant",
                "label": "Groq: fastest, shorter answers",
                "free": False,
            },
            {
                "id": "openai/gpt-oss-120b",
                "label": "Groq: slower, more thorough",
                "free": False,
            },
        ],
    },
    "nim": {
        "id": "nim",
        "label": "NVIDIA NIM",
        "env": "NVIDIA_API_KEY",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "site": "build.nvidia.com",
        "supports_web_search": False,
        "default_model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "models": [
            {
                "id": "nvidia/llama-3.1-nemotron-70b-instruct",
                "label": "NVIDIA NIM: good all-round answers",
                "free": False,
            },
            {
                "id": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
                "label": "NVIDIA NIM: slower, more thorough",
                "free": False,
            },
            {
                "id": "mistralai/mistral-large-2-instruct",
                "label": "NVIDIA NIM: strong writing, longer answers",
                "free": False,
            },
            {
                "id": "deepseek-ai/deepseek-v4.1-flash",
                "label": "NVIDIA NIM: fast and light",
                "free": False,
            },
        ],
    },
    "zen": {
        "id": "zen",
        "label": "OpenCode Zen",
        "env": "OPENCODE_API_KEY",
        "url": "https://opencode.ai/zen/v1/chat/completions",
        "site": "opencode.ai/auth",
        "supports_web_search": False,
        "default_model": "space-bunny-free",
        "models": [
            {
                "id": "space-bunny-free",
                "label": "Zen free: keeps no copy of your text, never trains on it",
                "free": True,
            },
            {
                "id": "longcat-2.5-preview-free",
                "label": "Zen free: keeps no copy of your text, never trains on it",
                "free": True,
            },
            {
                "id": "mimo-v2.6-flash-free",
                "label": "Zen free: your text may be used to improve the model",
                "free": True,
            },
            {
                "id": "mimo-v2.5-free",
                "label": "Zen free: your text may be used to improve the model",
                "free": True,
            },
            {
                "id": "ling-3.0-flash-fin-free",
                "label": "Zen free: your text may be used to improve the model",
                "free": True,
            },
            {
                "id": "big-pickle",
                "label": "Zen free: your text may be used to improve the model",
                "free": True,
            },
            {
                "id": "nemotron-3-ultra-free",
                "label": "Zen free: trial only - do not send private documents",
                "free": True,
            },
            {
                "id": "nemotron-3.5-lightning-free",
                "label": "Zen free: trial only - do not send private documents",
                "free": True,
            },
        ],
    },
}

PROVIDER_ORDER = ("openrouter", "groq", "nim", "zen")


def get_provider(provider_id: str | None) -> dict:
    return PROVIDERS.get(provider_id or "", PROVIDERS["openrouter"])


def all_models() -> list[dict]:
    models = []
    for provider_id in PROVIDER_ORDER:
        models.extend(PROVIDERS[provider_id]["models"])
    return models
