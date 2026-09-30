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
        "default_model": "qwen/qwen3.8-27b",
        "models": [
            {
                "id": "qwen/qwen3.8-27b",
                "label": "Groq: fast, good all-round answers, paid per use",
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
        "default_model": "openai/gpt-oss-20b",
        "models": [
            {
                "id": "openai/gpt-oss-20b",
                "label": "NVIDIA NIM: the model confirmed working on this account",
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
        ],
    },
    "kilo_code": {
        "id": "kilo_code",
        "label": "Kilo Code",
        "env": "KILO_CODE_API_KEY",
        "url": "https://api.kilocode.org/v1/chat/completions",
        "site": "kilocode.org",
        "supports_web_search": False,
        "default_model": "kilocode/llama-3.1-8b-instruct",
        "models": [
            {
                "id": "kilocode/llama-3.1-8b-instruct",
                "label": "Kilo Code: Llama 3.1 8B instruction-tuned model",
                "free": False,
            },
            {
                "id": "kilocode/mistral-7b-instruct",
                "label": "Kilo Code: Mistral 7B instruction-tuned model",
                "free": False,
            },
            {
                "id": "kilocode/phi-3.5-mini",
                "label": "Kilo Code: Phi-3.5 Mini (smaller, faster model)",
                "free": False,
            },
            {
                "id": "kilocode/llama-3.1-8b-instruct-free",
                "label": "Kilo Code: Llama 3.1 8B (free version)",
                "free": True,
            },
        ],
    },
    "local": {
        "id": "local",
        "label": "Local model (this computer)",
        "env": "LOCAL_LLM_API_KEY",
        "url": "http://127.0.0.1:11434/v1/chat/completions",
        "site": "ollama.com",
        "supports_web_search": False,
        "default_model": "gemma3:1b",
        "models": [
            {
                "id": "gemma3:1b",
                "label": "Local: private, runs on this computer, nothing sent online",
                "free": True,
            },
        ],
    },
    "kiloworks_ai": {
        "id": "kiloworks_ai",
        "label": "Cloudflare Workers AI",
        "env": "CLOUDFLARE_WORKERS_AI_API_KEY",
        "url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
        "site": "cloudflare.com/ai",
        "supports_web_search": False,
        "default_model": "@cf/meta/llama-3.1-8b-instruct-fp8",
        "models": [
            {
                "id": "@cf/meta/llama-3.1-8b-instruct-fp8",
                "label": "Cloudflare Workers AI: Llama 3.1 8B instruction-tuned",
                "free": True,
            },
            {
                "id": "@cf/mistral/mistral-7b-instruct-v0.2-lora",
                "label": "Cloudflare Workers AI: Mistral 7B instruction-tuned",
                "free": False,
            },
            {
                "id": "@cf/meta/llama-3.2-3b-instruct",
                "label": "Cloudflare Workers AI: Llama 3.2 3B (smaller, faster)",
                "free": True,
            },
        ],
    },
}

PROVIDER_ORDER = (
    "openrouter",
    "groq",
    "nim",
    "zen",
    "kilo_code",
    "kiloworks_ai",
    "local",
)


def get_provider(provider_id: str | None) -> dict:
    return PROVIDERS.get(provider_id or "", PROVIDERS["openrouter"])


def all_models() -> list[dict]:
    models = []
    for provider_id in PROVIDER_ORDER:
        models.extend(PROVIDERS[provider_id]["models"])
    return models
