import os

import httpx
from langchain_ollama import OllamaLLM


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _is_model_available(models_data: dict, model_name: str) -> bool:
    for model in models_data.get("models", []):
        name = model.get("name", "")
        if name == model_name or name.startswith(f"{model_name}:"):
            return True
    return False


def check_ollama_connection(base_url: str, model_name: str) -> None:
    url = f"{_normalize_base_url(base_url)}/api/tags"
    try:
        response = httpx.get(url, timeout=8.0)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Cannot reach Ollama server. Configure OLLAMA_BASE_URL to a reachable "
            "endpoint or run Ollama locally (default: http://127.0.0.1:11434)."
        ) from exc

    if not _is_model_available(data, model_name):
        raise RuntimeError(
            f"Ollama is reachable but model '{model_name}' is missing. "
            f"Run: ollama pull {model_name}"
        )


def get_llm() -> OllamaLLM:
    model_name = os.getenv("OLLAMA_MODEL", "phi3")
    base_url = _normalize_base_url(
        os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )

    check_ollama_connection(base_url, model_name)

    return OllamaLLM(
        model=model_name,
        base_url=base_url,
    )