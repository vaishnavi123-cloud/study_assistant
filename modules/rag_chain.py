import os

import httpx
from langchain_ollama import OllamaLLM


def _read_streamlit_secret(*keys):
    try:
        import streamlit as st
    except Exception:
        return None

    for key in keys:
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            return str(value)

    return None


def _get_setting(primary_key: str, *aliases: str):
    value = os.getenv(primary_key)
    if value:
        return value
    return _read_streamlit_secret(primary_key, *aliases)


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


def get_llm():
    model_name = _get_setting("OLLAMA_MODEL", "ollama_model") or "phi3"
    base_url = _normalize_base_url(
        _get_setting("OLLAMA_BASE_URL", "ollama_base_url")
        or "http://127.0.0.1:11434"
    )

    check_ollama_connection(base_url, model_name)
    return (
        OllamaLLM(
            model=model_name,
            base_url=base_url,
        ),
        "ollama",
    )