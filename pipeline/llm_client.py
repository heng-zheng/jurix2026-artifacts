from __future__ import annotations

from typing import Any, Dict

from .openrouter_client import OpenRouterClient


def chat_client_for_model(config: Dict[str, Any], model: Dict[str, str]) -> OpenRouterClient:
    provider = model.get("provider", "openrouter")
    if provider != "openrouter":
        raise RuntimeError(
            f"Unsupported model provider {provider!r}; the Justice workflow only supports OpenRouter."
        )
    return OpenRouterClient(config)
