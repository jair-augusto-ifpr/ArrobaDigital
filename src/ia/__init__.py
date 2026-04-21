"""Integração com OpenRouter (LLM) — laudo textual, análise visual e relatório."""

from .client import IAClient, IAConfig, IAError, load_ia_config

__all__ = ["IAClient", "IAConfig", "IAError", "load_ia_config"]
