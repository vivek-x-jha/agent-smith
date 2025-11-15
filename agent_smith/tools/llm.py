"""LLM abstraction supporting local heuristics and optional OpenAI access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import Settings, get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class BaseLLM(ABC):
    """LLM interface used by agents."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return a completion for the prompt."""


class LocalLLM(BaseLLM):
    """Deterministic heuristics for offline generation."""

    def complete(self, prompt: str, max_tokens: int = 512, **_: Any) -> str:
        important_lines = [line.strip() for line in prompt.splitlines() if len(line.split()) > 3]
        important_lines = important_lines[-8:]
        summary = " ".join(important_lines)
        summary = summary[-(max_tokens * 4) :]
        if not summary:
            summary = "Provide actionable study guidance."
        bullets = self._chunk_text(summary)
        return "\n".join(f"- {bullet}" for bullet in bullets if bullet)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 24) -> list[str]:
        tokens = text.split()
        chunks = [" ".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        return chunks[:6]


class OpenAILLM(BaseLLM):
    """Thin wrapper around `openai`'s Responses API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the openai extra to enable hosted models") from exc

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, **_: Any) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        parts = []
        for item in response.output:
            if item.type == "output_text":
                parts.append(item.text)
        return "".join(parts)


def get_llm(settings: Settings | None = None) -> BaseLLM:
    """Return the best available LLM client given current settings."""

    settings = settings or get_settings()
    if settings.openai_api_key:
        try:
            logger.info("llm_provider", provider="openai")
            return OpenAILLM(settings.openai_api_key)
        except Exception as exc:  # pragma: no cover - degrade gracefully
            logger.warning("openai_init_failed", error=str(exc))
    logger.info("llm_provider", provider="local")
    return LocalLLM()


__all__ = ["BaseLLM", "LocalLLM", "OpenAILLM", "get_llm"]
