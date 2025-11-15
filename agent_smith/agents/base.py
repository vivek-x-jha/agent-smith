"""Abstract Agent definitions and utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from ..logging_config import get_logger
from ..tools.llm import BaseLLM, get_llm

logger = get_logger(__name__)


class Message(TypedDict, total=False):
    """Simple schema for chat-style LLM prompts."""

    role: str
    content: str


class SupportsLLM(Protocol):
    """Protocol for LLM-like objects."""

    def complete(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - interface
        ...


@dataclass
class AgentContext:
    """Common context fields passed to agents."""

    goal_title: str
    learner_profile: str | None = None
    day_number: int = 1


class Agent(ABC):
    """Base class shared by all domain-specific agents."""

    def __init__(self, name: str, system_prompt: str, llm: BaseLLM | None = None) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm or get_llm()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary action and return structured data."""

    def build_prompt(self, user_prompt: str) -> str:
        """Combine the system prompt with a user instruction string."""

        return f"{self.system_prompt.strip()}\n\n{user_prompt.strip()}".strip()

    def call_llm(self, prompt: str, **kwargs: Any) -> str:
        """Send prompt to configured LLM and capture logs."""

        logger.info("agent_llm_call", agent=self.name)
        return self.llm.complete(prompt, **kwargs)


__all__ = ["Agent", "AgentContext", "Message"]
