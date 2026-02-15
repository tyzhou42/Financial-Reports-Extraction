"""Provider clients and helper utilities for LLM interactions."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Optional, Set, Type

from openai import OpenAI

try:
    from .config import ACTIVE_LLM_PROVIDERS, PROVIDER_API_KEY_ENV
except ImportError:
    from config import ACTIVE_LLM_PROVIDERS, PROVIDER_API_KEY_ENV  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


class BaseLLMAgent(ABC):
    """Abstract interface for LLM provider adapters."""

    provider: str

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the raw provider response text."""


class GeminiAgent(BaseLLMAgent):
    """Adapter for the Gemini API."""

    provider = "gemini"

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text


class GPTAgent(BaseLLMAgent):
    """Adapter for the OpenAI chat completions API."""

    provider = "openai"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(model_name)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required to use GPTAgent")

        self._client = OpenAI(api_key=key)
        self._response_format = response_format or {"type": "json_object"}

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=self._response_format,
        )

        if not response.choices:
            raise RuntimeError("OpenAI response contained no choices")

        message = response.choices[0].message
        if message is None or message.content is None:
            raise RuntimeError("OpenAI response missing message content")
        return message.content


LLM_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMAgent]] = {
    "gemini": GeminiAgent,
    "openai": GPTAgent,
}


@lru_cache(maxsize=None)
def get_llm_agent(provider: str, model_name: str) -> BaseLLMAgent:
    normalized = provider.strip().lower()
    agent_cls = LLM_PROVIDER_REGISTRY.get(normalized)
    if agent_cls is None:
        raise KeyError(f"Unsupported LLM provider: {provider}")
    return agent_cls(model_name=model_name)


def required_api_keys(active_providers: Optional[Set[str]] = None) -> Dict[str, str]:
    """Return environment variable names required by the active providers."""
    providers = active_providers or ACTIVE_LLM_PROVIDERS
    keys = {
        provider: PROVIDER_API_KEY_ENV[provider]
        for provider in providers
        if provider in PROVIDER_API_KEY_ENV
    }
    logger.debug("Required API keys resolved for providers: %s", sorted(keys))
    return keys
