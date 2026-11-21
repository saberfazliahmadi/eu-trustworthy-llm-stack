"""Abstract provider interfaces shared by the offline simulator and the live adapters."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Approximate token count (word + punctuation pieces); provider-agnostic and deterministic."""
    return len(_TOKEN_RE.findall(text or ""))


class GenerationRequest(BaseModel):
    """A single generation call. `task` and `metadata` are hints; live providers ignore them."""

    prompt: str
    system: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 512
    num_beams: int = 1
    do_sample: bool = False
    seed: int = 42
    task: str = "chat"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


class LLMProvider(ABC):
    """Text generation interface."""

    name: str = "abstract"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a completion for `request`."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Convenience wrapper returning only the text."""
        return self.generate(GenerationRequest(prompt=prompt, **kwargs)).text


class EmbeddingProvider(ABC):
    """Dense embedding interface; implementations must return L2-normalised float32 rows."""

    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into an (n, dim) array."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
