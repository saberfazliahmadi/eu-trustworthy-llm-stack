"""Central configuration: paths, seeds, provider selection via environment variables."""

from __future__ import annotations

import logging
import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent.parent
DATA_DIR = Path(os.getenv("TRUST_LLM_DATA_DIR", ROOT_DIR / "data"))
REPORTS_DIR = Path(os.getenv("TRUST_LLM_REPORTS_DIR", ROOT_DIR / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"
POLICIES_PATH = PACKAGE_DIR / "guardrails" / "policies.yaml"


class Settings(BaseModel):
    """Runtime settings resolved from environment variables."""

    llm_provider: str = Field(default="mock", description="mock | openai | hf")
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    guard_backend: str = Field(default="heuristic", description="heuristic | llm")
    seed: int = 42
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mock").lower(),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            hf_model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
            hf_embed_model=os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            guard_backend=os.getenv("GUARD_BACKEND", "heuristic").lower(),
            seed=int(os.getenv("SEED", "42")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def set_global_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a declared dependency
        pass


def configure_logging(level: str | None = None) -> None:
    """Structured key=value logging used across all modules."""
    logging.basicConfig(
        level=(level or get_settings().log_level).upper(),
        format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
    )


def get_llm_provider(settings: Settings | None = None):
    """Factory returning the configured LLM provider (mock by default)."""
    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        from trust_llm.providers.live_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            embed_model=settings.embed_model,
        )
    if settings.llm_provider == "hf":
        from trust_llm.providers.live_provider import HuggingFaceLocalProvider

        return HuggingFaceLocalProvider(model=settings.hf_model)
    from trust_llm.providers.mock_provider import MockLLMProvider

    return MockLLMProvider(seed=settings.seed)


def get_embedding_provider(settings: Settings | None = None):
    """Factory returning the configured embedding provider (deterministic hashing by default)."""
    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        from trust_llm.providers.live_provider import OpenAICompatibleEmbeddings

        return OpenAICompatibleEmbeddings(
            model=settings.embed_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    if settings.llm_provider == "hf":
        from trust_llm.providers.live_provider import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model=settings.hf_embed_model)
    from trust_llm.providers.mock_provider import HashingEmbeddingProvider

    return HashingEmbeddingProvider(seed=settings.seed)
