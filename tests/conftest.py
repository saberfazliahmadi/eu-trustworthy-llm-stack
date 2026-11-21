import os

import pytest

os.environ.setdefault("LLM_PROVIDER", "mock")

from trust_llm.providers.mock_provider import HashingEmbeddingProvider, MockLLMProvider  # noqa: E402
from trust_llm.rag.pipeline import RAGPipeline  # noqa: E402


@pytest.fixture(scope="session")
def llm() -> MockLLMProvider:
    return MockLLMProvider(seed=42)


@pytest.fixture(scope="session")
def embedder() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(seed=42)


@pytest.fixture(scope="session")
def rag(llm, embedder) -> RAGPipeline:
    return RAGPipeline(llm, embedder)


@pytest.fixture(scope="session")
def reliable_rag(llm, embedder) -> RAGPipeline:
    return RAGPipeline(llm, embedder, min_reliability="high")
