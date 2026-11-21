"""Live provider adapters: any OpenAI-compatible endpoint (OpenAI, Gemini OpenAI-compat, Ollama, vLLM)
and local Hugging Face models. Heavy dependencies are imported lazily so offline mode never needs them.

Environment:
    LLM_PROVIDER=openai  OPENAI_API_KEY=...  OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    LLM_PROVIDER=openai  OPENAI_BASE_URL=http://localhost:11434/v1  OPENAI_API_KEY=ollama  LLM_MODEL=llama3.1
    LLM_PROVIDER=hf      HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from trust_llm.providers.base import (
    EmbeddingProvider,
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    count_tokens,
)

logger = logging.getLogger(__name__)


def _build_messages(request: GenerationRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})
    return messages


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions adapter. Deterministic decoding = temperature 0 + fixed seed (judge mode)."""

    name = "openai-compatible"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        embed_model: str | None = None,
        timeout: float = 60.0,
        http_client: Any | None = None,
    ) -> None:
        """`http_client` accepts a custom `httpx.Client` (proxies, custom CA, or `httpx.MockTransport` in tests)."""
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install live extras: pip install -e '.[live]'") from exc
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM_PROVIDER=openai (use any string for Ollama).")
        self.model = model
        self.embed_model = embed_model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, http_client=http_client)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        deterministic = not request.do_sample or request.temperature == 0
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": 0.0 if deterministic else request.temperature,
            "top_p": 1.0 if deterministic else request.top_p,
            "seed": request.seed,
        }
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=text,
            model=resp.model or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) or count_tokens(request.prompt),
            completion_tokens=getattr(usage, "completion_tokens", None) or count_tokens(text),
            finish_reason=choice.finish_reason or "stop",
        )


class OpenAICompatibleEmbeddings(EmbeddingProvider):
    name = "openai-embeddings"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install live extras: pip install -e '.[live]'") from exc
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for live embeddings.")
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        arr = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        self.dim = arr.shape[1]
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)


class HuggingFaceLocalProvider(LLMProvider):
    """Local transformers causal LM. Judge mode uses num_beams=1, do_sample=False (greedy, reproducible)."""

    name = "huggingface-local"

    def __init__(self, model: str, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install HF extras: pip install -e '.[hf]'") from exc
        self._torch = torch
        self.model_id = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)
        self.model.eval()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        torch = self._torch
        torch.manual_seed(request.seed)
        messages = _build_messages(request)
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n\n".join(m["content"] for m in messages) + "\n\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = request.do_sample and request.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "num_beams": request.num_beams,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=request.temperature, top_p=request.top_p)
            if request.top_k:
                gen_kwargs["top_k"] = request.top_k
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerationResult(
            text=text.strip(),
            model=self.model_id,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
            finish_reason="length" if new_tokens.shape[0] >= request.max_tokens else "stop",
        )


class HuggingFaceEmbeddings(EmbeddingProvider):
    name = "huggingface-embeddings"

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install HF extras: pip install -e '.[hf]'") from exc
        self.model = SentenceTransformer(model)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return arr
