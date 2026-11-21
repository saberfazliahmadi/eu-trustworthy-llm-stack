"""FastAPI service: /health, /v1/chat (safety stack + grounded RAG + FActScore), /v1/audit/summary.

Run:  uvicorn trust_llm.api.app:app --reload
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from trust_llm import __version__
from trust_llm.config import REPORTS_DIR, get_embedding_provider, get_llm_provider, get_settings
from trust_llm.guardrails.safety_stack import EscalationQueue, LayerEvent, SafetyStack
from trust_llm.rag.fact_verifier import FactVerifier
from trust_llm.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

AI_DISCLOSURE = "You are interacting with an AI system (EU AI Act Art. 50). Hiring and HR decisions are always taken by humans."
ASR_CONFIDENCE_THRESHOLD = 0.85


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    user_id: str = "anonymous"
    use_rag: bool = True
    input_modality: Literal["text", "speech"] = "text"
    asr_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="speech-to-text confidence, if speech")


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    blocked: bool
    final_action: str
    blocked_at: str | None = None
    escalation_ticket: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    factscore: float | None = None
    unsupported_facts: list[str] = Field(default_factory=list)
    grounding_warning: bool = False
    requires_transcript_confirmation: bool = False
    transcript_confirmation_reason: str | None = None
    ai_disclosure: str = AI_DISCLOSURE
    trace: list[LayerEvent] = Field(default_factory=list)


class ChatService:
    def __init__(self, stack: SafetyStack, verifier: FactVerifier | None = None) -> None:
        self.stack = stack
        self.verifier = verifier or FactVerifier()

    def chat(self, message: str, user_id: str = "anonymous", use_rag: bool = True,
             input_modality: str = "text", asr_confidence: float | None = None) -> ChatResponse:
        res = self.stack.process(message, user_id=user_id, use_rag=use_rag)
        out = ChatResponse(answer=res.response, blocked=res.blocked, final_action=res.final_action, blocked_at=res.blocked_at,
                           escalation_ticket=res.escalation_ticket, trace=res.trace,
                           citations=[Citation(doc_id=r.doc_id, chunk_id=r.chunk_id, score=round(r.score, 4)) for r in res.retrieved])
        if not res.blocked and res.retrieved:
            core = res.response.split("\n\n")[0]  # exclude appended disclaimers from fact checking
            fs = self.verifier.score(core, [r.text for r in res.retrieved])
            if not fs.abstained:
                out.factscore = round(fs.factscore, 4)
                out.unsupported_facts = [f.text for f in fs.facts if f.label.value != "supported"]
                out.grounding_warning = fs.factscore < 0.5
        if input_modality == "speech" and (asr_confidence is None or asr_confidence < ASR_CONFIDENCE_THRESHOLD):
            # Accent fairness: low ASR confidence triggers a confirmation step, never a rejection or a lower score.
            out.requires_transcript_confirmation = True
            out.transcript_confirmation_reason = (
                f"Speech recognition confidence {asr_confidence if asr_confidence is not None else 'unknown'} < {ASR_CONFIDENCE_THRESHOLD}. "
                "Please confirm or correct the transcript before the answer is used; accented speech is never penalised."
            )
        return out


@lru_cache(maxsize=1)
def build_service() -> ChatService:
    llm, emb = get_llm_provider(), get_embedding_provider()
    rag = RAGPipeline(llm, emb, min_reliability="high")
    return ChatService(SafetyStack(llm, rag, escalations=EscalationQueue(REPORTS_DIR / "escalations.jsonl")))


app = FastAPI(title="EU Trustworthy LLM Stack - HR & Career Assistant", version=__version__,
              description="Safety-stacked, grounded HR policy and career-coaching assistant (EU AI Act high-risk use case).")


@app.get("/health")
def health() -> dict[str, str]:
    s = get_settings()
    return {"status": "ok", "version": __version__, "provider": s.llm_provider}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return build_service().chat(req.message, req.user_id, req.use_rag, req.input_modality, req.asr_confidence)


@app.get("/v1/audit/summary")
def audit_summary() -> dict:
    path = Path(REPORTS_DIR) / "audit_summary.json"
    if not path.exists():
        return {"available": False, "hint": "run `python -m trust_llm.cli run-all` to generate the audit"}
    data = json.loads(path.read_text(encoding="utf-8"))
    sec, rag = data["security"], data["rag"]
    return {
        "available": True,
        "meta": data["meta"],
        "gates": data["gates"],
        "headline": {
            "asr_raw_model": sec["raw_model"]["asr"],
            "asr_safety_stack": sec["safety_stack"]["asr"],
            "over_refusal_rate": sec["safety_stack"]["over_refusal_rate"],
            "factscore_no_retrieval": rag["no_retrieval_baseline"]["mean_factscore"],
            "factscore_grounded_rag": rag["rag_k3_reliable_sources"]["mean_factscore"],
            "screening_worst_group_raw": data["fairness"]["raw"]["worst_group_accuracy"],
            "screening_worst_group_blind": data["fairness"]["blind"]["worst_group_accuracy"],
        },
    }
