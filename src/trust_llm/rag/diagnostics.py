"""Structured RAG diagnostics tagging the four canonical RAG failure modes.

1. RETRIEVAL_FAILURE          - the gold evidence document is not in the top-k context.
2. EVIDENCE_MISINTERPRETATION - gold evidence was retrieved, yet the answer contradicts it (intrinsic hallucination).
3. CONTEXT_OVERLOAD           - context exceeds the token budget or the gold evidence is buried behind
                                distractors (rank >= overload_rank), and the answer misses the gold fact.
4. SOURCE_RELIABILITY         - a low-reliability / outdated source was retrieved and the answer relies on it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from trust_llm.rag.fact_verifier import FactLabel, FactScoreResult
from trust_llm.rag.pipeline import RAGAnswer


class FailureMode(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    EVIDENCE_MISINTERPRETATION = "evidence_misinterpretation"
    CONTEXT_OVERLOAD = "context_overload"
    SOURCE_RELIABILITY = "source_reliability"


class Diagnosis(BaseModel):
    question_id: str
    question: str
    gold_doc: str
    retrieved_docs: list[str]
    gold_rank: int | None
    gold_fact_hit: bool
    factscore: float
    failure_modes: list[FailureMode] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failure_modes


def _normalise(s: str) -> str:
    return " ".join(s.lower().replace(",", "").split())


def gold_fact_present(answer: str, gold_facts: list[str]) -> bool:
    a = _normalise(answer)
    return all(_normalise(f) in a for f in gold_facts)


def diagnose(
    question_id: str,
    rag: RAGAnswer,
    fs: FactScoreResult,
    gold_doc: str,
    gold_facts: list[str],
    context_token_budget: int = 160,
    overload_rank: int = 3,
) -> Diagnosis:
    retrieved_docs = [r.doc_id for r in rag.retrieved]
    gold_rank = next((r.rank for r in rag.retrieved if r.doc_id == gold_doc), None)
    hit = gold_fact_present(rag.answer, gold_facts)
    d = Diagnosis(
        question_id=question_id,
        question=rag.question,
        gold_doc=gold_doc,
        retrieved_docs=retrieved_docs,
        gold_rank=gold_rank,
        gold_fact_hit=hit,
        factscore=fs.factscore,
    )
    if gold_rank is None:
        d.failure_modes.append(FailureMode.RETRIEVAL_FAILURE)
        d.notes.append(f"gold doc '{gold_doc}' missing from top-{len(rag.retrieved)}")
    elif fs.n_intrinsic > 0:
        d.failure_modes.append(FailureMode.EVIDENCE_MISINTERPRETATION)
        bad = [f.text for f in fs.facts if f.label == FactLabel.INTRINSIC]
        d.notes.append(f"contradicts retrieved evidence: {bad[:2]}")
    if not hit and gold_rank is not None and (rag.context_tokens > context_token_budget or gold_rank >= overload_rank):
        d.failure_modes.append(FailureMode.CONTEXT_OVERLOAD)
        d.notes.append(f"context_tokens={rag.context_tokens}, gold_rank={gold_rank}")
    low = [r for r in rag.retrieved if r.reliability == "low"]
    if low and any(r.doc_id in rag.answer for r in low):
        d.failure_modes.append(FailureMode.SOURCE_RELIABILITY)
        d.notes.append(f"answer cites low-reliability source(s): {sorted({r.doc_id for r in low})}")
    return d


# ---------------------------------------------------------------------------- evaluation ----

class RAGConfigResult(BaseModel):
    config: str
    n_questions: int
    mean_factscore: float
    gold_fact_accuracy: float
    retrieval_hit_at_k: float | None
    intrinsic_facts: int
    extrinsic_facts: int
    supported_facts: int
    total_facts: int
    hallucination_rate: float
    failure_mode_counts: dict[str, int]
    diagnoses: list[Diagnosis]


def evaluate_rag_config(pipeline, questions: list[dict], config: str, mode: str = "rag", k: int = 3,
                        verifier=None) -> RAGConfigResult:
    """Run one configuration over the QA set; baseline answers are verified against the gold document."""
    from trust_llm.rag.fact_verifier import FactVerifier

    verifier = verifier or FactVerifier()
    diags: list[Diagnosis] = []
    sup = intr = extr = 0
    hits = 0
    for q in questions:
        ans = pipeline.answer(q["question"], mode=mode, k=k)
        sources = [h.text for h in ans.retrieved] if mode == "rag" else [pipeline.doc_by_id[q["gold_doc"]].text]
        fs = verifier.score(ans.answer, sources)
        sup, intr, extr = sup + fs.n_supported, intr + fs.n_intrinsic, extr + fs.n_extrinsic
        if mode == "rag":
            d = diagnose(q["id"], ans, fs, q["gold_doc"], q["gold_facts"])
            hits += int(d.gold_rank is not None)
        else:
            d = Diagnosis(question_id=q["id"], question=q["question"], gold_doc=q["gold_doc"], retrieved_docs=[],
                          gold_rank=None, gold_fact_hit=gold_fact_present(ans.answer, q["gold_facts"]), factscore=fs.factscore)
        diags.append(d)
    total = sup + intr + extr
    counts = {m.value: sum(m in d.failure_modes for d in diags) for m in FailureMode}
    return RAGConfigResult(
        config=config,
        n_questions=len(questions),
        mean_factscore=sum(d.factscore for d in diags) / len(diags),
        gold_fact_accuracy=sum(d.gold_fact_hit for d in diags) / len(diags),
        retrieval_hit_at_k=hits / len(questions) if mode == "rag" else None,
        intrinsic_facts=intr,
        extrinsic_facts=extr,
        supported_facts=sup,
        total_facts=total,
        hallucination_rate=(intr + extr) / total if total else 0.0,
        failure_mode_counts=counts,
        diagnoses=diags,
    )
