import json

import numpy as np
import pytest

from trust_llm.config import DATA_DIR
from trust_llm.rag.diagnostics import FailureMode, diagnose, evaluate_rag_config, gold_fact_present
from trust_llm.rag.fact_verifier import FactLabel, FactVerifier, decompose
from trust_llm.rag.pipeline import RAGAnswer, RetrievedChunk, chunk_document, load_documents

QUESTIONS = json.loads((DATA_DIR / "rag_eval.json").read_text(encoding="utf-8"))


def test_corpus_has_ten_policy_documents_with_metadata():
    docs = load_documents()
    assert len(docs) == 10
    assert {d.reliability for d in docs} == {"high", "low"}
    assert all(chunk_document(d) for d in docs)


def test_embeddings_are_deterministic_and_normalised(embedder):
    a = embedder.embed(["Wie lange dauert die Probezeit?", "vacation days"])
    b = embedder.embed(["Wie lange dauert die Probezeit?", "vacation days"])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_cosine_retrieval_finds_gold_doc_including_german_queries(rag):
    hits = sum(q["gold_doc"] in [h.doc_id for h in rag.retrieve(q["question"], k=3)] for q in QUESTIONS)
    assert hits / len(QUESTIONS) >= 0.9
    top = rag.retrieve("Wie viele Tage pro Woche darf ich im Homeoffice arbeiten?", k=1)[0]
    assert top.doc_id == "remote_work"
    scores = [h.score for h in rag.retrieve("vacation days", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_reliability_filter_excludes_low_reliability_sources(reliable_rag):
    for q in QUESTIONS:
        assert all(h.reliability == "high" for h in reliable_rag.retrieve(q["question"], k=5))


def test_grounded_answer_cites_sources(rag):
    ans = rag.answer("How many vacation days do full-time employees get per year?")
    assert "30 vacation days" in ans.answer
    assert "[vacation]" in ans.answer


def test_decomposition_strips_citations_and_preamble():
    facts = decompose("Based on the company policy documents: The probation period is 6 months. [probation] "
                      "Unused days must be taken by 31 March. [vacation]")
    assert facts == ["The probation period is 6 months", "Unused days must be taken by 31 March"]


def test_fact_verifier_detects_intrinsic_and_extrinsic_hallucinations():
    src = ["All new permanent employees have a probation period of 6 months. During the probation period the notice period is 2 weeks."]
    v = FactVerifier()
    assert v.verify_fact("The probation period is 6 months", src).label == FactLabel.SUPPORTED
    assert v.verify_fact("The probation period is 3 months", src).label == FactLabel.INTRINSIC
    assert v.verify_fact("Employees receive a company car after two years of service", src).label == FactLabel.EXTRINSIC
    res = v.score("The probation period is 3 months. Employees receive a free company car.", src)
    assert res.n_intrinsic == 1 and res.n_extrinsic == 1 and res.factscore == 0.0


def test_abstention_is_not_scored_as_hallucination():
    res = FactVerifier().score("I could not find this in the provided policy documents. Please contact the HR team.", ["x"])
    assert res.abstained and res.factscore == 1.0


def test_rag_beats_no_retrieval_baseline(rag):
    base = evaluate_rag_config(rag, QUESTIONS, "baseline", mode="no_retrieval")
    grounded = evaluate_rag_config(rag, QUESTIONS, "rag", mode="rag", k=3)
    assert grounded.mean_factscore > base.mean_factscore + 0.5
    assert grounded.gold_fact_accuracy > base.gold_fact_accuracy
    assert base.intrinsic_facts > grounded.intrinsic_facts


def test_diagnostics_tags_all_four_failure_modes():
    def chunk(doc, rank, rel="high", text="Vacation is 30 days per year."):
        return RetrievedChunk(chunk_id=f"{doc}#0", doc_id=doc, title=doc, text=text, reliability=rel, score=0.5, rank=rank)

    fv = FactVerifier()
    # retrieval failure
    a = RAGAnswer(question="q", answer="x", mode="rag", retrieved=[chunk("remote_work", 0)])
    assert FailureMode.RETRIEVAL_FAILURE in diagnose("q", a, fv.score(a.answer, ["x"]), "vacation", ["30 days"]).failure_modes
    # evidence misinterpretation
    src = "Full-time employees receive 30 vacation days per calendar year."
    a = RAGAnswer(question="q", answer="Full-time employees receive 32 vacation days per calendar year.", mode="rag",
                  retrieved=[chunk("vacation", 0, text=src)])
    d = diagnose("q", a, fv.score(a.answer, [src]), "vacation", ["30 vacation days"])
    assert FailureMode.EVIDENCE_MISINTERPRETATION in d.failure_modes
    # context overload: gold evidence buried at rank 5 of 8 and answer misses the gold fact
    chunks = [chunk(f"d{i}", i) for i in range(8)]
    chunks[5] = chunk("vacation", 5, text=src)
    a = RAGAnswer(question="q", answer="I could not find this.", mode="rag", retrieved=chunks, context_tokens=500)
    assert FailureMode.CONTEXT_OVERLOAD in diagnose("q", a, fv.score(a.answer, [src]), "vacation", ["30 vacation days"]).failure_modes
    # source reliability
    a = RAGAnswer(question="q", answer="Probation can be extended to 12 months [forum_rumors]", mode="rag",
                  retrieved=[chunk("probation", 0), chunk("forum_rumors", 1, rel="low")])
    assert FailureMode.SOURCE_RELIABILITY in diagnose("q", a, fv.score(a.answer, ["x"]), "probation", ["6 months"]).failure_modes


@pytest.mark.parametrize("answer,facts,expected", [("valid for 30 EUR per month", ["30 EUR"], True), ("76.000 to 92,000", ["76,000 to 92,000"], False)])
def test_gold_fact_matching(answer, facts, expected):
    assert gold_fact_present(answer, facts) is expected
