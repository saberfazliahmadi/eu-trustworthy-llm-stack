"""FActScore-style factuality: decompose an answer into atomic facts, verify each against sources.

Labels per atomic fact:
* SUPPORTED  - a source sentence entails the fact (topical overlap + all numbers / negations consistent).
* INTRINSIC  - the fact contradicts the source (same topic, conflicting number or negation).
* EXTRINSIC  - the fact cannot be verified from the source (no supporting or contradicting evidence).
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

from trust_llm.text_utils import content_tokens, extract_numbers, jaccard, split_sentences

_NEGATION_RE = re.compile(r"\b(not|never|no|cannot|can't|nicht|kein\w*|nie|invalid|without)\b", re.IGNORECASE)
_PREAMBLE_RE = re.compile(r"^(based on (the )?(company )?policy documents:|according to [^:]+:)\s*", re.IGNORECASE)
_CITATION_RE = re.compile(r"\s*\[[\w#\-]+\]")
_ABSTAIN_RE = re.compile(r"(could not find this|i (don't|do not) know|not (contained|stated) in the (provided )?documents|"
                         r"can't follow that request|can't help with)", re.IGNORECASE)


class FactLabel(str, Enum):
    SUPPORTED = "supported"
    INTRINSIC = "intrinsic_hallucination"
    EXTRINSIC = "extrinsic_hallucination"


class AtomicFact(BaseModel):
    text: str
    label: FactLabel
    evidence: str = ""
    overlap: float = 0.0


class FactScoreResult(BaseModel):
    facts: list[AtomicFact] = Field(default_factory=list)
    factscore: float = 1.0
    n_supported: int = 0
    n_intrinsic: int = 0
    n_extrinsic: int = 0
    abstained: bool = False


def decompose(answer: str) -> list[str]:
    """Split an answer into atomic, independently checkable claims."""
    text = _CITATION_RE.sub("", _PREAMBLE_RE.sub("", answer.strip()))
    facts: list[str] = []
    for sent in split_sentences(text):
        for clause in re.split(r";\s+|,\s+and\s+(?=\w+\s+\w+)|\.\s+", sent):
            clause = clause.strip(" .")
            if len(content_tokens(clause)) >= 3 and extract_numbers(clause) or len(content_tokens(clause)) >= 4:
                facts.append(clause)
    return facts


def _numbers_consistent(fact_nums: list[float], source_nums: list[float]) -> bool:
    return all(n in source_nums for n in fact_nums)


class FactVerifier:
    """Source-grounded verifier; thresholds tuned on the bundled HR corpus."""

    def __init__(self, support_threshold: float = 0.30, topic_threshold: float = 0.12) -> None:
        self.support_threshold = support_threshold
        self.topic_threshold = topic_threshold

    def verify_fact(self, fact: str, sources: list[str]) -> AtomicFact:
        f_toks = set(t for t in content_tokens(fact) if not t.startswith("concept:"))
        f_nums = extract_numbers(fact)
        f_neg = bool(_NEGATION_RE.search(fact))
        best_sent, best_overlap = "", 0.0
        for src in sources:
            for sent in split_sentences(src):
                s_toks = set(t for t in content_tokens(sent) if not t.startswith("concept:"))
                ov = jaccard(f_toks, s_toks)
                # recall-oriented overlap: share of fact tokens found in source sentence
                recall = len(f_toks & s_toks) / max(1, len(f_toks))
                score = max(ov, 0.8 * recall)
                if score > best_overlap:
                    best_sent, best_overlap = sent, score
        if best_overlap < self.topic_threshold:
            return AtomicFact(text=fact, label=FactLabel.EXTRINSIC, evidence="", overlap=best_overlap)
        s_nums = extract_numbers(best_sent)
        s_neg = bool(_NEGATION_RE.search(best_sent))
        nums_ok = _numbers_consistent(f_nums, s_nums)
        if best_overlap >= self.support_threshold and nums_ok and f_neg == s_neg:
            return AtomicFact(text=fact, label=FactLabel.SUPPORTED, evidence=best_sent, overlap=best_overlap)
        if (f_nums and s_nums and not nums_ok) or (f_neg != s_neg and best_overlap >= self.support_threshold):
            return AtomicFact(text=fact, label=FactLabel.INTRINSIC, evidence=best_sent, overlap=best_overlap)
        if f_nums and not nums_ok:
            # numbers asserted that appear nowhere in the closest evidence -> unverifiable additions
            return AtomicFact(text=fact, label=FactLabel.EXTRINSIC, evidence=best_sent, overlap=best_overlap)
        return AtomicFact(text=fact, label=FactLabel.EXTRINSIC, evidence=best_sent, overlap=best_overlap)

    def score(self, answer: str, sources: list[str]) -> FactScoreResult:
        if _ABSTAIN_RE.search(answer):
            return FactScoreResult(facts=[], factscore=1.0, abstained=True)
        facts = decompose(answer)
        if not facts:
            return FactScoreResult(facts=[], factscore=1.0, abstained=True)
        verified = [self.verify_fact(f, sources) for f in facts]
        n_sup = sum(f.label == FactLabel.SUPPORTED for f in verified)
        n_int = sum(f.label == FactLabel.INTRINSIC for f in verified)
        n_ext = sum(f.label == FactLabel.EXTRINSIC for f in verified)
        return FactScoreResult(
            facts=verified,
            factscore=n_sup / len(verified),
            n_supported=n_sup,
            n_intrinsic=n_int,
            n_extrinsic=n_ext,
        )
