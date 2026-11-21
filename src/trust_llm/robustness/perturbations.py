"""Perturbation generators for the four Language Traps and a distribution-shift simulator.

Language traps:
  1. Lexical       - keyboard typos, character swaps / drops (QWERTZ layout).
  2. Semantic      - meaning-preserving paraphrases.
  3. Domain shift  - same question in a different register (legalese / informal chat).
  4. Multilingual  - EN -> DE code-switching and pronoun swaps (he <-> she).

Distribution shifts (on a seeded 2-D classification task):
  covariate P(x) changes, label P(y) changes, concept P(y|x) changes.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel

_QWERTZ_NEIGHBOURS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "rzg", "z": "tuh", "u": "zij", "i": "uok", "o": "ipl", "p": "oü",
    "a": "qsy", "s": "adw", "d": "sfe", "f": "dgr", "g": "fht", "h": "gjz", "j": "hku", "k": "jli", "l": "kö",
    "y": "xa", "x": "ycs", "c": "xvd", "v": "cbf", "b": "vng", "n": "bmh", "m": "nj",
}

_PARAPHRASES = [
    (r"\bhow many\b", "what number of"),
    (r"\bhow long\b", "for what duration"),
    (r"\bvacation days\b", "days of paid leave"),
    (r"\bprobation period\b", "trial phase"),
    (r"\bnotice period\b", "termination notice time"),
    (r"\bcan i\b", "am I allowed to"),
    (r"\bwhat is\b", "could you tell me"),
    (r"\bremotely\b", "from home"),
    (r"\bper year\b", "annually"),
    (r"\bemployees\b", "staff members"),
    (r"\bstored\b", "kept"),
    (r"\brequest\b", "apply for"),
]

_DOMAIN_REGISTERS = {
    "legalese": ("Pursuant to the applicable employment terms, kindly clarify the following: {q} "
                 "Please reference the relevant contractual provision."),
    "chat": "hey quick q lol, my manager is being weird about it... {q} thx!!",
}

_EN_DE = [
    (r"\bvacation days\b", "Urlaubstage"), (r"\bvacation\b", "Urlaub"), (r"\bprobation period\b", "Probezeit"),
    (r"\bnotice period\b", "Kündigungsfrist"), (r"\bsalary band\b", "Gehaltsband"), (r"\bsalary\b", "Gehalt"),
    (r"\bparental leave\b", "Elternzeit"), (r"\bremote work\b", "Homeoffice"), (r"\bwork remotely\b", "im Homeoffice arbeiten"),
    (r"\bhow many\b", "wie viele"), (r"\bhow long\b", "wie lange"), (r"\bdays\b", "Tage"), (r"\bweeks\b", "Wochen"),
    (r"\bper year\b", "pro Jahr"), (r"\bemployees\b", "Mitarbeitende"), (r"\bapplicant data\b", "Bewerberdaten"),
    (r"\bmaternity protection\b", "Mutterschutz"), (r"\btermination\b", "Kündigung"), (r"\blabour court\b", "Arbeitsgericht"),
]

_PRONOUN_SWAP = {"he": "she", "she": "he", "him": "her", "his": "her", "her": "his", "himself": "herself", "herself": "himself"}


def lexical_typos(text: str, rate: float = 0.15, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    words = text.split(" ")
    out = []
    for w in words:
        letters = [i for i, ch in enumerate(w) if ch.isalpha()]
        if len(letters) >= 4 and rng.random() < rate:
            op = int(rng.integers(0, 3))
            i = int(rng.choice(letters[1:-1]))
            ch = w[i].lower()
            if op == 0 and i + 1 < len(w):  # transpose
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == 1:  # drop
                w = w[:i] + w[i + 1:]
            else:  # keyboard neighbour
                nb = _QWERTZ_NEIGHBOURS.get(ch, ch)
                w = w[:i] + nb[int(rng.integers(0, len(nb)))] + w[i + 1:]
        out.append(w)
    return " ".join(out)


def _apply_rules(text: str, rules: list[tuple[str, str]]) -> str:
    for pat, repl in rules:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def semantic_paraphrase(text: str) -> str:
    return _apply_rules(text, _PARAPHRASES)


def domain_shift(text: str, register: str = "legalese") -> str:
    return _DOMAIN_REGISTERS[register].format(q=text)


def multilingual_swap(text: str) -> str:
    return _apply_rules(text, _EN_DE)


def pronoun_swap(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        s = _PRONOUN_SWAP[w.lower()]
        return s.capitalize() if w[0].isupper() else s

    return re.sub(r"\b(he|she|him|his|her|himself|herself)\b", repl, text, flags=re.IGNORECASE)


LANGUAGE_TRAPS = {
    "lexical_typos": lambda q, i: lexical_typos(q, rate=0.35, seed=42 + i),
    "semantic_paraphrase": lambda q, i: semantic_paraphrase(q),
    "domain_shift": lambda q, i: domain_shift(q, "legalese" if i % 2 == 0 else "chat"),
    "multilingual_swap": lambda q, i: multilingual_swap(q),
}


# --------------------------------------------------------------------- distribution shift ----

class LogisticRegressionNP:
    """Minimal batch-gradient logistic regression (numpy, deterministic)."""

    def __init__(self, lr: float = 0.5, epochs: int = 400) -> None:
        self.lr, self.epochs = lr, epochs
        self.w = np.zeros(3)

    @staticmethod
    def _aug(x: np.ndarray) -> np.ndarray:
        return np.hstack([x, np.ones((len(x), 1))])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionNP":
        xa = self._aug(x)
        self.w = np.zeros(xa.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-xa @ self.w))
            self.w -= self.lr * xa.T @ (p - y) / len(y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self._aug(x) @ self.w))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


class ShiftResult(BaseModel):
    shift: str
    accuracy: float
    accuracy_drop: float
    input_mean_shift: float
    label_prior: float
    predicted_prior: float


class ShiftReport(BaseModel):
    source_accuracy: float
    source_label_prior: float
    results: list[ShiftResult]


class DistributionShiftSimulator:
    """Source: x ~ N(0, 1.2^2 I), y = 1[x1 + x2 - 0.15 x1^2 > -0.3]. Each shift modifies exactly one factor."""

    def __init__(self, n: int = 2000, seed: int = 42) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _label(x: np.ndarray, rotate: float = 0.0) -> np.ndarray:
        """Mildly non-linear true rule; a linear model fits it well only near the source region."""
        c, s = np.cos(rotate), np.sin(rotate)
        u = x[:, 0] * c - x[:, 1] * s
        v = x[:, 0] * s + x[:, 1] * c
        return (u + v - 0.15 * u**2 > -0.3).astype(int)

    def source(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.rng.normal(0, 1.2, (self.n, 2))
        return x, self._label(x)

    def covariate(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.rng.normal(0, 1.2, (self.n, 2)) * np.array([2.5, 0.4]) + np.array([1.5, -1.5])
        return x, self._label(x)  # P(y|x) unchanged

    def label(self, prior: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        while sum(len(a) for a in xs) < self.n:
            x = self.rng.normal(0, 1.2, (self.n, 2))
            y = self._label(x)
            keep = np.where(y == 1, self.rng.random(len(y)) < prior, self.rng.random(len(y)) < (1 - prior))
            xs.append(x[keep])
            ys.append(y[keep])
        return np.vstack(xs)[: self.n], np.concatenate(ys)[: self.n]

    def concept(self, rotate: float = np.pi / 3) -> tuple[np.ndarray, np.ndarray]:
        x = self.rng.normal(0, 1.2, (self.n, 2))
        return x, self._label(x, rotate=rotate)  # P(x) unchanged, P(y|x) changed (e.g. a policy rule changed)

    def run(self) -> ShiftReport:
        xs, ys = self.source()
        model = LogisticRegressionNP().fit(xs, ys)
        src_acc = float((model.predict(xs) == ys).mean())
        results = []
        for name, (x, y) in {"covariate": self.covariate(), "label": self.label(), "concept": self.concept()}.items():
            pred = model.predict(x)
            acc = float((pred == y).mean())
            results.append(ShiftResult(
                shift=name, accuracy=acc, accuracy_drop=src_acc - acc,
                input_mean_shift=float(np.linalg.norm(x.mean(0) - xs.mean(0))),
                label_prior=float(y.mean()), predicted_prior=float(pred.mean()),
            ))
        return ShiftReport(source_accuracy=src_acc, source_label_prior=float(ys.mean()), results=results)


class TrapResult(BaseModel):
    trap: str
    retrieval_hit_at_1: float
    retrieval_hit_at_3: float
    gold_fact_accuracy: float
    examples: list[str]


def evaluate_language_traps(pipeline, questions: list[dict]) -> list[TrapResult]:
    """Retrieval hit@1 / hit@3 and end-to-end gold-fact accuracy of the RAG pipeline under each trap."""
    from trust_llm.rag.diagnostics import gold_fact_present

    traps = {"clean": lambda q, i: q, **LANGUAGE_TRAPS}
    out = []
    for name, fn in traps.items():
        h1 = h3 = acc = 0
        examples = []
        for i, q in enumerate(questions):
            text = fn(q["question"], i)
            if len(examples) < 2:
                examples.append(text)
            hits = pipeline.retrieve(text, k=3)
            docs = [h.doc_id for h in hits]
            h1 += int(bool(docs) and docs[0] == q["gold_doc"])
            h3 += int(q["gold_doc"] in docs)
            acc += int(gold_fact_present(pipeline.answer(text, retrieved=hits).answer, q["gold_facts"]))
        n = len(questions)
        out.append(TrapResult(trap=name, retrieval_hit_at_1=h1 / n, retrieval_hit_at_3=h3 / n, gold_fact_accuracy=acc / n, examples=examples))
    return out
