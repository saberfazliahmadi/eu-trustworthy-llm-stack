"""Decoding strategies benchmarked on a real (tiny) language model trained on the HR policy corpus.

The LM is an interpolated bigram/unigram model with add-k smoothing - small enough to run anywhere,
real enough that greedy loops, beam search maximises sequence likelihood, and temperature / top-k
sampling trades likelihood for diversity exactly as in large models.

Implemented: greedy decoding (argmax), beam search (width k, length-normalised), sampling with
temperature and top-k, and held-out perplexity.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from pydantic import BaseModel

from trust_llm.evaluation.metrics import perplexity
from trust_llm.rag.pipeline import load_documents

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def corpus_sentences() -> list[list[str]]:
    sents = []
    for doc in load_documents():
        for line in doc.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for s in re.split(r"(?<=[.!?])\s+", line):
                toks = re.findall(r"[A-Za-zÄÖÜäöüß]+|\d+(?:[.,]\d+)?|[(),;:]", s.lower())
                if len(toks) >= 4:
                    sents.append(toks)
    return sents


class BigramLM:
    """p(w | v) = lambda * (c(v,w)+k)/(c(v)+k|V|) + (1-lambda) * p_unigram(w)."""

    def __init__(self, k: float = 0.05, lam: float = 0.85) -> None:
        self.k, self.lam = k, lam
        self.vocab: list[str] = []
        self.idx: dict[str, int] = {}
        self.bigram: dict[str, Counter] = defaultdict(Counter)
        self.unigram: Counter = Counter()

    def fit(self, sentences: list[list[str]]) -> "BigramLM":
        for s in sentences:
            seq = [BOS] + s + [EOS]
            self.unigram.update(seq[1:])
            for a, b in zip(seq, seq[1:]):
                self.bigram[a][b] += 1
        self.unigram[UNK] += 1  # reserve probability mass for out-of-vocabulary words
        self.vocab = sorted(set(self.unigram) | {BOS})
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        total = sum(self.unigram.values())
        self._uni = np.array([self.unigram[w] / total for w in self.vocab])
        self._cache: dict[str, np.ndarray] = {}
        return self

    def next_dist(self, prev: str) -> np.ndarray:
        if prev not in self._cache:
            counts = self.bigram.get(prev, Counter())
            c_prev = sum(counts.values())
            v = len(self.vocab)
            big = np.array([(counts[w] + self.k) / (c_prev + self.k * v) for w in self.vocab])
            p = self.lam * big + (1 - self.lam) * self._uni
            p[self.idx[BOS]] = 0.0
            self._cache[prev] = p / p.sum()
        return self._cache[prev]

    def generation_dist(self, prev: str) -> np.ndarray:
        """Distribution used for decoding: <unk> is never emitted."""
        p = self.next_dist(prev if prev in self.idx else UNK).copy()
        p[self.idx[UNK]] = 0.0
        return p / p.sum()

    def logprob(self, prev: str, word: str) -> float:
        prev = prev if prev in self.idx else UNK
        return float(np.log(self.next_dist(prev)[self.idx.get(word, self.idx[UNK])]))

    def sequence_logprobs(self, tokens: list[str], context: list[str] | None = None) -> list[float]:
        prev = (context or [BOS])[-1]
        lps = []
        for t in tokens:
            lps.append(self.logprob(prev, t))
            prev = t
        return lps

    def perplexity(self, sentences: list[list[str]]) -> float:
        lps = []
        for s in sentences:
            lps.extend(self.sequence_logprobs(s + [EOS]))
        return perplexity(lps)


def greedy_decode(lm: BigramLM, prefix: list[str], max_len: int = 20) -> list[str]:
    out, prev = [], prefix[-1]
    for _ in range(max_len):
        w = lm.vocab[int(np.argmax(lm.generation_dist(prev)))]
        if w == EOS:
            break
        out.append(w)
        prev = w
    return out


def beam_search(lm: BigramLM, prefix: list[str], beam_width: int = 4, max_len: int = 20, length_penalty: float = 0.7) -> list[str]:
    beams: list[tuple[float, list[str], bool]] = [(0.0, [], False)]
    for _ in range(max_len):
        cand: list[tuple[float, list[str], bool]] = []
        for score, seq, done in beams:
            if done:
                cand.append((score, seq, True))
                continue
            prev = seq[-1] if seq else prefix[-1]
            dist = lm.generation_dist(prev)
            for j in np.argsort(-dist)[:beam_width]:
                w = lm.vocab[int(j)]
                cand.append((score + math.log(dist[j]), seq if w == EOS else seq + [w], w == EOS))
        cand.sort(key=lambda c: c[0] / max(1, len(c[1])) ** length_penalty, reverse=True)
        beams = cand[:beam_width]
        if all(d for _, _, d in beams):
            break
    best = max(beams, key=lambda c: c[0] / max(1, len(c[1])) ** length_penalty)
    return best[1]


def sample_decode(lm: BigramLM, prefix: list[str], temperature: float = 1.0, top_k: int | None = None,
                  max_len: int = 20, rng: np.random.Generator | None = None) -> list[str]:
    rng = rng or np.random.default_rng(42)
    out, prev = [], prefix[-1]
    for _ in range(max_len):
        logits = np.log(lm.generation_dist(prev) + 1e-30) / max(temperature, 1e-6)
        if top_k:
            cutoff = np.sort(logits)[-top_k]
            logits = np.where(logits >= cutoff, logits, -np.inf)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        w = lm.vocab[int(rng.choice(len(p), p=p))]
        if w == EOS:
            break
        out.append(w)
        prev = w
    return out


def distinct_n(seqs: list[list[str]], n: int = 2) -> float:
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(seq: list[str]) -> float:
    return 1 - len(set(seq)) / len(seq) if seq else 0.0


class DecodingResult(BaseModel):
    method: str
    avg_logprob_per_token: float
    perplexity_under_model: float
    distinct_2: float
    repetition_rate: float
    avg_length: float
    example: str


class DecodingBenchReport(BaseModel):
    vocab_size: int
    n_train_sentences: int
    n_heldout_sentences: int
    heldout_perplexity: float
    unigram_heldout_perplexity: float
    results: list[DecodingResult]


PREFIXES = [["the", "probation"], ["employees"], ["vacation"], ["the", "notice"], ["applicant", "data"], ["parental", "leave"],
            ["salary"], ["remote", "work"], ["the", "company"], ["every"]]


def run_decoding_bench(seed: int = 42) -> DecodingBenchReport:
    rng = np.random.default_rng(seed)
    sents = corpus_sentences()
    order = rng.permutation(len(sents))
    n_held = max(5, len(sents) // 10)
    held = [sents[i] for i in order[:n_held]]
    train = [sents[i] for i in order[n_held:]]
    lm = BigramLM().fit(train)
    uni = BigramLM(lam=0.0).fit(train)
    configs = [
        ("greedy", lambda p: greedy_decode(lm, p)),
        ("beam_k2", lambda p: beam_search(lm, p, 2)),
        ("beam_k5", lambda p: beam_search(lm, p, 5)),
        ("sample_T0.7", lambda p: sample_decode(lm, p, 0.7, rng=rng)),
        ("sample_T1.0", lambda p: sample_decode(lm, p, 1.0, rng=rng)),
        ("sample_T1.5", lambda p: sample_decode(lm, p, 1.5, rng=rng)),
        ("sample_T1.0_top10", lambda p: sample_decode(lm, p, 1.0, top_k=10, rng=rng)),
    ]
    results = []
    for name, fn in configs:
        outs, lps = [], []
        for pre in PREFIXES:
            seq = fn([BOS] + pre)
            outs.append(seq)
            lps.extend(lm.sequence_logprobs(seq + [EOS], context=[BOS] + pre))
        results.append(DecodingResult(
            method=name,
            avg_logprob_per_token=float(np.mean(lps)),
            perplexity_under_model=perplexity(lps),
            distinct_2=distinct_n(outs, 2),
            repetition_rate=float(np.mean([repetition_rate(o) for o in outs])),
            avg_length=float(np.mean([len(o) for o in outs])),
            example=" ".join(PREFIXES[0] + outs[0]),
        ))
    return DecodingBenchReport(
        vocab_size=len(lm.vocab), n_train_sentences=len(train), n_heldout_sentences=len(held),
        heldout_perplexity=lm.perplexity(held), unigram_heldout_perplexity=uni.perplexity(held), results=results,
    )
