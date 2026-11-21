"""Word Embedding Association Test (Caliskan et al., 2017) on internal embeddings.

s(w, A, B) = mean_a cos(w, a) - mean_b cos(w, b)
effect size d = (mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)) / std_{w in X u Y} s(w, A, B)
p-value: one-sided permutation test over equal-size re-partitions of X u Y.

The internal embeddings are PPMI + truncated-SVD vectors learned from a seeded synthetic corpus of HR
sentences with stereotyped co-occurrence. Counterfactual Data Augmentation (gender-swapped copies of
every sentence) is applied as a mitigation and the effect size is re-measured.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
from pydantic import BaseModel

CAREER = ["management", "salary", "professional", "career", "office", "business", "leadership", "engineering"]
FAMILY = ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]
MALE = ["lukas", "jonas", "felix", "paul", "leon", "max", "elias", "noah"]
FEMALE = ["anna", "leonie", "mia", "emma", "lena", "sophie", "hannah", "marie"]
FILLER = ["the", "team", "meeting", "today", "project", "week", "plan", "talks", "about", "with", "new", "discusses"]


def synthetic_corpus(n_sentences: int = 6000, stereotype: float = 0.8, seed: int = 42) -> list[list[str]]:
    """Male names co-occur with career words (and female names with family words) with prob `stereotype`."""
    rng = np.random.default_rng(seed)
    corpus = []
    for _ in range(n_sentences):
        male = rng.random() < 0.5
        name = rng.choice(MALE if male else FEMALE)
        congruent = rng.random() < stereotype
        pool = (CAREER if male else FAMILY) if congruent else (FAMILY if male else CAREER)
        words = [str(name)] + list(rng.choice(FILLER, 3)) + list(rng.choice(pool, 2))
        rng.shuffle(words)
        corpus.append([str(w) for w in words])
    return corpus


def gender_swap_cda(corpus: list[list[str]]) -> list[list[str]]:
    """CDA: append a copy of every sentence with male<->female names swapped."""
    swap = dict(zip(MALE, FEMALE)) | dict(zip(FEMALE, MALE))
    return corpus + [[swap.get(w, w) for w in sent] for sent in corpus]


def ppmi_svd_embeddings(corpus: list[list[str]], dim: int = 24) -> dict[str, np.ndarray]:
    """Sentence-level co-occurrence -> positive PMI -> truncated SVD (deterministic)."""
    vocab = sorted({w for s in corpus for w in s})
    idx = {w: i for i, w in enumerate(vocab)}
    co = np.zeros((len(vocab), len(vocab)))
    for sent in corpus:
        ids = [idx[w] for w in sent]
        for a, b in combinations(ids, 2):
            if a != b:
                co[a, b] += 1
                co[b, a] += 1
    total = co.sum()
    row = co.sum(1, keepdims=True)
    col = co.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co * total) / (row * col))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    vecs = u[:, :dim] * np.sqrt(s[:dim])
    # deterministic sign convention
    signs = np.sign(vecs[np.abs(vecs).argmax(0), range(vecs.shape[1])])
    vecs = vecs * signs
    return {w: vecs[i] for w, i in idx.items()}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def association(w: np.ndarray, A: list[np.ndarray], B: list[np.ndarray]) -> float:
    return float(np.mean([_cos(w, a) for a in A]) - np.mean([_cos(w, b) for b in B]))


class WEATResult(BaseModel):
    effect_size: float
    test_statistic: float
    p_value: float
    n_permutations: int


def weat(emb: dict[str, np.ndarray], X: list[str], Y: list[str], A: list[str], B: list[str],
         n_permutations: int = 2000, seed: int = 42) -> WEATResult:
    Av, Bv = [emb[a] for a in A], [emb[b] for b in B]
    s = {w: association(emb[w], Av, Bv) for w in X + Y}
    sx, sy = [s[w] for w in X], [s[w] for w in Y]
    stat = float(np.sum(sx) - np.sum(sy))
    diff, sd = float(np.mean(sx) - np.mean(sy)), float(np.std(sx + sy, ddof=1))
    # Exactly symmetric embeddings (e.g. after CDA) give 0/0 up to float noise: report d = 0.
    d = 0.0 if abs(diff) < 1e-9 or sd < 1e-12 else diff / sd
    rng = np.random.default_rng(seed)
    pool = np.array(sx + sy)
    # Tolerance makes the test platform-independent: when all associations are float noise (~1e-16, e.g. after CDA),
    # an exact `>=` would be decided by BLAS/LAPACK rounding instead of by the data.
    tol = 1e-9 * max(1.0, float(np.abs(pool).sum()))
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(pool))
        if pool[perm[: len(X)]].sum() - pool[perm[len(X):]].sum() >= stat - tol:
            exceed += 1
    return WEATResult(effect_size=d, test_statistic=stat, p_value=(exceed + 1) / (n_permutations + 1), n_permutations=n_permutations)


class WEATStudy(BaseModel):
    biased_corpus: WEATResult
    cda_corpus: WEATResult
    hashing_embeddings: WEATResult
    corpus_size: int
    top_cooccurrence: dict[str, list[str]]


def run_weat_study(seed: int = 42, embedder=None) -> WEATStudy:
    corpus = synthetic_corpus(seed=seed)
    biased = weat(ppmi_svd_embeddings(corpus), CAREER, FAMILY, MALE, FEMALE, seed=seed)
    cda = weat(ppmi_svd_embeddings(gender_swap_cda(corpus)), CAREER, FAMILY, MALE, FEMALE, seed=seed)
    if embedder is None:
        from trust_llm.providers.mock_provider import HashingEmbeddingProvider

        embedder = HashingEmbeddingProvider(seed=seed)
    words = CAREER + FAMILY + MALE + FEMALE
    vecs = embedder.embed(words)
    hashing = weat(dict(zip(words, vecs)), CAREER, FAMILY, MALE, FEMALE, seed=seed)
    counts: dict[str, Counter] = {"lukas": Counter(), "anna": Counter()}
    for sent in corpus:
        for nm in counts:
            if nm in sent:
                counts[nm].update(w for w in sent if w in CAREER + FAMILY)
    return WEATStudy(
        biased_corpus=biased,
        cda_corpus=cda,
        hashing_embeddings=hashing,
        corpus_size=len(corpus),
        top_cooccurrence={k: [w for w, _ in v.most_common(4)] for k, v in counts.items()},
    )
