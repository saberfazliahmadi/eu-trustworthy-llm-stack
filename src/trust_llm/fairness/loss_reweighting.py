"""Inverse-frequency group loss reweighting in PyTorch + diagnostics of its four known failure modes.

Per-sample losses use ``F.cross_entropy(logits, y, reduction="none", ignore_index=-100)`` so samples
without a label (-100) contribute exactly zero loss and zero weight. Group weights
``w_g = N_valid / (G * n_g)`` give every group an equal epoch-level share (1/G) of the total loss weight.

Failure modes diagnosed (each computed, not asserted):
1. Tokenizer fragmentation  - under token-level losses, groups whose text fragments into more tokens
                              silently regain extra weight; equal sample shares != equal token shares.
2. Pretraining bias         - reweighting the classifier head leaves associations in frozen
                              pretrained embeddings untouched.
3. Label-noise amplification - up-weighting a small group also up-weights its noisy labels.
4. Biased audit metric      - if group membership for the audit is inferred by a noisy proxy, the
                              measured worst-group accuracy is itself biased.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

IGNORE_INDEX = -100


def inverse_frequency_weights(groups: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample weights w_i = N_valid / (G * n_{g(i)}); zero for ignored samples."""
    valid = labels != IGNORE_INDEX
    weights = torch.zeros_like(groups, dtype=torch.float32)
    present = torch.unique(groups[valid])
    n_valid = int(valid.sum())
    for g in present:
        mask = (groups == g) & valid
        weights[mask] = n_valid / (len(present) * int(mask.sum()))
    return weights


def weighted_group_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, labels, reduction="none", ignore_index=IGNORE_INDEX)
    valid = (labels != IGNORE_INDEX).float()
    return (per_sample * weights * valid).sum() / (weights * valid).sum().clamp_min(1e-12)


def group_weight_shares(groups: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> dict[int, float]:
    valid = labels != IGNORE_INDEX
    total = float((weights * valid).sum())
    return {int(g): float((weights * ((groups == g) & valid)).sum()) / total for g in torch.unique(groups[valid])}


class BagClassifier(nn.Module):
    """Mean-pooled embedding bag + linear head (optionally with frozen 'pretrained' embeddings)."""

    def __init__(self, vocab_size: int, dim: int = 16, n_classes: int = 2, pretrained: torch.Tensor | None = None) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, dim, mode="mean", padding_idx=0)
        if pretrained is not None:
            self.emb.weight.data.copy_(pretrained)
            self.emb.weight.requires_grad_(False)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(tokens))


class SyntheticGroupData(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    tokens: torch.Tensor
    labels: torch.Tensor
    groups: torch.Tensor
    vocab_size: int


def make_group_data(n: int = 1600, minority_share: float = 0.12, seq_len: int = 12, seed: int = 42,
                    unlabeled_share: float = 0.05, vocab_size: int = 120) -> SyntheticGroupData:
    """Two groups whose label signal lives in *different* token ranges (e.g. different languages/dialects).

    Majority signal tokens: 1-10 (class 0) / 11-20 (class 1), three per sequence. Minority: one token from
    21-30 (class 0) / 26-35 (class 1) - weaker and partially ambiguous.
    A spurious majority-correlated token 41 appears with class 1 mostly in the majority group.
    """
    rng = np.random.default_rng(seed)
    groups = (rng.random(n) < minority_share).astype(np.int64)
    labels = rng.integers(0, 2, n)
    toks = rng.integers(50, vocab_size, (n, seq_len))
    for i in range(n):
        if groups[i] == 0:
            lo, k = (1 if labels[i] == 0 else 11), 3
        else:
            lo, k = (21 if labels[i] == 0 else 26), 1  # weak, partially overlapping minority signal
        pos = rng.choice(seq_len, k, replace=False)
        toks[i, pos] = rng.integers(lo, lo + 10, k)
        if groups[i] == 0 and labels[i] == 1 and rng.random() < 0.8:
            toks[i, rng.integers(seq_len)] = 41
        if groups[i] == 1 and labels[i] == 0 and rng.random() < 0.6:
            toks[i, rng.integers(seq_len)] = 41  # spurious cue points the wrong way for the minority
    y = labels.copy()
    y[rng.random(n) < unlabeled_share] = IGNORE_INDEX
    return SyntheticGroupData(tokens=torch.tensor(toks), labels=torch.tensor(y), groups=torch.tensor(groups), vocab_size=vocab_size)


def train(data: SyntheticGroupData, reweight: bool, epochs: int = 40, lr: float = 0.05, seed: int = 42,
          pretrained: torch.Tensor | None = None) -> BagClassifier:
    torch.manual_seed(seed)
    model = BagClassifier(data.vocab_size, pretrained=pretrained)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    if reweight:
        w = inverse_frequency_weights(data.groups, data.labels)
    else:
        w = (data.labels != IGNORE_INDEX).float()
    for _ in range(epochs):
        opt.zero_grad()
        loss = weighted_group_loss(model(data.tokens), data.labels, w)
        loss.backward()
        opt.step()
    return model


def group_accuracy(model: nn.Module, data: SyntheticGroupData, groups: torch.Tensor | None = None) -> dict[int, float]:
    groups = data.groups if groups is None else groups
    with torch.no_grad():
        pred = model(data.tokens).argmax(-1)
    valid = data.labels != IGNORE_INDEX
    out = {}
    for g in torch.unique(groups):
        m = (groups == g) & valid
        out[int(g)] = float((pred[m] == data.labels[m]).float().mean())
    return out


def per_group_gradient_norm_share(model: nn.Module, data: SyntheticGroupData, weights: torch.Tensor) -> dict[int, float]:
    """Norm of the gradient contributed by each group's weighted loss (diagnostic, not expected to be equal)."""
    norms = {}
    valid = (data.labels != IGNORE_INDEX).float()
    total_w = (weights * valid).sum()
    for g in torch.unique(data.groups):
        model.zero_grad()
        per = F.cross_entropy(model(data.tokens), data.labels, reduction="none", ignore_index=IGNORE_INDEX)
        mask = (data.groups == g).float()
        ((per * weights * valid * mask).sum() / total_w).backward()
        norms[int(g)] = float(torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)))
    s = sum(norms.values())
    return {k: v / s for k, v in norms.items()}


# ----------------------------------------------------------------------------- failure modes ----

def greedy_subword_tokenize(word: str, vocab: set[str]) -> list[str]:
    """WordPiece-style greedy longest-match; unknown characters become single-char pieces."""
    word = word.lower()
    pieces, i = [], 0
    while i < len(word):
        for j in range(len(word), i, -1):
            piece = word[i:j] if i == 0 else "##" + word[i:j]
            if piece in vocab or j == i + 1:
                pieces.append(piece)
                i = j
                break
    return pieces


def build_subword_vocab(corpus_words: list[str], max_len: int = 6) -> set[str]:
    vocab: set[str] = set()
    for w in corpus_words:
        w = w.lower()
        for n in range(2, max_len + 1):
            if len(w) >= n:
                vocab.add(w[:n])
            for i in range(1, len(w) - n + 1):
                vocab.add("##" + w[i:i + n])
    return vocab


class ReweightingReport(BaseModel):
    weight_share_unweighted: dict[int, float]
    weight_share_weighted: dict[int, float]
    equal_shares_verified: bool
    gradient_norm_share_weighted: dict[int, float]
    group_sizes: dict[int, int]
    n_ignored: int
    acc_unweighted: dict[int, float]
    acc_weighted: dict[int, float]
    worst_group_unweighted: float
    worst_group_weighted: float
    # failure modes
    fragmentation_tokens_per_name: dict[str, float]
    token_level_share_minority: float
    pretrained_bias_before: float
    pretrained_bias_after: float
    noisy_weight_share_unweighted: float
    noisy_weight_share_weighted: float
    minority_acc_noisy_unweighted: float
    minority_acc_noisy_weighted: float
    true_worst_group_acc: float
    proxy_audited_worst_group_acc: float


GERMAN_NAMES = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann",
                "Koch", "Richter", "Klein", "Wolf", "Neumann", "Schwarz", "Zimmermann", "Braun", "Krüger", "Hofmann"]
INTERNATIONAL_NAMES = ["Yılmaz", "Okafor", "Nguyen", "Kowalczyk", "Haddad", "Chukwuemeka", "Abdelrahman", "Papadopoulos",
                       "Oyelaran", "Vaidyanathan"]


def run_reweighting_study(seed: int = 42) -> ReweightingReport:
    torch.manual_seed(seed)
    data = make_group_data(seed=seed)
    test = make_group_data(seed=seed + 1, unlabeled_share=0.0)
    unw = (data.labels != IGNORE_INDEX).float()
    w = inverse_frequency_weights(data.groups, data.labels)
    share_u = group_weight_shares(data.groups, data.labels, unw)
    share_w = group_weight_shares(data.groups, data.labels, w)
    equal = all(abs(v - 1 / len(share_w)) < 1e-6 for v in share_w.values())

    m_u = train(data, reweight=False, seed=seed)
    m_w = train(data, reweight=True, seed=seed)
    acc_u, acc_w = group_accuracy(m_u, test), group_accuracy(m_w, test)
    grad_share = per_group_gradient_norm_share(m_w, data, w)

    # (1) Tokenizer fragmentation: vocab learned on German surnames fragments international ones.
    vocab = build_subword_vocab(GERMAN_NAMES * 3)
    tpn_de = float(np.mean([len(greedy_subword_tokenize(n, vocab)) for n in GERMAN_NAMES]))
    tpn_int = float(np.mean([len(greedy_subword_tokenize(n, vocab)) for n in INTERNATIONAL_NAMES]))
    # With token-level loss averaging, each group's share is proportional to (sample weight x tokens).
    n0, n1 = int(((data.groups == 0) & (data.labels != IGNORE_INDEX)).sum()), int(((data.groups == 1) & (data.labels != IGNORE_INDEX)).sum())
    base_len = 12
    tok0, tok1 = base_len + tpn_de, base_len + tpn_int
    w0, w1 = (n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)
    token_share_min = (w1 * n1 * tok1) / (w0 * n0 * tok0 + w1 * n1 * tok1)

    # (2) Pretraining bias: frozen embeddings with an injected association direction.
    g = torch.Generator().manual_seed(seed)
    pre = torch.randn(data.vocab_size, 16, generator=g) * 0.3
    bias_dir = torch.zeros(16)
    bias_dir[0] = 1.0
    pre[21:41] += 0.8 * bias_dir  # minority-signal tokens share a direction with ...
    pre[60:70] += 0.8 * bias_dir  # ... an arbitrary 'negative attribute' token block
    def assoc(emb: torch.Tensor) -> float:
        a = F.normalize(emb[21:41], dim=-1)
        b = F.normalize(emb[60:70], dim=-1)
        c = F.normalize(emb[80:90], dim=-1)
        return float((a @ b.T).mean() - (a @ c.T).mean())
    before = assoc(pre)
    m_pre = train(data, reweight=True, seed=seed, pretrained=pre.clone())
    after = assoc(m_pre.emb.weight.detach())

    # (3) Label-noise amplification: flip 25% of minority labels.
    rng = np.random.default_rng(seed)
    noisy_labels = data.labels.clone()
    minority_idx = torch.nonzero((data.groups == 1) & (data.labels != IGNORE_INDEX)).flatten().numpy()
    flip = rng.choice(minority_idx, int(0.25 * len(minority_idx)), replace=False)
    noisy_labels[flip] = 1 - noisy_labels[flip]
    noisy_mask = torch.zeros_like(noisy_labels, dtype=torch.bool)
    noisy_mask[flip] = True
    noisy = SyntheticGroupData(tokens=data.tokens, labels=noisy_labels, groups=data.groups, vocab_size=data.vocab_size)
    wn = inverse_frequency_weights(noisy.groups, noisy.labels)
    un = (noisy.labels != IGNORE_INDEX).float()
    share_noisy_u = float((un * noisy_mask).sum() / un.sum())
    share_noisy_w = float((wn * noisy_mask).sum() / wn.sum())
    mn_u, mn_w = train(noisy, reweight=False, seed=seed), train(noisy, reweight=True, seed=seed)
    acc_noisy_u = group_accuracy(mn_u, test)[1]  # evaluated on clean held-out labels
    acc_noisy_w = group_accuracy(mn_w, test)[1]

    # (4) Biased audit metric: group inferred by a proxy with 30% minority miss-rate.
    proxy = test.groups.clone()
    miss = torch.nonzero(test.groups == 1).flatten().numpy()
    miss = rng.choice(miss, int(0.3 * len(miss)), replace=False)
    proxy[miss] = 0
    true_worst = min(acc_w.values())
    proxy_worst = min(group_accuracy(m_w, test, groups=proxy).values())

    return ReweightingReport(
        weight_share_unweighted=share_u,
        weight_share_weighted=share_w,
        equal_shares_verified=equal,
        gradient_norm_share_weighted=grad_share,
        group_sizes={0: int((data.groups == 0).sum()), 1: int((data.groups == 1).sum())},
        n_ignored=int((data.labels == IGNORE_INDEX).sum()),
        acc_unweighted=acc_u,
        acc_weighted=acc_w,
        worst_group_unweighted=min(acc_u.values()),
        worst_group_weighted=min(acc_w.values()),
        fragmentation_tokens_per_name={"german": tpn_de, "international": tpn_int},
        token_level_share_minority=token_share_min,
        pretrained_bias_before=before,
        pretrained_bias_after=after,
        noisy_weight_share_unweighted=share_noisy_u,
        noisy_weight_share_weighted=share_noisy_w,
        minority_acc_noisy_unweighted=acc_noisy_u,
        minority_acc_noisy_weighted=acc_noisy_w,
        true_worst_group_acc=true_worst,
        proxy_audited_worst_group_acc=proxy_worst,
    )
