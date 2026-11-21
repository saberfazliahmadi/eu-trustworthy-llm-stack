import torch
import torch.nn.functional as F

from trust_llm.fairness.counterfactuals import blind_cv, cda_augment, counterfactual_flip_test, load_fairness_data, neutralize_proxies, render_cv, swap_names
from trust_llm.fairness.loss_reweighting import (
    IGNORE_INDEX, group_weight_shares, inverse_frequency_weights, run_reweighting_study, weighted_group_loss,
)
from trust_llm.fairness.subgroup_audit import audit, run_screening
from trust_llm.fairness.weat import CAREER, FAMILY, FEMALE, MALE, gender_swap_cda, ppmi_svd_embeddings, synthetic_corpus, weat
from trust_llm.providers.mock_provider import MockLLMProvider

DATA = load_fairness_data()


def test_dataset_shape():
    assert len(DATA["profiles"]) == 20 and len(DATA["identities"]) == 8
    assert ["Spencer Miller", "Latisha Washington"] in DATA["counterfactual_name_pairs"]


def test_name_swap_and_proxy_neutralisation():
    t = "Lukas Müller met Ayşe Kaya."
    assert swap_names(t, "Lukas Müller", "Ayşe Kaya") == "Ayşe Kaya met Lukas Müller."
    assert "Elternzeit" not in neutralize_proxies("Elternzeit 2023", DATA["proxy_terms"])
    cv = render_cv(DATA["profiles"][2], DATA["identities"][6], DATA["job"])
    b = blind_cv(cv, DATA["proxy_terms"])
    assert "Ayşe" not in b and "she/her" not in b and "Elternzeit" not in b and "[CANDIDATE]" in b


def test_cda_doubles_matching_examples():
    texts, labels = cda_augment(["Anna Schmidt is a strong engineer", "No name here"], [1, 0], [("Anna Schmidt", "Lukas Müller")])
    assert len(texts) == 3 and "Lukas Müller is a strong engineer" in texts and labels == [1, 0, 1]


def test_overall_accuracy_trap_and_significant_intersectional_gap(llm):
    rep = audit(run_screening(llm), "raw")
    assert rep.n == 160
    assert rep.accuracy_trap
    assert rep.worst_group == "female|international"
    inter = next(t for t in rep.gap_tests if "|" in t.comparison and t.metric == "accuracy")
    assert inter.result.significant and abs(inter.result.z) > 1.96


def test_blind_screening_removes_gap(llm):
    rep = audit(run_screening(llm, blind=True), "blind")
    accs = {g.group: g.accuracy for g in rep.intersectional}
    assert max(accs.values()) - min(accs.values()) < 1e-9
    assert not rep.accuracy_trap


def test_bias_free_model_has_no_counterfactual_flips():
    assert counterfactual_flip_test(MockLLMProvider(bias_strength=0.0)).flip_rate_raw == 0.0
    biased = counterfactual_flip_test(MockLLMProvider(bias_strength=1.0))
    assert biased.flip_rate_raw > 0.1 and biased.flip_rate_blind == 0.0


def test_inverse_frequency_weights_equal_shares_and_ignore_index():
    groups = torch.tensor([0] * 90 + [1] * 10)
    labels = torch.tensor([1] * 85 + [IGNORE_INDEX] * 5 + [0] * 10)
    w = inverse_frequency_weights(groups, labels)
    shares = group_weight_shares(groups, labels, w)
    assert abs(shares[0] - 0.5) < 1e-6 and abs(shares[1] - 0.5) < 1e-6
    assert torch.all(w[85:90] == 0)
    logits = torch.randn(100, 2, generator=torch.Generator().manual_seed(42))
    per = F.cross_entropy(logits, labels, reduction="none", ignore_index=IGNORE_INDEX)
    assert torch.all(per[85:90] == 0)
    loss = weighted_group_loss(logits, labels, w)
    valid = labels != IGNORE_INDEX
    expected = (per[valid] * w[valid]).sum() / w[valid].sum()
    assert torch.isclose(loss, expected)


def test_reweighting_study_and_failure_mode_diagnostics():
    r = run_reweighting_study()
    assert r.equal_shares_verified
    assert r.worst_group_weighted >= r.worst_group_unweighted
    assert r.fragmentation_tokens_per_name["international"] > r.fragmentation_tokens_per_name["german"]
    assert abs(r.token_level_share_minority - 0.5) > 0.02
    assert r.pretrained_bias_after == r.pretrained_bias_before  # frozen bias untouched
    assert r.noisy_weight_share_weighted > 2 * r.noisy_weight_share_unweighted


def test_weat_detects_bias_and_cda_removes_it():
    corpus = synthetic_corpus(n_sentences=3000)
    biased = weat(ppmi_svd_embeddings(corpus), CAREER, FAMILY, MALE, FEMALE, n_permutations=500)
    debiased = weat(ppmi_svd_embeddings(gender_swap_cda(corpus)), CAREER, FAMILY, MALE, FEMALE, n_permutations=500)
    assert biased.effect_size > 1.0 and biased.p_value < 0.05
    assert abs(debiased.effect_size) < 0.2
    assert debiased.p_value > 0.5  # symmetric null: no evidence of bias, independent of BLAS rounding
