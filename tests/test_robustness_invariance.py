import json

from trust_llm.config import DATA_DIR
from trust_llm.providers.mock_provider import MockLLMProvider
from trust_llm.robustness.invariance import (
    SelfConsistency, build_translation_cases, detect_noun_gender, run_cv_invariance, run_translation_invariance,
)
from trust_llm.robustness.perturbations import (
    DistributionShiftSimulator, domain_shift, evaluate_language_traps, lexical_typos, multilingual_swap, pronoun_swap, semantic_paraphrase,
)

QUESTIONS = json.loads((DATA_DIR / "rag_eval.json").read_text(encoding="utf-8"))


def test_perturbations_are_deterministic_and_change_text():
    q = "How many vacation days do employees get per year?"
    assert lexical_typos(q, 0.5, seed=1) == lexical_typos(q, 0.5, seed=1) != q
    assert semantic_paraphrase(q) == "what number of days of paid leave do staff members get annually?"
    assert domain_shift(q, "chat").startswith("hey")
    assert "Urlaubstage" in multilingual_swap(q)
    assert pronoun_swap("She said he would send her notes.") == "He said she would send his notes."


def test_language_traps_degrade_but_do_not_destroy_retrieval(reliable_rag):
    res = {t.trap: t for t in evaluate_language_traps(reliable_rag, QUESTIONS)}
    assert set(res) == {"clean", "lexical_typos", "semantic_paraphrase", "domain_shift", "multilingual_swap"}
    assert res["clean"].retrieval_hit_at_3 == 1.0
    assert min(t.retrieval_hit_at_3 for t in res.values()) >= 0.75


def test_distribution_shift_signatures():
    rep = DistributionShiftSimulator(seed=42).run()
    r = {s.shift: s for s in rep.results}
    assert rep.source_accuracy > 0.95
    assert r["covariate"].input_mean_shift > 1.0 and r["covariate"].accuracy_drop > 0.1
    assert abs(r["label"].label_prior - rep.source_label_prior) > 0.3
    assert r["concept"].input_mean_shift < 0.2 and r["concept"].accuracy_drop > 0.2


def test_noun_gender_detection():
    assert detect_noun_gender("Die Ingenieurin sagte, dass sie ...", ["Ingenieur"], ["Ingenieurin"]) == "f"
    assert detect_noun_gender("Der Ingenieur sagte, dass er ...", ["Ingenieur"], ["Ingenieurin"]) == "m"


def test_high_average_accuracy_masks_low_invariance(llm):
    assert len(build_translation_cases()) == 120
    rep = run_translation_invariance(llm)
    assert rep.accuracy > 0.7
    assert rep.invariance_rate < rep.accuracy - 0.15
    assert rep.gender_gap > 0  # 'she' handled worse than 'he'


def test_unbiased_translator_is_fully_invariant():
    rep = run_translation_invariance(MockLLMProvider(translation_bias=0.0))
    assert rep.accuracy == 1.0 and rep.invariance_rate == 1.0 and rep.gender_gap == 0.0


def test_self_consistency_is_reproducible(llm):
    sc = SelfConsistency(llm, n=5, temperature=0.8)
    a = run_translation_invariance(llm, self_consistency=sc)
    b = run_translation_invariance(llm, self_consistency=sc)
    assert a.model_dump() == b.model_dump()


def test_cv_gender_swap_invariance(llm):
    rep = run_cv_invariance(llm)
    assert rep.n_pairs == 80
    assert rep.invariance_rate < rep.accuracy
    assert run_cv_invariance(MockLLMProvider(bias_strength=0.0)).gender_gap == 0.0
