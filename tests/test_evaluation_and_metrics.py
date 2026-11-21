import json
import math

import numpy as np
import pytest
from scipy import stats

from trust_llm.evaluation.decoding_bench import BigramLM, beam_search, corpus_sentences, greedy_decode, run_decoding_bench, sample_decode
from trust_llm.evaluation.llm_judge import (
    DisagreementInspector, JudgeConfig, LLMJudge, agreement_report, build_judge_prompt, load_safety_eval,
    parse_judge_output, run_judge, stability_experiment,
)
from trust_llm.evaluation.metrics import (
    accuracy, cohens_kappa, mcnemar_test, multiclass_prf, perplexity, spearman_rho, thresholded_kappa, two_proportion_z_test,
)


def test_perplexity_uniform():
    assert perplexity([math.log(1 / 4)] * 10) == pytest.approx(4.0)


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(42)
    x = rng.integers(1, 6, 50).tolist()
    y = (np.array(x) + rng.integers(-1, 2, 50)).tolist()
    assert spearman_rho(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-10)


def test_cohens_kappa_known_values():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    # p_o = 0.7, p_e = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa = 0.4
    a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 1, 0, 1, 0, 0, 0, 1]
    assert cohens_kappa(a, b) == pytest.approx(0.4)
    assert thresholded_kappa([1, 2, 4, 5], [2, 1, 3, 5], threshold=3) == 1.0


def test_mcnemar_exact_and_chi2():
    a = [True] * 10 + [False] * 2 + [True] * 20
    b = [False] * 10 + [True] * 2 + [True] * 20
    r = mcnemar_test(a, b)
    assert (r.b, r.c, r.method) == (10, 2, "exact")
    assert r.p_value == pytest.approx(stats.binomtest(2, 12, 0.5).pvalue)
    r2 = mcnemar_test(a * 3, b * 3)
    assert r2.method == "chi2_cc"
    assert r2.p_value == pytest.approx(stats.chi2.sf((abs(30 - 6) - 1) ** 2 / 36, 1))


def test_micro_equals_accuracy_and_macro_differs():
    y_true = ["a"] * 8 + ["b"] * 2 + ["c"] * 2
    y_pred = ["a"] * 8 + ["a", "b"] + ["a", "a"]
    rep = multiclass_prf(y_true, y_pred)
    assert rep.micro_identity_holds
    assert rep.micro.f1 == pytest.approx(accuracy(y_true, y_pred))
    assert rep.macro.f1 < rep.micro.f1


def test_two_proportion_z_test():
    r = two_proportion_z_test(38, 40, 28, 40)
    assert r.z == pytest.approx(2.94, abs=0.02) and r.significant
    assert not two_proportion_z_test(30, 40, 29, 40).significant


def test_judge_prompt_has_four_parts():
    p = build_judge_prompt("role", "q", "r")
    for part in ("### ROLE", "### TASK", "### OUTPUT SCHEMA", "### INPUT"):
        assert part in p


@pytest.mark.parametrize("raw,ok", [
    ('{"score": 4, "label": "safe", "rationale": "x"}', True),
    ('```json\n{"score": 2, "label": "unsafe", "rationale": "x"}\n```', True),
    ("Score: 4/5 - safe", False),
    ('{"score": 4, "label": "safe", "rationale": "trunc', False),
    ('{"score": "four", "label": "safe"}', False),
    ('{"score": 7, "label": "safe"}', False),
])
def test_parse_judge_output(raw, ok):
    assert (parse_judge_output(raw) is not None) is ok


def test_judges_deterministic_raw_outputs_persisted(llm, tmp_path):
    rows = load_safety_eval()
    assert len(rows) == 40
    j = LLMJudge(llm, JudgeConfig(name="a", persona="strict"))
    r1, r2 = run_judge(j, rows, tmp_path), run_judge(j, rows, tmp_path, tag="_2")
    assert r1.parsing_failures == 0
    assert [o.raw for o in r1.outputs] == [o.raw for o in r2.outputs]
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40 and "raw" in json.loads(lines[0])


def test_high_temperature_produces_instability(llm):
    rows = load_safety_eval()
    rep = stability_experiment(llm, rows, None, temperature=1.5, n_runs=3)
    assert rep.deterministic_identical_rate == 1.0 and rep.deterministic_parsing_failures == 0
    assert sum(rep.high_temp_parsing_failures) > 0
    assert rep.high_temp_mean_score_std > 0


def test_disagreement_inspector_finds_consensus_errors(llm):
    rows = load_safety_eval()
    a = run_judge(LLMJudge(llm, JudgeConfig(name="a", persona="strict")), rows)
    b = run_judge(LLMJudge(llm, JudgeConfig(name="b", persona="lenient")), rows)
    insp = DisagreementInspector().inspect(rows, a, b)
    assert all(abs(d.score_a - d.score_b) >= 2 for d in insp.judge_disagreements)
    assert insp.consensus_errors, "both-judges-wrong cases must be surfaced"
    assert all(d.gold_label == "unsafe" for d in insp.consensus_errors)
    rep = agreement_report(rows, a, b)
    assert -1 <= rep.spearman_a_b <= 1 and rep.n_both_parsed == 40


def test_decoding_strategies():
    lm = BigramLM().fit(corpus_sentences())
    prefix = ["<s>", "the", "probation"]
    g = greedy_decode(lm, prefix)
    b = beam_search(lm, prefix, beam_width=5)
    assert g == greedy_decode(lm, prefix)  # argmax is deterministic
    s1 = sample_decode(lm, prefix, 1.0, rng=np.random.default_rng(42))
    s2 = sample_decode(lm, prefix, 1.0, rng=np.random.default_rng(42))
    assert s1 == s2  # seeded sampling is reproducible
    lp = lambda seq: sum(lm.sequence_logprobs(seq, context=prefix)) / max(1, len(seq))  # noqa: E731
    assert lp(b) >= lp(g) - 1e-9 or len(b) != len(g)


def test_decoding_bench_tradeoff():
    rep = run_decoding_bench()
    res = {r.method: r for r in rep.results}
    assert rep.heldout_perplexity < rep.unigram_heldout_perplexity
    assert res["greedy"].avg_logprob_per_token > res["sample_T1.5"].avg_logprob_per_token
    assert res["sample_T1.5"].distinct_2 > res["greedy"].distinct_2
    assert res["sample_T1.0_top10"].avg_logprob_per_token > res["sample_T1.0"].avg_logprob_per_token
