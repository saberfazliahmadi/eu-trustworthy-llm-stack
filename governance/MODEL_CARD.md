# Model Card - "Lotse" HR Policy & Career-Coaching Assistant

> **Disclaimer:** Educational & engineering reference prototype using a fictional employer (`ExampleHR Demo GmbH`) and synthetic data. Not legal advice, and not certified or approved for production use in high-risk HR decisions.

Format follows Mitchell et al. (2019), *Model Cards for Model Reporting*, extended with the system-level
information required by EU AI Act Art. 11 / Annex IV. All numbers are produced by
`python -m trust_llm.cli run-all` (seed 42) and are reproduced from `reports/audit_summary.json`.

## 1. System details

| Item | Value |
|---|---|
| System | Retrieval-augmented LLM assistant wrapped in a 5-layer safety stack |
| Version | 1.0.0 |
| Core LLM | Provider-agnostic: offline deterministic simulator (`mock`), any OpenAI-compatible endpoint (OpenAI, Gemini, Ollama, vLLM) or a local Hugging Face model |
| Retrieval | Dense cosine-similarity top-k over 10 policy documents; reliability metadata filter (`min_reliability="high"`) |
| Guard | LlamaGuard-style prompt **and** response classification over an 8-category HR hazard taxonomy (heuristic backend by default, LLM backend via `GUARD_BACKEND=llm`) |
| Policies | 14 declarative rules (`policies.yaml`), each with `trigger`, `action`, `rationale` |
| Human oversight | `ESCALATE_TO_HUMAN` action -> HITL ticket queue (JSONL audit log) |
| Developer / deployer | Portfolio reference implementation for a fictional employer (ExampleHR Demo GmbH) |

## 2. Intended use

* Answer employee and applicant questions about the employer's HR policies (probation, vacation, notice periods, parental leave,
  remote work, salary bands, applicant data protection, offer rules, equal treatment) **with citations**.
* Give formative feedback on mock-interview answers (STAR structure) and CV wording.
* Produce *recommendations* that support - never replace - a trained recruiter (GDPR Art. 22, EU AI Act Art. 14).

## 3. Out-of-scope / prohibited uses

* Autonomous hiring, rejection, promotion or termination decisions (escalated to a human by policy `consequential_hr_decision`).
* Individual legal advice (redirected by `legal_advice_redirect` to works council / Fachanwalt für Arbeitsrecht).
* Emotion recognition of candidates or employees - prohibited at the workplace by EU AI Act Art. 5(1)(f); the system processes text only.
* Screening on AGG-protected characteristics (refused by `discriminatory_screening`, `protected_attribute_probing`).
* Crisis counselling (escalated with TelefonSeelsorge contact data).

## 4. Evaluation results (offline mode, deterministic simulator)

| Dimension | Metric | Result |
|---|---|---|
| Grounding | FActScore - no-retrieval baseline -> grounded RAG (reliable sources) | 0.125 -> **0.875** |
| Grounding | Gold-fact accuracy - baseline -> grounded RAG | 0.062 -> **0.875** |
| Grounding | Atomic facts hallucinated (intrinsic + extrinsic share) - baseline -> RAG | 0.885 -> 0.128 |
| Security | Attack Success Rate, 34 attacks - raw model -> safety stack | 0.971 -> **0.029** (1 successful attack: A34, indirect CV injection quoted verbatim) |
| Security | Majority-vote ASR (3 reviewers) - raw -> stack | 0.941 -> 0.029 |
| Over-refusal | 12 benign controls refused by the stack | **0.000** |
| Fairness | CV screening overall accuracy (raw model) | 0.800 |
| Fairness | Worst intersectional group (female x international), raw | **0.700** (z = 2.94 vs male x german, p = 0.003) |
| Fairness | Equal-opportunity TPR, male x german vs female x international (raw) | 1.000 vs 0.500 (z = 4.00, p < 0.001) |
| Fairness | Counterfactual name-swap flip rate - raw -> blind screening | 0.217 -> **0.000** |
| Fairness | Worst intersectional accuracy with blind screening (deployed configuration) | 0.950 (no significant gap) |
| Robustness | EN->DE profession translation: avg accuracy vs. invariance rate (INV) | 0.767 vs **0.533** (gender gap Delta = 0.10) |
| Robustness | Retrieval hit@3 under typos / paraphrase / domain shift / code-switching | 0.94 / 0.94 / 1.00 / 0.94 |
| Evaluation | LLM-judge agreement (Spearman rho / thresholded kappa A-B) | 0.841 / 0.935 |
| Evaluation | Deterministic judge parsing failures / reproducibility | 0 / 100 % identical |
| Evaluation | Judge parsing failures at T = 1.5 (3 seeds, 40 rows each) | 17, 15, 14 |
| Evaluation | Consensus errors (both judges agree, both wrong) | 6 of 40 rows |

## 5. Known limitations

1. **Offline metrics come from a behavioural simulator.** The `MockLLMProvider` reproduces documented LLM failure modes
   (instruction following of injected text, stereotyped translation, biased screening, hallucinated parametric answers)
   deterministically. The numbers above verify that the *pipelines, defences and audit statistics work*; they are not a
   measurement of any commercial model. Re-run with `LLM_PROVIDER=openai` (or `hf`) to audit a real model.
2. The heuristic guard has limited recall on subtle harms: prompt-hazard multiclass accuracy 0.50 (macro-F1 0.535) and
   recall 0.353 on unsafe *responses* in `safety_eval.csv`. Polite, cue-free discriminatory answers pass it - which is why
   the stack has five layers and humans review consequential outputs.
3. Grounded answers can still **misread** retrieved evidence (evidence misinterpretation in 4 of 16 questions, e.g.
   "33 March" instead of "31 March"); the API surfaces a per-answer FActScore and unsupported facts.
4. Self-consistency (majority vote over 5 samples) **reduced** translation invariance from 0.533 to 0.433: majority voting
   amplifies systematic stereotypes instead of averaging them out.
5. The gold labels of `safety_eval.csv` were authored by a single annotator; borderline rows (judge accuracy 0.50) need
   multi-annotator adjudication before production use.
6. The knowledge base covers one fictional employer; answers are not valid for other employers or collective agreements.

## 6. Ethical considerations

Recruiting AI is high-risk because errors compound historical discrimination. The default deployment therefore uses
**blind screening** (names, pronouns and proxy terms such as *Elternzeit* removed), keeps a recruiter in the loop for every
recommendation, informs users that they are interacting with an AI system, and never penalises accented speech (low ASR
confidence triggers a transcript-confirmation step instead of a lower score).
