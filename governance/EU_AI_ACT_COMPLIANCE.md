# EU AI Act Requirements Mapping (Engineering Self-Assessment)

> **Disclaimer:** Educational & engineering reference prototype using a fictional employer (`ExampleHR Demo GmbH`) and synthetic data. Not legal advice, and not certified or approved for production use in high-risk HR decisions.

Regulation (EU) 2024/1689 (Artificial Intelligence Act). This document maps each obligation relevant to this system
to the concrete engineering artefact that implements or evidences it. It is an engineering self-assessment, not legal advice.

## 1. Risk classification

| Question | Answer |
|---|---|
| Prohibited practice (Art. 5)? | **No.** No emotion recognition at the workplace (Art. 5(1)(f)), no biometric categorisation by protected attributes, no social scoring. Processing is text-only. |
| High-risk (Art. 6(2) + Annex III)? | **Yes.** Annex III point 4(a): AI intended for recruitment or selection, in particular to analyse and filter applications and evaluate candidates (CV pre-screening feature). Annex III point 4(b) is avoided by design: termination / promotion decisions are escalated to humans and never produced by the system. |
| Art. 6(3) derogation? | Not claimed. The screening feature influences the outcome of decision making and involves profiling of natural persons. |
| Transparency obligations (Art. 50)? | **Yes** - conversational system interacting with natural persons. |
| General-purpose AI model provider (Chapter V)? | No - the system *uses* GPAI models via providers; GPAI obligations sit with the model provider. |

**Applicable dates.** Regulation (EU) 2024/1689 was published in the Official Journal on 12 July 2024 and entered into
force on 1 August 2024 (Art. 113).

| Date | Provisions that apply |
|---|---|
| 2 February 2025 | Chapters I-II: definitions, AI literacy (Art. 4), prohibited practices (Art. 5) |
| 2 August 2025 | Notified bodies, GPAI model obligations (Chapter V), governance, penalties |
| 2 August 2026 | General date of application, incl. transparency obligations (Art. 50) |
| 2 December 2027 | High-risk obligations for stand-alone Annex III systems - **applies to this system** (Annex III 4(a)) |
| 2 August 2028 | High-risk obligations for AI in products under Annex I harmonisation legislation |

The Annex III and Annex I dates (2 December 2027 / 2 August 2028) are those set by the Digital Omnibus on AI,
Regulation (EU) 2026/1744 (OJ, 24 July 2026), which amended the original Art. 113 dates (2 August 2026 and
2 August 2027). See also: European Commission, AI Act Service Desk, *Timeline for the implementation of the EU AI Act*.

## 2. Provider obligations for high-risk systems (Chapter III, Section 2)

| Article | Requirement | Implementation in this repository | Evidence |
|---|---|---|---|
| Art. 9 | Risk management system across the lifecycle | Hazard taxonomy S1-S8; red-team suite (34 attacks, CIA x 3 surfaces); trust gates block releases on regression | `security/`, `reports/06_red_team.md`, `.github/workflows/trust-gate.yml` |
| Art. 10(2)(f)-(g), 10(5) | Examine data for possible biases; detect, prevent, mitigate | Disaggregated + intersectional audit, two-proportion z-tests, counterfactual name swaps, CDA, WEAT, inverse-frequency reweighting with failure-mode diagnostics | `fairness/`, `reports/04_fairness.md` |
| Art. 10(3) | Data relevant, representative, free of errors as far as possible | Data statement, gold-label definitions independent of identity, reliability metadata on sources | `governance/DATA_STATEMENT.md` |
| Art. 11 + Annex IV | Technical documentation | Model card, architecture in README, mechanistic interpretability lab (probing / activation patching) | `governance/MODEL_CARD.md`, `labs/03_*.ipynb` |
| Art. 12 | Automatic logging / record-keeping | Structured key=value logs; per-request layer trace returned by the API; HITL tickets persisted as JSONL; raw judge outputs persisted | `api/app.py` (`trace`), `reports/escalations.jsonl`, `reports/raw_judge_outputs/` |
| Art. 13 | Transparency & instructions for use to deployers | Intended / out-of-scope use, metrics and limitations | `governance/MODEL_CARD.md` sections 2-5 |
| Art. 14 | Human oversight (understand, monitor, override, stop) | `ESCALATE_TO_HUMAN` for crisis, harassment, consequential HR decisions; recommendations only; recruiter reviews every screening result; per-answer FActScore and unsupported-fact list help humans detect automation bias | `guardrails/policies.yaml`, `guardrails/safety_stack.py` (Layer 5) |
| Art. 15(1)-(3) | Accuracy, declared metrics | FActScore, gold-fact accuracy, judge agreement with gold, declared in model card | `reports/01_rag_factuality.md`, `reports/03_llm_judge.md` |
| Art. 15(4) | Robustness, incl. feedback-loop and error resilience | 4 language traps, covariate / label / concept shift simulator, metamorphic invariance tests | `robustness/`, `reports/05_robustness.md` |
| Art. 15(5) | Cybersecurity: data poisoning, adversarial examples, confidentiality attacks | Context quarantine for indirect injection in retrieved docs/CVs, canary-token leakage detection, exfiltration-link stripping, encoding-aware guard, token-bucket rate limiting | `guardrails/safety_stack.py`, `reports/06_red_team.md` |
| Art. 17 | Quality management system | Versioned policies, tests (pytest), CI trust gates, reproducible seeds | `tests/`, CI workflow |

## 3. Deployer obligations (employer using the system)

| Article | Obligation | How the system supports it |
|---|---|---|
| Art. 26(1)-(2) | Use according to instructions; assign competent human oversight | Model card intended-use section; HITL queue |
| Art. 26(5)-(6) | Monitor operation; keep logs >= 6 months | JSONL logs and layer traces; `/v1/audit/summary` for monitoring |
| Art. 26(7) | Inform workers' representatives and affected workers before workplace use | See German co-determination section below |
| Art. 26(11) | Inform natural persons subject to high-risk decisions that AI is used | `ai_disclosure` field in every API response |
| Art. 27 | Fundamental rights impact assessment | Mandatory only for public bodies / providers of public services and certain Annex III 5 uses; recommended here as best practice - the fairness and red-team reports provide its quantitative inputs |
| Art. 50(1) | Disclose AI interaction | Policy `ai_interaction_disclosure` appends the notice to every answer |
| Art. 86 | Right to explanation of individual decision-making | Citations, per-fact verification and the CV screening rationale are returned for the human decision maker |

## 4. Interplay with GDPR and German employment law

| Law | Requirement | Implementation |
|---|---|---|
| GDPR Art. 22 | No solely automated decisions with legal effect | Screening output = recommendation; rejection decisions escalated (`consequential_hr_decision`) |
| GDPR Art. 5(1)(c) | Data minimisation | PII redaction (IBAN, tax ID, phone, private e-mail) before the LLM and in outputs; SFT prompt masking discussed in lab 01 |
| GDPR Art. 9 | Special-category data | `protected_attribute_probing` refuses eliciting pregnancy, religion, health, disability |
| GDPR Art. 35 | DPIA for high-risk processing | Audit reports supply the risk evidence for the DPIA |
| AGG sections 1, 7, 11 | No discrimination in hiring / job ads | `discriminatory_screening` refusal; blind screening; intersectional audit |
| BetrVG section 87(1) no. 6 | Co-determination for technical systems able to monitor employees | Works-council agreement required before rollout; logs are pseudonymised by `user_id` |
| BetrVG section 95 | Selection guidelines (Auswahlrichtlinien) need works-council consent | Screening criteria are declared in `cv_fairness_samples.json` -> `job` and must be agreed with the Betriebsrat |
| BetrVG section 90(1) no. 3, section 80(3) | Information on planned AI use; right to an AI expert | Model card and this mapping are the information package |

## 5. Residual risks accepted for the portfolio scope

* Heuristic guard misses subtle, cue-free harms (response-guard recall 0.353 on `safety_eval.csv`) -> mitigated by human review, not eliminated.
* One indirect CV injection (A34) succeeded against the full stack by being quoted verbatim inside a grounded summary -> next control:
  instruction-pattern scrubbing of quoted context in Layer 4.
* Offline metrics are simulator-based; a production conformity assessment must re-run every audit against the deployed model.
