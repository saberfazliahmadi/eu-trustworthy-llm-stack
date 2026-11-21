"""Declarative policy engine. Rules live in policies.yaml; each rule has trigger, action, rationale."""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from trust_llm.config import POLICIES_PATH

logger = logging.getLogger(__name__)


class Action(str, Enum):
    REDACT = "REDACT"
    REDIRECT = "REDIRECT"
    REFUSE = "REFUSE"
    ADD_DISCLAIMER = "ADD_DISCLAIMER"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    ALLOW = "ALLOW"


# Terminal actions stop the pipeline; precedence resolves conflicts (highest first).
ACTION_PRECEDENCE = [Action.REFUSE, Action.ESCALATE_TO_HUMAN, Action.REDIRECT, Action.REDACT, Action.ADD_DISCLAIMER, Action.ALLOW]
TERMINAL_ACTIONS = {Action.REFUSE, Action.ESCALATE_TO_HUMAN, Action.REDIRECT}


class Trigger(BaseModel):
    type: Literal["regex", "keywords", "guard_category", "always"]
    patterns: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "Trigger":
        if self.type in ("regex", "keywords") and not self.patterns:
            raise ValueError(f"trigger type '{self.type}' requires patterns")
        if self.type == "guard_category" and not self.categories:
            raise ValueError("guard_category trigger requires categories")
        if self.type == "regex":
            for p in self.patterns:
                re.compile(p)
        return self


class Rule(BaseModel):
    id: str
    trigger: Trigger
    action: Action
    rationale: str = Field(min_length=10)
    stages: list[Literal["input", "output"]] = Field(default_factory=lambda: ["input"])
    params: dict[str, Any] = Field(default_factory=dict)


class PolicySet(BaseModel):
    version: int = 1
    rules: list[Rule]


class RuleMatch(BaseModel):
    rule_id: str
    action: Action
    rationale: str
    matched_text: str = ""


class PolicyDecision(BaseModel):
    stage: str
    final_action: Action = Action.ALLOW
    text: str
    matches: list[RuleMatch] = Field(default_factory=list)
    message: str | None = None
    disclaimers: list[str] = Field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.final_action in TERMINAL_ACTIONS


def load_policies(path: Path | None = None) -> PolicySet:
    data = yaml.safe_load(Path(path or POLICIES_PATH).read_text(encoding="utf-8"))
    return PolicySet.model_validate(data)


class PolicyEngine:
    """Evaluates rules for a stage and applies REDACT / ADD_DISCLAIMER transforms and terminal actions."""

    def __init__(self, policies: PolicySet | None = None, path: Path | None = None) -> None:
        self.policies = policies or load_policies(path)
        self._compiled = {
            r.id: [re.compile(p) for p in r.trigger.patterns] if r.trigger.type == "regex" else []
            for r in self.policies.rules
        }

    @property
    def rules(self) -> list[Rule]:
        return self.policies.rules

    def _match(self, rule: Rule, text: str, guard_categories: set[str]) -> str | None:
        t = rule.trigger
        if t.type == "always":
            return "*"
        if t.type == "regex":
            for rx in self._compiled[rule.id]:
                m = rx.search(text)
                if m:
                    return m.group(0)
            return None
        if t.type == "keywords":
            low = text.lower()
            return next((k for k in t.patterns if k.lower() in low), None)
        hit = guard_categories & set(t.categories)
        return ",".join(sorted(hit)) if hit else None

    def evaluate(
        self,
        text: str,
        stage: Literal["input", "output"],
        guard_categories: set[str] | None = None,
        views: list[str] | None = None,
    ) -> PolicyDecision:
        """Evaluate all rules of `stage`. `views` are extra normalised renderings (decoded, de-leeted)."""
        guard_categories = guard_categories or set()
        decision = PolicyDecision(stage=stage, text=text)
        scan_views = [text] + [v for v in (views or []) if v != text]
        for rule in self.rules:
            if stage not in rule.stages:
                continue
            matched = None
            for view in scan_views if rule.action != Action.REDACT else [text]:
                matched = self._match(rule, view, guard_categories)
                if matched:
                    break
            if not matched:
                continue
            decision.matches.append(RuleMatch(rule_id=rule.id, action=rule.action, rationale=rule.rationale, matched_text=matched[:120]))
            if rule.action == Action.REDACT:
                repl = rule.params.get("replacement", "[REDACTED]")
                for rx in self._compiled[rule.id]:
                    decision.text = rx.sub(repl, decision.text)
            elif rule.action == Action.ADD_DISCLAIMER:
                msg = rule.params.get("message", "")
                if msg and msg not in decision.disclaimers:
                    decision.disclaimers.append(msg)
        if decision.matches:
            actions = {m.action for m in decision.matches}
            decision.final_action = next(a for a in ACTION_PRECEDENCE if a in actions)
            if decision.final_action in TERMINAL_ACTIONS:
                rule_id = next(m.rule_id for m in decision.matches if m.action == decision.final_action)
                rule = next(r for r in self.rules if r.id == rule_id)
                decision.message = rule.params.get("message")
        logger.debug("policy stage=%s action=%s rules=%s", stage, decision.final_action, [m.rule_id for m in decision.matches])
        return decision
