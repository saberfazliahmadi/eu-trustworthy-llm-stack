"""Repository hygiene: YAML files parse, CI / docs invoke only real CLI commands, no machine-specific paths."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from trust_llm.cli import MODULES

ROOT = Path(__file__).resolve().parents[1]
CLI_COMMANDS = {"run-all", "trust-gate", "chat", *MODULES}
YAML_FILES = [
    "docker-compose.yml",
    "src/trust_llm/guardrails/policies.yaml",
    ".github/workflows/trust-gate.yml",
]


@pytest.mark.parametrize("rel_path", YAML_FILES)
def test_yaml_files_parse(rel_path):
    data = yaml.safe_load((ROOT / rel_path).read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data


def test_workflow_structure():
    wf = yaml.safe_load((ROOT / ".github/workflows/trust-gate.yml").read_text(encoding="utf-8"))
    triggers = wf.get("on", wf.get(True))  # YAML 1.1 reads a bare `on` key as boolean True
    assert "push" in triggers and "pull_request" in triggers
    job = wf["jobs"]["test-and-audit"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    assert all("run" in s or "uses" in s for s in job["steps"])


@pytest.mark.parametrize("rel_path", [".github/workflows/trust-gate.yml", "README.md", "docker-compose.yml"])
def test_documented_cli_commands_exist(rel_path):
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    used = re.findall(r"trust_llm\.cli\"?,?\s*\"?([a-z][a-z-]*)", text)
    assert used, f"no CLI invocation found in {rel_path}"
    unknown = sorted(set(used) - CLI_COMMANDS)
    assert not unknown, f"{rel_path} calls unknown CLI commands: {unknown}"


def test_no_machine_specific_paths_in_published_files():
    pattern = re.compile(r"[A-Za-z]:\\{1,2}Users\\{1,2}|/home/[a-z]+/|/Users/[A-Za-z]+/")
    globs = ["*.md", "*.toml", "*.yml", "governance/*.md", "reports/*.md", "reports/*.json", "labs/*.ipynb", "src/**/*.py"]
    hits = [
        str(p.relative_to(ROOT))
        for g in globs
        for p in ROOT.glob(g)
        if pattern.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not hits, f"absolute local paths found in: {hits}"
