"""Unified CLI.

    python -m trust_llm.cli run-all          # every audit -> reports/*.md, reports/figures/*.png, reports/audit_summary.json
    python -m trust_llm.cli rag|guardrails|judge|fairness|robustness|security|decoding
    python -m trust_llm.cli trust-gate       # CI gate: exits 1 if any trust gate fails (runs audits if needed)
    python -m trust_llm.cli chat "Wie lange dauert die Probezeit?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trust_llm import audits
from trust_llm.config import REPORTS_DIR, configure_logging

MODULES = {
    "rag": audits.audit_rag,
    "guardrails": audits.audit_guardrails,
    "judge": audits.audit_judge,
    "fairness": audits.audit_fairness,
    "robustness": audits.audit_robustness,
    "security": audits.audit_security,
    "decoding": audits.audit_decoding,
}


def _print_gates(gates: list[dict]) -> bool:
    ok = True
    for g in gates:
        status = "PASS" if g["passed"] else "FAIL"
        ok &= g["passed"]
        print(f"[{status}] {g['gate']}: {g['value']}")
    return ok


def _utf8_stdout() -> None:
    """Avoid UnicodeEncodeError on legacy Windows consoles (cp1252) when printing German/Turkish names."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(prog="trust_llm", description="Trustworthy LLM audit stack")
    parser.add_argument("command", choices=["run-all", "trust-gate", "chat", *MODULES])
    parser.add_argument("text", nargs="?", default="", help="message for the chat command")
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "chat":
        from trust_llm.api.app import build_service

        svc = build_service()
        print(json.dumps(svc.chat(args.text or "Wie lange dauert die Probezeit?").model_dump(), indent=2, ensure_ascii=False))
        return 0

    ctx = audits.AuditContext(reports_dir=args.reports_dir)
    if args.command == "run-all":
        summary = audits.run_all(ctx)
        print(f"Reports written to {ctx.reports_dir} (runtime {summary['meta']['runtime_seconds']} s)")
        _print_gates(summary["gates"])
        return 0
    if args.command == "trust-gate":
        path = (args.reports_dir or REPORTS_DIR) / "audit_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else audits.run_all(ctx)
        return 0 if _print_gates(summary.get("gates") or audits.evaluate_gates(summary)) else 1
    result = MODULES[args.command](ctx)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
