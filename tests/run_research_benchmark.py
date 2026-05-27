#!/usr/bin/env python3
"""
Family-based research benchmark with small honest baselines.

This runner is intentionally modest. Its purpose is to establish methodology:
family labels, verdict metrics, baseline comparison, and reproducible JSON
output. The benchmark data itself is only a seed set.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ahankara.layer import Ahankara
from manas.layer import Manas


ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "tests" / "data" / "research_benchmark.json"
OUT_PATH = ROOT / "tests" / "logs" / "research_benchmark_summary.json"


VERDICTS = {"YES", "NO", "UNKNOWN", "CONFLICT", "INVALID"}


def normalize_answer(text: str) -> str:
    lower = text.lower()
    if lower.startswith("yes"):
        return "YES"
    if lower.startswith("no"):
        return "NO"
    if "conflicting" in lower or "conflict" in lower:
        return "CONFLICT"
    if "invalid" in lower or "malformed" in lower:
        return "INVALID"
    return "UNKNOWN"


class ClosedWorldDirectBaseline:
    """Direct exact-match baseline. Unknown direct facts become NO."""

    def __init__(self):
        self.manas = Manas()
        self.facts = set()
        self.negative_facts = set()

    def teach(self, text: str):
        p = self.manas.parse(text)
        key = (tuple(sorted(p.get("entities", []))), tuple(sorted(p.get("predicates", []))))
        if p.get("polarity") == -1:
            self.negative_facts.add(key)
        else:
            self.facts.add(key)

    def ask(self, text: str) -> str:
        p = self.manas.parse(text)
        key = (tuple(sorted(p.get("entities", []))), tuple(sorted(p.get("predicates", []))))
        if key in self.negative_facts:
            return "NO"
        if key in self.facts:
            return "YES"
        return "NO"


class OpenWorldDirectBaseline(ClosedWorldDirectBaseline):
    """Direct exact-match baseline. Unknown direct facts remain UNKNOWN."""

    def ask(self, text: str) -> str:
        p = self.manas.parse(text)
        key = (tuple(sorted(p.get("entities", []))), tuple(sorted(p.get("predicates", []))))
        if key in self.negative_facts and key in self.facts:
            return "CONFLICT"
        if key in self.negative_facts:
            return "NO"
        if key in self.facts:
            return "YES"
        return "UNKNOWN"


class EpistemeFull:
    def __init__(self):
        self.engine = Ahankara()

    def teach(self, text: str):
        self.engine.process(text)

    def ask(self, text: str) -> str:
        return normalize_answer(self.engine.ask(text))


class EpistemeNoActivation(EpistemeFull):
    """Ablation placeholder: activation fields exist but no decay is applied."""


def run_system(system_cls, cases):
    rows = []
    started = time.perf_counter()
    for case in cases:
        system = system_cls()
        for fact in case["facts"]:
            system.teach(fact)
        actual = system.ask(case["query"])
        rows.append({
            "id": case["id"],
            "family": case["family"],
            "expected": case["expected"],
            "actual": actual,
            "passed": actual == case["expected"],
        })
    elapsed = time.perf_counter() - started
    return rows, elapsed


def summarize(rows, elapsed):
    by_family = defaultdict(lambda: {"passed": 0, "total": 0})
    false_positive = 0
    false_negative = 0
    unknown_correct = 0
    conflict_correct = 0

    for row in rows:
        fam = by_family[row["family"]]
        fam["total"] += 1
        fam["passed"] += int(row["passed"])
        expected = row["expected"]
        actual = row["actual"]
        if actual == "YES" and expected != "YES":
            false_positive += 1
        if actual in {"NO", "UNKNOWN"} and expected == "YES":
            false_negative += 1
        if actual == expected == "UNKNOWN":
            unknown_correct += 1
        if actual == expected == "CONFLICT":
            conflict_correct += 1

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    return {
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total else 0.0,
        "false_positive_rate": false_positive / total if total else 0.0,
        "false_negative_rate": false_negative / total if total else 0.0,
        "unknown_correct": unknown_correct,
        "conflict_correct": conflict_correct,
        "proof_validity_rate": None,
        "runtime_seconds": elapsed,
        "memory_size": None,
        "families": {
            key: {
                "passed": value["passed"],
                "total": value["total"],
                "accuracy": value["passed"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(by_family.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    args = parser.parse_args()

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    systems = {
        "closed_world_direct": ClosedWorldDirectBaseline,
        "open_world_direct": OpenWorldDirectBaseline,
        "episteme_no_activation": EpistemeNoActivation,
        "episteme_full": EpistemeFull,
    }

    summary = {"systems": {}, "rows": {}}
    for name, cls in systems.items():
        rows, elapsed = run_system(cls, cases)
        summary["systems"][name] = summarize(rows, elapsed)
        summary["rows"][name] = rows

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    full_accuracy = summary["systems"]["episteme_full"]["accuracy"]
    print(json.dumps(summary["systems"], indent=2))
    print(f"Summary: {OUT_PATH}")
    if full_accuracy < args.min_accuracy:
        print(f"FAIL: episteme_full accuracy {full_accuracy:.3f} below {args.min_accuracy:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
