#!/usr/bin/env python3
"""
Reproduce current Episteme checks.

This script intentionally runs the same public checks used in the research
status file and writes a machine-readable summary. It does not hide failures:
the process exits non-zero if any required check fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "logs"
OUT_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = OUT_DIR / "reproducibility_summary.json"


def run(name: str, cmd: list[str]) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "started_at": started,
        "output_tail": proc.stdout[-6000:],
    }


def main() -> int:
    checks = [
        ("unit_tests", [sys.executable, "-m", "unittest", "discover", "tests"]),
        ("scientific_benchmark", [sys.executable, "tests/run_scientific_benchmark.py"]),
        ("strict_benchmark", [sys.executable, "tests/benchmark_strict.py"]),
            (
                "symbolic_reasoning_benchmark",
                [sys.executable, "scripts/run_symbolic_reasoning_benchmarks.py", "--limit", "5"],
            ),
        (
            "parser_benchmark",
            [sys.executable, "scripts/run_parser_benchmarks.py", "--limit", "5"],
        ),
        (
            "aggregate_benchmarks",
            [sys.executable, "scripts/aggregate_benchmarks.py"],
        ),
        (
            "generate_paper_eval_artifacts",
            [sys.executable, "scripts/generate_paper_eval_artifacts.py"],
        ),
        ("showcase", [sys.executable, "showcase_episteme.py"]),
    ]

    results = [run(name, cmd) for name, cmd in checks]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "results": results,
        "success": all(item["returncode"] == 0 for item in results),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for item in results:
        status = "PASS" if item["returncode"] == 0 else "FAIL"
        print(f"{status}: {item['name']} (returncode={item['returncode']})")
    print(f"Summary: {SUMMARY_PATH}")

    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
