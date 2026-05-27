"""Triage mismatches: replay each mismatched benchmark case and record proofs.

Usage:
    python3 scripts/triage_mismatches.py
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahankara.layer import Ahankara
from buddhi.layer import Buddhi, AnswerProof
from chitta.graph import ChittaGraph
from manas.layer import Manas


RESULTS = Path("tests/logs/adversarial_results.jsonl")
BENCH = Path("tests/benchmarks/adversarial_benchmarks_natural.json")
OUT_DIR = Path("tests/logs/proofs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIAGE_MD = Path("docs/ADVERSARIAL_TRIAGE_FULL.md")


def proof_to_dict(proof: AnswerProof) -> dict:
    return {
        "query": proof.query,
        "verdict": proof.verdict,
        "steps": [
            {"step_id": s.step_id, "rule": s.rule, "inputs": s.inputs, "output": s.output, "confidence": s.confidence}
            for s in proof.steps
        ],
        "conflicts": [
            {"predicate": c.predicate, "positive": c.positive, "negative": c.negative, "delta": c.delta, "resolution": c.resolution}
            for c in proof.conflicts
        ],
        "metadata": proof.metadata or {}
    }


def guess_root_cause(proof: AnswerProof) -> str:
    if not proof.steps:
        return "no_derivation"
    # grounding heuristics
    first = proof.steps[0]
    if first.rule in ("grounding_check", "entity_existence_check"):
        return "entity_grounding"
    if proof.conflicts:
        return "conflict_resolution"
    # parser fallback
    return "insufficient_inference"


def run():
    results = [json.loads(l) for l in open(RESULTS) if l.strip()]
    bench = json.loads(BENCH.read_text())
    cases = {c['id']: c for c in bench}

    grouped: dict[str, list[dict]] = defaultdict(list)

    report_lines = ["# Adversarial Triage (full)\n"]

    for r in results:
        exp = str(r.get('expected')).strip().lower()
        obs = str(r.get('observed')).strip().lower()
        if exp == obs:
            continue

        cid = r.get('id')
        case = cases.get(cid)
        if not case:
            continue

        # Rebuild fresh pipeline
        ch = ChittaGraph()
        man = Manas(llm_backend='mock')
        bud = Buddhi(ch)
        ah = Ahankara(man, bud, ch)

        teachings = case.get('premises', [])
        for t in teachings:
            try:
                ah.process(t)
            except Exception:
                # swallow teaching errors to continue triage
                pass

        query = case.get('query')
        parsed = man.parse(query)
        proof = bud.answer(parsed)

        out_path = OUT_DIR / f"proof_{cid}.json"
        out_path.write_text(json.dumps({"id": cid, "case": case, "replay_verdict": proof.verdict, "proof": proof_to_dict(proof)}, indent=2))

        root = guess_root_cause(proof)
        grouped[case.get('category')].append({
            "id": cid,
            "title": case.get('title'),
            "expected": case.get('expected'),
            "observed": r.get('observed'),
            "replay": proof.verdict,
            "root_cause": root,
            "proof_path": str(out_path)
        })

    # Render triage markdown
    for cat, items in grouped.items():
        report_lines.append(f"## {cat} \n")
        for it in items:
            report_lines.append(f"- **{it['id']}**: {it['title']} — expected: **{it['expected']}**, observed: **{it['observed']}**, replay: **{it['replay']}** — root cause: *{it['root_cause']}* ([proof]({it['proof_path']}))")
        report_lines.append("\n")

    TRIAGE_MD.write_text("\n".join(report_lines))
    print(f"Wrote triage to {TRIAGE_MD} and proofs to {OUT_DIR}")


if __name__ == '__main__':
    run()
