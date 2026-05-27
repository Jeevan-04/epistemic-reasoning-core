#!/usr/bin/env python3
"""Run a controlled symbolic reasoning benchmark.

This benchmark bypasses Manas entirely so we can evaluate Buddhi/Chitta on
aligned symbolic inputs. It is the first half of the split benchmark plan:

- symbolic reasoning benchmark: structured inputs -> verdict
- parser benchmark: natural language -> structured inputs (to be added next)
"""

from __future__ import annotations

import json
import os
import re
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "tests" / "benchmarks" / "reasoning_symbolic_benchmarks.json"
DEFAULT_OUT = ROOT / "tests" / "logs" / "reasoning_symbolic_results.jsonl"
DEFAULT_PROOF_DIR = ROOT / "tests" / "logs" / "symbolic_proofs"


def split_args(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*,\s*", text.strip()) if part.strip()]


def parse_clause(text: str) -> dict:
    clause = text.strip().rstrip(".")
    match = re.match(r"^(?P<name>[a-z_]+)\((?P<args>.*)\)$", clause, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported symbolic clause: {text}")

    name = match.group("name").lower()
    args = split_args(match.group("args"))

    if name == "is_a":
        if len(args) != 2:
            raise ValueError(f"is_a expects 2 args, got {len(args)}: {text}")
        subject, obj = args
        return {
            "template": "is_a",
            "canonical": {
                "relation_type": "is_a",
                "subject": subject,
                "object": obj,
                "entities": [subject, obj],
            },
            "entities": [subject, obj],
            "predicates": ["is_a"],
            "polarity": 1,
            "parser_confidence": 1.0,
            "raw_text": text,
            "epistemic_type": "OBSERVATION",
        }

    if name == "default":
        if len(args) < 2:
            raise ValueError(f"default expects at least 2 args, got {len(args)}: {text}")
        subject, predicate = args[0], args[1]
        polarity = 1
        rank = None
        source = None
        source_reliability = None
        for extra in args[2:]:
            extra_l = extra.lower()
            if extra_l in {"-", "neg", "negative"}:
                polarity = -1
            elif extra_l in {"+", "pos", "positive"}:
                polarity = 1
            elif extra_l.startswith("rank="):
                rank = int(extra_l.split("=", 1)[1])
            elif extra_l.startswith("source="):
                source = extra.split("=", 1)[1]
            elif extra_l.startswith("reliability=") or extra_l.startswith("r="):
                source_reliability = float(extra_l.split("=", 1)[1])
        canonical = {"predicate": predicate, "entities": [subject]}
        if rank is not None:
            canonical["rank"] = rank
        if source is not None:
            canonical["source"] = source
        return {
            "template": "default",
            "canonical": canonical,
            "entities": [subject],
            "predicates": [predicate],
            "polarity": polarity,
            "parser_confidence": 1.0,
            "raw_text": text,
            **({"source": source} if source is not None else {}),
            **({"source_reliability": source_reliability} if source_reliability is not None else {}),
            "epistemic_type": "EXCEPTION" if polarity < 0 else "DEFAULT",
        }

    if name == "query":
        if len(args) != 2:
            raise ValueError(f"query expects 2 args, got {len(args)}: {text}")
        subject, predicate = args
        relation_type = predicate if predicate.startswith("can_") else f"can_{predicate}"
        return {
            "template": "relation",
            "canonical": {
                "relation_type": relation_type,
                "entities": [subject],
                "subject": subject,
                "object": predicate,
            },
            "entities": [subject],
            "predicates": [relation_type],
            "polarity": 1,
            "parser_confidence": 1.0,
            "raw_text": text,
        }

    raise ValueError(f"Unsupported symbolic clause type: {name}")


def serialize_proof(proof) -> dict:
    return json.loads(json.dumps(asdict(proof), default=str))


def proof_signature(proof) -> dict:
    return {
        "verdict": proof.verdict,
        "steps": [
            {"rule": step.rule, "output": step.output, "confidence": step.confidence}
            for step in proof.steps
        ],
        "conflicts": [asdict(conflict) for conflict in proof.conflicts],
        "arguments": proof.metadata.get("arguments", []),
    }


def classify_pipeline_failure(parse_ok: bool, proof, stage_rules: list[str], trace_determinism: bool, parse_error: str | None, verdict: str | None) -> tuple[str | None, str | None]:
    if not parse_ok:
        if parse_error:
            return "parser", "malformed_query"
        return "parser", "malformed_query"

    if proof is None:
        return "argumentation", "zero_arguments"

    step_outputs = " ".join((step.output or "") for step in getattr(proof, "steps", []))
    if verdict == "unknown":
        if "No relevant beliefs found" in step_outputs:
            return "grounding", "missing_anchor"
        if any(rule == "grounding_check" for rule in stage_rules):
            return "grounding", "missing_anchor"
        return "argumentation", "zero_arguments"

    if verdict == "conflict":
        return "defeat", "unresolved_tie"

    if not trace_determinism:
        return "replay", "nondeterministic_reconstruction"

    return None, None


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf8"))


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR)
    args = parser.parse_args()

    cases = load_cases(args.bench)
    if args.limit > 0:
        cases = cases[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.proof_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        chitta = ChittaGraph()
        buddhi = Buddhi(chitta)

        parse_ok = True
        parse_error = None
        try:
            for premise in case.get("premises", []):
                proposal = parse_clause(premise)
                chitta.add_belief_from_proposal(proposal)
            query_proposal = parse_clause(case["query"])
        except Exception as exc:
            parse_ok = False
            parse_error = f"{exc.__class__.__name__}: {exc}"
            query_proposal = None

        if parse_ok and query_proposal is not None:
            proof = buddhi.answer(query_proposal)
            replay_proof = buddhi.answer(query_proposal)
            verdict = proof.verdict
            proof_path = args.proof_dir / f"proof_{case['id']}.json"
            proof_path.write_text(json.dumps(serialize_proof(proof), indent=2), encoding="utf8")
            stage_rules = [step.rule for step in proof.steps]
            grounding_success = verdict != "unknown" and "entity_existence_check" not in stage_rules
            argument_count = len(proof.metadata.get("arguments", []))
            conflict_count = len(proof.conflicts)
            reasoning_depth = len(proof.steps)
            avg_attack_degree = (conflict_count / max(argument_count, 1)) if argument_count else 0.0
            trace_determinism = proof_signature(proof) == proof_signature(replay_proof)
            replay_verification_status = "passed" if trace_determinism else "failed"
            failure_stage, failure_reason = classify_pipeline_failure(parse_ok, proof, stage_rules, trace_determinism, parse_error, verdict)
        else:
            proof = None
            verdict = "unknown"
            proof_path = None
            stage_rules = []
            grounding_success = False
            argument_count = 0
            conflict_count = 0
            reasoning_depth = 0
            avg_attack_degree = 0.0
            trace_determinism = False
            replay_verification_status = "not_applicable"
            failure_stage, failure_reason = classify_pipeline_failure(parse_ok, None, stage_rules, trace_determinism, parse_error, verdict)

        results.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "title": case.get("title"),
                "expected": case.get("expected"),
                "observed": verdict,
                "dsl_parse_success": parse_ok,
                "parse_error": parse_error,
                "grounding_success": grounding_success,
                "argument_count": argument_count,
                "conflict_count": conflict_count,
                "avg_reasoning_depth": reasoning_depth,
                "avg_attack_degree": avg_attack_degree,
                "replay_verification_status": replay_verification_status,
                "trace_determinism": trace_determinism,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "stage_rules": stage_rules,
                "proof_ref": str(proof_path) if proof_path else None,
            }
        )

    with args.out.open("w", encoding="utf8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(results)} symbolic results to {args.out}")
    print(f"Proofs in {args.proof_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())