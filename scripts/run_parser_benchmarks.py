#!/usr/bin/env python3
"""Run a parser benchmark: NL -> BeliefProposal (Manas) and replay via Buddhi.

This script parses natural language benchmark cases using `Manas`, records
per-case translation artifacts, and (optionally) replays the translated
proposals through `ChittaGraph` + `Buddhi` to obtain grounding/verification
metrics. Outputs a JSONL results file and per-case translation files.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean
from dataclasses import asdict

from manas.layer import Manas
from chitta.graph import ChittaGraph
from buddhi.layer import Buddhi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "tests" / "benchmarks" / "parser_benchmarks.json"
DEFAULT_OUT = ROOT / "tests" / "logs" / "parser_benchmark_results.jsonl"
DEFAULT_TRANSLATIONS = ROOT / "tests" / "logs" / "parser_translations"
DEFAULT_PROOF_DIR = ROOT / "tests" / "logs" / "parser_proofs"


def load_cases(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def proposal_to_symbolic_clause(proposal: dict) -> dict:
    """Convert a Manas BeliefProposal to a compact symbolic dict similar to
    what the symbolic runner expects. This is lossy but useful for replay.
    """
    tpl = proposal.get("template")
    can = proposal.get("canonical", {})
    if tpl == "is_a":
        entities = proposal.get("entities", [])
        subj = entities[0] if entities else can.get("subject")
        obj = entities[1] if len(entities) > 1 else can.get("object")
        return {"type": "is_a", "subject": subj, "object": obj, "raw": proposal.get("raw_text"), "parser_confidence": proposal.get("parser_confidence", 0.0)}
    if tpl == "default":
        subj = can.get("entities", [None])[0]
        pred = can.get("predicate")
        return {"type": "default", "subject": subj, "predicate": pred, "polarity": proposal.get("polarity", 1), "rank": can.get("rank"), "raw": proposal.get("raw_text"), "parser_confidence": proposal.get("parser_confidence", 0.0)}
    if tpl == "relation":
        # capability or relation; canonical.relation_type often holds predicate
        rtype = can.get("relation_type") or (proposal.get("predicates") or [None])[0]
        subj = (proposal.get("entities") or [None])[0]
        return {"type": "relation", "subject": subj, "relation_type": rtype, "raw": proposal.get("raw_text"), "parser_confidence": proposal.get("parser_confidence", 0.0), "polarity": proposal.get("polarity", 1)}
    # Fallback
    return {"type": tpl or "unknown", "raw": proposal.get("raw_text"), "parser_confidence": proposal.get("parser_confidence", 0.0)}

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


def classify_pipeline_failure(parser_success: bool, proof, stage_rules: list[str], trace_determinism: bool, parse_errors: list[str]) -> tuple[str | None, str | None]:
    if not parser_success:
        if any("low_confidence" in err for err in parse_errors):
            return "parser", "low_confidence_ambiguity"
        if any("unknown" in err for err in parse_errors):
            return "parser", "unresolved_placeholder"
        return "parser", "malformed_query"

    if proof is None:
        return "argumentation", "zero_arguments"

    verdict = proof.verdict
    step_outputs = " ".join((step.output or "") for step in getattr(proof, "steps", []))
    if verdict == "unknown":
        if "No relevant beliefs found" in step_outputs:
            return "grounding", "missing_anchor"
        if any(rule == "grounding_check" for rule in stage_rules):
            return "grounding", "missing_anchor"
        if any(rule == "applicability_check" for rule in stage_rules):
            return "grounding", "predicate_mismatch"
        return "argumentation", "zero_arguments"

    if verdict == "conflict":
        return "defeat", "unresolved_tie"

    if not trace_determinism:
        return "replay", "nondeterministic_reconstruction"

    return None, None


def main() -> int:
    p = ArgumentParser()
    p.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    p.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR)
    args = p.parse_args()

    cases = load_cases(args.bench)
    if args.limit > 0:
        cases = cases[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.translations.mkdir(parents=True, exist_ok=True)
    args.proof_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        manas = Manas(llm_backend="mock")
        parse_errors = []
        premise_proposals = []
        premise_translations = []
        confidences = []

        for premise in case.get("premises", []):
            prop = manas.parse(premise)
            premise_proposals.append(prop)
            premise_translations.append(proposal_to_symbolic_clause(prop))
            confidences.append(prop.get("parser_confidence", 0.0))
            # Detect obvious parse fallback
            if prop.get("parser_confidence", 0.0) < 0.2 or prop.get("canonical", {}).get("relation_type") == "unknown":
                parse_errors.append(prop.get("canonical", {}).get("error", "low_confidence"))

        query_prop = manas.parse(case.get("query", ""))
        query_translation = proposal_to_symbolic_clause(query_prop)
        confidences.append(query_prop.get("parser_confidence", 0.0))
        if query_prop.get("parser_confidence", 0.0) < 0.2 or query_prop.get("canonical", {}).get("relation_type") == "unknown":
            parse_errors.append(query_prop.get("canonical", {}).get("error", "low_confidence"))

        parser_success = len(parse_errors) == 0
        avg_conf = mean(confidences) if confidences else 0.0

        # Save translation artifact
        trans_path = args.translations / f"{case['id']}_translation.json"
        trans_path.write_text(json.dumps({"premises": premise_translations, "query": query_translation}, indent=2), encoding="utf8")

        # Replay via Chitta + Buddhi if parse succeeded
        if parser_success:
            chitta = ChittaGraph()
            buddhi = Buddhi(chitta)
            for prop in premise_proposals:
                chitta.add_belief_from_proposal(prop)
            proof = buddhi.answer(query_prop)
            replay_proof = buddhi.answer(query_prop)
            verdict = proof.verdict
            proof_path = args.proof_dir / f"proof_{case['id']}.json"
            proof_path.write_text(json.dumps(asdict(proof), indent=2, default=str), encoding="utf8")
            stage_rules = [s.rule for s in proof.steps]
            grounding_success = verdict != "unknown" and "entity_existence_check" not in stage_rules
            argument_count = len(proof.metadata.get("arguments", [])) if hasattr(proof, "metadata") else 0
            conflict_count = len(proof.conflicts) if hasattr(proof, "conflicts") else 0
            reasoning_depth = len(proof.steps)
            avg_attack_degree = (conflict_count / max(argument_count, 1)) if argument_count else 0.0
            trace_determinism = proof_signature(proof) == proof_signature(replay_proof)
            replay_verification_status = "passed" if trace_determinism else "failed"
            failure_stage, failure_reason = classify_pipeline_failure(parser_success, proof, stage_rules, trace_determinism, parse_errors)
        else:
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
            failure_stage, failure_reason = classify_pipeline_failure(parser_success, None, stage_rules, trace_determinism, parse_errors)

        results.append(
            {
                "id": case["id"],
                "title": case.get("title"),
                "expected": case.get("expected"),
                "observed": verdict,
                "parser_success": parser_success,
                "avg_parser_confidence": avg_conf,
                "num_premise_proposals": len(premise_proposals),
                "grounding_success": grounding_success,
                "argument_count": argument_count,
                "conflict_count": conflict_count,
                "avg_reasoning_depth": reasoning_depth,
                "avg_attack_degree": avg_attack_degree,
                "replay_verification_status": replay_verification_status,
                "trace_determinism": trace_determinism,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "translation_ref": str(trans_path),
                "proof_ref": str(proof_path) if proof_path else None,
            }
        )

    with args.out.open("w", encoding="utf8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} parser results to {args.out}")
    print(f"Translations in {args.translations}")
    print(f"Proofs in {args.proof_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
