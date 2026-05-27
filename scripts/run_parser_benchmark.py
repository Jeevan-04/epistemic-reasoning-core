#!/usr/bin/env python3
"""Run the parser benchmark half of the split evaluation.

This benchmark measures how well Manas translates natural-language cases into
structured symbolic clauses. It emits:

- per-case parser metrics in JSONL,
- a translated symbolic benchmark for replay, and
- proof traces for replayable cases.
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph
from manas.layer import Manas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "tests" / "benchmarks" / "adversarial_benchmarks_natural.json"
DEFAULT_OUT = ROOT / "tests" / "logs" / "parser_benchmark_results.jsonl"
DEFAULT_TRANSLATED = ROOT / "tests" / "logs" / "parser_translated_symbolic_benchmarks.json"
DEFAULT_PROOF_DIR = ROOT / "tests" / "logs" / "parser_proofs"
DEFAULT_SUMMARY = ROOT / "tests" / "logs" / "parser_benchmark_summary.json"

ARTICLE_WORDS = {"a", "an", "the", "both", "either", "this", "that", "these", "those"}


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf8"))


def strip_punct(text: str) -> str:
    return text.strip().strip(".?!,;:")


def normalize_symbol(text: str | None) -> str | None:
    if text is None:
        return None
    value = strip_punct(str(text)).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or None


def split_conjuncts(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\band\b|,", text, flags=re.IGNORECASE)]
    cleaned = []
    for part in parts:
        token = strip_punct(part)
        token = re.sub(r"^(?:both|either)\s+", "", token, flags=re.IGNORECASE)
        token = re.sub(r"^(?:a|an|the)\s+", "", token, flags=re.IGNORECASE)
        if token:
            cleaned.append(token)
    return cleaned


def parse_meta(text: str) -> tuple[str, dict[str, str]]:
    m = re.match(r"^(?P<src>[^(:]+)\s*\((?P<meta>[^\)]+)\)\s*:\s*(?P<body>.+)$", text)
    if m:
        meta: dict[str, str] = {"source": strip_punct(m.group("src"))}
        for part in m.group("meta").split(","):
            if "=" in part:
                key, value = [piece.strip() for piece in part.split("=", 1)]
                meta[key.lower()] = value
        return m.group("body").strip(), meta

    m = re.match(r"^(?P<src>[^:]+)\s*:\s*(?P<body>.+)$", text)
    if m:
        body = m.group("body").strip()
        if body.endswith("."):
            body = body[:-1]
        return body, {"source": strip_punct(m.group("src"))}

    m = re.match(r"^According to\s+(?P<src>.+?),\s*(?P<body>.+)$", text, flags=re.IGNORECASE)
    if m:
        body = m.group("body").strip()
        if body.endswith("."):
            body = body[:-1]
        return body, {"source": strip_punct(m.group("src"))}

    return text.strip(), {}


def parse_reliability(meta: dict[str, str]) -> float | None:
    for key in ("reliability", "r", "confidence", "weight"):
        if key in meta:
            try:
                return float(meta[key])
            except ValueError:
                return None
    return None


def clause_key(clause: dict) -> tuple:
    template = clause.get("template")
    polarity = clause.get("polarity", 1)
    canonical = clause.get("canonical", {}) if isinstance(clause.get("canonical"), dict) else {}
    source_reliability = clause.get("source_reliability")

    if template == "is_a":
        return (
            template,
            normalize_symbol(canonical.get("subject")),
            normalize_symbol(canonical.get("object")),
            polarity,
        )

    if template == "relation":
        relation_type = normalize_symbol(canonical.get("relation_type") or (clause.get("predicates") or [None])[0])
        entities = tuple(sorted(normalize_symbol(entity) for entity in clause.get("entities", []) if normalize_symbol(entity)))
        return (template, relation_type, entities, polarity)

    predicate = normalize_symbol(canonical.get("predicate") or (clause.get("predicates") or [None])[0])
    subject = normalize_symbol((clause.get("entities") or [None])[0])
    rank = canonical.get("rank")
    return (template, subject, predicate, polarity, rank, source_reliability)


def build_clause(
    template: str,
    subject: str,
    predicate: str,
    *,
    polarity: int = 1,
    rank: int | None = None,
    source: str | None = None,
    source_reliability: float | None = None,
    raw_text: str = "",
) -> dict:
    subject_n = normalize_symbol(subject)
    predicate_n = normalize_symbol(predicate)
    if not subject_n or not predicate_n:
        raise ValueError(f"Unsupported clause components: {subject!r}, {predicate!r}")

    clause = {
        "template": template,
        "canonical": {"entities": [subject_n]},
        "entities": [subject_n],
        "predicates": [predicate_n],
        "polarity": polarity,
        "parser_confidence": 1.0,
        "raw_text": raw_text,
    }

    if template == "is_a":
        clause["canonical"] = {
            "relation_type": "is_a",
            "subject": subject_n,
            "object": predicate_n,
            "entities": [subject_n, predicate_n],
        }
        clause["entities"] = [subject_n, predicate_n]
        clause["predicates"] = ["is_a"]
    elif template == "relation":
        clause["canonical"] = {
            "relation_type": predicate_n if predicate_n.startswith("can_") else f"can_{predicate_n}",
            "subject": subject_n,
            "object": predicate_n,
            "entities": [subject_n],
        }
        clause["predicates"] = [clause["canonical"]["relation_type"]]

    if rank is not None:
        clause["canonical"]["rank"] = rank
    if source is not None:
        clause["source"] = source
        clause["canonical"]["source"] = source
    if source_reliability is not None:
        clause["source_reliability"] = source_reliability

    return clause


def proposal_to_clauses(proposal: dict) -> list[dict]:
    template = proposal.get("template")
    canonical = proposal.get("canonical", {}) if isinstance(proposal.get("canonical"), dict) else {}
    raw_text = proposal.get("raw_text", "")
    source = proposal.get("source")
    source_reliability = proposal.get("source_reliability")
    if isinstance(source_reliability, str):
        try:
            source_reliability = float(source_reliability)
        except ValueError:
            source_reliability = None

    if template == "is_a":
        subject = canonical.get("subject") or (proposal.get("entities") or [None])[0]
        obj = canonical.get("object")
        if not obj and len(proposal.get("entities", [])) > 1:
            obj = proposal["entities"][1]
        if not subject or not obj:
            raise ValueError(f"Incomplete is_a proposal: {proposal}")

        extra_objects = []
        if len(proposal.get("entities", [])) > 2:
            extra_objects = [entity for entity in proposal["entities"][2:] if entity and normalize_symbol(entity) != normalize_symbol(subject)]
        if len(canonical.get("entities", [])) > 2:
            extra_objects.extend(canonical["entities"][2:])

        clauses = [build_clause("is_a", subject, obj, polarity=proposal.get("polarity", 1), source=source, source_reliability=source_reliability, raw_text=raw_text)]
        for extra in extra_objects:
            extra_n = normalize_symbol(extra)
            if extra_n and extra_n != normalize_symbol(obj):
                clauses.append(build_clause("is_a", subject, extra, polarity=proposal.get("polarity", 1), source=source, source_reliability=source_reliability, raw_text=raw_text))
        return clauses

    if template == "relation":
        relation_type = canonical.get("relation_type") or (proposal.get("predicates") or [None])[0]
        if not relation_type:
            raise ValueError(f"Incomplete relation proposal: {proposal}")
        subject = canonical.get("subject") or (proposal.get("entities") or [None])[0]
        object_text = canonical.get("object") or relation_type
        if not subject:
            raise ValueError(f"Incomplete relation proposal: {proposal}")
        return [
            {
                "template": "relation",
                "canonical": {
                    "relation_type": normalize_symbol(relation_type) if normalize_symbol(relation_type) and normalize_symbol(relation_type).startswith("can_") else f"can_{normalize_symbol(relation_type)}",
                    "subject": normalize_symbol(subject),
                    "object": normalize_symbol(object_text),
                    "entities": [normalize_symbol(subject)],
                    **({"source": source} if source else {}),
                },
                "entities": [normalize_symbol(subject)],
                "predicates": [f"can_{normalize_symbol(relation_type).removeprefix('can_')}"] if normalize_symbol(relation_type) else ["can_unknown"],
                "polarity": proposal.get("polarity", 1),
                "parser_confidence": proposal.get("parser_confidence", 1.0),
                "raw_text": raw_text,
                **({"source": source} if source else {}),
                **({"source_reliability": source_reliability} if source_reliability is not None else {}),
            }
        ]

    predicate = canonical.get("predicate") or (proposal.get("predicates") or [None])[0]
    if not predicate:
        raise ValueError(f"Incomplete default proposal: {proposal}")
    subject = (proposal.get("entities") or [None])[0]
    rank = canonical.get("rank")
    return [
        build_clause(
            "default",
            subject,
            predicate,
            polarity=proposal.get("polarity", 1),
            rank=rank,
            source=source,
            source_reliability=source_reliability,
            raw_text=raw_text,
        )
    ]


def clause_to_dsl(clause: dict) -> str:
    if clause["template"] == "is_a":
        return f"is_a({clause['canonical']['subject']}, {clause['canonical']['object']})"

    if clause["template"] == "relation":
        subject = clause["canonical"].get("subject", "unknown")
        predicate = clause["canonical"].get("object", "unknown")
        predicate = predicate[4:] if isinstance(predicate, str) and predicate.startswith("can_") else predicate
        return f"query({subject}, {predicate})"

    subject = clause.get("entities", ["unknown"])[0]
    predicate = clause["canonical"].get("predicate", clause.get("predicates", ["unknown"])[0])
    extras = []
    if clause.get("polarity", 1) < 0:
        extras.append("-")
    if clause["canonical"].get("rank") is not None:
        extras.append(f"rank={clause['canonical']['rank']}")
    if clause.get("source"):
        extras.append(f"source={normalize_symbol(clause['source'])}")
    if clause.get("source_reliability") is not None:
        extras.append(f"reliability={clause['source_reliability']}")
    if extras:
        return f"default({subject}, {predicate}, {', '.join(extras)})"
    return f"default({subject}, {predicate})"


def raw_to_gold_clauses(text: str) -> list[dict]:
    body, meta = parse_meta(text)
    source = meta.get("source")
    source_reliability = parse_reliability(meta)
    stripped = strip_punct(body)

    m = re.match(r"^(?P<sub>.+?)\s+is\s+both\s+an?\s+(?P<a>.+?)\s+and\s+an?\s+(?P<b>.+)$", stripped, flags=re.IGNORECASE)
    if m:
        return [
            build_clause("is_a", m.group("sub"), m.group("a"), source=source, source_reliability=source_reliability, raw_text=text),
            build_clause("is_a", m.group("sub"), m.group("b"), source=source, source_reliability=source_reliability, raw_text=text),
        ]

    m = re.match(r"^(?P<sub>.+?)\s+is\s+(?:an?\s+)?instance\s+of\s+(?P<parents>.+)$", stripped, flags=re.IGNORECASE)
    if m:
        clauses = []
        for parent in split_conjuncts(m.group("parents")):
            clauses.append(build_clause("is_a", m.group("sub"), parent, source=source, source_reliability=source_reliability, raw_text=text))
        return clauses

    m = re.match(r"^(?P<sub>.+?)\s+inherits\s+from\s+(?P<parents>.+)$", stripped, flags=re.IGNORECASE)
    if m:
        clauses = []
        for parent in split_conjuncts(m.group("parents")):
            clauses.append(build_clause("is_a", m.group("sub"), parent, source=source, source_reliability=source_reliability, raw_text=text))
        return clauses

    m = re.match(r"^(?P<sub>.+?)\s+is\s+(?:a|an|the)?\s*(?P<parent>.+)$", stripped, flags=re.IGNORECASE)
    if m and " and " not in stripped.lower():
        return [build_clause("is_a", m.group("sub"), m.group("parent"), source=source, source_reliability=source_reliability, raw_text=text)]

    m = re.match(r"^(?P<sub>.+?)\s+(?:do\s+not|does\s+not|did\s+not|not)\s+(?P<pred>.+)$", stripped, flags=re.IGNORECASE)
    if m:
        return [build_clause("default", m.group("sub"), m.group("pred"), polarity=-1, source=source, source_reliability=source_reliability, raw_text=text)]

    m = re.match(r"^(?P<sub>.+?)\s+typically\s+(?P<body>.+)$", stripped, flags=re.IGNORECASE)
    if m:
        subject = m.group("sub")
        body_text = m.group("body").strip()
        rank = None
        rank_match = re.search(r"\(\s*rank\s*=?\s*(?P<rank>\d+)\s*\)$", body_text, flags=re.IGNORECASE)
        if rank_match:
            rank = int(rank_match.group("rank"))
            body_text = re.sub(r"\(\s*rank\s*=?\s*\d+\s*\)$", "", body_text, flags=re.IGNORECASE).strip()
        neg = bool(re.match(r"^(?:do\s+not|does\s+not|did\s+not|not)\s+", body_text, flags=re.IGNORECASE))
        if neg:
            body_text = re.sub(r"^(?:do\s+not|does\s+not|did\s+not|not)\s+", "", body_text, flags=re.IGNORECASE).strip()
        return [build_clause("default", subject, body_text, polarity=-1 if neg else 1, rank=rank, source=source, source_reliability=source_reliability, raw_text=text)]

    m = re.match(r"^(?P<sub>.+?)\s+move\s+by\s+(?P<mode>.+)$", stripped, flags=re.IGNORECASE)
    if m:
        clause = build_clause("default", m.group("sub"), "move", source=source, source_reliability=source_reliability, raw_text=text)
        clause["canonical"]["mode"] = normalize_symbol(m.group("mode"))
        return [clause]

    m = re.match(r"^(?P<sub>.+?)\s+(?P<pred>[A-Za-z0-9_\-]+)\s*(?:\(\s*rank\s*=?\s*(?P<rank>\d+)\s*\))?$", stripped, flags=re.IGNORECASE)
    if m:
        rank = int(m.group("rank")) if m.group("rank") else None
        return [build_clause("default", m.group("sub"), m.group("pred"), rank=rank, source=source, source_reliability=source_reliability, raw_text=text)]

    m = re.match(r"^(?:Do|Does|Did|Is|Are|Was|Were)\s+(?P<sub>.+?)\s+(?P<pred>.+?)\??$", stripped, flags=re.IGNORECASE)
    if m:
        relation_type = normalize_symbol(m.group("pred")) or "unknown"
        subject = normalize_symbol(m.group("sub")) or "unknown"
        return [
            {
                "template": "relation",
                "canonical": {
                    "relation_type": f"can_{relation_type}" if not relation_type.startswith("can_") else relation_type,
                    "subject": subject,
                    "object": relation_type,
                    "entities": [subject],
                },
                "entities": [subject],
                "predicates": [f"can_{relation_type}" if not relation_type.startswith("can_") else relation_type],
                "polarity": 1,
                "parser_confidence": 1.0,
                "raw_text": text,
                **({"source": source} if source else {}),
                **({"source_reliability": source_reliability} if source_reliability is not None else {}),
            }
        ]

    m = re.match(r"^How\s+do(?:es)?\s+(?P<sub>.+?)\s+(?P<pred>.+?)\??$", stripped, flags=re.IGNORECASE)
    if m:
        subject = normalize_symbol(m.group("sub")) or "unknown"
        predicate = normalize_symbol(m.group("pred")) or "unknown"
        return [
            {
                "template": "relation",
                "canonical": {
                    "relation_type": predicate,
                    "subject": subject,
                    "object": predicate,
                    "entities": [subject],
                    "query_kind": "how",
                },
                "entities": [subject],
                "predicates": [predicate],
                "polarity": 1,
                "parser_confidence": 1.0,
                "raw_text": text,
            }
        ]

    # Fallback to Manas for anything we did not encode explicitly.
    proposal = proposal_to_clauses(Manas(llm_backend="mock").parse(text))
    return proposal


def serialize_proof(proof) -> dict:
    return json.loads(json.dumps(asdict(proof), default=str))


def run_case(case: dict, manas: Manas, proof_dir: Path) -> tuple[dict, dict]:
    premise_rows = []
    gold_premises: list[dict] = []
    parsed_premises: list[dict] = []
    premise_matches = 0
    premise_supported = 0

    for premise in case.get("premises", []):
        gold_clauses = raw_to_gold_clauses(premise)
        gold_premises.extend(gold_clauses)
        try:
            parsed = manas.parse(premise)
            parsed_clauses = proposal_to_clauses(parsed)
            parsed_premises.extend(parsed_clauses)
            gold_keys = Counter(clause_key(clause) for clause in gold_clauses)
            parsed_keys = Counter(clause_key(clause) for clause in parsed_clauses)
            match = parsed_keys == gold_keys
            premise_supported += 1
            premise_matches += int(match)
            premise_rows.append(
                {
                    "raw": premise,
                    "gold": [clause_to_dsl(clause) for clause in gold_clauses],
                    "parsed": [clause_to_dsl(clause) for clause in parsed_clauses],
                    "parse_success": match,
                    "parse_error": None,
                    "parser_confidence": parsed.get("parser_confidence"),
                }
            )
        except Exception as exc:
            premise_rows.append(
                {
                    "raw": premise,
                    "gold": [clause_to_dsl(clause) for clause in gold_clauses],
                    "parsed": None,
                    "parse_success": False,
                    "parse_error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    query_text = case.get("query", "")
    query_gold = raw_to_gold_clauses(query_text)
    query_parsed = proposal_to_clauses(manas.parse(query_text))
    query_match = Counter(clause_key(clause) for clause in query_gold) == Counter(clause_key(clause) for clause in query_parsed)

    translated_case = {
        "id": case.get("id"),
        "category": case.get("category"),
        "title": case.get("title"),
        "expected": case.get("expected"),
        "premises": [clause_to_dsl(clause) for clause in gold_premises],
        "query": clause_to_dsl(query_gold[0]) if query_gold else None,
        "notes": case.get("notes"),
    }

    replay_verdict = "unknown"
    proof_ref = None
    replayable = bool(gold_premises and query_gold)
    if replayable:
        chitta = ChittaGraph()
        buddhi = Buddhi(chitta)
        for clause in gold_premises:
            chitta.add_belief_from_proposal(clause)
        proof = buddhi.answer(query_gold[0])
        replay_verdict = proof.verdict
        proof_ref = proof_dir / f"proof_{case['id']}.json"
        proof_ref.write_text(json.dumps(serialize_proof(proof), indent=2), encoding="utf8")

    row = {
        "id": case.get("id"),
        "category": case.get("category"),
        "title": case.get("title"),
        "expected": case.get("expected"),
        "replay_verdict": replay_verdict,
        "parse_success": bool(gold_premises) and bool(query_gold) and premise_supported == len(case.get("premises", [])) and query_match,
        "premise_parse_match_rate": round(premise_matches / max(1, len(case.get("premises", []))), 3),
        "premise_count": len(case.get("premises", [])),
        "query_parse_success": query_match,
        "premise_results": premise_rows,
        "gold_query": [clause_to_dsl(clause) for clause in query_gold],
        "parsed_query": [clause_to_dsl(clause) for clause in query_parsed],
        "proof_ref": str(proof_ref) if proof_ref else None,
    }

    return row, translated_case


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--translated-out", type=Path, default=DEFAULT_TRANSLATED)
    parser.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    cases = load_cases(args.bench)
    if args.limit > 0:
        cases = cases[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.proof_dir.mkdir(parents=True, exist_ok=True)

    manas = Manas(llm_backend="mock")
    rows = []
    translated_cases = []
    replayable_cases = 0
    replayed_cases = 0
    total_premises = 0
    parsed_premises = 0
    matched_premises = 0
    matched_queries = 0
+    fully_parsed_cases = 0
+
    for case in cases:
        row, translated = run_case(case, manas, args.proof_dir)
        rows.append(row)
        translated_cases.append(translated)
        total_premises += len(case.get("premises", []))
        parsed_premises += sum(1 for premise_result in row["premise_results"] if premise_result.get("parsed") is not None)
        matched_premises += sum(1 for premise_result in row["premise_results"] if premise_result.get("parse_success"))
        matched_queries += int(row["query_parse_success"])
        if row["gold_query"] and row["premise_results"]:
            replayable_cases += 1
        if row["proof_ref"]:
            replayed_cases += 1
+        if row["parse_success"]:
+            fully_parsed_cases += 1

    with args.out.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    args.translated_out.write_text(json.dumps(translated_cases, indent=2), encoding="utf8")

    summary = {
        "cases_total": len(cases),
        "premises_total": total_premises,
        "premises_parsed": parsed_premises,
        "premises_matched": matched_premises,
        "queries_matched": matched_queries,
        "replayable_cases": replayable_cases,
        "replayed_cases": replayed_cases,
        "fully_parsed_cases": fully_parsed_cases,
        "premise_match_rate": round(matched_premises / max(1, total_premises), 3),
        "query_match_rate": round(matched_queries / max(1, len(cases)), 3),
        "success": replayed_cases == replayable_cases,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf8")

    print(f"Wrote {len(rows)} parser results to {args.out}")
    print(f"Translated benchmark written to {args.translated_out}")
    print(f"Proofs in {args.proof_dir}")
    print(f"Summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
