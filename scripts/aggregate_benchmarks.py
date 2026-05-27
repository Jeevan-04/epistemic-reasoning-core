#!/usr/bin/env python3
"""
Aggregate benchmark JSONL results into CSV/JSON/Markdown summaries and per-case pipeline outcomes.

Outputs:
- tests/logs/bench_compare_results.csv
- tests/logs/bench_compare_results.json
- tests/logs/bench_summary.md
- tests/logs/aggregated_outputs/pipeline_outcomes.jsonl

This script is conservative and uses heuristics to attribute failures when explicit
failure labels are not available. It also enriches each record with a `pipeline_outcome`
field for reproducible debugging.
"""
import os
import json
import glob
import csv
from collections import defaultdict, Counter
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT, 'tests', 'logs')
BENCH_DIR = os.path.join(ROOT, 'tests', 'benchmarks')
OUT_CSV = os.path.join(LOG_DIR, 'bench_compare_results.csv')
OUT_JSON = os.path.join(LOG_DIR, 'bench_compare_results.json')
OUT_MD = os.path.join(LOG_DIR, 'bench_summary.md')
OUT_PIPELINE_DIR = os.path.join(LOG_DIR, 'aggregated_outputs')
os.makedirs(OUT_PIPELINE_DIR, exist_ok=True)
OUT_PIPELINE = os.path.join(OUT_PIPELINE_DIR, 'pipeline_outcomes.jsonl')


def load_bench_defs():
    mapping = {}
    # load parser and symbolic benchmark definitions to map id->category
    for path in glob.glob(os.path.join(BENCH_DIR, '*.json')):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        if 'id' in entry and 'category' in entry:
                            mapping[entry['id']] = entry['category']
        except Exception:
            continue
    return mapping


def read_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed lines
                continue


def canonical_final_verdict(rec):
    for k in ('observed', 'final_verdict', 'verdict'):
        if k in rec:
            return rec[k]
    return rec.get('observed', rec.get('final_verdict', None))


def determine_failure_type(rec):
    # Prefer explicit execution-time labels when present.
    if 'failure_stage' in rec or 'failure_reason' in rec:
        stage = rec.get('failure_stage')
        reason = rec.get('failure_reason')
        # Successful rows should not be counted as failures.
        if stage is None and reason is None:
            return None
        return (stage or 'unknown', reason or 'unknown')

    # Improved failure attribution using translation and proof artifacts when available.
    # Priority: explicit parser failure -> parser evidence -> grounding evidence -> argumentation -> defeat -> replay
    # 1. Parser explicit
    if 'parser_success' in rec and rec.get('parser_success') is False:
        # try to refine reason by inspection of translation
        tr = rec.get('translation_ref')
        if not tr or not os.path.exists(tr):
            return ('parser', 'no_translation')
        try:
            with open(tr, 'r') as f:
                t = json.load(f)
                # malformed if no premises or query
                if not t.get('premises') or not t.get('query'):
                    return ('parser', 'malformed_translation')
                # unresolved placeholders (X, Y) detection
                text = json.dumps(t)
                if 'X' in text or 'Y' in text or 'x' in text and 'subject' in text:
                    return ('parser', 'unresolved_placeholder')
                # low confidence
                confidences = []
                for p in t.get('premises', []):
                    if 'parser_confidence' in p:
                        try:
                            confidences.append(float(p['parser_confidence']))
                        except Exception:
                            pass
                if confidences and mean(confidences) < 0.6:
                    return ('parser', 'low_confidence_ambiguity')
        except Exception:
            return ('parser', 'malformed_translation')
        return ('parser', 'unknown_parser_error')

    # 2. Grounding evidence from proofs
    proof_path = rec.get('proof_ref') or rec.get('proof')
    if proof_path and os.path.exists(proof_path):
        try:
            with open(proof_path, 'r') as f:
                proof = json.load(f)
                steps = proof.get('steps', [])
                for s in steps:
                    rule = s.get('rule','').lower()
                    out = (s.get('output') or '').lower()
                    if 'applicability_check' in rule:
                        # detect pred or entity zeros
                        if 'pred=0' in out or 'pred=0.00' in out:
                            return ('grounding', 'predicate_mismatch')
                        if 'entity=0' in out or 'entity=0.00' in out:
                            return ('grounding', 'entity_normalization')
                    if 'grounding_check' in rule:
                        if 'direct grounding: 0' in out:
                            return ('grounding', 'missing_anchor')
                # conflicts and tie detection
                conflicts = proof.get('conflicts', [])
                if conflicts and str(proof.get('verdict','')).lower() == 'conflict':
                    return ('defeat', 'unresolved_tie')
        except Exception:
            pass

    # 3. Argumentation level
    if rec.get('argument_count', 0) == 0:
        # if grounding appears to have failed earlier, label grounding, else argumentation
        if 'grounding_success' in rec and not rec.get('grounding_success'):
            return ('grounding', 'missing_anchor')
        return ('argumentation', 'zero_arguments')

    # 4. Replay / determinism
    if rec.get('trace_determinism') is False or (rec.get('replay_verification_status') and str(rec.get('replay_verification_status')).lower() != 'passed'):
        return ('replay', 'nondeterministic_reconstruction')

    return ('other', 'unknown')


def enrich_and_collect():
    bench_map = load_bench_defs()
    # Only aggregate the two benchmark streams that define the paper's split-evaluation story.
    results_files = [
        os.path.join(LOG_DIR, 'parser_benchmark_results.jsonl'),
        os.path.join(LOG_DIR, 'reasoning_symbolic_results.jsonl'),
    ]
    per_case = []
    for path in results_files:
        mode = 'symbolic' if 'symbolic' in os.path.basename(path) else 'nl'
        for rec in read_jsonl(path):
            rec = dict(rec)
            rec['_source_file'] = os.path.basename(path)
            rec['mode'] = mode
            rec_id = rec.get('id')
            rec['category'] = rec.get('category') or bench_map.get(rec_id, 'unknown')

            # pipeline outcome
            pipeline = {
                'parse_success': bool(rec.get('parser_success')) if 'parser_success' in rec else (None if mode=='symbolic' else None),
                'grounding_success': bool(rec.get('grounding_success', False)),
                'argument_construction': rec.get('argument_count', 0) > 0,
                'reasoning_executed': (rec.get('argument_count', 0) > 0) or (rec.get('avg_reasoning_depth', 0) > 0),
                'final_verdict': canonical_final_verdict(rec)
            }
            rec['pipeline_outcome'] = pipeline
            fa = determine_failure_type(rec)
            # normalize to dict {'stage':.., 'reason':..}; keep successful rows empty
            if isinstance(fa, tuple) and len(fa) == 2:
                rec['failure_attribution'] = {'stage': fa[0], 'reason': fa[1]}
            elif fa is None:
                rec['failure_attribution'] = None
            else:
                rec['failure_attribution'] = {'stage': 'unknown', 'reason': str(fa)}
            per_case.append(rec)

    # write per-case pipeline outcomes
    with open(OUT_PIPELINE, 'w') as f:
        for r in per_case:
            json.dump({'id': r.get('id'), 'pipeline_outcome': r['pipeline_outcome'], 'failure_attribution': r['failure_attribution']}, f)
            f.write('\n')

    return per_case


def aggregate(per_case):
    by_category = defaultdict(list)
    by_mode = defaultdict(list)
    for r in per_case:
        by_category[r['category']].append(r)
        by_mode[r['mode']].append(r)

    def compute_stats(list_rec):
        n = len(list_rec)
        if n == 0:
            return None
        parse_success = [1 for r in list_rec if r.get('pipeline_outcome', {}).get('parse_success') is True]
        grounding_success = [1 for r in list_rec if r.get('pipeline_outcome', {}).get('grounding_success')]
        reasoning_success = [1 for r in list_rec if (canonical_final_verdict(r) is not None and canonical_final_verdict(r) != 'unknown' and 'expected' in r and str(canonical_final_verdict(r)).upper() == str(r.get('expected','')).upper())]
        avg_args = [r.get('argument_count', 0) for r in list_rec]
        avg_attacks = [r.get('avg_attack_degree', r.get('avg_attack', 0.0)) for r in list_rec]
        replay_verified = [1 for r in list_rec if r.get('replay_verification_status','').lower() == 'passed' or r.get('replay_verification','') is True]
        return {
            'count': n,
            'parse_success_pct': round(100 * (sum(parse_success) / n), 1) if any('parser_success' in r for r in list_rec) else 'N/A',
            'grounding_success_pct': round(100 * (sum(grounding_success) / n), 1) if any('grounding_success' in r for r in list_rec) else 'N/A',
            'reasoning_success_pct': round(100 * (sum(reasoning_success) / n), 1),
            'avg_args': round(mean(avg_args), 2) if avg_args else 0.0,
            'avg_attacks': round(mean(avg_attacks), 2) if avg_attacks else 0.0,
            'replay_verified_pct': round(100 * (sum(replay_verified) / n), 1)
        }

    cat_stats = {cat: compute_stats(lst) for cat, lst in by_category.items()}
    mode_stats = {mode: compute_stats(lst) for mode, lst in by_mode.items()}

    # failure distribution (use only actual failures)
    failure_counter = Counter(
        (r.get('failure_attribution', {}).get('stage', 'unknown'), r.get('failure_attribution', {}).get('reason', 'unknown'))
        for r in per_case
        if r.get('failure_attribution')
    )

    return cat_stats, mode_stats, failure_counter


def write_outputs(cat_stats, mode_stats, failure_counter, per_case):
    # CSV by category
    with open(OUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['Category', 'Count', 'Parse Success %', 'Grounding Success %', 'Reasoning Success %', 'Avg Args', 'Avg Attacks', 'Replay Verified %']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for cat, st in sorted(cat_stats.items()):
            if st is None:
                continue
            writer.writerow({
                'Category': cat,
                'Count': st['count'],
                'Parse Success %': st['parse_success_pct'],
                'Grounding Success %': st['grounding_success_pct'],
                'Reasoning Success %': st['reasoning_success_pct'],
                'Avg Args': st['avg_args'],
                'Avg Attacks': st['avg_attacks'],
                'Replay Verified %': st['replay_verified_pct']
            })

    # JSON summary
    # Convert failure_counter to list of dicts for JSON-friendliness
    failure_list = []
    total_failures = sum(failure_counter.values())
    for (stage, reason), cnt in failure_counter.items():
        failure_list.append({'stage': stage, 'reason': reason, 'count': cnt, 'pct': round(100 * cnt / total_failures, 1) if total_failures else 0.0})

    summary = {
        'by_category': cat_stats,
        'by_mode': mode_stats,
        'failure_distribution': failure_list
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    # Markdown summary with simple tables
    with open(OUT_MD, 'w') as f:
        f.write('# Benchmark Summary\n\n')
        f.write('## Category-wise Metrics\n\n')
        f.write('| Category | Count | Parse Success % | Grounding Success % | Reasoning Success % | Avg Args | Avg Attacks | Replay Verified % |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for cat, st in sorted(cat_stats.items()):
            if st is None:
                continue
            f.write(f"| {cat} | {st['count']} | {st['parse_success_pct']} | {st['grounding_success_pct']} | {st['reasoning_success_pct']} | {st['avg_args']} | {st['avg_attacks']} | {st['replay_verified_pct']} |\n")

        f.write('\n## Mode comparison\n\n')
        f.write('| Mode | Count | Parse Success % | Grounding Success % | Reasoning Success % | Avg Args | Avg Attacks | Replay Verified % |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for mode, st in sorted(mode_stats.items()):
            if st is None:
                continue
            f.write(f"| {mode} | {st['count']} | {st['parse_success_pct']} | {st['grounding_success_pct']} | {st['reasoning_success_pct']} | {st['avg_args']} | {st['avg_attacks']} | {st['replay_verified_pct']} |\n")

        f.write('\n## Failure distribution\n\n')
        f.write('| Stage | Reason | Count | % |\n')
        f.write('|---|---|---:|---:|\n')
        total = sum(failure_counter.values())
        for (stage, reason), cnt in failure_counter.most_common():
            pct = round(100 * cnt / total, 1) if total else 0.0
            f.write(f"| {stage} | {reason} | {cnt} | {pct}% |\n")


def main():
    per_case = enrich_and_collect()
    cat_stats, mode_stats, failure_counter = aggregate(per_case)
    write_outputs(cat_stats, mode_stats, failure_counter, per_case)
    print('Wrote:', OUT_CSV, OUT_JSON, OUT_MD, OUT_PIPELINE)


if __name__ == '__main__':
    main()
