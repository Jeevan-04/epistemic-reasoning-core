"""Append benchmark rows into docs/TRACE_EXAMPLES.md in canonical form.

Reads `tests/logs/bench_compare_results.json`, resolves the matching proof trace,
and appends a canonical markdown row unless that trace already exists.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from scripts.export_trace_examples_from_traces import (
        _argument_vector,
        _category_for_trace,
        _input_for_trace,
        _load_json,
        _notes,
        _root_cause,
        _source_reliability,
    )
except Exception:
    # Allow running this file directly (python scripts/append_bench_results.py)
    from export_trace_examples_from_traces import (
        _argument_vector,
        _category_for_trace,
        _input_for_trace,
        _load_json,
        _notes,
        _root_cause,
        _source_reliability,
    )

LOG_JSON = Path('tests/logs/bench_compare_results.json')
TRACE_MD = Path('docs/TRACE_EXAMPLES.md')
ROOT = Path(__file__).resolve().parents[1]


def load_results():
    if not LOG_JSON.exists():
        print('No bench_compare_results.json found in tests/logs/')
        return []
    with LOG_JSON.open() as f:
        return json.load(f)


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'`', '', text)
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text


def _latest_proof_for_query(query: str) -> Path | None:
    slug = _slugify(query)
    candidates = sorted(Path('tests/logs').glob(f'proof_{slug}_*.json'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def append_rows(results):
    TRACE_MD.parent.mkdir(parents=True, exist_ok=True)
    existing = TRACE_MD.read_text() if TRACE_MD.exists() else ''
    marker = '\n## Why These Rows Matter\n'
    table_prefix, table_suffix = existing, ''
    if marker in existing:
        table_prefix, table_suffix = existing.split(marker, 1)
        table_suffix = marker + table_suffix

    appended = 0
    skipped = 0
    new_rows: list[str] = []
    existing_ids = [int(match.group(1)) for match in re.finditer(r'^\| (\d+) \|', table_prefix, re.MULTILINE)]
    next_id = max(existing_ids, default=0) + 1

    for r in results:
        scenario = r.get('scenario')
        proof_path = _latest_proof_for_query(scenario)
        if not proof_path:
            skipped += 1
            print(f'Skipped {scenario}: no matching proof trace found')
            continue

        trace = _load_json(proof_path)
        if not trace:
            skipped += 1
            print(f'Skipped {scenario}: could not load proof trace {proof_path}')
            continue

        trace_ref = str(proof_path)
        if trace_ref in existing or trace_ref in ''.join(new_rows):
            skipped += 1
            print(f'Skipped {scenario}: already present in TRACE_EXAMPLES.md')
            continue

        proof = trace.get('proof', {}) if isinstance(trace.get('proof'), dict) else {}
        details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
        argument_meta = proof.get('metadata', {}).get('arguments', []) if isinstance(proof.get('metadata'), dict) else []

        input_text = _input_for_trace(trace, proof_path)
        category = _category_for_trace(trace, proof_path)
        expected = 'N/A'
        observed = str(trace.get('verdict', proof.get('verdict', 'UNKNOWN'))).upper()
        verdict = 'INFO'
        row_id = str(next_id + appended)
        failure_severity = 'N/A'
        reproducibility_status = 'deterministic'
        row = (
            f"| {row_id} | {category} | `{input_text}` | {expected} | {observed} | {verdict} | "
            f"{_root_cause(trace)} | {_argument_vector(argument_meta)} | {_source_reliability(argument_meta)} | "
            f"{failure_severity} | {reproducibility_status} | {trace_ref} | {_notes(trace)} |\n"
        )
        new_rows.append(row)
        appended += 1
        print(f'Appended {scenario}: {trace_ref}')

    if appended:
        updated = table_prefix
        if updated and not updated.endswith('\n'):
            updated += '\n'
        updated += ''.join(new_rows)
        if table_suffix:
            if not updated.endswith('\n'):
                updated += '\n'
            updated += table_suffix.lstrip('\n')
        TRACE_MD.write_text(updated)

    print(f'Append summary: {appended} appended, {skipped} skipped')


def main():
    results = load_results()
    if not results:
        return
    append_rows(results)


if __name__ == '__main__':
    main()
