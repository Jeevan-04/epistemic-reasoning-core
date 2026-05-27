"""Canonicalize and deduplicate docs/FAILURE_TABLES.md.

This script rewrites the failure table into a single deduped markdown table with
additional columns for argument summaries and source reliability.

It reads:
- docs/FAILURE_TABLES.md
- tests/logs/**/*.json (proof traces)

Usage:
    PYTHONPATH=. python scripts/canonicalize_failure_tables.py
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FAILURE_MD = ROOT / 'docs' / 'FAILURE_TABLES.md'
LOG_DIRS = [ROOT / 'tests' / 'logs', ROOT / 'tests' / 'logs' / 'pytest']


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find_trace(trace_ref: str) -> Path | None:
    candidate = ROOT / trace_ref
    if candidate.exists():
        return candidate
    for log_dir in LOG_DIRS:
        maybe = log_dir / Path(trace_ref).name
        if maybe.exists():
            return maybe
    return None


def _slugify_query(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r'`', '', cleaned)
    cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned


def _find_latest_trace_from_input(input_text: str) -> Path | None:
    match = re.search(r'Query:\s*(.+)$', input_text)
    if not match:
        return None
    query = match.group(1).strip('` ')
    query = query.rstrip('.').strip()
    slug = _slugify_query(query)
    if not slug:
        return None
    candidates = []
    for log_dir in LOG_DIRS:
        candidates.extend(log_dir.glob(f'proof_{slug}_*.json'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _compact_argument_summary(proof: dict[str, Any]) -> tuple[str, str]:
    meta = (proof or {}).get('proof', {}).get('metadata', {}) if proof else {}
    arguments = meta.get('arguments', []) if isinstance(meta, dict) else []
    if not arguments:
        return '-', '-'

    summaries = []
    reliabilities = []
    for arg in arguments:
        if not isinstance(arg, dict):
            continue
        spec = arg.get('specificity', '?')
        rank = arg.get('rank', '?')
        rel = arg.get('source_reliability', '?')
        act = arg.get('activation', '?')
        neg = '!' if arg.get('is_negative') else '+'
        claim = arg.get('claim', 'unknown')
        summaries.append(f"{neg}{claim}[s={spec},r={rank},rel={rel},a={act}]")
        if isinstance(rel, (int, float)):
            reliabilities.append(rel)

    source_reliability = str(round(sum(reliabilities) / len(reliabilities), 3)) if reliabilities else '-'
    return ' ; '.join(summaries), source_reliability


def _normalize_input(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('`', '').strip())


def _parse_existing_rows(md_text: str):
    rows = []
    in_table = False
    for line in md_text.splitlines():
        if line.startswith('| id |'):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith('|'):
            # stop once table ends
            if rows:
                break
            continue
        if line.startswith('|---'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 9:
            continue
        rows.append(cells)
    return rows


def _normalize_row(cells: list[str]) -> dict[str, str]:
    """Normalize either 9-col legacy rows or 11-col canonical rows."""
    if len(cells) >= 11:
        return {
            'id': cells[0],
            'category': cells[1],
            'input': cells[2],
            'expected': cells[3],
            'observed': cells[4],
            'verdict': cells[5],
            'root_cause': cells[6],
            'argument_vector': cells[7],
            'source_reliability': cells[8],
            'trace_ref': cells[9],
            'notes': cells[10],
        }
    return {
        'id': cells[0] if len(cells) > 0 else '-',
        'category': cells[1] if len(cells) > 1 else '-',
        'input': cells[2] if len(cells) > 2 else '-',
        'expected': cells[3] if len(cells) > 3 else '-',
        'observed': cells[4] if len(cells) > 4 else '-',
        'verdict': cells[5] if len(cells) > 5 else '-',
        'root_cause': cells[6] if len(cells) > 6 else '-',
        'argument_vector': '-',
        'source_reliability': '-',
        'trace_ref': cells[7] if len(cells) > 7 else '-',
        'notes': cells[8] if len(cells) > 8 else '-',
    }


def _dedupe_key(cells: list[str]) -> tuple[str, str]:
    row = _normalize_row(cells)
    trace_ref = row['trace_ref']
    input_text = _normalize_input(row['input'])
    return (trace_ref, input_text)


def main():
    if not FAILURE_MD.exists():
        raise SystemExit(f"Missing {FAILURE_MD}")

    md_text = FAILURE_MD.read_text()
    existing_rows = _parse_existing_rows(md_text)

    deduped: "OrderedDict[tuple[str, str], list[str]]" = OrderedDict()
    for cells in existing_rows:
        deduped[_dedupe_key(cells)] = cells

    canonical_rows = []
    for idx, cells in enumerate(deduped.values(), start=1):
        row = _normalize_row(cells)
        # Canonicalize to 11 columns:
        # id | category | input | expected | observed | verdict | root_cause | argument_vector | source_reliability | trace_ref | notes
        trace_ref = row['trace_ref']
        trace_file = _find_trace(trace_ref)
        proof = _load_json(trace_file) if trace_file else None
        if not proof or not proof.get('proof', {}).get('metadata'):
            latest = _find_latest_trace_from_input(row['input'])
            if latest:
                proof = _load_json(latest)
                trace_file = latest
        arg_vec, src_rel = _compact_argument_summary(proof or {})
        if trace_file:
            try:
                trace_ref = str(trace_file.relative_to(ROOT))
            except Exception:
                trace_ref = str(trace_file)

        category = row['category']
        input_text = row['input']
        expected = row['expected']
        observed = row['observed']
        verdict = row['verdict']
        root_cause = row['root_cause']
        notes = row['notes']

        if proof:
            proof_verdict = str(proof.get('verdict', '')).upper()
            if proof_verdict:
                observed = proof_verdict
                verdict = 'PASS' if observed == expected else 'FAIL'

        if re.fullmatch(r'\d+(?:\.\d+)?', notes.strip()):
            notes = '-'

        # If this row came from an old benchmark append line, columns are shifted.
        if category == 'Benchmark' and len(cells) >= 9:
            root_cause = row['root_cause']
            trace_ref = row['trace_ref']
            notes = row['notes']

        canonical_rows.append([
            str(idx), category, input_text, expected, observed, verdict,
            root_cause, arg_vec, src_rel, trace_ref, notes,
        ])

    out = []
    out.append('# Failure Tables (Canonicalized)')
    out.append('')
    out.append('This document records observed behavior on benchmark scenarios with reproducible')
    out.append('proof traces. Rows below are deduplicated and enriched with argument metadata.')
    out.append('')
    out.append('| id | category | input | expected | observed | verdict | root_cause | argument_vector | source_reliability | trace_ref | notes |')
    out.append('|---:|---|---|---|---|---|---|---|---:|---|---|')

    for row in canonical_rows:
        out.append('| ' + ' | '.join(row) + ' |')

    out.append('')
    out.append('## Current Gaps')
    out.append('')
    out.append('- `parser`: one confirmed issue in cyclic case where no relevant beliefs are retrieved.')
    out.append('- `inheritance`: still need additional non-cyclic inheritance edge cases (multiple parents, deep overrides).')
    out.append('- `argument explosion`: no confirmed timeout trace yet from current mini benchmark.')

    FAILURE_MD.write_text('\n'.join(out) + '\n')
    print(f'Canonicalized {len(canonical_rows)} rows into {FAILURE_MD}')


if __name__ == '__main__':
    main()
