"""Export docs/FAILURE_TABLES.md directly from failing proof traces.

This script scans `tests/logs/proof_*.json`, extracts canonical failure rows,
deduplicates them by trace reference, and writes a markdown table section.

Usage:
    PYTHONPATH=. python scripts/export_failure_table_from_traces.py
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / 'tests' / 'logs'
FAILURE_MD = ROOT / 'docs' / 'FAILURE_TABLES.md'


def _failure_severity(trace: dict[str, Any]) -> str:
    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    severity = details.get('failure_severity')
    if severity:
        return str(severity)
    proof = trace.get('proof', {}) if isinstance(trace.get('proof'), dict) else {}
    text = ' '.join(
        str(item) for item in [trace.get('scenario', ''), details.get('category', ''), proof.get('verdict', '')]
    ).lower()
    if any(keyword in text for keyword in ['timeout', 'explosion']):
        return 'CRITICAL'
    if any(keyword in text for keyword in ['nixon', 'conflict']):
        return 'MAJOR'
    if any(keyword in text for keyword in ['parser', 'normalization']):
        return 'MODERATE'
    return 'MINOR'


def _reproducibility_status(trace: dict[str, Any]) -> str:
    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    status = details.get('reproducibility_status')
    if status:
        return str(status)
    proof = trace.get('proof', {}) if isinstance(trace.get('proof'), dict) else {}
    if proof.get('timeout') or trace.get('timeout'):
        return 'environment-dependent'
    return 'deterministic'


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'`', '', text)
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text


def _category_for_trace(trace: dict[str, Any], path: Path) -> str:
    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    if details.get('category'):
        return str(details['category'])

    scenario = str(trace.get('scenario', path.stem))
    scenario_l = scenario.lower()
    if 'nixon' in scenario_l:
        return 'Nixon diamond'
    if 'penguin' in scenario_l:
        return 'Penguin exception'
    if 'cycle' in scenario_l or 'a fly' in scenario_l:
        return 'Cyclic inheritance propagation'
    return scenario


def _input_for_trace(trace: dict[str, Any], path: Path) -> str:
    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    teachings = details.get('teachings') if isinstance(details.get('teachings'), list) else None
    query = details.get('query') or trace.get('scenario', path.stem)
    if teachings:
        teaching_text = '; '.join(str(t) for t in teachings)
        return f"{teaching_text}; Query: {query}"
    return str(query)


def _source_reliability(arguments: list[dict[str, Any]]) -> str:
    values = [arg.get('source_reliability') for arg in arguments if isinstance(arg.get('source_reliability'), (int, float))]
    if not values:
        return '-'
    return str(round(sum(values) / len(values), 3))


def _argument_vector(arguments: list[dict[str, Any]]) -> str:
    if not arguments:
        return '-'
    parts = []
    for arg in arguments:
        if not isinstance(arg, dict):
            continue
        if 'claim' not in arg and 'winner' not in arg:
            continue
        if arg.get('winner') == 'conflict':
            parts.append(
                f"conflict[pos={arg.get('positive_specificity', '?')},neg={arg.get('negative_specificity', '?')}]"
            )
            continue
        claim = arg.get('claim', 'unknown')
        neg = '!' if arg.get('is_negative') else '+'
        parts.append(
            f"{neg}{claim}[s={arg.get('specificity', '?')},r={arg.get('rank', '?')},"
            f"rel={arg.get('source_reliability', '?')},a={arg.get('activation', '?')}]"
        )
    return ' ; '.join(parts) if parts else '-'


def _root_cause(trace: dict[str, Any]) -> str:
    proof = trace.get('proof', {}) if isinstance(trace.get('proof'), dict) else {}
    steps = proof.get('steps', []) if isinstance(proof.get('steps'), list) else []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        rule = str(step.get('rule', '')).strip()
        if rule:
            return rule
    return 'trace_only'


def _notes(trace: dict[str, Any]) -> str:
    proof = trace.get('proof', {}) if isinstance(trace.get('proof'), dict) else {}
    if isinstance(proof.get('metadata'), dict) and proof['metadata'].get('arguments'):
        return 'trace-backed'
    return '-'


def _trace_query(trace: dict[str, Any], path: Path) -> str:
    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    if details.get('query'):
        return str(details['query'])
    scenario = str(trace.get('scenario', path.stem))
    return scenario


def _dedupe_key(trace: dict[str, Any], path: Path) -> str:
    return _slugify(_trace_query(trace, path))


def load_trace_rows() -> list[list[str]]:
    rows = OrderedDict()
    for path in sorted(LOG_DIR.glob('proof_*.json')):
        if path.parent.name == 'pytest':
            continue
        trace = _load_json(path)
        if not trace:
            continue
        proof = trace.get('proof', {}) if isinstance(trace.get('proof'), dict) else {}
        expected_raw = trace.get('expected')
        expected = str(expected_raw).upper() if expected_raw else 'N/A'
        arguments = []
        if isinstance(proof.get('metadata'), dict):
            arguments = proof['metadata'].get('arguments', []) if isinstance(proof['metadata'].get('arguments'), list) else []
        input_text = _input_for_trace(trace, path)
        observed = str(trace.get('verdict', proof.get('verdict', 'UNKNOWN'))).upper()
        is_failure = expected != 'N/A' and observed != expected
        if not is_failure and str(proof.get('verdict', '')).upper() == 'FAIL':
            is_failure = True
        if not is_failure:
            continue
        verdict = 'FAIL'
        row = [
            '',
            _category_for_trace(trace, path),
            f'`{input_text}`',
            expected,
            observed,
            verdict,
            _failure_severity(trace),
            _reproducibility_status(trace),
            _root_cause(trace),
            _argument_vector(arguments),
            _source_reliability(arguments),
            str(path.relative_to(ROOT)),
            _notes(trace),
        ]
        key = _dedupe_key(trace, path)
        current = rows.get(key)
        if current is None:
            rows[key] = row + [str(path.stat().st_mtime)]
        else:
            current_mtime = float(current[-1])
            if path.stat().st_mtime >= current_mtime:
                rows[key] = row + [str(path.stat().st_mtime)]

    canonical_rows = []
    for idx, row in enumerate(rows.values(), start=1):
        row[0] = str(idx)
        canonical_rows.append(row[:-1])
    return canonical_rows


def render_markdown(rows: list[list[str]]) -> str:
    out = []
    out.append('# Failure Tables (Exported From Traces)')
    out.append('')
    out.append('This document is generated directly from failing proof traces under `tests/logs/`.')
    out.append('')
    out.append('| id | category | input | expected | observed | verdict | failure_severity | reproducibility_status | root_cause | argument_vector | source_reliability | trace_ref | notes |')
    out.append('|---:|---|---|---|---|---|---|---|---|---|---:|---|---|')
    if rows:
        for row in rows:
            out.append('| ' + ' | '.join(row) + ' |')
    else:
        out.append('No confirmed failures captured yet.')
    out.append('')
    out.append('## Failure Criteria')
    out.append('')
    out.append('- Incorrect verdicts.')
    out.append('- Instability across repeated runs.')
    out.append('- Parser normalization failures.')
    out.append('- Contradictory resolution bugs.')
    out.append('- Argument explosion or timeouts.')
    out.append('')
    return '\n'.join(out)


def main():
    rows = load_trace_rows()
    FAILURE_MD.write_text(render_markdown(rows))
    print(f'Exported {len(rows)} trace-backed rows to {FAILURE_MD}')


if __name__ == '__main__':
    main()
