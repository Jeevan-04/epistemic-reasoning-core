"""Export docs/TRACE_EXAMPLES.md directly from proof traces.

This script scans `tests/logs/proof_*.json`, extracts canonical trace-example
rows, deduplicates them by trace reference, and writes a markdown table.

Usage:
    PYTHONPATH=. python scripts/export_trace_examples_from_traces.py
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    from scripts.export_failure_table_from_traces import (
        _argument_vector,
        _category_for_trace,
        _input_for_trace,
        _load_json,
        _notes,
        _root_cause,
        _source_reliability,
    )
except Exception:
    # Allow running this file directly (python scripts/export_trace_examples_from_traces.py)
    from export_failure_table_from_traces import (
        _argument_vector,
        _category_for_trace,
        _input_for_trace,
        _load_json,
        _notes,
        _root_cause,
        _source_reliability,
    )


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / 'tests' / 'logs'
TRACE_EXAMPLES_MD = ROOT / 'docs' / 'TRACE_EXAMPLES.md'


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'`', '', text)
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text


def _reproducibility_status(trace: dict[str, Any]) -> str:
    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    status = details.get('reproducibility_status')
    if status:
        return str(status)

    proof = trace.get('proof', {}) if isinstance(trace.get('proof'), dict) else {}
    if proof.get('timeout') or trace.get('timeout'):
        return 'environment-dependent'
    return 'deterministic'


def _failure_severity(trace: dict[str, Any]) -> str:
    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    severity = details.get('failure_severity')
    if severity:
        return str(severity)
    return 'N/A'


def _trace_query(trace: dict[str, Any], path: Path) -> str:
    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    if details.get('query'):
        return str(details['query'])
    return str(trace.get('scenario', path.stem))


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
        arguments = []
        if isinstance(proof.get('metadata'), dict):
            arguments = proof['metadata'].get('arguments', []) if isinstance(proof['metadata'].get('arguments'), list) else []
        input_text = _input_for_trace(trace, path)
        expected = str(trace.get('expected', 'N/A')).upper() if trace.get('expected') else 'N/A'
        observed = str(trace.get('verdict', proof.get('verdict', 'UNKNOWN'))).upper()
        verdict = 'PASS' if expected != 'N/A' and observed == expected else ('FAIL' if expected != 'N/A' else 'INFO')
        row = [
            '',
            _category_for_trace(trace, path),
            f'`{input_text}`',
            expected,
            observed,
            verdict,
            _root_cause(trace),
            _argument_vector(arguments),
            _source_reliability(arguments),
            _failure_severity(trace),
            _reproducibility_status(trace),
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
    out.append('# Trace Examples (Exported From Traces)')
    out.append('')
    out.append('This document is generated directly from proof traces under `tests/logs/`.')
    out.append('')
    out.append('| id | category | input | expected | observed | verdict | root_cause | argument_vector | source_reliability | failure_severity | reproducibility_status | trace_ref | notes |')
    out.append('|---:|---|---|---|---|---|---|---|---:|---|---|---|---|')
    for row in rows:
        out.append('| ' + ' | '.join(row) + ' |')
    out.append('')
    out.append('## Why These Rows Matter')
    out.append('')
    out.append('- These are positive validation artifacts, not failures.')
    out.append('- They are useful for explainability, replay, and regression comparisons.')
    out.append('')
    return '\n'.join(out)


def main():
    rows = load_trace_rows()
    TRACE_EXAMPLES_MD.write_text(render_markdown(rows))
    print(f'Exported {len(rows)} trace examples to {TRACE_EXAMPLES_MD}')


if __name__ == '__main__':
    main()