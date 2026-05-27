#!/usr/bin/env python3
"""
Generate a failure table markdown file from the adversarial benchmark results.

Reads `tests/logs/adversarial_results.jsonl` and writes `docs/ADVERSARIAL_FAILURES.md`.
"""
import json
import os
from pathlib import Path

IN = Path('tests/logs/adversarial_results.jsonl')
OUT = Path('docs/ADVERSARIAL_FAILURES.md')


def load_results(path):
    rows = []
    with open(path, 'r', encoding='utf8') as f:
        for l in f:
            if not l.strip():
                continue
            rows.append(json.loads(l))
    return rows


def render_table(rows):
    hdr = (
        "# Adversarial Failures\n\n"
        "This table lists benchmark cases where the observed verdict diverged from the expected semantics.\n\n"
        "| id | title | category | expected | observed | trace | notes |\n"
        "|---|---|---|---|---|---|---|\n"
    )

    lines = [hdr]
    for r in rows:
        if 'error' in r:
            obs = f"ERROR: {r['error']}"
        else:
            obs = r.get('observed', '')
        trace = r.get('trace_ref') or ''
        title = r.get('title', '')
        cat = r.get('category', '')
        exp = r.get('expected', '')
        notes = ''
        lines.append(f"| {r.get('id')} | {title} | {cat} | {exp} | {obs} | {trace} | {notes} |\n")

    return ''.join(lines)


def main():
    if not IN.exists():
        print('No adversarial results file:', IN)
        return
    rows = load_results(IN)
    # Select only mismatches (observed != expected or errors)
    mismatches = []
    for r in rows:
        if 'error' in r:
            mismatches.append(r)
            continue
        exp = str(r.get('expected','')).strip().lower()
        obs = str(r.get('observed','')).strip().lower()
        if exp != obs:
            mismatches.append(r)

    out = render_table(mismatches)
    os.makedirs(OUT.parent, exist_ok=True)
    with open(OUT, 'w', encoding='utf8') as f:
        f.write(out)
    print(f'Wrote {len(mismatches)} mismatch rows to {OUT}')


if __name__ == '__main__':
    main()
