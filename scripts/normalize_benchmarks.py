#!/usr/bin/env python3
"""
Normalize compact benchmark shorthand into natural-language premises.

Reads `tests/benchmarks/adversarial_benchmarks.json` and writes
`tests/benchmarks/adversarial_benchmarks_natural.json`.
"""
import json
import re
from pathlib import Path

IN = Path('tests/benchmarks/adversarial_benchmarks.json')
OUT = Path('tests/benchmarks/adversarial_benchmarks_natural.json')


def normalize_premise(p: str) -> str:
    s = p.strip()

    # Source with meta: SourceName (r=0.95): Statement
    m = re.match(r"^(?P<src>[^\(:]+)\s*\((?P<meta>[^\)]+)\)\s*:\s*(?P<stmt>.+)$", s)
    if m:
        src = m.group('src').strip()
        meta = m.group('meta')
        stmt = m.group('stmt').strip().rstrip('.')
        # try to extract reliability
        reli = None
        for part in meta.split(','):
            kv = part.strip()
            if kv.startswith('r=') or kv.startswith('reliability='):
                try:
                    reli = float(kv.split('=')[1])
                except Exception:
                    reli = None
        if reli is not None:
            return f"According to {src} (reliability={reli}), {stmt}."
        return f"According to {src}, {stmt}."

    # Arrow: A->B
    m = re.match(r"^(?P<a>\w+)\s*->\s*(?P<b>\w+)(?:\s*\((?P<rest>.*)\))?\.?$", s)
    if m:
        a = m.group('a')
        b = m.group('b')
        return f"{a} is a {b}."

    # Typically: "X typically P" or "X typically not P"
    m = re.match(r"^(?P<subject>[^:]+?)\s+typically\s+(?P<neg>not\s+)?(?P<p>.+)$", s)
    if m:
        subj = m.group('subject').strip()
        neg = m.group('neg')
        p = m.group('p').strip().rstrip('.')
        if neg:
            return f"Typically, {subj} do not {p}." if ' ' not in subj else f"Typically, {subj} do not {p}."
        return f"Typically, {subj} {p}."

    # Already a sentence or fallback
    if s.endswith('.') or s.endswith('?') or s.endswith('!'):
        return s

    # General fallback: append a period
    return s + '.'


def main():
    data = json.loads(IN.read_text(encoding='utf8'))
    out = []
    for case in data:
        new = dict(case)
        new_p = [normalize_premise(p) for p in case.get('premises', [])]
        new['premises'] = new_p
        out.append(new)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding='utf8')
    print(f'Wrote normalized benchmarks to {OUT}')


if __name__ == '__main__':
    main()
