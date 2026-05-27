#!/usr/bin/env python3
"""
Normalize shorthand benchmark premises into natural-language sentences.
Writes `tests/benchmarks/adversarial_benchmarks_nl.json`.
"""
import json
import re
from pathlib import Path

IN = Path('tests/benchmarks/adversarial_benchmarks.json')
OUT = Path('tests/benchmarks/adversarial_benchmarks_nl.json')


def convert_premise(p):
    t = p.strip()
    # A -> B
    m = re.match(r'^(?P<a>\w+)\s*->\s*(?P<b>\w+)(?:\s*\((?P<rest>.*)\))?\.?$', t)
    if m:
        a = m.group('a')
        b = m.group('b')
        return f"{a} are {b}."

    # Source (r=0.95): Statement
    m = re.match(r"^(?P<src>[^\(:]+)\s*\((?P<meta>[^\)]+)\)\s*:\s*(?P<stmt>.+)$", t)
    if m:
        src = m.group('src').strip()
        stmt = m.group('stmt').strip()
        return f"According to {src}, {stmt}"

    # "X typically P" or "X typically not P"
    m = re.match(r"^(?P<subject>[^:]+?)\s+typically\s+(?P<neg>not\s+)?(?P<p>.+)$", t)
    if m:
        subj = m.group('subject').strip()
        neg = bool(m.group('neg'))
        ptext = m.group('p').strip().rstrip('.')
        if neg:
            return f"Typically, {subj} do not {ptext}."
        else:
            # attempt make grammatical
            return f"Typically, {subj} {ptext}."

    # Numeric weak reports etc.
    m = re.match(r"^(?P<num>\d+) weak reports each .* say (?P<what>.+)$", t)
    if m:
        what = m.group('what').strip()
        return f"Several weak reports say {what}."

    # Conjunctions, phrase fragments — fallback: ensure ends with period
    if not t.endswith('.'):
        return t + '.'
    return t


def main():
    data = json.loads(IN.read_text(encoding='utf8'))
    for case in data:
        new_prem = []
        for p in case.get('premises', []):
            new_prem.append(convert_premise(p))
        case['premises'] = new_prem
        # Also normalize query if it's shorthand
        if 'query' in case:
            q = case['query']
            if not q.endswith('?') and ' ' in q:
                # heuristics: if query starts with Do/Is/How, keep; else append question
                if not re.match(r'^(Do|Does|Is|Are|How|Why)\b', q, re.IGNORECASE):
                    case['query'] = q.rstrip('.') + '?'
    OUT.write_text(json.dumps(data, indent=2), encoding='utf8')
    print(f'Wrote normalized benchmarks to {OUT}')


if __name__ == '__main__':
    main()
