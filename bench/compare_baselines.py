"""
Simple baseline comparison harness.

Runs three lightweight baselines on a small scenario set:
 - Episteme (current engine)
 - Naive graph traversal (simple BFS inheritance)
 - Direct lookup (exact belief match)

Currently a single-file runner that outputs JSON results.
"""
import time
import json
from typing import List, Tuple
import os

from chitta.graph import ChittaGraph
from buddhi.layer import Buddhi
from manas.layer import Manas
from tests.utils.save_proof import save_proof


def _scenario_category(query: str) -> str:
    q = query.lower()
    if 'nixon' in q:
        return 'Nixon diamond'
    if 'penguin' in q:
        return 'Penguin exception'
    if 'a fly' in q or 'cycle' in q:
        return 'Cyclic inheritance propagation'
    return query


def run_episteme(teachings: List[str], query: str):
    ch = ChittaGraph()
    ai = Buddhi(ch)
    manas = Manas()
    for t in teachings:
        ai.think(manas.parse(t))
    start = time.time()
    proof = ai.answer(manas.parse(query))
    end = time.time()
    summary = {
        'verdict': str(proof.verdict).upper(),
        'time': end - start,
        'steps': len(getattr(proof, 'steps', []))
    }
    # save full proof trace for later analysis
    try:
            save_proof(
                proof,
                query,
                summary['verdict'],
                details={
                    'category': _scenario_category(query),
                    'teachings': teachings,
                    'query': query,
                },
            )
    except Exception:
        pass
    return summary


def run_direct_lookup(teachings: List[str], query: str):
    # Direct exact-match baseline: if any taught belief matches query literal
    manas = Manas()
    beliefs = []
    for t in teachings:
        p = manas.parse(t)
        beliefs.append(p)
    q = manas.parse(query)
    start = time.time()
    for b in beliefs:
        # crude check: same predicate and subject/object
        if b.get('canonical') == q.get('canonical'):
            end = time.time()
            return {'verdict': 'YES', 'time': end - start}
    end = time.time()
    return {'verdict': 'UNKNOWN', 'time': end - start}


def run_naive_traversal(teachings: List[str], query: str, max_depth=5):
    # Build simple taxonomic edges and property map
    manas = Manas()
    isa = {}
    props = {}
    negs = set()
    for t in teachings:
        p = manas.parse(t)
        can = p.get('canonical', {})
        if p.get('template') == 'is_a' and can.get('subject') and can.get('object'):
            s = can['subject']; o = can['object']
            isa.setdefault(s, []).append(o)
        else:
            # property
            subj = can.get('subject') or (p.get('entities') or [None])[0]
            pred = (can.get('relation_type') or can.get('predicate_type') or p.get('predicates', [None])[0])
            if subj and pred:
                if p.get('polarity', 1) < 0:
                    negs.add((subj, pred))
                else:
                    props.setdefault(subj, []).append(pred)

    q = manas.parse(query)
    subj = q.get('canonical', {}).get('subject') or (q.get('entities') or [None])[0]
    pred = (q.get('canonical', {}).get('relation_type') or q.get('predicates', [None])[0])

    # BFS up to depth
    from collections import deque
    visited = set()
    dq = deque([(subj, 0)])
    start = time.time()
    found_pos = False
    found_neg = False
    while dq:
        node, d = dq.popleft()
        if node is None or d > max_depth:
            continue
        if (node, pred) in negs:
            found_neg = True
        if node in props and pred in props.get(node, []):
            found_pos = True
        visited.add(node)
        for parent in isa.get(node, []):
            if parent not in visited:
                dq.append((parent, d+1))
    end = time.time()
    if found_pos and not found_neg:
        return {'verdict': 'YES', 'time': end-start}
    if found_neg and not found_pos:
        return {'verdict': 'NO', 'time': end-start}
    if found_pos and found_neg:
        return {'verdict': 'CONFLICT', 'time': end-start}
    return {'verdict': 'UNKNOWN', 'time': end-start}


SCENARIOS = [
    ({'teachings': ["Birds can fly", "Penguins are birds", "Penguins cannot fly"], 'query': "Do penguins fly?"}),
    ({'teachings': ["Quakers are pacifists", "Republicans are not pacifists", "Nixon is a quaker", "Nixon is a republican"], 'query': "Is Nixon a pacifist?"}),
    ({'teachings': ["A is B", "B is C", "C is A", "C can fly"], 'query': "Do A fly?"}),
]


def main():
    results = []
    for s in SCENARIOS:
        teachings = s['teachings']
        query = s['query']
        e = run_episteme(teachings, query)
        d = run_direct_lookup(teachings, query)
        n = run_naive_traversal(teachings, query)
        results.append({'scenario': query, 'teachings': teachings, 'episteme': e, 'direct': d, 'naive': n})
    # Print and export results
    out_json = 'tests/logs/bench_compare_results.json'
    out_csv = 'tests/logs/bench_compare_results.csv'
    os.makedirs('tests/logs', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # CSV
    import csv
    keys = ['scenario', 'episteme_verdict', 'episteme_time', 'episteme_steps',
            'direct_verdict', 'direct_time', 'naive_verdict', 'naive_time']
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(keys)
        for r in results:
            writer.writerow([
                r['scenario'],
                r['episteme'].get('verdict'),
                r['episteme'].get('time'),
                r['episteme'].get('steps'),
                r['direct'].get('verdict'),
                r['direct'].get('time'),
                r['naive'].get('verdict'),
                r['naive'].get('time'),
            ])

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
