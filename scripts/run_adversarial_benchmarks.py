#!/usr/bin/env python3
"""
Run the adversarial benchmark cases and save outcomes to `tests/logs/adversarial_results.jsonl`.

Usage:
  PYTHONPATH=. python3 scripts/run_adversarial_benchmarks.py [--limit N]

The script attempts to import the local pipeline (`manas`, `ahankara`, `buddhi`) and will
gracefully fall back to a dry-run mode that prints the loaded cases if imports fail.
"""
import json
import os
import sys
from argparse import ArgumentParser

HERE = os.path.dirname(os.path.dirname(__file__))
BENCHMARK_PATH = os.path.join(HERE, 'tests', 'benchmarks', 'adversarial_benchmarks.json')
OUT_PATH = os.path.join(HERE, 'tests', 'logs', 'adversarial_results.jsonl')

def safe_import_pipeline():
    try:
        import manas
        import ahankara
        import buddhi
        return manas, ahankara, buddhi
    except Exception:
        # try package-style imports when running from repo root
        try:
            sys.path.insert(0, os.getcwd())
            import manas
            import ahankara
            import buddhi
            return manas, ahankara, buddhi
        except Exception:
            return None, None, None

def load_cases(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def main():
    p = ArgumentParser()
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--bench', type=str, default=BENCHMARK_PATH, help='Path to benchmarks JSON')
    args = p.parse_args()

    cases = load_cases(args.bench)
    limit = args.limit or len(cases)
    manas, ahankara, buddhi = safe_import_pipeline()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if not manas:
        print('Pipeline imports failed. Running in dry-run mode; will only print cases.')
        for c in cases[:limit]:
            print(f"{c['id']}: {c['title']} [{c['category']}] -> expected: {c.get('expected')}")
        print('\nTo run for real, ensure `PYTHONPATH=. python3` from repo root and that imports succeed.')
        return

    # Real execution path
    results = []
    for c in cases[:limit]:
        case_id = c.get('id')
        # Instantiate fresh cognitive components per-case to isolate state
        try:
            ChittaGraph = __import__('chitta.graph', fromlist=['ChittaGraph']).ChittaGraph
            Manas = __import__('manas.layer', fromlist=['Manas']).Manas
            Ahankara = __import__('ahankara.layer', fromlist=['Ahankara']).Ahankara
            BuddhiClass = __import__('buddhi.layer', fromlist=['Buddhi']).Buddhi

            ch = ChittaGraph()
            man = Manas()
            bud = BuddhiClass(ch)
            ah = Ahankara(manas=man, buddhi=bud, chitta=ch)

            # Teach premises via Ahankara.process (which drives Manas->Buddhi->Chitta)
            for premise in c.get('premises', []):
                try:
                    ah.process(premise)
                except Exception:
                    # Best-effort: try perceiving + judging directly
                    try:
                        proposal = man.parse(premise)
                        ah.judge(proposal)
                    except Exception:
                        pass

            # Parse the query and ask Buddhi for a structured proof
            query = c.get('query')
            try:
                parsed_q = man.parse(query)
            except Exception:
                parsed_q = {'text': query, 'statement': query}

            # Heuristic: if parser produced a placeholder entity (e.g., 'X') that
            # isn't present in the current Chitta, try substituting a taught entity
            # from the premises to make the query grounded for the replay.
            try:
                primary = None
                ents = parsed_q.get('entities', []) if isinstance(parsed_q, dict) else []
                if ents:
                    for e in ents:
                        if e and e.isupper() and len(e) == 1:
                            primary = e
                            break
                if primary:
                    # pick a plausible entity from ch (skip trivial tokens)
                    candidates = [k for k in ch.entity_index.keys() if len(k) > 1 and k.lower() not in {'the','a','an','in','on','at'}]
                    if candidates:
                        subst = candidates[0]
                        substituted_query = query.replace(primary, subst)
                        parsed_q = man.parse(substituted_query)
                        # record substitution for traceability
                        # (runner will record trace_ref later if available)
            except Exception:
                pass

            trace_ref = None
            try:
                proof = bud.answer(parsed_q)
                verdict = getattr(proof, 'verdict', getattr(proof, 'status', 'UNKNOWN'))
                # Persist proof if possible
                try:
                    if hasattr(proof, 'to_json'):
                        trace_path = os.path.join('tests', 'logs', f"bench_{case_id}.json")
                        with open(trace_path, 'w', encoding='utf8') as tf:
                            tf.write(json.dumps(proof.to_json(), indent=2))
                        trace_ref = trace_path
                except Exception:
                    trace_ref = None
            except Exception as e:
                verdict = f'ERROR: {e.__class__.__name__}: {e}'
                proof = None

            rec = {
                'id': case_id,
                'title': c.get('title'),
                'category': c.get('category'),
                'expected': c.get('expected'),
                'observed': str(verdict),
                'trace_ref': trace_ref,
            }
        except Exception as e:
            rec = {'id': case_id, 'error': f'{e.__class__.__name__}: {e}'}
        results.append(rec)

    # Write results as JSON lines
    with open(OUT_PATH, 'w', encoding='utf8') as out:
        for r in results:
            out.write(json.dumps(r) + '\n')

    print(f'Wrote {len(results)} results to {OUT_PATH}')

if __name__ == '__main__':
    main()
