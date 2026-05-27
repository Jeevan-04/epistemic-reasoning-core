"""Quick runner for focused multi-inheritance cases (mi-001..mi-005).

Exits with non-zero status if any case does not produce the expected verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahankara.layer import Ahankara
from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph
from manas.layer import Manas

BENCH = Path("tests/benchmarks/adversarial_benchmarks_natural.json")
cases = {c['id']: c for c in json.loads(BENCH.read_text())}
targets = ['mi-001', 'mi-002', 'mi-003', 'mi-004', 'mi-005']

ok = True
for cid in targets:
    case = cases[cid]
    ch = ChittaGraph()
    man = Manas(llm_backend='mock')
    bud = Buddhi(ch)
    ah = Ahankara(man, bud, ch)

    for p in case.get('premises', []):
        try:
            ah.process(p)
        except Exception:
            pass

    parsed = man.parse(case.get('query'))
    proof = bud.answer(parsed)
    got = str(proof.verdict).strip().lower()
    expect = str(case.get('expected')).strip().lower()
    print(f"{cid}: expected={expect} replay={got}")
    if expect != got:
        print(f"  MISMATCH: {cid} expected {expect} but got {got}")
        ok = False

sys.exit(0 if ok else 2)
