"""Debug exporter for mi-001 to inspect Chitta after teaching premises."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahankara.layer import Ahankara
from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph
from manas.layer import Manas

bench = json.loads(Path('tests/benchmarks/adversarial_benchmarks_natural.json').read_text())
case = next(c for c in bench if c['id']=='mi-001')

ch = ChittaGraph()
man = Manas(llm_backend='mock')
bud = Buddhi(ch)
ah = Ahankara(man, bud, ch)

for p in case['premises']:
    try:
        ah.process(p)
    except Exception as e:
        print('teach error', e)

print('--- BELIEFS ---')
for bid, b in ch.beliefs.items():
    print('ID:', bid)
    print(' template:', getattr(b,'template',None))
    print(' canonical:', getattr(b,'canonical',None))
    print(' entities:', getattr(b,'entities',None))
    print(' subject:', getattr(b,'subject',None))
    print(' object:', getattr(b,'object',None))
    print(' predicates:', getattr(b,'predicates',None))
    print(' statement_text:', getattr(b,'statement_text',None))
    print(' polarity:', getattr(b,'polarity',None))
    print(' epistemic_state:', getattr(b,'epistemic_state',None))
    print(' source_reliability:', getattr(b,'source_reliability',None))
    print(' confidence:', getattr(b,'confidence',None))
    print('---')

print('\n--- ENTITY INDEX ---')
print(ch.entity_index)
