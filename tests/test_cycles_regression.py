import os

from manas.layer import Manas
from chitta.graph import ChittaGraph
from buddhi.layer import Buddhi


def test_cyclic_inheritance_propagation():
    """Regression: ensure capability propagates through a cycle A->B->C->A."""
    manas = Manas()
    ch = ChittaGraph()
    ai = Buddhi(ch)

    teachings = ["A is B", "B is C", "C is A", "C can fly"]
    query = "Do A fly?"

    # Ingest teachings
    for t in teachings:
        proposal = manas.parse(t)
        ai.think(proposal)

    # Ask query
    qproposal = manas.parse(query)
    proof = ai.answer(qproposal)

    # Expect YES (ability should propagate through cycle)
    assert str(proof.verdict).upper() == 'YES', (
        f"Cyclic propagation failed: got {proof.verdict}; proof steps: {getattr(proof, 'steps', None)}"
    )
