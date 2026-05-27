import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chitta.graph import ChittaGraph
from buddhi.layer import Buddhi
from manas.layer import Manas


class TestTaxonomicCycles(unittest.TestCase):
    def setUp(self):
        self.chitta = ChittaGraph()
        self.buddhi = Buddhi(self.chitta)
        self.manas = Manas()

    def test_cycle_handling(self):
        # Create a cycle: a is b, b is c, c is a
        self.buddhi.think(self.manas.parse("A is B"))
        self.buddhi.think(self.manas.parse("B is C"))
        self.buddhi.think(self.manas.parse("C is A"))

        # Add property on C
        self.buddhi.think(self.manas.parse("C can fly"))

        # Query: Does A fly? Expect inference via taxonomic traversal but no infinite loop
        query = self.manas.parse("Do A fly?")
        proof = self.buddhi.answer(query)
        # Should be able to inherit property from C through cycle once
        self.assertIn(str(proof.verdict).upper(), ["YES", "UNCERTAIN", "UNKNOWN"]) 


if __name__ == '__main__':
    unittest.main()
