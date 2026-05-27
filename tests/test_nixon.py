import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chitta.graph import ChittaGraph
from buddhi.layer import Buddhi
from manas.layer import Manas


class TestNixonDiamond(unittest.TestCase):
    def setUp(self):
        self.chitta = ChittaGraph()
        self.buddhi = Buddhi(self.chitta)
        self.manas = Manas()

    def test_nixon_diamond_conflict(self):
        # Quakers are pacifists (default / positive)
        self.buddhi.think(self.manas.parse("Quakers are pacifists"))

        # Republicans are not pacifists (default / negative)
        self.buddhi.think(self.manas.parse("Republicans are not pacifists"))

        # Nixon is both a Quaker and a Republican
        self.buddhi.think(self.manas.parse("Nixon is a quaker"))
        self.buddhi.think(self.manas.parse("Nixon is a republican"))

        # Query: Is Nixon a pacifist? Expect a horizontal conflict
        query = self.manas.parse("Is Nixon a pacifist?")
        proof = self.buddhi.answer(query)
        self.assertEqual(str(proof.verdict).upper(), "CONFLICT")


if __name__ == '__main__':
    unittest.main()
