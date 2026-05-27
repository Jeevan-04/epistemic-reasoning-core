"""
Mini benchmark runner for key scenarios: Penguin specificity, Nixon diamond,
taxonomic cycles, and basic abstention. Prints simple pass/fail summary.

Run: `python -m pytest tests/benchmark_mini.py -q` or `python tests/benchmark_mini.py`
"""
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chitta.graph import ChittaGraph
from buddhi.layer import Buddhi
from manas.layer import Manas


class MiniBenchmark(unittest.TestCase):
    def setUp(self):
        self.chitta = ChittaGraph()
        self.buddhi = Buddhi(self.chitta)
        self.manas = Manas()

    def runScenario(self, teachings, query_text, expected_verdicts):
        for t in teachings:
            self.buddhi.think(self.manas.parse(t))
        proof = self.buddhi.answer(self.manas.parse(query_text))
        return str(proof.verdict).upper() in [v.upper() for v in expected_verdicts]

    def test_penguin_specificity(self):
        teachings = ["Birds can fly", "Penguins are birds", "Penguins cannot fly"]
        ok = self.runScenario(teachings, "Do penguins fly?", ["NO"])
        self.assertTrue(ok)

    def test_nixon_diamond(self):
        teachings = ["Quakers are pacifists", "Republicans are not pacifists", "Nixon is a quaker", "Nixon is a republican"]
        ok = self.runScenario(teachings, "Is Nixon a pacifist?", ["CONFLICT"])
        self.assertTrue(ok)

    def test_cycle(self):
        teachings = ["A is B", "B is C", "C is A", "C can fly"]
        ok = self.runScenario(teachings, "Do A fly?", ["YES", "UNKNOWN", "UNCERTAIN"])
        self.assertTrue(ok)


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
