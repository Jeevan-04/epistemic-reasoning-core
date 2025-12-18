"""
HRE — Hypothetical Reasoning Engine

Epistemic sandbox for counterfactual reasoning.
"""

from .hre import HRE
from .hypothetical_proof import (
    HypotheticalAnswerProof,
    HypotheticalAssumption,
    HypotheticalDerivationStep,
    HypotheticalConflict,
)

__all__ = [
    "HRE",
    "HypotheticalAnswerProof",
    "HypotheticalAssumption",
    "HypotheticalDerivationStep",
    "HypotheticalConflict",
]
