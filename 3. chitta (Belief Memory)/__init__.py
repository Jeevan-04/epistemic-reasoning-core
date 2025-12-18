"""
CHITTA — Belief Memory System

Chitta is the hypergraph memory store for MARC.
It holds beliefs, relations, provenance, and epistemic state.

Core components:
- Belief: atomic unit of knowledge
- ChittaGraph: hypergraph storage with indexes
- Edge system: relations between beliefs

Philosophy:
- Only ONE node type: belief
- Everything else is metadata or relations
- No rules as nodes, no inference nodes, no question nodes
- Clean epistemology
"""

from .belief import Belief
from .graph import ChittaGraph
from .utils import (
    generate_belief_id,
    generate_edge_id,
    logit,
    sigmoid,
    now_utc,
    validate_confidence,
    validate_epistemic_state,
    validate_template,
)

__all__ = [
    "Belief",
    "ChittaGraph",
    "generate_belief_id",
    "generate_edge_id",
    "logit",
    "sigmoid",
    "now_utc",
    "validate_confidence",
    "validate_epistemic_state",
    "validate_template",
]

__version__ = "0.1.0"
