"""
MANAS — Understanding and Parsing Engine for MARC

Manas (Sanskrit: "mind", "sensory processor") is the understanding module.

Philosophy:
- Understanding ≠ Thinking
- Stateless (no memory)
- NO access to Chitta
- Outputs untrusted BeliefProposal

Core Function:
text → BeliefProposal

BeliefProposal Schema:
{
    "template": str,
    "canonical": dict,
    "entities": list[str],
    "predicates": list[str],
    "polarity": +1 | -1,
    "parser_confidence": float,
    "raw_text": str
}
"""

try:
    from .manas import Manas, BELIEF_PROPOSAL_SCHEMA
except ImportError:
    from manas import Manas, BELIEF_PROPOSAL_SCHEMA

__all__ = ["Manas", "BELIEF_PROPOSAL_SCHEMA"]

__version__ = "0.1.0"
