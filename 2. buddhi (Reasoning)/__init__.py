"""
BUDDHI — Reasoning and Judgment Engine for MARC

Buddhi (Sanskrit: "intellect", "discernment") is the thinking module.

Philosophy:
- Thinking = judgment over competing beliefs
- No LLM (pure algorithmic reasoning)
- Explicit, inspectable decisions
- Can say "I don't know"
- ONLY writer to Chitta

Core Algorithm:
1. Focus → Restrict to subgraph
2. Generate → Create candidates
3. Evaluate → Judgment function
4. Select → Choose winner
5. Revise → Update Chitta

Judgment Function:
J(b) = 0.40·c + 0.20·s + 0.15·tanh(S) - 0.20·tanh(C) + 0.05·tanh(a)

Where:
- c = confidence
- s = specificity (1/(1+depth))
- S = support strength
- C = conflict strength
- a = activation
"""

try:
    from .buddhi import Buddhi, W_CONF, W_SPEC, W_SUP, W_CON, W_ACT, WIN_MARGIN, MIN_SCORE
except ImportError:
    from buddhi import Buddhi, W_CONF, W_SPEC, W_SUP, W_CON, W_ACT, WIN_MARGIN, MIN_SCORE

__all__ = [
    "Buddhi",
    "W_CONF",
    "W_SPEC",
    "W_SUP",
    "W_CON",
    "W_ACT",
    "WIN_MARGIN",
    "MIN_SCORE",
]

__version__ = "0.1.0"
