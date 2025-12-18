"""
AHANKARA — Self and Execution Orchestrator for MARC

Ahankara (Sanskrit: "I-maker", "ego") is the orchestration module.

Philosophy:
- SELF is not data, not memory, not belief
- SELF is the DOER, the scheduler
- Pure orchestration, no logic

Core Loop:
text → Manas → BeliefProposal → Buddhi → Chitta

Components:
- Manas: understanding
- Buddhi: reasoning
- Chitta: memory
"""

try:
    from .ahankara import Ahankara
except ImportError:
    from ahankara import Ahankara

__all__ = ["Ahankara"]

__version__ = "0.1.0"
