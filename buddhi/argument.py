from dataclasses import dataclass, field
from typing import List, Any, Optional
from datetime import datetime
from buddhi.belief import EpistemicType

@dataclass
class Argument:
    """
    Formal epistemic argument object representing a constructed path of reasoning.
    Used in defeasible argumentation frameworks to compute attacks and defeats.
    """
    claim: str                  # The conclusion (e.g., a belief ID, or a string summary)
    supports: List[str]         # Belief IDs used in construction (the premises)
    path: List[str]             # Taxonomic path (is_a links) connecting entity to source
    rank: int                   # Epistemic rank (AXIOM > EXCEPTION > OBSERVATION > DEFAULT)
    specificity: int            # Distance metric from the query entity (lower is more specific)
    activation: float           # Weakest link confidence / strength of the argument chain
    provenance: List[Any]       # Origin traces / history
    source_reliability: float = 1.0
    recency: Optional[datetime] = None
    is_negative: bool = False   # True if this argument argues AGAINST the queried predicate

    @staticmethod
    def get_rank(etype: EpistemicType) -> int:
        """Map EpistemicType to integer rank for defeat resolution."""
        if etype == EpistemicType.AXIOM: return 3
        if etype == EpistemicType.EXCEPTION: return 2
        if etype == EpistemicType.OBSERVATION: return 2
        if etype == EpistemicType.DEFAULT: return 1
        return 0

    def defeats(self, other: "Argument") -> bool:
        """
        Determines if THIS argument defeats the OTHER argument.
        Defeat ordering (Lexicographic):
        1. Specificity (Vertical): smaller `specificity` value is stronger.
        2. Source Reliability: larger `source_reliability` is stronger.
        3. Activation: larger `activation` is stronger.
        4. Epistemic Rank: larger `rank` value is stronger.
        5. Recency: more recent `recency` (later datetime) is stronger.

        Returns True iff `self` strictly wins the lexicographic comparison.
        """
        # 1. Specificity (smaller is stronger)
        if self.specificity != other.specificity:
            return self.specificity < other.specificity

        # 2. Source Reliability (larger is stronger)
        if self.source_reliability != other.source_reliability:
            return self.source_reliability > other.source_reliability

        # 3. Activation (larger is stronger)
        if self.activation != other.activation:
            return self.activation > other.activation

        # 4. Epistemic Rank (larger is stronger)
        if self.rank != other.rank:
            return self.rank > other.rank

        # 5. Recency (more recent is stronger)
        if self.recency is not None or other.recency is not None:
            # Treat missing recency as very old
            t_self = self.recency.timestamp() if self.recency else 0
            t_other = other.recency.timestamp() if other.recency else 0
            if t_self != t_other:
                return t_self > t_other

        # Fully tied -> neither strictly defeats the other
        return False
