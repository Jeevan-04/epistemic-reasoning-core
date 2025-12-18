"""
BELIEF — Atomic Unit of Knowledge

A Belief is a proposition that the system holds with some degree of confidence.

Key principles:
- Single node type (no kinds, no variants)
- Immutable canonical structure
- Mutable epistemic state (confidence, activation)
- Full provenance tracking
- Edge management handled by graph
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Import core types
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from CORE_TYPES import Quantifier, Polarity, extract_quantifier_and_polarity, validate_quantifier, validate_polarity
except ImportError:
    # Fallback if not available
    Quantifier = None
    Polarity = None
    extract_quantifier_and_polarity = None
    validate_quantifier = lambda x: "UNSPECIFIED"
    validate_polarity = lambda x: "POSITIVE"

try:
    from .utils import (
        extract_entities,
        extract_predicates,
        generate_belief_id,
        now_utc,
        to_iso,
        validate_canonical,
        validate_confidence,
        validate_epistemic_state,
        validate_template,
    )
except ImportError:
    from utils import (
        extract_entities,
        extract_predicates,
        generate_belief_id,
        now_utc,
        to_iso,
        validate_canonical,
        validate_confidence,
        validate_epistemic_state,
        validate_template,
    )


@dataclass
class ProvenanceEntry:
    """
    A single record in a belief's provenance chain.
    
    Provenance tracks where beliefs come from and how they evolved.
    Think of it like a commit history for knowledge.
    """
    
    op: str  # what happened: parsed | inferred | revised | merged | dreamed
    from_source: str | list[str]  # where it came from (component name or belief IDs)
    score: float  # how much this source contributed to confidence
    timestamp: datetime = field(default_factory=now_utc)  # when this happened
    metadata: dict[str, Any] = field(default_factory=dict)  # any extra details
    
    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "from": self.from_source,
            "score": self.score,
            "at": to_iso(self.timestamp),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ProvenanceEntry:
        from .utils import from_iso
        return cls(
            op=data["op"],
            from_source=data["from"],
            score=data["score"],
            timestamp=from_iso(data["at"]) if "at" in data else now_utc(),
            metadata=data.get("metadata", {}),
        )


class Belief:
    """
    The fundamental unit of knowledge in MARC.
    
    A belief is a proposition (statement) that MARC holds with some level of confidence.
    For example: "Penguins cannot fly" with confidence 0.95.
    
    Every belief has:
    - Content: what the belief says (template + canonical representation)
    - Epistemic state: asserted (believed), unknown (uncertain), or hypothetical (speculative)
    - Confidence: how sure we are [0.0 to 1.0]
    - Provenance: where it came from (was it taught? inferred? merged?)
    
    Beliefs are designed to be:
    - Simple: one node type for everything
    - Traceable: full provenance history
    - Mutable in confidence: beliefs can strengthen or weaken
    - Immutable in meaning: the canonical form doesn't change
    """
    
    def __init__(
        self,
        *,
        template: str,
        canonical: dict,
        confidence: float,
        epistemic_state: str = "asserted",
        original_text: str | None = None,
        statement_text: str | None = None,
        source: dict | None = None,
        moral_value: float | None = None,
        metadata: dict | None = None,
        belief_id: str | None = None,
        quantifier: str | None = None,
        polarity: str | None = None,
        epistemic_class: str | None = None,  # NEW: STRUCTURAL/BEHAVIORAL/ABSTRACT
    ):
        """
        Create a new belief.
        
        Args:
            template: belief type (relation, event, is_a, etc.)
            canonical: machine-readable semantic frame
            confidence: initial confidence [0.0, 1.0]
            epistemic_state: asserted | unknown | hypothetical
            original_text: original user input
            statement_text: normalized statement
            source: source metadata dict
            moral_value: moral score [0.0, 1.0] or None
            metadata: additional metadata
            belief_id: optional ID (generated if None)
            quantifier: ALL/SOME/MOST/FEW/NONE/UNSPECIFIED
            polarity: POSITIVE/NEGATIVE
            epistemic_class: STRUCTURAL/BEHAVIORAL/ABSTRACT (controls propagation)
        """
        # Identity
        self.id = belief_id or generate_belief_id()
        
        # Semantic content (IMMUTABLE after creation)
        self.template = validate_template(template)
        self._canonical = validate_canonical(copy.deepcopy(canonical))
        
        # Epistemic class (controls inference rules)
        self.epistemic_class = epistemic_class  # Store enum or string
        
        # QUANTIFIER LOGIC (v1.0 critical fix)
        # Auto-detect if not provided
        if quantifier is None and statement_text:
            if extract_quantifier_and_polarity:
                auto_quantifier, auto_polarity = extract_quantifier_and_polarity(statement_text)
                self.quantifier = validate_quantifier(auto_quantifier) if validate_quantifier else "UNSPECIFIED"
                self.polarity = validate_polarity(auto_polarity) if validate_polarity else "POSITIVE"
            else:
                self.quantifier = "UNSPECIFIED"
                self.polarity = "POSITIVE"
        else:
            self.quantifier = validate_quantifier(quantifier) if validate_quantifier else (quantifier or "UNSPECIFIED")
            self.polarity = validate_polarity(polarity) if validate_polarity else (polarity or "POSITIVE")
        
        # Epistemic state (MUTABLE)
        self._epistemic_state = validate_epistemic_state(epistemic_state)
        self._confidence = validate_confidence(confidence)
        
        # Text representations
        self.original_text = original_text
        self.statement_text = statement_text or original_text
        
        # Moral value (computed by Buddhi, stored here)
        self.moral_value = moral_value
        
        # Source tracking
        self.source = source or {}
        
        # Provenance chain
        self.provenance: list[ProvenanceEntry] = []
        
        # VERSIONING: Track confidence history
        self.versions: list[dict] = []  # Full epistemic trace
        
        # Evidence accumulation (Problem #1 fix)
        self.evidence_count = 1  # Number of times this assertion was repeated
        
        # Dynamics
        self.activation = 0.0  # usage counter
        self.created_at = now_utc()
        self.updated_at = self.created_at
        self.active = True
        
        # ═══════════════════════════════════════════════════════════
        # EPISTEMIC LEARNING (confidence dynamics)
        # ═══════════════════════════════════════════════════════════
        
        # Confidence decay: unused beliefs lose explanatory power
        self.last_accessed = self.created_at  # Last time belief was used in reasoning
        self.decay_rate = 0.001  # Confidence decay per day (configurable)
        
        # Confidence reinforcement: successful use increases confidence
        self.usage_count = 0  # Number of times belief successfully answered a question
        self.success_count = 0  # Number of times answer was correct (if validated)
        self.failure_count = 0  # Number of times answer was incorrect (if validated)
        
        # Justification chains: track which beliefs support this one
        self.supported_by: list[str] = []  # Belief IDs that justify this belief
        self.supports: list[str] = []  # Belief IDs that this belief justifies
        
        # Relations (managed by graph, not by belief)
        # Format: {relation_type: [target_belief_ids]}
        self.edges_out: dict[str, list[str]] = {}
        self.edges_in: dict[str, list[str]] = {}
        
        # Additional metadata
        self.metadata = metadata or {}
        
        # Cache extracted entities/predicates
        self._entities_cache: set[str] | None = None
        self._predicates_cache: set[str] | None = None
    
    # ═══════════════════════════════════════════════════════════════
    # PROPERTIES (controlled access)
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def canonical(self) -> dict:
        """Canonical structure (read-only, returns deep copy)."""
        return copy.deepcopy(self._canonical)
    
    @property
    def epistemic_state(self) -> str:
        """Current epistemic state."""
        return self._epistemic_state
    
    @epistemic_state.setter
    def epistemic_state(self, value: str):
        """Update epistemic state with validation."""
        self._epistemic_state = validate_epistemic_state(value)
        self.updated_at = now_utc()
    
    @property
    def confidence(self) -> float:
        """Current confidence level."""
        return self._confidence
    
    @confidence.setter
    def confidence(self, value: float):
        """Update confidence with validation and versioning."""
        # VERSIONING: Save old state before updating
        if self._confidence != value:
            self.versions.append({
                "confidence": self._confidence,
                "epistemic_state": self._epistemic_state,
                "timestamp": now_utc(),
                "reason": "confidence_update",  # Will be set by caller if needed
            })
        
        self._confidence = validate_confidence(value)
        self.updated_at = now_utc()
    
    # ═══════════════════════════════════════════════════════════════
    # ENTITY & PREDICATE EXTRACTION
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def entities(self) -> set[str]:
        """Extract all entities from canonical (cached)."""
        if self._entities_cache is None:
            self._entities_cache = extract_entities(self._canonical)
        return self._entities_cache
    
    @property
    def predicates(self) -> set[str]:
        """Extract all predicates from canonical (cached)."""
        if self._predicates_cache is None:
            self._predicates_cache = extract_predicates(self._canonical)
        return self._predicates_cache
    
    # ═══════════════════════════════════════════════════════════════
    # BINARY RELATION SUPPORT (subject/object extraction)
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def subject(self) -> str | None:
        """
        Extract subject from binary relations (is_a, has_attr, etc.).
        
        For is_a: "Bats are mammals" -> subject="bat", object="mammal"
        For has_attr: "Gold is shiny" -> subject="gold", object="shiny"
        
        Returns None for non-binary templates.
        """
        if self.template not in ["is_a", "has_attr", "relation"]:
            return None
        
        # Extract from canonical (preserves order)
        return self._canonical.get('subject')
    
    @property
    def object(self) -> str | None:
        """
        Extract object from binary relations.
        
        For is_a: "Bats are mammals" -> object="mammal"
        For has_attr: "Gold is shiny" -> object="shiny"
        
        Returns None for non-binary templates.
        """
        if self.template not in ["is_a", "has_attr", "relation"]:
            return None
        
        # Extract from canonical (preserves order)
        return self._canonical.get('object')
    
    @property
    def relation(self) -> str | None:
        """Extract primary predicate for binary relations."""
        if self.template not in ["is_a", "has_attr", "relation"]:
            return None
        
        predicates_list = list(self.predicates)
        if predicates_list:
            return predicates_list[0]
        return self.template
    
    # ═══════════════════════════════════════════════════════════════
    # ACTIVATION & USAGE TRACKING
    # ═══════════════════════════════════════════════════════════════
    
    def touch(self, activation_delta: float = 1.0):
        """
        Mark belief as used (updates activation and recency).
        
        Args:
            activation_delta: amount to increase activation by
        """
        self.activation += activation_delta
        self.updated_at = now_utc()
    
    def decay_activation(self, factor: float):
        """
        Apply multiplicative decay to activation.
        
        Args:
            factor: decay multiplier in [0, 1]
        """
        self.activation *= max(0.0, min(1.0, factor))
    
    # ═══════════════════════════════════════════════════════════════
    # EPISTEMIC LEARNING (confidence dynamics)
    # ═══════════════════════════════════════════════════════════════
    
    def apply_decay(self, current_time: datetime | None = None) -> float:
        """
        Apply time-based confidence decay.
        
        Unused beliefs lose explanatory power over time.
        This keeps the system plastic and allows new evidence to win.
        
        Returns:
            New confidence value after decay
        """
        if current_time is None:
            current_time = now_utc()
        
        # Calculate time elapsed since last access (in days)
        elapsed = (current_time - self.last_accessed).total_seconds() / 86400.0
        
        # Apply exponential decay: conf_new = conf_old * exp(-decay_rate * time)
        import math
        decay_factor = math.exp(-self.decay_rate * elapsed)
        
        old_confidence = self._confidence
        self._confidence = max(0.0, old_confidence * decay_factor)
        
        return self._confidence
    
    def reinforce(self, boost: float = 0.05, success: bool = True) -> float:
        """
        Reinforce confidence when belief successfully answers a question.
        
        Args:
            boost: Confidence increase amount (default 0.05)
            success: Whether the answer was validated as correct
        
        Returns:
            New confidence value after reinforcement
        """
        self.usage_count += 1
        self.last_accessed = now_utc()
        
        if success:
            self.success_count += 1
            # Boost confidence (with ceiling at 1.0)
            self._confidence = min(1.0, self._confidence + boost)
        else:
            self.failure_count += 1
            # Penalize confidence on failure
            self._confidence = max(0.0, self._confidence - boost * 2)
        
        self.updated_at = now_utc()
        return self._confidence
    
    def add_justification(self, supporting_belief_id: str) -> None:
        """
        Add a justification edge: this belief is supported by another belief.
        
        Example:
            "Bat produces milk" is supported by "Bat is mammal" and "Mammals produce milk"
        """
        if supporting_belief_id not in self.supported_by:
            self.supported_by.append(supporting_belief_id)
    
    # ═══════════════════════════════════════════════════════════════
    # PROVENANCE TRACKING
    # ═══════════════════════════════════════════════════════════════
    
    def add_provenance(
        self,
        op: str,
        from_source: str | list[str],
        score: float,
        metadata: dict | None = None,
    ):
        """
        Add a provenance entry.
        
        Args:
            op: operation type
            from_source: source component or belief IDs
            score: confidence contribution
            metadata: additional metadata
        """
        entry = ProvenanceEntry(
            op=op,
            from_source=from_source,
            score=score,
            metadata=metadata or {},
        )
        self.provenance.append(entry)
    
    # ═══════════════════════════════════════════════════════════════
    # EDGE HELPERS (graph manages, belief exposes)
    # ═══════════════════════════════════════════════════════════════
    
    def _add_edge_out(self, relation: str, target_id: str):
        """Internal: add outgoing edge (called by graph)."""
        if relation not in self.edges_out:
            self.edges_out[relation] = []
        if target_id not in self.edges_out[relation]:
            self.edges_out[relation].append(target_id)
    
    def _add_edge_in(self, relation: str, source_id: str):
        """Internal: add incoming edge (called by graph)."""
        if relation not in self.edges_in:
            self.edges_in[relation] = []
        if source_id not in self.edges_in[relation]:
            self.edges_in[relation].append(source_id)
    
    def _remove_edge_out(self, relation: str, target_id: str):
        """Internal: remove outgoing edge (called by graph)."""
        if relation in self.edges_out and target_id in self.edges_out[relation]:
            self.edges_out[relation].remove(target_id)
    
    def _remove_edge_in(self, relation: str, source_id: str):
        """Internal: remove incoming edge (called by graph)."""
        if relation in self.edges_in and source_id in self.edges_in[relation]:
            self.edges_in[relation].remove(source_id)
    
    # ═══════════════════════════════════════════════════════════════
    # DEACTIVATION
    # ═══════════════════════════════════════════════════════════════
    
    def deactivate(self):
        """Mark belief as inactive (soft delete)."""
        self.active = False
        self.updated_at = now_utc()
    
    def reactivate(self):
        """Reactivate a deactivated belief."""
        self.active = True
        self.updated_at = now_utc()
    
    # ═══════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════════════════════════════
    
    def to_dict(self) -> dict:
        """Export belief to dictionary (for JSON serialization)."""
        # Extract clean quantifier/polarity values
        quantifier_value = "UNSPECIFIED"
        if hasattr(self, 'quantifier'):
            q = self.quantifier
            if hasattr(q, 'value'):
                quantifier_value = q.value
            else:
                quantifier_value = str(q).replace("Quantifier.", "")
        
        polarity_value = "POSITIVE"
        if hasattr(self, 'polarity'):
            p = self.polarity
            if hasattr(p, 'value'):
                polarity_value = p.value
            else:
                polarity_value = str(p).replace("Polarity.", "")
        
        return {
            "id": self.id,
            "epistemic_state": self.epistemic_state,
            "template": self.template,
            "canonical": self._canonical,
            "original_text": self.original_text,
            "statement_text": self.statement_text,
            "confidence": self.confidence,
            "quantifier": quantifier_value,
            "polarity": polarity_value,
            "moral_value": self.moral_value,
            "source": self.source,
            "provenance": [p.to_dict() for p in self.provenance],
            "edges_out": self.edges_out,
            "edges_in": self.edges_in,
            "activation": self.activation,
            "created_at": to_iso(self.created_at),
            "updated_at": to_iso(self.updated_at),
            "active": self.active,
            # Epistemic learning
            "last_accessed": to_iso(self.last_accessed),
            "decay_rate": self.decay_rate,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "supported_by": self.supported_by,
            "supports": self.supports,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Belief:
        """Create belief from dictionary."""
        try:
            from .utils import from_iso
        except ImportError:
            from utils import from_iso
        
        belief = cls(
            belief_id=data["id"],
            template=data["template"],
            canonical=data["canonical"],
            confidence=data["confidence"],
            epistemic_state=data.get("epistemic_state", "asserted"),
            original_text=data.get("original_text"),
            statement_text=data.get("statement_text"),
            source=data.get("source", {}),
            moral_value=data.get("moral_value"),
            metadata=data.get("metadata", {}),
            quantifier=data.get("quantifier", "UNSPECIFIED"),
            polarity=data.get("polarity", "POSITIVE"),
        )
        
        # Restore provenance
        if "provenance" in data:
            belief.provenance = [
                ProvenanceEntry.from_dict(p) for p in data["provenance"]
            ]
        
        # Restore edges (graph will validate these)
        belief.edges_out = data.get("edges_out", {})
        belief.edges_in = data.get("edges_in", {})
        
        # Restore dynamics
        belief.activation = data.get("activation", 0.0)
        belief.active = data.get("active", True)
        
        if "created_at" in data:
            belief.created_at = from_iso(data["created_at"])
        if "updated_at" in data:
            belief.updated_at = from_iso(data["updated_at"])
        
        # Restore epistemic learning fields
        if "last_accessed" in data:
            belief.last_accessed = from_iso(data["last_accessed"])
        else:
            belief.last_accessed = belief.created_at
        
        belief.decay_rate = data.get("decay_rate", 0.001)
        belief.usage_count = data.get("usage_count", 0)
        belief.success_count = data.get("success_count", 0)
        belief.failure_count = data.get("failure_count", 0)
        belief.supported_by = data.get("supported_by", [])
        belief.supports = data.get("supports", [])
        
        return belief
    
    # ═══════════════════════════════════════════════════════════════
    # REPR & DEBUG
    # ═══════════════════════════════════════════════════════════════
    
    def __repr__(self) -> str:
        return (
            f"<Belief {self.id} "
            f"state={self.epistemic_state} "
            f"conf={self.confidence:.3f} "
            f"template={self.template}>"
        )
    
    def __str__(self) -> str:
        return self.statement_text or repr(self)
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Belief {self.id}\n"
            f"  State: {self.epistemic_state}\n"
            f"  Confidence: {self.confidence:.3f}\n"
            f"  Template: {self.template}\n"
            f"  Text: {self.statement_text}\n"
            f"  Entities: {self.entities}\n"
            f"  Predicates: {self.predicates}\n"
            f"  Activation: {self.activation:.2f}\n"
            f"  Created: {self.created_at}\n"
            f"  Active: {self.active}"
        )
