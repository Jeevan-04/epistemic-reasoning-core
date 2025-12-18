"""
COMPREHENSIVE REASONING LOGGER

Logs every reasoning step for transparency and debugging.

For each question, tracks:
- Focus set size
- Applicable beliefs
- Rejected beliefs (with reason)
- Epistemic class filter
- Confidence blocks
- Belief competition
- Taxonomic propagation
- Analogical hypotheses
- Final verdict + confidence

Required breakdowns:
- Where restraint triggered (no applicable beliefs)
- Where confidence blocked an answer (low score)
- Where multiple beliefs competed (conflicts)
- Where epistemic class prevented propagation (abstract/behavioral)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FocusRecord:
    """Record of focus phase."""
    query: str
    focus_set_size: int
    focus_beliefs: list[str]  # Belief IDs
    focus_method: str  # entity_overlap | semantic_search | etc


@dataclass
class ApplicabilityRecord:
    """Record of applicability check."""
    applicable_count: int
    rejected_count: int
    applicable_beliefs: list[dict]  # {id, reason, overlap}
    rejected_beliefs: list[dict]  # {id, reason, overlap}


@dataclass
class EpistemicClassRecord:
    """Record of epistemic class filtering."""
    query_class: str  # structural | behavioral | abstract
    filter_applied: bool
    exact_matches: list[str]  # Belief IDs
    filtered_out: list[str]  # Belief IDs


@dataclass
class ConfidenceRecord:
    """Record of confidence-based decisions."""
    threshold: float
    winning_belief: str | None
    winning_score: float | None
    runner_up: str | None
    runner_up_score: float | None
    margin: float


@dataclass
class ConflictRecord:
    """Record of belief competition."""
    competing_beliefs: list[dict]  # {id, predicate, confidence, polarity}
    resolution: str  # prefer_positive | prefer_negative | uncertain
    delta: float


@dataclass
class TaxonomicRecord:
    """Record of taxonomic inference."""
    chains: list[dict]  # {child, parent, grandparent, belief_ids}
    propagations: list[dict]  # {from, to, predicate, confidence}


@dataclass
class AnalogicalRecord:
    """Record of analogical hypotheses."""
    hypotheses: list[dict]  # {statement, mapping, confidence}
    accepted: list[str]  # Hypothesis IDs
    rejected: list[str]  # Hypothesis IDs


@dataclass
class ReasoningLog:
    """Complete log for a single query."""
    query: str
    query_entities: list[str]
    query_predicates: list[str]
    query_epistemic_class: str
    
    # Phase records
    focus: FocusRecord | None = None
    applicability: ApplicabilityRecord | None = None
    epistemic_class: EpistemicClassRecord | None = None
    confidence: ConfidenceRecord | None = None
    conflicts: list[ConflictRecord] = field(default_factory=list)
    taxonomic: TaxonomicRecord | None = None
    analogical: AnalogicalRecord | None = None
    
    # Final outcome
    verdict: str = "unknown"
    final_confidence: float = 0.0
    answer_text: str = ""
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        result = {
            "query": self.query,
            "query_entities": self.query_entities,
            "query_predicates": self.query_predicates,
            "query_epistemic_class": self.query_epistemic_class,
            "verdict": self.verdict,
            "confidence": self.final_confidence,
            "answer": self.answer_text,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.focus:
            result["focus"] = asdict(self.focus)
        if self.applicability:
            result["applicability"] = asdict(self.applicability)
        if self.epistemic_class:
            result["epistemic_class"] = asdict(self.epistemic_class)
        if self.confidence:
            result["confidence"] = asdict(self.confidence)
        if self.conflicts:
            result["conflicts"] = [asdict(c) for c in self.conflicts]
        if self.taxonomic:
            result["taxonomic"] = asdict(self.taxonomic)
        if self.analogical:
            result["analogical"] = asdict(self.analogical)
        
        result["metadata"] = self.metadata
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class ReasoningLogger:
    """
    Comprehensive logging for reasoning process.
    
    Tracks all decisions, filters, and competitions.
    """
    
    def __init__(self, output_dir: Path | None = None):
        """
        Args:
            output_dir: Directory to save logs (None = memory only)
        """
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs: list[ReasoningLog] = []
    
    def create_log(
        self,
        query: str,
        query_entities: list[str],
        query_predicates: list[str],
        query_epistemic_class: str
    ) -> ReasoningLog:
        """Create a new log for a query."""
        log = ReasoningLog(
            query=query,
            query_entities=query_entities,
            query_predicates=query_predicates,
            query_epistemic_class=query_epistemic_class,
        )
        self.logs.append(log)
        return log
    
    def save_log(self, log: ReasoningLog) -> None:
        """Save log to file."""
        if self.output_dir:
            filename = f"log_{len(self.logs):04d}_{log.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            filepath.write_text(log.to_json())
    
    def generate_summary(self) -> dict:
        """
        Generate summary statistics across all logs.
        
        Returns:
            Summary dict with breakdowns
        """
        total = len(self.logs)
        if total == 0:
            return {"total_queries": 0}
        
        # Count verdicts
        verdict_counts = {}
        for log in self.logs:
            verdict_counts[log.verdict] = verdict_counts.get(log.verdict, 0) + 1
        
        # Track restraint triggers
        restraint_no_focus = sum(1 for log in self.logs if log.focus and log.focus.focus_set_size == 0)
        restraint_no_applicable = sum(1 for log in self.logs 
                                     if log.applicability and log.applicability.applicable_count == 0)
        restraint_epistemic = sum(1 for log in self.logs
                                 if log.epistemic_class and log.epistemic_class.filter_applied
                                 and len(log.epistemic_class.exact_matches) == 0)
        restraint_low_confidence = sum(1 for log in self.logs
                                      if log.confidence and log.confidence.winning_score is not None
                                      and log.confidence.winning_score < 0.25)
        
        # Track belief competition
        conflicts_total = sum(len(log.conflicts) for log in self.logs)
        conflicts_resolved_positive = sum(
            sum(1 for c in log.conflicts if c.resolution == "prefer_positive")
            for log in self.logs
        )
        conflicts_resolved_negative = sum(
            sum(1 for c in log.conflicts if c.resolution == "prefer_negative")
            for log in self.logs
        )
        conflicts_uncertain = sum(
            sum(1 for c in log.conflicts if c.resolution == "uncertain")
            for log in self.logs
        )
        
        # Track epistemic class prevention
        epistemic_blocked = {}
        for log in self.logs:
            if log.epistemic_class and log.epistemic_class.filter_applied:
                cls = log.query_epistemic_class
                epistemic_blocked[cls] = epistemic_blocked.get(cls, 0) + 1
        
        # Track taxonomic propagation
        taxonomic_inferences = sum(
            len(log.taxonomic.propagations) if log.taxonomic else 0
            for log in self.logs
        )
        
        # Track analogical hypotheses
        analogical_generated = sum(
            len(log.analogical.hypotheses) if log.analogical else 0
            for log in self.logs
        )
        analogical_accepted = sum(
            len(log.analogical.accepted) if log.analogical else 0
            for log in self.logs
        )
        
        return {
            "total_queries": total,
            "verdicts": verdict_counts,
            "restraint_triggers": {
                "no_focus": restraint_no_focus,
                "no_applicable": restraint_no_applicable,
                "epistemic_filter": restraint_epistemic,
                "low_confidence": restraint_low_confidence,
                "total": restraint_no_focus + restraint_no_applicable + restraint_epistemic + restraint_low_confidence,
            },
            "belief_competition": {
                "total_conflicts": conflicts_total,
                "resolved_positive": conflicts_resolved_positive,
                "resolved_negative": conflicts_resolved_negative,
                "uncertain": conflicts_uncertain,
            },
            "epistemic_class_blocking": epistemic_blocked,
            "taxonomic_inferences": taxonomic_inferences,
            "analogical_hypotheses": {
                "generated": analogical_generated,
                "accepted": analogical_accepted,
                "rejected": analogical_generated - analogical_accepted,
            },
        }
    
    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.generate_summary()
        
        print("=" * 80)
        print("REASONING SUMMARY")
        print("=" * 80)
        print(f"\nTotal Queries: {summary['total_queries']}")
        
        print("\nVERDICTS:")
        for verdict, count in summary['verdicts'].items():
            pct = count / summary['total_queries'] * 100
            print(f"  {verdict:12s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nRESTRAINT TRIGGERS:")
        restraint = summary['restraint_triggers']
        for trigger, count in restraint.items():
            if trigger != 'total':
                pct = count / summary['total_queries'] * 100 if summary['total_queries'] > 0 else 0
                print(f"  {trigger:20s}: {count:4d} ({pct:5.1f}%)")
        print(f"  {'Total':20s}: {restraint['total']:4d}")
        
        print("\nBELIEF COMPETITION:")
        comp = summary['belief_competition']
        print(f"  Total conflicts:      {comp['total_conflicts']:4d}")
        print(f"  Resolved (positive):  {comp['resolved_positive']:4d}")
        print(f"  Resolved (negative):  {comp['resolved_negative']:4d}")
        print(f"  Uncertain:            {comp['uncertain']:4d}")
        
        print("\nEPISTEMIC CLASS BLOCKING:")
        for cls, count in summary['epistemic_class_blocking'].items():
            print(f"  {cls:12s}: {count:4d}")
        
        print(f"\nTAXONOMIC INFERENCES: {summary['taxonomic_inferences']}")
        
        print(f"\nANALOGICAL HYPOTHESES:")
        analog = summary['analogical_hypotheses']
        print(f"  Generated:  {analog['generated']:4d}")
        print(f"  Accepted:   {analog['accepted']:4d}")
        print(f"  Rejected:   {analog['rejected']:4d}")
        
        print("=" * 80)
