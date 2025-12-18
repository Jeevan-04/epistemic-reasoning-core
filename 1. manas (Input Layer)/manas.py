"""
MANAS — Understanding and Parsing Engine

Manas is Sanskrit for "mind" or "sensory processor."

In MARC, Manas is:
- The ONLY module that uses LLM
- Converts raw text → structured BeliefProposal
- Stateless (no memory access)
- NO access to Chitta
- NO belief lookup
- NO reasoning

Philosophy:
- Understanding ≠ Thinking
- Manas answers: "What is being said?"
- Buddhi answers: "Should I believe it?"

Output: BeliefProposal (untrusted, structured meaning)
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from .manas_utils import normalize_entity, detect_intent, detect_modality
    from .predicate_normalizer import get_predicate_normalizer, get_entity_normalizer
except ImportError:
    from manas_utils import normalize_entity, detect_intent, detect_modality
    from predicate_normalizer import get_predicate_normalizer, get_entity_normalizer


# ═══════════════════════════════════════════════════════════════════
# BELIEF PROPOSAL SCHEMA
# ═══════════════════════════════════════════════════════════════════

# This is what Manas outputs after parsing text — think of it as a structured claim
# that hasn't been judged yet. It's like when you hear something and understand the
# words, but haven't decided if you believe it.
BELIEF_PROPOSAL_SCHEMA = {
    "template": "relation | is_a | has_attr | event | action | epistemic | temporal | causal",
    "canonical": {},  # the core meaning in machine-readable format
    "entities": [],   # who or what this is about (e.g., ["penguin", "bird"])
    "predicates": [], # what properties or actions are claimed (e.g., ["can_fly"])
    "polarity": +1,   # whether this is a positive claim (+1) or negative (-1)
    "parser_confidence": 0.0,  # how confident the parser is that it understood correctly
    "raw_text": "",   # the exact original text we received
}


# ═══════════════════════════════════════════════════════════════════
# MANAS CLASS
# ═══════════════════════════════════════════════════════════════════

class Manas:
    """
    Understanding and parsing engine for MARC.
    
    Responsibilities:
    - Parse natural language → structured BeliefProposal
    - Extract entities, predicates, relations
    - Classify template type
    - Detect polarity (positive/negative)
    - Assign parser confidence (NOT belief confidence)
    
    Invariants:
    - STATELESS (no memory)
    - NO access to Chitta
    - NO belief lookup
    - NO reasoning
    - ONLY module that uses LLM
    """
    
    def __init__(self, llm_backend: str = "mock"):
        """
        Initialize Manas with LLM backend.
        
        Args:
            llm_backend: "mock" | "openai" | "anthropic" | "local"
        """
        self.llm_backend = llm_backend
        
        # Statistics
        self.stats = {
            "parses": 0,
            "successes": 0,
            "failures": 0,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN PARSING ENTRY POINT
    # ═══════════════════════════════════════════════════════════════
    
    def parse(self, text: str) -> dict:
        """
        Parse natural language → BeliefProposal.
        
        This is the ONLY public method.
        
        Adds:
        - Entity normalization (prevent graph fragmentation)
        - Intent detection (query vs assertion)
        - Modality markers (strong/weak/default)
        
        Args:
            text: raw natural language input
        
        Returns:
            BeliefProposal dict
        """
        self.stats["parses"] += 1
        
        try:
            if self.llm_backend == "mock":
                proposal = self._parse_mock(text)
            else:
                proposal = self._parse_llm(text)
            
            # ═══════════════════════════════════════════════════════════
            # PERCEPTION LAYER: Semantic Normalization (CRITICAL)
            # ═══════════════════════════════════════════════════════════
            # Here's a key insight: "Fish live in water" and "Do fish live in water?"
            # are asking about the SAME thing, just phrased differently. The perception
            # layer catches these variations and maps them to a single representation.
            # We're not reasoning about truth here — just recognizing that different
            # phrasings point to the same underlying concept.
            
            # Grab our normalization tools
            entity_norm = get_entity_normalizer()
            pred_norm = get_predicate_normalizer()
            
            # STEP 1: Extract and normalize entities FROM PREDICATE CONTEXT
            # The tricky part: figuring out what entities are involved depends heavily on
            # what the sentence is actually saying. "Birds fly" has different entities than
            # "Birds can fly" even though both mention "birds" and "fly".
            pred_result = pred_norm.normalize_with_confidence(text)
            extracted_entities = pred_result.get("entities", [])
            
            # Combine entities from predicate analysis with what the parser found
            raw_entities = proposal.get("entities", [])
            all_entities = list(set(extracted_entities + raw_entities))
            
            # Remove common verbs that sneak in as entities (they're actions, not things)
            verb_blacklist = {'live', 'lives', 'breathe', 'breathes', 'exist', 'exists', 'fly', 'flies', 'swim', 'swims', 'do', 'does', 'did', 'not'}
            all_entities = [e for e in all_entities if e.lower() not in verb_blacklist]
            
            # Clean up the entities — turn "Birds", "birds", "bird" all into "bird"
            normalized_entities = list(set(entity_norm.normalize_list(all_entities)))
            proposal["entities"] = normalized_entities
            
            # STEP 2: Normalize predicates (THE CRITICAL FIX)
            # This was a tough bug to fix: "Fish live in water" (stating a habitat fact)
            # and "Do fish live in water?" (asking a question) were getting treated as
            # different predicates. They should map to the same underlying relation.
            canonical_predicate = pred_result["canonical"]
            pred_type = pred_result["type"]
            pred_confidence = pred_result["confidence"]
            epistemic_class = pred_result.get("epistemic_class")
            
            # Replace whatever predicate the parser found with the normalized version
            proposal["predicates"] = [canonical_predicate]
            proposal["epistemic_class"] = epistemic_class  # save this for later inference control
            
            # Update the canonical form to use our cleaned-up versions
            if "canonical" in proposal:
                proposal["canonical"]["relation_type"] = canonical_predicate
                proposal["canonical"]["predicate_type"] = pred_type
                proposal["canonical"]["entities"] = normalized_entities
                proposal["canonical"]["epistemic_class"] = epistemic_class
                
                # If there's a subject mentioned, normalize it too ("Birds" → "bird")
                if "subject" in proposal["canonical"] and proposal["canonical"]["subject"]:
                    proposal["canonical"]["subject"] = entity_norm.normalize(proposal["canonical"]["subject"])
                # Same for object ("Mammals" → "mammal")
                if "object" in proposal["canonical"] and proposal["canonical"]["object"]:
                    proposal["canonical"]["object"] = entity_norm.normalize(proposal["canonical"]["object"])
            
            # The parser's confidence is only as good as its weakest link — if we're
            # not sure about the predicate, that uncertainty affects the whole parse
            proposal["parser_confidence"] = min(
                proposal.get("parser_confidence", 0.8),
                pred_confidence
            )
            
            # ═══════════════════════════════════════════════════════════
            
            # Double-check that all entities are actually lowercase (important for matching)
            assert all(e == e.lower() for e in proposal["entities"]), \
                f"Entity normalization failed: {proposal['entities']}"
            
            # Figure out if this is a question or a statement
            proposal["intent"] = detect_intent(text)
            
            # REFINEMENT 2.1: Query canonicalization
            # When someone asks "Do penguins fly?", we don't want to pollute the belief
            # space with a negated claim. Questions are neutral — they're requests for
            # information, not assertions. So we strip away negation markers for queries.
            if proposal["intent"] == "query":
                if "canonical" in proposal:
                    proposal["canonical"]["negated"] = None  # queries don't assert anything
                # We keep polarity for matching purposes, but mark this as a query
                proposal["query_mode"] = True
            
            # Detect modality: how strongly is this being claimed?
            # "Penguins definitely can't fly" (strong) vs "Penguins probably can't fly" (weak)
            proposal["modality"] = detect_modality(text)
            
            # Modality affects how confident we are in the parse
            if proposal["modality"] == "strong":
                # Strong language → boost confidence a bit (but cap at 0.95)
                proposal["parser_confidence"] = min(0.95, proposal.get("parser_confidence", 0.5) * 1.1)
            elif proposal["modality"] == "weak":
                # Hedging language → reduce confidence (but don't go below 0.2)
                proposal["parser_confidence"] = max(0.2, proposal.get("parser_confidence", 0.5) * 0.7)
            
            # Validate schema
            self._validate_proposal(proposal)
            
            self.stats["successes"] += 1
            return proposal
        
        except Exception as e:
            self.stats["failures"] += 1
            # Fallback to basic parse
            return self._parse_fallback(text, error=str(e))
    
    # ═══════════════════════════════════════════════════════════════
    # MOCK PARSER (FOR TESTING)
    # ═══════════════════════════════════════════════════════════════
    
    def _parse_mock(self, text: str) -> dict:
        """
        Mock parser for testing (rule-based).
        
        Handles common patterns without LLM.
        
        Args:
            text: input text
        
        Returns:
            BeliefProposal dict
        """
        text_lower = text.lower().strip()
        
        # Pattern 1: "X can Y" / "X cannot Y" / "Can X Y?"
        if " can " in text_lower or " cannot " in text_lower or text_lower.startswith("can "):
            return self._parse_capability(text)
        
        # Pattern 2: "X is a Y" / "X are Y"
        if " is a " in text_lower or " are " in text_lower or " is an " in text_lower:
            return self._parse_is_a(text)
        
        # Pattern 3: "X has Y" / "X have Y" / "Do X have Y?"
        if " has " in text_lower or " have " in text_lower or text_lower.startswith("do "):
            return self._parse_has_attr(text)
        
        # Default: generic relation
        return self._parse_generic(text)
    
    def _parse_capability(self, text: str) -> dict:
        """Parse capability statements like 'Birds can fly' or 'Penguins cannot fly'."""
        text_lower = text.lower()
        # Check if this is a negative statement
        negated = "cannot" in text_lower or "can't" in text_lower or "can not" in text_lower
        
        # Easier to work with "can not" than "cannot" when splitting words
        normalized = text.replace("cannot", "can not").replace("can't", "can not")
        words = normalized.split()
        
        # Extract the entity and action from the sentence
        entity = None
        predicate = None
        
        if "can" in [w.lower() for w in words]:
            # Find where "can" appears in the sentence
            can_idx = next(i for i, w in enumerate(words) if w.lower() == "can")
            
            # Two common patterns:
            # 1. "Birds can fly" — entity before "can"
            # 2. "Can birds fly?" — entity after "can"
            if can_idx == 0:
                # Question format: "Can penguins swim?"
                if can_idx + 1 < len(words):
                    entity = words[can_idx + 1].strip(".,!?")
                    # The action comes after the entity
                    action_idx = can_idx + 2
                else:
                    action_idx = can_idx + 1
            else:
                # Statement format: "Penguins can swim"
                entity = words[can_idx - 1].strip(".,!?")
                # Skip over "not" if it appears after "can"
                action_idx = can_idx + 1
                if action_idx < len(words) and words[action_idx].lower() == "not":
                    action_idx += 1
            
            if action_idx < len(words):
                predicate = words[action_idx].strip(".,!?")
        
        # Build the predicate name: "can_fly", "can_swim", etc.
        # This is the same whether the statement is positive or negative — the polarity
        # field tracks whether we're affirming or denying the capability
        predicate_name = f"can_{predicate}" if predicate else "capability"
        
        return {
            "template": "relation",
            "canonical": {
                "relation_type": predicate_name,
                "entities": [entity] if entity else [],
                "negated": negated,
            },
            "entities": [entity] if entity else [],
            "predicates": [predicate_name],  # Same predicate regardless of polarity
            "polarity": -1 if negated else +1,
            "parser_confidence": 0.85,
            "raw_text": text,
        }
    
    def _parse_is_a(self, text: str) -> dict:
        """Parse taxonomic statements like 'A penguin is a bird'."""
        text_lower = text.lower()
        
        # Look for negation markers ("is not", "isn't", etc.)
        is_negative = any(neg in text_lower for neg in [
            "is not", "are not", "isn't", "aren't", "was not", "were not", "wasn't", "weren't"
        ])
        polarity = -1 if is_negative else +1
        
        # Split into words for processing
        words = text.split()
        
        # Extract subject and object from the sentence
        subject = None
        obj = None
        
        for i, word in enumerate(words):
            if word.lower() in ["is", "are", "was", "were"]:
                # The subject comes before the verb (e.g., "penguin" in "A penguin is...")
                if i > 0:
                    subject = words[i - 1].strip(".,!?")
                
                # The object comes after "a", "an", or "not" (e.g., "bird" in "...is a bird")
                j = i + 1
                while j < len(words) and words[j].lower() in ["a", "an", "not"]:
                    j += 1
                if j < len(words):
                    obj = words[j].strip(".,!?")
                break
        
        return {
            "template": "is_a",
            "canonical": {
                "relation_type": "is_a",
                "entities": [subject, obj] if subject and obj else [],
                "subject": subject,
                "object": obj,
            },
            "entities": [subject, obj] if subject and obj else [],
            "predicates": ["is_a"],
            "polarity": polarity,
            "parser_confidence": 0.90,
            "raw_text": text,
        }
    
    def _parse_has_attr(self, text: str) -> dict:
        """Parse attribute statements like 'Fish have gills'."""
        words = text.split()
        
        subject = None
        attr = None
        polarity = +1
        
        # Check for various negation patterns
        text_lower = text.lower()
        is_negative = any(neg in text_lower for neg in [
            "do not", "does not", "did not", "don't", "doesn't", "didn't",
            "cannot", "can't", "can not",
            "have not", "has not", "haven't", "hasn't",
            "have no", "has no"
        ])
        if is_negative:
            polarity = -1
        
        # Find the subject and what attribute they have
        for i, word in enumerate(words):
            if word.lower() in ["has", "have"]:
                # Work backwards to find the actual subject (skip negation words like "not")
                if i > 0:
                    j = i - 1
                    while j >= 0 and words[j].lower() in ["not", "no", "don't", "doesn't", "do", "does"]:
                        j -= 1
                    if j >= 0:
                        subject = words[j].strip(".,!?")
                
                # The attribute comes right after "has"/"have"
                if i + 1 < len(words):
                    attr = words[i + 1].strip(".,!?")
                break
        
        # Important fix: store the full predicate ("has_gills") not just "has"
        # This makes it easier to extract the attribute later when querying
        full_predicate = f"has_{attr}" if attr else "has"
        
        return {
            "template": "has_attr",
            "canonical": {
                "relation_type": full_predicate,  # "has_gills", not just "has"
                "entities": [subject] if subject else [],
                "attribute": attr,
            },
            "entities": [subject] if subject else [],
            "predicates": [full_predicate],
            "polarity": polarity,
            "parser_confidence": 0.80,
            "raw_text": text,
        }
    
    def _parse_generic(self, text: str) -> dict:
        """Fallback parser when we can't match any specific pattern."""
        words = text.split()
        
        # Try to grab a couple entities by filtering out common stop words
        stop_words = {'is', 'are', 'was', 'were', 'be', 'been', 'the', 'a', 'an'}
        entities = [w for w in words[:3] if w.lower() not in stop_words][:2]
        
        return {
            "template": "relation",
            "canonical": {
                "relation_type": "generic",
                "entities": entities,
            },
            "entities": entities,
            "predicates": ["generic"],
            "polarity": +1,
            "parser_confidence": 0.50,
            "raw_text": text,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # LLM PARSER (FUTURE)
    # ═══════════════════════════════════════════════════════════════
    
    def _parse_llm(self, text: str) -> dict:
        """
        LLM-backed parser (not implemented yet).
        
        Eventually this will use a real language model (GPT-4, Claude, or a local model)
        to handle more complex sentences that the pattern matcher can't parse.
        
        For now, we just fall back to the mock parser.
        """
        # TODO: Implement actual LLM parsing
        return self._parse_mock(text)
    
    # ═══════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    def _validate_proposal(self, proposal: dict):
        """
        Make sure the BeliefProposal has all required fields.
        
        This catches bugs early — if we're missing something critical,
        better to fail here than have weird errors downstream.
        
        Args:
            proposal: BeliefProposal dict
        
        Raises:
            ValueError: if something is missing or wrong type
        """
        required_fields = [
            "template",
            "canonical",
            "entities",
            "predicates",
            "polarity",
            "parser_confidence",
            "raw_text",
        ]
        
        for field in required_fields:
            if field not in proposal:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate types
        if not isinstance(proposal["canonical"], dict):
            raise ValueError("canonical must be dict")
        
        if not isinstance(proposal["entities"], list):
            raise ValueError("entities must be list")
        
        if not isinstance(proposal["predicates"], list):
            raise ValueError("predicates must be list")
        
        if proposal["polarity"] not in [+1, -1]:
            raise ValueError("polarity must be +1 or -1")
        
        if not (0.0 <= proposal["parser_confidence"] <= 1.0):
            raise ValueError("parser_confidence must be in [0, 1]")
    
    def _parse_fallback(self, text: str, error: str = "") -> dict:
        """
        Fallback parser when all else fails.
        
        Returns minimal BeliefProposal.
        """
        return {
            "template": "relation",
            "canonical": {
                "relation_type": "unknown",
                "error": error,
            },
            "entities": [],
            "predicates": [],
            "polarity": +1,
            "parser_confidence": 0.1,
            "raw_text": text,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════
    
    def get_stats(self) -> dict:
        """Get parsing statistics."""
        return self.stats.copy()
    
    def __repr__(self) -> str:
        return (
            f"<Manas "
            f"backend={self.llm_backend} "
            f"parses={self.stats['parses']} "
            f"success_rate={self.stats['successes']/(self.stats['parses'] or 1):.2%}>"
        )
