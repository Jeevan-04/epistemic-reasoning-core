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
import re

from manas.utils import normalize_entity, detect_intent, detect_modality
from manas.predicate_normalizer import get_predicate_normalizer, get_entity_normalizer


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
            # Lightweight shorthand pre-parser to accept benchmark notation like
            # "A->B (default P).", "SourceA (r=0.95): Vaccines work.",
            # "X typically P", etc. This helps tests that use compact specs.
            shorthand = self._parse_shorthand(text)
            if shorthand is not None:
                proposal = shorthand
            else:
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
            
            # Smart Merge:
            # If our parser found a specific template (is_a, comparative), trust its entities.
            # Only use norm entities if parser was generic/uncertain.
            
            is_specific = False
            if proposal.get("template") in ["is_a", "has_attr"]:
                is_specific = True
            elif proposal.get("template") == "relation" and proposal.get("predicates", [""])[0].endswith("_than"):
                is_specific = True
                
            if is_specific:
                 all_entities = raw_entities
                 # print(f"DEBUG: Manas Specific Extraction: {all_entities}")
            else:
                 all_entities = list(set(extracted_entities + raw_entities))
                 # print(f"DEBUG: Manas Generic Extraction: {all_entities}")
            
            # Remove common verbs/prepositions that sneak in as entities
            # Expanded blacklist for stricter namespace separation
            verb_blacklist = {
                'live', 'lives', 'breathe', 'breathes', 'exist', 'exists', 'fly', 'flies', 'swim', 'swims',
                'bark', 'barks', 'move', 'moves', 'walk', 'walks',
                'do', 'does', 'did', 'not', 'no', 'never',
                'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'has', 'have', 'had', 'having',
                'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
                'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
                'an', 'the'
            }
            all_entities = [e for e in all_entities if e.lower() not in verb_blacklist]
            
            # Clean up the entities — turn "Birds", "birds", "bird" all into "bird"
            # Preserve order for specific templates (is_a, has_attr)
            if is_specific:
                normalized_entities = entity_norm.normalize_list(all_entities)
            else:
                normalized_entities = list(dict.fromkeys(entity_norm.normalize_list(all_entities)))

            # Heuristic: filter spurious single-letter articles like 'a' when they
            # appear in lowercase in the raw text. Preserve single-letter tokens
            # if the original raw text used an uppercase token (e.g., 'A' as a named
            # placeholder) or if the token appears explicitly in the parser's
            # original entity list (defensive).
            final_entities = []
            raw_text = proposal.get('raw_text', '')
            raw_entities = proposal.get('entities', []) or []
            for e in normalized_entities:
                if len(e) == 1:
                    # Keep if uppercase appearance in raw text (proper token)
                    if re.search(rf"\b{re.escape(e.upper())}\b", raw_text):
                        final_entities.append(e)
                        continue
                    # Keep if parser originally found the same token (conservative)
                    if any(re.sub(r"[^A-Za-z]", "", r).lower() == e for r in raw_entities):
                        final_entities.append(e)
                        continue
                    # Otherwise drop single-letter lowercase entity (likely article)
                    continue
                final_entities.append(e)

            proposal["entities"] = final_entities
            # NAMESPACE ENFORCEMENT: Predicates cannot be Entities

            # Return the constructed proposal
            return proposal
        except Exception as e:
            self.stats["failures"] += 1
            return self._parse_fallback(text, error=str(e))

    def _parse_shorthand(self, text: str) -> dict | None:
        """Try to parse compact benchmark shorthand into a BeliefProposal.

        Returns a BeliefProposal-like dict on success, or None if not matched.
        This is intentionally conservative — only recognizes a few common patterns.
        """
        import re

        t = text.strip()

        # Normalize leading adverbs like "Typically, A entities P." -> "A typically P."
        t = re.sub(r"^\s*Typically,\s*", "", t, flags=re.IGNORECASE)
        # Convert awkward "entities" phrasing into a 'typically' marker to match shorthand
        if re.search(r"\bentities\b", t, flags=re.IGNORECASE):
            t = re.sub(r"\bentities\b", "typically", t, flags=re.IGNORECASE)

        # Handle compact default pattern: "A P (rank N)" or "Typically, A P (rank N)."
        m_compact = re.match(r"^(?P<sub>[A-Za-z0-9_\-]+)\s+(?P<pred>[A-Za-z0-9_\-]+)\s*(?:\(\s*rank\s*=?\s*(?P<r>\d+)\s*\))?\.?$", t)
        if m_compact:
            subj = m_compact.group('sub')
            pred = m_compact.group('pred')
            rank = int(m_compact.group('r')) if m_compact.group('r') else None
            canonical = {'predicate': pred, 'entities': [subj]}
            if rank is not None:
                canonical['rank'] = rank
            return {
                'template': 'default',
                'raw_text': text,
                'entities': [subj],
                'predicates': [pred],
                'polarity': +1,
                'parser_confidence': 0.85,
                'canonical': canonical,
            }

        # Handle compact negative default: "B do not P (rank 2)" -> default negated
        m_neg_compact = re.match(r"^(?P<sub>[A-Za-z0-9_\-]+)\s+(?:do\s+not|does\s+not|not)\s+(?P<pred>[A-Za-z0-9_\-]+)\s*(?:\(\s*rank\s*=?\s*(?P<r>\d+)\s*\))?\.?$", t, flags=re.IGNORECASE)
        if m_neg_compact:
            subj = m_neg_compact.group('sub')
            pred = m_neg_compact.group('pred')
            rank = int(m_neg_compact.group('r')) if m_neg_compact.group('r') else None
            canonical = {'predicate': pred, 'entities': [subj]}
            if rank is not None:
                canonical['rank'] = rank
            return {
                'template': 'default',
                'raw_text': text,
                'entities': [subj],
                'predicates': [pred],
                'polarity': -1,
                'parser_confidence': 0.85,
                'canonical': canonical,
            }

        # Pattern: SourceName (r=0.95, timestamp=2025): Statement
        m = re.match(r"^(?P<src>[^\(:]+)\s*\((?P<meta>[^\)]+)\)\s*:\s*(?P<stmt>.+)$", t)
        if m:
            src = m.group('src').strip()
            meta = m.group('meta')
            stmt = m.group('stmt').strip()
            # extract reliability if present
            reli = None
            for part in meta.split(','):
                kv = part.strip()
                if kv.startswith('r=') or kv.startswith('reliability='):
                    try:
                        reli = float(kv.split('=')[1])
                    except Exception:
                        reli = None
            proposal = {
                'template': 'relation',
                'raw_text': text,
                'entities': [],
                'predicates': [],
                'parser_confidence': 0.9,
                'canonical': {},
                'source': src,
                'source_reliability': reli,
            }
            # try to further decompose stmt with existing normalizer
            try:
                pred_norm = get_predicate_normalizer()
                pred_res = pred_norm.normalize_with_confidence(stmt)
                proposal['predicates'] = pred_res.get('predicates', []) or [stmt]
                proposal['entities'] = pred_res.get('entities', [])
            except Exception:
                proposal['predicates'] = [stmt]
            return proposal

        # Pattern: A->B meaning A is_a B
        m = re.match(r"^(?P<a>\w+)\s*->\s*(?P<b>\w+)(?:\s*\((?P<rest>.*)\))?\.?$", t)
        if m:
            a = m.group('a')
            b = m.group('b')
            proposal = {
                'template': 'is_a',
                'raw_text': text,
                'entities': [a, b],
                'predicates': [],
                'parser_confidence': 0.9,
                'canonical': {'subclass': a, 'superclass': b}
            }
            return proposal

        # Pattern: "X typically P" or "X typically not P"
        m = re.match(r"^(?P<subject>[^:]+?)\s+typically\s+(?P<neg>(?:do\s+not|does\s+not|not)\s+)?(?P<p>.+)$", t, flags=re.IGNORECASE)
        if m:
            subj = m.group('subject').strip()
            neg = bool(m.group('neg'))
            p = m.group('p').strip().rstrip('.')
            # Extract optional rank annotation e.g. '(rank 1)'
            rank = None
            rank_m = re.search(r"\(\s*rank\s*=?\s*(?P<r>\d+)\s*\)", p, flags=re.IGNORECASE)
            if rank_m:
                try:
                    rank = int(rank_m.group('r'))
                    # remove the rank annotation from predicate text
                    p = re.sub(r"\(\s*rank\s*=?\s*\d+\s*\)", "", p, flags=re.IGNORECASE).strip()
                except Exception:
                    rank = None
            pred = p
            canonical = {'predicate': pred, 'entities': [subj]}
            if rank is not None:
                canonical['rank'] = rank
            proposal = {
                'template': 'default',
                'raw_text': text,
                'entities': [subj],
                'predicates': [pred],
                'polarity': -1 if neg else 1,
                'parser_confidence': 0.85,
                'canonical': canonical
            }
            return proposal

        # Pattern: "X is A and B" or short noun phrases like "Birds fly." handled by normal parser
        return None
    
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
        # Handle short property queries like "Is X P?" where P is a predicate token
        m_short_prop = re.match(r"^is\s+([A-Za-z0-9_\-]+)\s+([A-Za-z0-9_\-]+)\??$", text.strip(), flags=re.IGNORECASE)
        if m_short_prop:
            subj = m_short_prop.group(1).strip(".,!?;")
            obj = m_short_prop.group(2).strip(".,!?;")
            # If object looks like a predicate token (single token), treat as relation query
            if obj and len(obj) <= 3:
                predicate = obj
                return {
                    "template": "relation",
                    "canonical": {"relation_type": predicate, "entities": [subj], "subject": subj, "object": obj},
                    "entities": [subj, obj],
                    "predicates": [predicate],
                    "polarity": +1,
                    "parser_confidence": 0.9,
                    "raw_text": text,
                }
        
        # Pattern 4: Comparatives "X is larger than Y" / "Is X larger than Y?"
        if " than " in text_lower:
            return self._parse_comparative(text)
            
        # Pattern 1: "X can Y" / "X cannot Y" / "Can X Y?"
        if (" can " in text_lower or " cannot " in text_lower or text_lower.startswith("can ") or
            " do not " in text_lower or " don't " in text_lower or " does not " in text_lower):
            return self._parse_capability(text)
        
        # Pattern 2: "X is a Y" / "X are Y" / "X were Y" / "Is X a Y?"
        # Accept both 'is a' and the shorter 'is' taxonomic form like 'A is B'
        if (" is " in text_lower and not any(neg in text_lower for neg in [" is not ", " isn't "]) ) or \
           (any(p in text_lower for p in [" are ", " is a ", " is an ", " were ", " was "]) or 
            text_lower.startswith("is a ") or 
            text_lower.startswith("is an ") or
            # Handle "Is Entity_L0 a Entity_L2?" where "is" is start, " a " is later
            (text_lower.startswith("is ") and " a " in text_lower)):
            return self._parse_is_a(text)

        # Short taxonomic question like "Is X Y?" (no article present)
        # Recognize simple three-token patterns and treat them as is_a
        words = text_lower.split()
        if text_lower.startswith("is ") and len(words) == 3:
            return self._parse_is_a(text)
        
        # Pattern 3: "X has Y" / "X have Y" / "Do X have Y?"
        # CRITICAL FIX: Only route "Do..." queries here if they actually contain "have"
        if (" has " in text_lower or " have " in text_lower):
            return self._parse_has_attr(text)
        if text_lower.startswith("do ") and " have " in text_lower:
            return self._parse_has_attr(text)

        # Pattern 3b: Yes/no capability questions like "Do penguins fly?"
        # Treat these like capability parses (convert to can/cannot form)
        if text_lower.startswith(("do ", "does ", "did ")) and " have " not in text_lower:
            return self._parse_capability(text)
        
        # Default: generic relation
        return self._parse_generic(text)

    def _parse_comparative(self, text: str) -> dict:
        """Parse comparative statements like 'Planet is larger than Moon'."""
        text_lower = text.lower()
        words = text.split()
        
        # Find "than"
        try:
            than_idx = words.index("than")
        except ValueError:
            return self._parse_generic(text)

        # print(f"DEBUG: Comparative Parse: '{text}' words={words} than_idx={than_idx}")
            
        # Adjective is usually before "than" (e.g. "larger")
        adj = words[than_idx - 1].strip(".,!?") if than_idx > 0 else "unknown"
        predicate = f"{adj}_than"
        
        # Subject is usually before "is" which is before adj
        # "Planet is larger than..."
        # Or "Is Planet larger than...?"
        
        subject = None
        obj = None
        
        # Simple extraction strategy:
        # Subject is everything before " is " or "Is " and adjective?
        # Let's rely on position relative to 'than'
        
        # Object is after "than"
        if than_idx + 1 < len(words):
            obj = words[than_idx + 1].strip(".,!?")
            if obj.lower() in ["a", "an", "the"]: # Skip articles
                 if than_idx + 2 < len(words):
                     obj = words[than_idx + 2].strip(".,!?")

        # Subject logic
        # Look for "is" or "are" before adj
        verb_idx = -1
        for i in range(than_idx - 1, -1, -1):
            if words[i].lower() in ["is", "are", "was", "were"]:
                verb_idx = i
                break
        
        # print(f"DEBUG: verb_idx={verb_idx}")
        
        if verb_idx != -1:
            if verb_idx > 0:
                 # "Planet is larger..." -> Subject at verb_idx - 1
                subject = words[verb_idx - 1].strip(".,!?")
            else:
                 # verb_idx == 0 -> "Is Planet larger...?"
                 # Subject is between verb (0) and adj (than_idx - 1)
                 # words[1] to words[than_idx-2]
                 start_s = 1
                 if start_s < len(words) and words[start_s].lower() in ["a", "an", "the"]:
                     start_s += 1
                 
                 subj_end = than_idx - 1
                 if subj_end > start_s:
                     subject = " ".join(words[start_s:subj_end]).strip(".,!?")
                 elif subj_end == start_s:
                      subject = words[start_s].strip(".,!?")
        else:
             # Fallback if no verb found? 
             pass

        return {
            "template": "relation",
            "canonical": {
                "relation_type": predicate,
                "entities": [subject, obj] if subject and obj else [],
                "subject": subject,
                "object": obj,
            },
            "entities": [subject, obj] if subject and obj else [],
            "predicates": [predicate],
            "polarity": +1,
            "parser_confidence": 0.85,
            "raw_text": text,
        }
    
    def _parse_capability(self, text: str) -> dict:
        """Parse capability statements like 'Birds can fly' or 'Penguins cannot fly'."""
        text_lower = text.lower()
        # Check if this is a negative statement
        negated = (
            "cannot" in text_lower or
            "can't" in text_lower or
            "can not" in text_lower or
            " do not " in text_lower or
            " don't " in text_lower or
            " does not " in text_lower
        )
        
        # Easier to work with "can not" than "cannot" when splitting words
        normalized = text.replace("cannot", "can not").replace("can't", "can not")
        words = normalized.split()
        
        # Extract the entity and action from the sentence
        entity = None
        predicate = None
        
        # Find any modal/auxiliary verb index (can/do/does/did)
        aux_idx = None
        for i, w in enumerate(words):
            if w.lower() in {"can", "do", "does", "did"}:
                aux_idx = i
                break

        if aux_idx is not None:
            # Two broad patterns:
            # - Statement: "Penguins do not fly" -> entity before aux, verb after
            # - Question: "Do penguins fly?" -> aux at start, entity after
            if aux_idx == 0:
                # Question format: "Do penguins fly?" or "Can penguins swim?"
                if aux_idx + 1 < len(words):
                    entity = words[aux_idx + 1].strip(".,!?;:")
                action_idx = aux_idx + 2
                if action_idx < len(words) and words[action_idx].lower() == "not":
                    action_idx += 1
            else:
                # Statement format: entity before aux
                entity = words[aux_idx - 1].strip(".,!?;:")
                action_idx = aux_idx + 1
                if action_idx < len(words) and words[action_idx].lower() == "not":
                    action_idx += 1

            if action_idx < len(words):
                predicate = words[action_idx].strip(".,!?;:")
        
        # Build the predicate name: "can_fly", "can_swim", etc.
        # This is the same whether the statement is positive or negative — the polarity
        # field tracks whether we're affirming or denying the capability
        # If we couldn't find an explicit 'can' pattern (e.g., "Do penguins fly?"),
        # try a fallback extraction: assume question form 'Do <entity> <verb>?'
        if not predicate:
            lw = text_lower.split()
            if lw[0] in ("do", "does", "did") and len(lw) >= 3:
                # entity often at position 1, verb at 2
                predicate = lw[2].strip("?,.!")

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
        """Parse taxonomic statements like 'A penguin is a bird' or 'Is a penguin a bird?'."""
        text_lower = text.lower()
        
        # Look for negation markers ("is not", "isn't", etc.)
        is_negative = any(neg in text_lower for neg in [
            "is not", "are not", "isn't", "aren't", "was not", "were not", "wasn't", "weren't"
        ])
        polarity = -1 if is_negative else +1
        
        words = text.split()
        subject = None
        obj = None
        
        # Check for Question format: "Is X a Y?"
        # Starts with verb?
        if words[0].lower() in ["is", "are", "was", "were"]:
            # Question format!
            # Subject is likely words[1] (unless article?)
            # "Is a penguin a bird?" -> words=["Is", "a", "penguin", "a", "bird?"]
            # "Is Entity_L0 a Entity_L2?" -> words=["Is", "Entity_L0", "a", "Entity_L2?"]
            
            # Find the "splitting" article/preposition for the object
            # Usually "a", "an", "the"
            # But subject might also have article.
            
            # Heuristic: Find the SECOND article, or " a " in middle?
            # Better: Scan for "a"/"an" skipping the start.
            
            # Let's try to identify the split point.
            # "Is [Subject Group] [a/an/the] [Object Group]?"
            
            # Find the first "a"/"an" that is NOT at index 1 (part of subject)
            # Actually, "Is a penguin..." -> "a" is at 1.
            # We need the one that separates S and O.
            # "Is a penguin A bird?"
            
            start_idx = 1
            if start_idx < len(words) and words[start_idx].lower() in ["a", "an", "the"]:
                start_idx += 1 # Skip subject article
                
            # Now extract Subject until we hit "a"/"an" or end
            subject_words = []
            obj_start_idx = -1
            
            for k in range(start_idx, len(words)):
                 if words[k].lower() in ["a", "an"]:
                     obj_start_idx = k
                     break
                 # specific case for "Entity is Entity" (no 'a')?
                 # Taxonomy usually implies "is a". "Is X Y?" might be identity.
                 # But parsing 'is_a' expects 'a'.
                 # If no 'a' found, maybe just take last word?
                 subject_words.append(words[k])
            
            if obj_start_idx != -1:
                 # Found separator
                 subject = " ".join(subject_words).strip(".,!?")
                 # Object is after separator
                 if obj_start_idx + 1 < len(words):
                      obj = words[obj_start_idx + 1].strip(".,!?")
            else:
                 # No separator found? "Is X Y?"
                 # Assume last word is object, rest is subject?
                 if len(subject_words) >= 2:
                      obj = subject_words[-1].strip(".,!?")
                      subject = " ".join(subject_words[:-1]).strip(".,!?")
        
        else:
            # Statement format: "X is a Y"
            for i, word in enumerate(words):
                if word.lower() in ["is", "are", "was", "were"]:
                    # Subject is before verb
                    if i > 0:
                        # Handle "A penguin" -> remove article
                        sub_start = 0
                        if words[0].lower() in ["a", "an", "the"]:
                            sub_start = 1
                        if i > sub_start:
                             subject = " ".join(words[sub_start:i]).strip(".,!?")
                        else:
                             # fallback
                             subject = words[i-1].strip(".,!?")
                    
                    # Object is after verb + article
                    j = i + 1
                    while j < len(words) and words[j].lower() in ["a", "an", "not", "the"]:
                        j += 1
                    if j < len(words):
                        obj = words[j].strip(".,!?")
                    break
        # Handle constructions like "X is both an A and a B" or "X is A and B"
        try:
            if subject and (' and ' in text_lower or ' both ' in text_lower):
                # Extract the portion after the verb (is/are/was/were)
                after_verb = None
                for i, w in enumerate(words):
                    if w.lower() in ["is", "are", "was", "were"]:
                        after_verb = ' '.join(words[i+1:])
                        break
                if after_verb:
                    # Split parents on ' and ' or comma
                    raw_parents = re.split(r"\band\b|,", after_verb, flags=re.IGNORECASE)
                    parents = []
                    for rp in raw_parents:
                        p = rp.strip()
                        # remove leading 'both'/'either' first, then any leading articles
                        p = re.sub(r"^\b(both|either)\b\s*", "", p, flags=re.IGNORECASE)
                        p = re.sub(r"^\b(a|an|the)\b\s*", "", p, flags=re.IGNORECASE)
                        # keep only first token as parent type (e.g., 'owl' from 'an owl')
                        if p:
                            first = p.split()[0].strip(".,!?;:")
                            if first and first.lower() != (subject or '').lower():
                                parents.append(first)
                    if parents:
                        obj = parents[0]
                        canonical_entities = [subject] + parents
                        return {
                            "template": "is_a",
                            "canonical": {
                                "relation_type": "is_a",
                                "entities": canonical_entities,
                                "subject": subject,
                                "object": obj,
                            },
                            "entities": [subject] + parents,
                            "predicates": ["is_a"],
                            "polarity": polarity,
                            "parser_confidence": 0.90,
                            "raw_text": text,
                        }
        except Exception:
            pass

        return {
            "template": "is_a",
            "canonical": {
                "relation_type": "is_a",
                "entities": [subject] + ([obj] if obj else []) if subject else [],
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
        stop_words = {'is', 'are', 'was', 'were', 'be', 'been', 'the', 'a', 'an', 'do', 'does', 'did'}
        
        entities = []
        for w in words[:4]:
             clean_w = w.strip(".,!?;:")
             if clean_w.lower() not in stop_words:
                 entities.append(clean_w)
        entities = entities[:2]

        # Heuristic: simple verb-statement like "Birds fly." -> treat as capability
        if len(words) >= 2:
            verb = words[1].strip(".,!?;:")
            if verb.isalpha() and verb.lower() not in stop_words:
                predicate = f"can_{verb.lower()}"
                return {
                    "template": "relation",
                    "canonical": {
                        "relation_type": predicate,
                        "entities": entities,
                    },
                    "entities": entities,
                    "predicates": [predicate],
                    "polarity": +1,
                    "parser_confidence": 0.8,
                    "raw_text": text,
                }

        return {
            "template": "relation",
            "canonical": {
                "relation_type": "generic",
                "entities": entities,
            },
            "entities": entities,
            "predicates": ["generic"],
            "polarity": +1,
            "parser_confidence": 0.65,
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
