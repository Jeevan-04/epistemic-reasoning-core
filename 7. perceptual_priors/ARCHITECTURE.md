# Perceptual Priors — Observational Knowledge

**STATUS**: AUXILIARY KNOWLEDGE SOURCE (not core module)

**MARC Architecture**:
- **CORE MODULES** (1-6): Manas, Buddhi, Chitta, HRE, Ahankara, Sakshin — system cannot function without these
- **AUXILIARY KNOWLEDGE SOURCES** (7-8): Perceptual Priors, Geographic Memory — optional external knowledge that enriches reasoning

---

## What are Perceptual Priors?

**Perceptual Priors** are MARC's **non-inferable observational knowledge** — facts you learn through DIRECT OBSERVATION, not reasoning.

Think of Perceptual Priors like knowing "the sky is blue" — you didn't INFER this from taxonomy or logic. You just OBSERVED it.

**Critical Distinction**: Perceptual Priors are NOT a core reasoning module. They are an **auxiliary knowledge source** that plugs into the core architecture. MARC can function without Perceptual Priors (it would just refuse perceptual queries and fall back to Buddhi).

---

## The Core Problem: Why Do We Need Perceptual Priors?

### The Human Analogy

When you're asked "Is gold shiny?", your brain doesn't reason:

```
❌ INFERENCE APPROACH (Wrong):
1. Gold is a metal
2. Metals are shiny
3. Therefore, gold is shiny

Problem: You never learned "metals are shiny" as a general rule.
```

Instead, your brain RECALLS:

```
✓ OBSERVATIONAL APPROACH (Correct):
1. I have SEEN gold
2. I OBSERVED it was shiny
3. Therefore, gold is shiny

This is PERCEPTUAL MEMORY, not logical inference.
```

**Key Insight**: Some knowledge is **observational** (learned through senses), not **inferential** (derived through reasoning).

### The Grounding Problem (Without Perceptual Priors)

**Scenario**: System taught "Gold is shiny" as a fact.

**Without Perceptual Priors**:
```python
# Store as normal belief
chitta.add(Belief(
    entities=['gold'],
    predicates=['shiny'],
    confidence=0.9,
    source='user_input'
))

# Query: "Is gold shiny?"
buddhi.answer("Is gold shiny?")
→ "Yes" (retrieves belief)

# But now query: "Is shiny gold metallic?"
buddhi.answer("Is shiny gold metallic?")
→ Tries to compose "shiny" + "gold" → "shiny gold"
→ Might HALLUCINATE answer via unbounded composition

# Problem: System doesn't know "shiny" is PERCEPTUAL property
# (Can't be composed into new entities)
```

**With Perceptual Priors**:
```python
# Store as perceptual prior
perceptual_priors.add_property('gold', 'shiny', confidence=0.85)

# Query: "Is gold shiny?"
perceptual_priors.has_property('gold', 'shiny')
→ "Yes (Perceptual observation, confidence: 0.85)"

# Query: "Is shiny gold metallic?"
# Manas parses: entities=['shiny', 'gold'], predicate='metallic'
perceptual_priors.has_property('shiny gold', 'metallic')
→ None (entity 'shiny gold' not in perceptual index)
→ Falls back to Buddhi, which refuses (ungrounded)
→ "I do not know"

# Benefit: Perceptual properties don't participate in composition
# → No hallucination ✓
```

---

## Architecture: How Perceptual Priors Work

### High-Level Structure

```
┌───────────────────────────────────────────────────────────────┐
│              PERCEPTUAL PRIORS (Observation Store)            │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         ENTITY-PROPERTY INDEX                           │  │
│  │                                                         │  │
│  │  {                                                      │  │
│  │    'gold': {                                            │  │
│  │      'shiny': 0.85,          # Observed: gold is shiny │  │
│  │      'metallic': 0.90        # Observed: gold is metal │  │
│  │    },                                                   │  │
│  │    'sky': {                                             │  │
│  │      'blue': 0.95            # Observed: sky is blue   │  │
│  │    },                                                   │  │
│  │    'water': {                                           │  │
│  │      'liquid': 0.98,         # Observed: water liquid  │  │
│  │      'transparent': 0.90     # Observed: water clear   │  │
│  │    }                                                    │  │
│  │  }                                                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  PROPERTIES:                                                  │
│  • Non-Inferable: Can't be derived from taxonomy             │
│  • Non-Inheritable: Properties don't flow down hierarchy     │
│  • Direct Observation: Learned through senses, not reasoning │
│  • Confidence Ceiling: Max 85% (perceptual uncertainty)      │
│                                                               │
│  QUERY INTERFACE:                                             │
│  • has_property(entity, property) → confidence or None       │
│  • add_property(entity, property, confidence)                │
│  • format_answer(entity, property, confidence) → string      │
└───────────────────────────────────────────────────────────────┘
```

---

## Deep Dive: The Core Principles

### Principle 1: Non-Inferable

**Definition**: Perceptual properties CANNOT be derived from other knowledge.

**Examples**:

```python
# INFERABLE (not perceptual):
"Bats produce milk"
→ Can be derived: bat → mammal → produces milk
→ This is REASONING, not observation
→ NOT a perceptual prior

# NON-INFERABLE (perceptual):
"Gold is shiny"
→ CANNOT be derived from "gold is metal"
→ You must OBSERVE gold to know it's shiny
→ This IS a perceptual prior
```

**Mathematical Property**:

$$
\text{Perceptual}(p) \iff \nexists \text{Beliefs } B : B \vdash p
$$

(Property $p$ is perceptual if NO beliefs $B$ can derive it)

**Implementation**:
```python
def is_perceptual(property_name: str) -> bool:
    """
    Check if property is perceptual (observational).
    
    Perceptual properties:
      - Sensory: shiny, blue, loud, smooth, sweet
      - Physical states: liquid, solid, hot, cold
      - Appearances: transparent, opaque, bright
    
    Non-perceptual properties:
      - Taxonomic: is_a, instance_of
      - Functional: produces_X, has_X (can inherit)
      - Relational: located_in, part_of
    """
    perceptual_markers = [
        # Visual
        'shiny', 'bright', 'transparent', 'opaque', 'color',
        'blue', 'red', 'green', 'yellow',
        
        # Physical state
        'liquid', 'solid', 'gas',
        
        # Texture
        'smooth', 'rough', 'soft', 'hard',
        
        # Thermal
        'hot', 'cold', 'warm',
        
        # Auditory
        'loud', 'quiet',
        
        # Gustatory
        'sweet', 'sour', 'bitter'
    ]
    
    prop_lower = property_name.lower()
    
    for marker in perceptual_markers:
        if marker in prop_lower:
            return True
    
    return False
```

### Principle 2: Non-Inheritable

**Definition**: Perceptual properties DO NOT flow down taxonomic hierarchies.

**Example**:

```python
# Suppose we taught: "Metals are shiny" (hypothetical)
# And: "Gold is a metal"

# Traditional inheritance (WRONG):
"Is gold shiny?"
→ gold → metal → shiny (inherited)
→ "Yes"

# Problem: Not all metals are shiny!
# (Iron can be rusty/dull, aluminum can be oxidized)

# Perceptual approach (CORRECT):
"Is gold shiny?"
→ Check perceptual_priors.has_property('gold', 'shiny')
→ If stored: "Yes" (observed directly)
→ If not stored: "I do not know" (refuse to infer)

# Benefit: No false inheritance
```

**Why Non-Inheritable?**

**Observational properties are INSTANCE-SPECIFIC**, not CATEGORY-GENERAL.

- "This gold sample is shiny" (specific observation)
- ≠ "All metals are shiny" (false generalization)

**Implementation**:
```python
# Perceptual priors DO NOT check taxonomy
def has_property(entity: str, property: str) -> Optional[float]:
    """
    Check if entity has perceptual property.
    
    DOES NOT check ancestors (no inheritance).
    """
    entity_lower = entity.lower()
    property_lower = property.lower()
    
    # Direct lookup only
    if entity_lower in self.properties:
        return self.properties[entity_lower].get(property_lower)
    
    return None  # Not found (NO inheritance attempt)
```

### Principle 3: Confidence Ceiling

**Definition**: Perceptual observations have MAX 85% confidence (epistemic humility).

**Why 85%?**

**Perceptual uncertainty**:
- Lighting conditions vary
- Observer memory fades
- Properties can change (oxidation, tarnish)
- Subjective perception ("Is this blue or teal?")

**Mathematical Property**:

$$
\forall p \in \text{Perceptual}: \text{confidence}(p) \leq 0.85
$$

**Implementation**:
```python
def add_property(self, entity: str, property: str, confidence: float = 0.85):
    """
    Add perceptual property observation.
    
    Enforces confidence ceiling.
    """
    entity_lower = entity.lower()
    property_lower = property.lower()
    
    # Enforce confidence ceiling
    confidence = min(confidence, 0.85)  # ← MAX 85%
    
    # Store
    if entity_lower not in self.properties:
        self.properties[entity_lower] = {}
    
    self.properties[entity_lower][property_lower] = confidence
```

**Example**:
```python
# Attempt to add with high confidence
perceptual_priors.add_property('gold', 'shiny', confidence=0.99)

# Actual stored confidence: 0.85 (capped)
perceptual_priors.has_property('gold', 'shiny')
→ 0.85  # Not 0.99
```

---

## Deep Dive: The Query Algorithm

### Entity-Property Matching

**Goal**: Check if entity has perceptual property.

**Algorithm**:
```python
class PerceptualPriors:
    def __init__(self):
        self.properties: Dict[str, Dict[str, float]] = {}
    
    def has_property(self, entity: str, property: str) -> Optional[float]:
        """
        Check if entity has perceptual property.
        
        Returns:
          Confidence (float) if property observed
          None if not observed
        """
        entity_lower = entity.lower()
        property_lower = property.lower()
        
        # Direct lookup (no inference, no inheritance)
        if entity_lower in self.properties:
            return self.properties[entity_lower].get(property_lower)
        
        return None
    
    def format_answer(self, entity: str, property: str, confidence: float) -> str:
        """
        Format perceptual answer.
        
        Emphasizes observational nature.
        """
        return f"Yes (Perceptual observation, confidence: {confidence:.2f})"
```

**Example**:
```python
# Query: "Is gold shiny?"
entities = ['gold']
property = 'shiny'

confidence = perceptual_priors.has_property('gold', 'shiny')
→ 0.85

answer = perceptual_priors.format_answer('gold', 'shiny', 0.85)
→ "Yes (Perceptual observation, confidence: 0.85)"
```

### Entity-Pair Handling (Compound Queries)

**Problem**: Manas sometimes parses properties as entities.

```python
Query: "Is gold shiny?"

Manas parsing:
  entities = ['gold', 'shiny']  # ← 'shiny' parsed as entity!
  predicates = ['generic']
```

**Solution**: Check both single entity and entity pairs.

**Algorithm**:
```python
def check_query(self, entities: List[str], predicates: List[str]) -> Optional[str]:
    """
    Check if query matches perceptual prior.
    
    Handles:
      - Single entity + property predicate: "Is gold shiny?"
        entities=['gold'], predicates=['shiny']
      
      - Entity pair + generic predicate: "Is gold shiny?"
        entities=['gold', 'shiny'], predicates=['generic']
    """
    # Case 1: Property in predicates
    if len(entities) == 1 and predicates:
        entity = entities[0]
        for predicate in predicates:
            if is_perceptual(predicate):
                confidence = self.has_property(entity, predicate)
                if confidence:
                    return self.format_answer(entity, predicate, confidence)
    
    # Case 2: Property in entities (parsed as entity)
    if len(entities) == 2 and 'generic' in predicates:
        # Check both orderings
        # "Is gold shiny?" → entities=['gold', 'shiny']
        # "Is shiny gold?" → entities=['shiny', 'gold']
        
        entity1, entity2 = entities
        
        # Try: entity1 has property entity2
        if is_perceptual(entity2):
            confidence = self.has_property(entity1, entity2)
            if confidence:
                return self.format_answer(entity1, entity2, confidence)
        
        # Try: entity2 has property entity1
        if is_perceptual(entity1):
            confidence = self.has_property(entity2, entity1)
            if confidence:
                return self.format_answer(entity2, entity1, confidence)
    
    return None  # Not a perceptual query
```

**Example**:
```python
# Query: "Is gold shiny?"
# Manas parses: entities=['gold', 'shiny'], predicates=['generic']

check_query(['gold', 'shiny'], ['generic'])
→ Tries: has_property('gold', 'shiny') → 0.85 ✓
→ Returns: "Yes (Perceptual observation, confidence: 0.85)"
```

---

## Integration with MARC

### Priority in Query Resolution

**Ahankara's Query Pipeline**:

```python
def query_answer(self, query_text: str) -> str:
    # Phase 1: Perception
    proposal = self.manas.parse(query_text)
    
    # Phase 2a: PERCEPTUAL PRIORS (Priority 1)
    if self.perceptual_priors:
        answer = self.perceptual_priors.check_query(
            proposal['entities'],
            proposal['predicates']
        )
        if answer:
            return answer  # ← EARLY RETURN (highest priority)
    
    # Phase 2b: Geographic Memory (Priority 2)
    # ...
    
    # Phase 3: Buddhi (Priority 3)
    # ...
```

**Why Highest Priority?**

**Epistemology**:
$$
\text{Observation} > \text{Inference}
$$

If you've OBSERVED something, trust that over DERIVED knowledge.

**Example**:
```python
# Suppose (hypothetically):
# - Perceptual prior: "Gold is shiny" (confidence: 0.85)
# - Inference: "Gold is dull" (via some weird reasoning)

Query: "Is gold shiny?"

# WITHOUT priority:
buddhi.answer("Is gold shiny?")
→ Might return "No" (via faulty inference)

# WITH priority:
perceptual_priors.check_query(...)
→ Returns "Yes (Perceptual observation, confidence: 0.85)"
→ EARLY RETURN (buddhi never called)

# Benefit: Observation trumps inference ✓
```

---

## What Perceptual Priors Are NOT

### NOT a Reasoning Engine

```python
# Perceptual priors do NOT reason
perceptual_priors.has_property('bat', 'produces_milk')
→ None (not stored, no inference attempted)

# Buddhi would reason:
buddhi.answer("Do bats produce milk?")
→ "Yes" (via taxonomic inheritance)

# Perceptual priors: lookup only, no derivation
```

### NOT Inheritable

```python
# Even if stored:
perceptual_priors.add_property('mammal', 'furry', 0.85)

# Query about descendant:
perceptual_priors.has_property('bat', 'furry')
→ None (no inheritance, even though bat → mammal)

# This is INTENTIONAL (no false generalization)
```

### NOT Compositional

```python
# Perceptual priors stored:
# - 'gold' → 'shiny'
# - 'gold' → 'metallic'

# Query: "Is shiny gold valuable?"
# Manas: entities=['shiny', 'gold'], predicates=['valuable']

perceptual_priors.check_query(['shiny', 'gold'], ['valuable'])
→ None (entity 'shiny gold' not stored)

# Benefit: No composition → No hallucination ✓
```

---

## Summary: Perceptual Priors in One Diagram

```
╔═══════════════════════════════════════════════════════════════╗
║              PERCEPTUAL PRIORS (Observational Knowledge)      ║
║                                                               ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │         ENTITY-PROPERTY INDEX (Direct Observation)       │ ║
║  │                                                          │ ║
║  │  gold → {shiny: 0.85, metallic: 0.90}                   │ ║
║  │  sky → {blue: 0.95}                                      │ ║
║  │  water → {liquid: 0.98, transparent: 0.90}               │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                           │                                   ║
║                           ▼                                   ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │  QUERY: has_property(entity, property)                   │ ║
║  │  • Direct lookup (O(1))                                  │ ║
║  │  • NO inference                                          │ ║
║  │  • NO inheritance                                        │ ║
║  │  • NO composition                                        │ ║
║  └──────────────────┬───────────────────────────────────────┘ ║
║                     │                                         ║
║                     ▼                                         ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │  ANSWER (if found)                                       │ ║
║  │  "Yes (Perceptual observation, confidence: 0.85)"       │ ║
║  │                                                          │ ║
║  │  REFUSAL (if not found)                                 │ ║
║  │  None → Falls back to Buddhi                            │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                               ║
║  THREE PRINCIPLES:                                            ║
║   1. Non-Inferable (can't be derived)                        ║
║   2. Non-Inheritable (don't flow down taxonomy)              ║
║   3. Confidence Ceiling (max 85%)                            ║
║                                                               ║
║  INTEGRATION:                                                 ║
║   Priority 1 in Ahankara's query pipeline                    ║
║   (Observation > Inference)                                  ║
║                                                               ║
║  EXAMPLES:                                                    ║
║   ✓ "Is gold shiny?" → Perceptual                           ║
║   ✓ "Is water liquid?" → Perceptual                         ║
║   ✗ "Do bats produce milk?" → NOT perceptual (inference)    ║
║   ✗ "Is London in Europe?" → NOT perceptual (geographic)    ║
║                                                               ║
║  PHILOSOPHY:                                                  ║
║   "Observation ≠ Inference"                                  ║
║   "Some knowledge is SEEN, not DERIVED"                      ║
║   "Epistemic humility: Perception has limits (≤85%)"         ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**Key Insight**: Perceptual Priors are MARC's "sensory memory" — facts learned through observation, stored without inference, retrieved without reasoning. They represent the boundary between perception and cognition.

**IMPORTANT**: Perceptual Priors are an **AUXILIARY** module, not CORE. The 6 core modules (Manas, Buddhi, Chitta, HRE, Ahankara, Sakshin) are sufficient for MARC to function. Perceptual Priors ENRICH the system with observational knowledge but are NOT required for basic operation.

**Perceptual Priors are FROZEN**: This is the production observational knowledge architecture.
