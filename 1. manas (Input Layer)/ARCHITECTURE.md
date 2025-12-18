# Manas — The Input Layer (मनस्)

## What is Manas?

**Manas** (Sanskrit: मनस्, "mind" or "perception") is the **perceptual gateway** of MARC. It transforms raw natural language into structured meaning that the reasoning engine (Buddhi) can work with.


Think of Manas like your **sensory cortex** — it doesn't think or reason, it just perceives and structures what it sees.

---

## The Core Problem: Why Do We Need Manas?

### The Human Analogy

When you hear "Bats are mammals," your brain doesn't process it as a raw string of letters. It immediately structures it:

```
PERCEPTION LAYER (Manas):
Input: "Bats are mammals"
↓
Structured Understanding:
- Subject: "bat"
- Relation: IS-A (taxonomic)
- Object: "mammal"
- Polarity: POSITIVE
- Confidence: HIGH (simple declarative)
```

Your reasoning brain (Buddhi) then receives **structured facts**, not **raw text**.

### Why Not Skip This Step?

**❌ Without Manas** (feeding raw text to reasoning):
```python
buddhi.reason("Do bats produce milk?")
# Buddhi has to:
# 1. Parse the question
# 2. Extract entities
# 3. Identify the query type
# 4. THEN reason
# Result: Logic contaminated with parsing concerns
```

**✅ With Manas** (separation of concerns):
```python
# Manas handles perception
proposal = manas.parse("Do bats produce milk?")
# → {entities: ['bat'], predicates: ['produces_milk'], query: True}

# Buddhi handles ONLY logic
answer = buddhi.answer(proposal)
# → Pure reasoning, no parsing
```

**Key Insight**: Humans separate perception from reasoning. MARC does too.

---

## Architecture: How Manas Works

### High-Level Flow

```
┌─────────────────────────────────────────────┐
│         NATURAL LANGUAGE INPUT              │
│     "Bats are mammals"                      │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│         MANAS (Perception Layer)            │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  1. Entity Extraction                │   │
│  │     "Bats" → "bat" (normalized)      │   │
│  │     "mammals" → "mammal"             │   │
│  └─────────────┬───────────────────────┘   │
│                │                            │
│  ┌─────────────▼───────────────────────┐   │
│  │  2. Relation Detection               │   │
│  │     "are" → IS-A (taxonomic)         │   │
│  └─────────────┬───────────────────────┘   │
│                │                            │
│  ┌─────────────▼───────────────────────┐   │
│  │  3. Polarity Detection               │   │
│  │     No negation → POSITIVE           │   │
│  └─────────────┬───────────────────────┘   │
│                │                            │
│  ┌─────────────▼───────────────────────┐   │
│  │  4. Template Classification          │   │
│  │     IS-A pattern → taxonomic         │   │
│  └─────────────┬───────────────────────┘   │
│                │                            │
│  ┌─────────────▼───────────────────────┐   │
│  │  5. Confidence Estimation            │   │
│  │     Simple statement → 0.9           │   │
│  └─────────────────────────────────────┘   │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│       STRUCTURED BELIEF PROPOSAL            │
│                                             │
│  {                                          │
│    entities: ['bat', 'mammal'],             │
│    predicates: ['is_a'],                    │
│    polarity: POSITIVE,                      │
│    template: 'taxonomic',                   │
│    confidence: 0.9,                         │
│    canonical: {...}                         │
│  }                                          │
└─────────────┬───────────────────────────────┘
              │
              ▼
         TO BUDDHI (Reasoning)
```

---

## Deep Dive: The Algorithm

### Step 1: Entity Extraction

**Goal**: Find the "things" being talked about.

**Human Analog**: When you hear "Dogs chase cats," you immediately identify TWO entities: dog, cat.

**Algorithm**:
```python
def extract_entities(text: str) -> List[str]:
    """
    Extract and normalize entities.
    
    Normalization rules:
    1. Lowercase (Bats → bat)
    2. Singularize (bats → bat, mice → mouse)
    3. Remove determiners (the bat → bat)
    """
    # Simplified version
    words = text.lower().split()
    entities = []
    
    for word in words:
        # Skip stop words
        if word in ['is', 'are', 'the', 'a', 'an']:
            continue
        
        # Normalize plural → singular
        normalized = singularize(word)  # bats → bat
        entities.append(normalized)
    
    return entities
```

**Why Normalize?**

Humans understand that "bat", "bats", "Bat", "BATS" all refer to the SAME concept. Manas does the same.

```
Input:   "Bats are mammals"
         "A bat is a mammal"
         "The bat is mammal"
         
All produce: entities = ['bat', 'mammal']
```

**Why This Matters**: Buddhi can now recognize that "bat" in different contexts is the SAME entity.

### Step 2: Relation Detection

**Goal**: Identify HOW entities relate to each other.

**Human Analog**: In "Bats are mammals," you immediately recognize this is a **taxonomic** (IS-A) relation, not a spatial or temporal relation.

**Relation Types**:

| Pattern | Relation Type | Example |
|---------|--------------|---------|
| "X is Y" | TAXONOMIC | "Bat is mammal" |
| "X has Y" | FUNCTIONAL | "Bat has wings" |
| "X in Y" | SPATIAL | "Bat in cave" |
| "X produces Y" | FUNCTIONAL | "Mammal produces milk" |

**Algorithm**:
```python
def detect_relation(text: str) -> str:
    """
    Detect relation type from linguistic patterns.
    
    Uses pattern matching on verbs/prepositions.
    """
    text_lower = text.lower()
    
    # Taxonomic patterns
    if any(pattern in text_lower for pattern in ['is a', 'are', 'is']):
        return 'is_a'
    
    # Functional patterns
    if 'has' in text_lower or 'have' in text_lower:
        return 'has_' + extract_property(text)
    
    if 'produces' in text_lower or 'produce' in text_lower:
        return 'produces_' + extract_object(text)
    
    # Default
    return 'generic'
```

**Why Pattern Matching?**

Humans recognize linguistic patterns instantly. "X is Y" → IS-A. "X has Y" → POSSESSION/PROPERTY.

This is **perception**, not **reasoning**. Manas doesn't need to understand WHY bats are mammals, just that the sentence STRUCTURE indicates a taxonomic relation.

### Step 3: Polarity Detection

**Goal**: Determine if the statement is POSITIVE or NEGATIVE.

**Human Analog**: "Bats have wings" (POSITIVE) vs "Bats do not have gills" (NEGATIVE).

**Algorithm**:
```python
def detect_polarity(text: str) -> int:
    """
    Detect statement polarity.
    
    Returns:
        +1 for positive
        -1 for negative
    """
    negation_words = ['not', 'no', 'never', "don't", "doesn't", "cannot"]
    
    text_lower = text.lower()
    
    # Check for negation words
    for neg in negation_words:
        if neg in text_lower:
            return -1  # NEGATIVE
    
    return +1  # POSITIVE (default)
```

**Why This Matters**:

Negation fundamentally changes meaning:
- "Mammals have gills" (FALSE belief)
- "Mammals do NOT have gills" (TRUE belief)

Manas captures this difference structurally.

### Step 4: Template Classification

**Goal**: Categorize the statement type for efficient processing.

**Templates** (like human cognitive schemas):

```
TAXONOMIC:
  Pattern: "X is a Y"
  Example: "Bat is a mammal"
  Schema: {subject: X, relation: IS-A, object: Y}

FUNCTIONAL:
  Pattern: "X has/does Y"
  Example: "Bat has wings"
  Schema: {subject: X, relation: HAS, property: Y}

BEHAVIORAL:
  Pattern: "X does Y"
  Example: "Bat flies"
  Schema: {subject: X, action: Y}
```

**Why Templates?**

Humans use **cognitive schemas** to understand the world. When you hear "X is a Y," your brain activates the IS-A template.

MARC does the same — templates enable efficient, pattern-based processing.

### Step 5: Confidence Estimation

**Goal**: How sure are we about this parse?

**Confidence Factors**:
```python
def estimate_confidence(parse_result: dict) -> float:
    """
    Estimate parsing confidence based on clarity.
    
    High confidence (0.8-1.0):
    - Simple declarative: "Bats are mammals"
    - Clear pattern match
    
    Medium confidence (0.5-0.8):
    - Complex structure: "Bats, which fly, are mammals"
    - Ambiguous phrasing
    
    Low confidence (0.0-0.5):
    - Hypothetical: "If bats were birds..."
    - Uncertain phrasing: "Maybe bats..."
    """
    confidence = 0.9  # Base confidence
    
    # Reduce for complexity
    if len(parse_result['entities']) > 3:
        confidence -= 0.2
    
    # Reduce for hypotheticals
    if any(word in text.lower() for word in ['if', 'maybe', 'perhaps']):
        confidence -= 0.3
    
    # Reduce for ambiguity
    if parse_result['template'] == 'unknown':
        confidence -= 0.4
    
    return max(0.1, confidence)
```

**Why Confidence?**

Humans don't perceive everything with equal certainty. "Bats are mammals" is CLEAR. "Bats might possibly be mammals" is UNCERTAIN.

Manas models this human uncertainty.

---

## Special Feature: Interrogative Handling

### The Challenge: Questions vs Assertions

**Humans distinguish**:
- Statement: "Bats are mammals" → STORE THIS
- Question: "Are bats mammals?" → QUERY THIS

**Manas does the same**:

```python
def classify_intent(text: str) -> str:
    """
    Classify input as assertion or query.
    
    Returns:
        'assertion' or 'query'
    """
    # Question markers
    if text.strip().endswith('?'):
        return 'query'
    
    # Interrogative words at start
    question_words = ['what', 'who', 'where', 'when', 'why', 'how', 
                     'is', 'are', 'do', 'does', 'can', 'will']
    
    first_word = text.lower().split()[0]
    if first_word in question_words:
        return 'query'
    
    return 'assertion'
```

**Subject Recovery for Questions**:

```
Statement: "Bats are mammals"
  Subject: bats
  Predicate: are mammals

Question: "Are bats mammals?"
  Interrogative: "Are"
  RECOVERED Subject: bats  ← Manas extracts this
  Predicate: mammals
```

**Why This Matters**:

Buddhi needs to know:
- **Assertions** → Add to memory
- **Queries** → Search memory

Manas provides this distinction.

---

## The Math: Why Conservative Perception?

### Design Philosophy: False Negatives > False Positives

**Manas follows this principle**:

$$
P(\text{reject valid input}) > P(\text{accept invalid input})
$$

**Why?**

**Option A: Permissive Parsing** (accept everything)
```
Input: "jdkfj sdjfk sdfjk" → Parse anyway?
Risk: Garbage in → Buddhi reasons about nonsense
```

**Option B: Conservative Parsing** (reject unclear input)
```
Input: "Bats... uh... like... maybe mammals?"
Manas: confidence = 0.3 → Mark as UNCERTAIN
Buddhi: Treats with appropriate skepticism
```

**Human Analogy**:

When you hear garbled speech, you say "I didn't catch that" rather than inventing what you think was said.

Manas does the same: **better to admit uncertainty than hallucinate structure**.

### Confidence Threshold

```python
PARSE_CONFIDENCE_THRESHOLD = 0.4

if confidence < PARSE_CONFIDENCE_THRESHOLD:
    return {
        'status': 'uncertain',
        'reason': 'Unable to parse with confidence',
        'raw_text': text
    }
```

**Result**: Manas refuses to parse gibberish, protecting Buddhi from contamination.

---

## Pluralization: The Hidden Complexity

### The Problem

English pluralization is **irregular**:

```
Regular:    bat → bats (add 's')
Irregular:  mouse → mice (vowel change)
            fish → fish (no change)
            analysis → analyses (Greek origin)
```

### Why This Matters

```
Input 1: "Bats are mammals"
Input 2: "A bat is a mammal"

Without normalization:
  Entity in Input 1: "bats"
  Entity in Input 2: "bat"
  Buddhi sees: TWO DIFFERENT ENTITIES ❌

With normalization:
  Both → "bat"
  Buddhi sees: SAME ENTITY ✓
```

### The Algorithm

```python
def singularize(word: str) -> str:
    """
    Convert plural to singular.
    
    Handles:
    - Regular plurals (bats → bat)
    - -es endings (boxes → box)
    - -ies endings (categories → category)
    - -ves endings (wolves → wolf)
    - Irregular (mice → mouse, fish → fish)
    """
    word_lower = word.lower()
    
    # Irregular forms (lookup table)
    irregulars = {
        'mice': 'mouse',
        'geese': 'goose',
        'children': 'child',
        'fish': 'fish'
    }
    
    if word_lower in irregulars:
        return irregulars[word_lower]
    
    # -ies → -y (categories → category)
    if word_lower.endswith('ies'):
        return word_lower[:-3] + 'y'
    
    # -ves → -f (wolves → wolf)
    if word_lower.endswith('ves'):
        return word_lower[:-3] + 'f'
    
    # -es → check if base ends in s,x,z,ch,sh
    if word_lower.endswith('es'):
        return word_lower[:-2]
    
    # -s → remove (bats → bat)
    if word_lower.endswith('s'):
        return word_lower[:-1]
    
    return word
```

**Why Not Use a Library?**

1. **Transparency**: Every transformation is explicit
2. **Control**: We know EXACTLY what happens
3. **Auditability**: No black-box dependency
4. **Minimalism**: Only handle what we need

---

## Predicate Normalization

### The Challenge: Compound Predicates

Human language uses compound forms:

```
"Bats produce milk"    → predicate: produces_milk
"Eagles have beaks"    → predicate: has_beaks
"Fish breathe water"   → predicate: breathes_water
```

### Why Preserve Compounds?

**Option A: Split everything**
```
"Bats produce milk" → entities: [bat, milk], predicate: produce
Problem: Lost the connection between "produce" and "milk"
```

**Option B: Compound predicates**
```
"Bats produce milk" → entities: [bat], predicate: produces_milk
Benefit: Relation is atomic and meaningful
```

**MARC uses Option B**: Predicates capture the FULL relation.

### Pattern-Based Normalization

```python
def normalize_predicate(verb: str, obj: str) -> str:
    """
    Create compound predicate from verb + object.
    
    Examples:
        produce + milk → produces_milk
        have + wings → has_wings
        breathe + air → breathes_air
    """
    # Normalize verb to singular third-person
    verb_normalized = singularize_verb(verb)  # produce/produces → produces
    
    # Normalize object
    obj_normalized = singularize(obj)  # wings → wing
    
    # Combine
    return f"{verb_normalized}_{obj_normalized}"
```

**Result**:
- "Mammals produce milk" → `produces_milk`
- "Bats have wings" → `has_wings`

These become **atomic predicates** that Buddhi can reason about.

---

## The Perception Contract

### What Manas Guarantees

1. **Structure**: Every parse produces a consistent schema
2. **Normalization**: Entities are canonicalized (plural → singular)
3. **Confidence**: Every parse has an uncertainty estimate
4. **Intent Classification**: Assertions vs queries are distinguished
5. **Polarity**: Negations are captured structurally

### What Manas Does NOT Do

1. **NO REASONING**: Manas doesn't verify if "Bats are mammals" is TRUE
2. **NO MEMORY**: Manas doesn't store anything
3. **NO INFERENCE**: Manas doesn't derive new facts
4. **NO JUDGMENT**: Manas doesn't evaluate plausibility

**Manas is PURE PERCEPTION**.

---

## Error Handling: When Perception Fails

### Graceful Degradation

```python
def parse(text: str) -> dict:
    """
    Parse with graceful degradation.
    
    Returns a parse even for unparseable input,
    but marks it as low confidence.
    """
    try:
        # Attempt full parse
        result = full_parse(text)
        
        if result['confidence'] > 0.4:
            return result
        else:
            # Low confidence → mark as uncertain
            return {
                'status': 'uncertain',
                'entities': [],
                'predicates': ['unknown'],
                'confidence': result['confidence'],
                'raw_text': text
            }
    except Exception as e:
        # Parse failure → return minimal structure
        return {
            'status': 'failed',
            'entities': [],
            'predicates': ['unknown'],
            'confidence': 0.0,
            'raw_text': text,
            'error': str(e)
        }
```

**Philosophy**: Never crash, always return SOMETHING.

Even if parsing fails, Buddhi gets a structured response (with confidence = 0.0).

---

## Why This Design? (Feynman's Question)

### "Why not just use an LLM to parse?"

**Because LLMs are black boxes.**

With Manas:
- ✅ Every transformation is **explicit**
- ✅ Every decision is **traceable**
- ✅ Every failure mode is **understood**
- ✅ No hallucination risk

With LLM parsing:
- ❌ Opaque transformations
- ❌ Unpredictable failures
- ❌ Hallucination risk
- ❌ No formal guarantees

### "Why not use dependency parsing?"

**Because dependency parsers are complex and fragile.**

```
Full dependency parse:
  "Bats are mammals that fly at night"
  → Complex tree with 15+ nodes
  → Fragile to minor variations
  → Overkill for simple relations

Manas pattern matching:
  "Bats are mammals" → IS-A(bat, mammal)
  → 3 tokens, clear mapping
  → Robust to variations
  → Sufficient for reasoning
```

**KISS principle**: Simple patterns handle 80% of cases.

### "Why separate Manas from Buddhi?"

**Because perception ≠ reasoning** (in humans and in MARC).

```
INTEGRATED (Bad):
  parse_and_reason("Bats are mammals")
  → Tangled concerns
  → Hard to debug
  → Can't verify reasoning separately

SEPARATED (Good):
  proposal = manas.parse("Bats are mammals")
  result = buddhi.reason(proposal)
  → Clean separation
  → Easy to test each layer
  → Can swap Manas implementation
```

**Modularity enables clarity.**

---

## Summary: Manas in One Diagram

```
╔══════════════════════════════════════════════════════════╗
║                   MANAS (Perception)                     ║
║                                                          ║
║  INPUT: Natural Language                                 ║
║    ↓                                                     ║
║  NORMALIZE: Plurals, case, determiners                   ║
║    ↓                                                     ║
║  EXTRACT: Entities, relations, polarity                  ║
║    ↓                                                     ║
║  CLASSIFY: Template, intent, confidence                  ║
║    ↓                                                     ║
║  OUTPUT: Structured Belief Proposal                      ║
║                                                          ║
║  GUARANTEES:                                             ║
║   • Conservative (false negative > false positive)       ║
║   • Transparent (every step explicit)                    ║
║   • Confident (uncertainty quantified)                   ║
║   • Consistent (same input → same output)                ║
║                                                          ║
║  DOES NOT:                                               ║
║   • Reason about truth                                   ║
║   • Store beliefs                                        ║
║   • Make inferences                                      ║
║                                                          ║
║  PHILOSOPHY:                                             ║
║   "Perception is NOT reasoning"                          ║
║   "Structure is NOT meaning"                             ║
║   "Better to admit uncertainty than hallucinate"         ║
╚══════════════════════════════════════════════════════════╝
```

---

**Key Insight**: Manas is the bridge between messy human language and clean logical reasoning. It's not intelligent — it's perceptive. And that's exactly what it should be.
