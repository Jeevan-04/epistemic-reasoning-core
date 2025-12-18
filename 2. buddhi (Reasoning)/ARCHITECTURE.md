# Buddhi — The Reasoning Engine (बुद्धि)

## What is Buddhi?

**Buddhi** (Sanskrit: बुद्धि, "intellect" or "discriminative knowledge") is the **epistemic calculus engine** of MARC. It determines what is TRUE, what is FALSE, and what is UNKNOWN — with formal proofs.

Think of Buddhi like your **logical cortex** — it doesn't perceive or remember, it just REASONS.

---

## The Core Problem: Why Do We Need Buddhi?

### The Human Analogy

When you're asked "Do bats produce milk?", your brain doesn't just retrieve a fact. It REASONS:

```
REASONING CHAIN:
1. Recall: "Bats are mammals" (taxonomic fact)
2. Recall: "Mammals produce milk" (property fact)
3. INFER: Bats inherit mammal properties
4. CONCLUDE: Yes, bats produce milk

This is REASONING, not RETRIEVAL.
```

Your brain follows **inference rules** (taxonomic inheritance) to derive NEW knowledge from EXISTING knowledge.

### Why Not Just Database Lookup?

**❌ Database approach**:
```sql
SELECT answer FROM facts WHERE question = "Do bats produce milk?"
-- Returns: NULL (not directly stored)
```

**✅ Reasoning approach**:
```python
# Known facts:
beliefs = [
    "Bats are mammals",
    "Mammals produce milk"
]

# Inference rule (taxonomic inheritance):
if is_a(X, Y) and has_property(Y, P):
    then has_property(X, P)

# Apply rule:
is_a(bat, mammal) ∧ has_property(mammal, produces_milk)
→ has_property(bat, produces_milk)
# Answer: YES
```

**Key Insight**: Humans don't store every possible fact. They store PRINCIPLES and DERIVE facts through reasoning.

---

## Architecture: How Buddhi Works

### High-Level Flow

```
┌─────────────────────────────────────────────────────┐
│        QUERY: "Do bats produce milk?"               │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│           BUDDHI (Reasoning Engine)                 │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  PHASE 1: DIRECT MATCH                      │   │
│  │  Search: belief("bat", "produces_milk")     │   │
│  │  Result: NOT FOUND                          │   │
│  └─────────────┬───────────────────────────────┘   │
│                │                                    │
│  ┌─────────────▼───────────────────────────────┐   │
│  │  PHASE 2: RELATION FRAME CHECK              │   │
│  │  Predicate: "produces_milk"                 │   │
│  │  Frame: FUNCTIONAL(inherits=True)           │   │
│  │  Decision: Can inherit taxonomically        │   │
│  └─────────────┬───────────────────────────────┘   │
│                │                                    │
│  ┌─────────────▼───────────────────────────────┐   │
│  │  PHASE 3: TAXONOMIC TRAVERSAL               │   │
│  │  Find ancestors of "bat"                    │   │
│  │  Found: bat → mammal                        │   │
│  └─────────────┬───────────────────────────────┘   │
│                │                                    │
│  ┌─────────────▼───────────────────────────────┐   │
│  │  PHASE 4: NEGATION DOMINANCE CHECK          │   │
│  │  Search: mammal produces_milk (NEGATIVE)?   │   │
│  │  Result: NO negation found                  │   │
│  └─────────────┬───────────────────────────────┘   │
│                │                                    │
│  ┌─────────────▼───────────────────────────────┐   │
│  │  PHASE 5: POSITIVE INHERITANCE              │   │
│  │  Search: mammal produces_milk (POSITIVE)?   │   │
│  │  Result: FOUND!                             │   │
│  └─────────────┬───────────────────────────────┘   │
│                │                                    │
│  ┌─────────────▼───────────────────────────────┐   │
│  │  PHASE 6: PROOF CONSTRUCTION                │   │
│  │  Verdict: YES                               │   │
│  │  Derivation:                                │   │
│  │    1. bat IS-A mammal (taxonomic)           │   │
│  │    2. mammal produces_milk (property)       │   │
│  │    3. FUNCTIONAL frame allows inheritance   │   │
│  │  Confidence: 0.9                            │   │
│  └─────────────────────────────────────────────┘   │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│               ANSWER PROOF                          │
│                                                     │
│  Verdict: YES                                       │
│  Reasoning: Taxonomic inheritance via mammal        │
│  Steps: [direct_match → inheritance → found]        │
│  Confidence: 0.9                                    │
└─────────────────────────────────────────────────────┘
```

---

## Deep Dive: The Inference Algorithm

### Step 1: Direct Match (The Fast Path)

**Goal**: Check if the answer is directly stored.

**Human Analog**: When asked "Is 2+2 = 4?", you don't DERIVE this — you directly recall it.

**Algorithm**:
```python
def direct_match(entities: List[str], predicates: List[str]) -> Optional[Belief]:
    """
    Check if belief exists directly in memory.
    
    This is the O(1) lookup path.
    """
    # Generate belief key
    key = canonical_key(entities, predicates)
    
    # Lookup in belief index
    belief = chitta.beliefs.get(key)
    
    if belief and belief.active:
        return belief
    
    return None
```

**Example**:
```python
Query: "Do bats have wings?"
Direct match: belief("bat", "has_wings") → FOUND
Answer: YES (no inference needed)
```

**Why Start Here?**

**Performance**: Direct lookup is O(1). Inference is O(n).

If the answer is stored directly, we skip expensive reasoning.

### Step 2: Relation Frame Check (Structural Semantics)

**Goal**: Determine if this predicate CAN be inherited.

**The Core Innovation**: Not all relations behave the same way.

#### The Relation Frame Theory

**Traditional AI** (flat predicates):
```
All predicates are equal:
  is_a(bat, mammal)
  located_in(london, europe)
  produces_milk(mammal)
  
No distinction in behavior.
```

**MARC** (relation frames):
```
Predicates have INTRINSIC PROPERTIES:

TAXONOMIC (is_a):
  • Transitive: X→Y, Y→Z ⟹ X→Z
  • Inherits: Properties flow down
  • Negation blocks: Negative blocks positive

SPATIAL (located_in):
  • Transitive: X→Y, Y→Z ⟹ X→Z  
  • Does NOT inherit: Location doesn't propagate
  • No negation blocking

FUNCTIONAL (produces_milk, has_wings):
  • NOT transitive
  • Inherits: Properties flow down taxonomy
  • Negation blocks: Negative blocks positive

STATE (is_liquid, is_solid):
  • NOT transitive
  • Does NOT inherit: Context-dependent
  • No negation blocking
```

**The Math**:

$$
\text{RelationFrame}(p) = \{k, t, i, n\}
$$

Where:
- $k$ = kind (TAXONOMIC, SPATIAL, FUNCTIONAL, STATE)
- $t$ = transitive (bool)
- $i$ = inherits (bool)
- $n$ = negation\_blocks (bool)

**Algorithm**:
```python
@dataclass
class RelationFrame:
    kind: RelationKind
    transitive: bool = False
    inherits: bool = False
    negation_blocks: bool = False
    
    @staticmethod
    def for_predicate(predicate: str) -> RelationFrame:
        """
        Map predicate to its structural frame.
        
        This defines HOW the relation behaves.
        """
        pred_lower = predicate.lower()
        
        # TAXONOMIC (is_a, instance_of)
        if pred_lower in ['is_a', 'instance_of']:
            return RelationFrame(
                kind=TAXONOMIC,
                transitive=True,
                inherits=True,
                negation_blocks=True
            )
        
        # FUNCTIONAL (produces_X, has_X, breathes_X)
        if any(pattern in pred_lower for pattern in ['produces', 'has_', 'breathe']):
            return RelationFrame(
                kind=FUNCTIONAL,
                transitive=False,
                inherits=True,      # ← KEY: Allows inheritance
                negation_blocks=True
            )
        
        # SPATIAL (located_in, part_of)
        if any(pattern in pred_lower for pattern in ['located', 'contain']):
            return RelationFrame(
                kind=SPATIAL,
                transitive=True,
                inherits=False,     # ← KEY: No inheritance
                negation_blocks=False
            )
        
        # STATE (is_liquid, is_solid)
        if any(pattern in pred_lower for pattern in ['liquid', 'solid', 'gas']):
            return RelationFrame(
                kind=STATE,
                transitive=False,
                inherits=False,     # ← KEY: Context-dependent
                negation_blocks=False
            )
        
        # GENERIC (default: conservative inheritance)
        return RelationFrame(
            kind=GENERIC,
            inherits=True,
            negation_blocks=True
        )
```

**Why This Matters**:

```python
# Query 1: "Do bats produce milk?"
frame = RelationFrame.for_predicate("produces_milk")
# → FUNCTIONAL(inherits=True)
# Decision: Check ancestors ✓

# Query 2: "Is London in Europe?"
frame = RelationFrame.for_predicate("located_in")
# → SPATIAL(inherits=False)
# Decision: NO inheritance (even though transitive)
#           Must be explicitly stored or in external memory
```

**Human Analogy**:

You instinctively know:
- Biological properties (produces milk) → inherit down taxonomy
- Locations (London in UK) → DON'T inherit down taxonomy
  (Just because London is a city doesn't mean all cities are in the UK!)

MARC models this distinction with **structural semantics**.

### Step 3: Taxonomic Traversal

**Goal**: Find ancestors of the entity in the taxonomic hierarchy.

**Human Analog**: When reasoning about bats, you traverse: bat → mammal → animal

**Algorithm**:
```python
def find_taxonomic_ancestors(entity: str, max_depth: int = 5) -> List[Tuple[str, List[str]]]:
    """
    Find all taxonomic ancestors via IS-A relations.
    
    Returns:
        List of (ancestor, path) tuples
        
    Example:
        find_taxonomic_ancestors("bat")
        → [("mammal", ["bat", "mammal"]),
           ("animal", ["bat", "mammal", "animal"])]
    """
    ancestors = []
    visited = set()
    
    # BFS traversal
    queue = [(entity, [entity])]
    
    while queue and len(ancestors) < max_depth:
        current, path = queue.pop(0)
        
        if current in visited:
            continue  # Cycle detection
        visited.add(current)
        
        # Find IS-A relations
        for belief in chitta.beliefs.values():
            if (belief.active and 
                entity_in_belief(current, belief) and
                'is_a' in belief.predicates):
                
                # Extract parent entity
                parent = extract_parent(belief, current)
                if parent and parent not in visited:
                    new_path = path + [parent]
                    ancestors.append((parent, new_path))
                    queue.append((parent, new_path))
    
    return ancestors
```

**Example**:
```
Beliefs in memory:
  - "Bat is a mammal"
  - "Mammal is an animal"
  - "Animal is alive"

find_taxonomic_ancestors("bat") →
  [
    ("mammal", ["bat", "mammal"]),
    ("animal", ["bat", "mammal", "animal"]),
    ("alive", ["bat", "mammal", "animal", "alive"])
  ]
```

**Why BFS?**

Breadth-First Search finds the SHORTEST path to each ancestor.

If there are multiple paths (bat → mammal → animal and bat → vertebrate → animal), BFS finds the shortest.

**Cycle Detection**:

```python
visited = set()  # Track seen entities

if current in visited:
    continue  # Prevent infinite loops
```

Prevents circular hierarchies: A → B → C → A

### Step 4: Negation Dominance (The Critical Rule)

**Goal**: Check if an inherited NEGATION blocks a POSITIVE inference.

**The Problem** (without negation dominance):
```
Facts:
  - Bats are mammals
  - Mammals do NOT have gills

Query: "Do bats have gills?"

Naive reasoning:
  1. Bat is mammal ✓
  2. Mammal has predicate "has_gills"? Yes (NEGATIVE polarity)
  3. Inherit... → ???
  
Wrong answer: "Unknown" (or worse, "Yes")
```

**The Solution** (negation dominance):
```
Rule: Inherited NEGATIONS block POSITIVE inheritance.

If ancestor has NEGATIVE belief for predicate:
  AND frame.negation_blocks = True
  → Answer is NO (not unknown)
```

**The Math**:

$$
\text{NegationDominance}(e, p) = \begin{cases}
\text{NO} & \text{if } \exists a \in \text{ancestors}(e): \text{belief}(a, p, \text{NEG}) \land \text{frame}(p).\text{negation\_blocks} \\
\text{continue} & \text{otherwise}
\end{cases}
$$

**Algorithm**:
```python
def check_negation_dominance(
    entity: str,
    predicate: str,
    ancestors: List[Tuple[str, List[str]]],
    frame: RelationFrame
) -> Optional[str]:
    """
    Check if inherited negation blocks positive answer.
    
    Returns:
        "no" if negation blocks
        None if no blocking negation
    """
    # Only applies if frame allows negation blocking
    if not frame.negation_blocks:
        return None
    
    # Search ancestors for NEGATIVE beliefs
    for ancestor, path in ancestors:
        beliefs = find_beliefs(ancestor, predicate)
        
        for belief in beliefs:
            if belief.polarity == NEGATIVE:
                # FOUND NEGATION → BLOCKS
                return "no"
    
    return None  # No blocking negation
```

**Example**:
```python
Query: "Do bats have gills?"

Step 1: Get frame
  frame = RelationFrame.for_predicate("has_gills")
  → FUNCTIONAL(negation_blocks=True)

Step 2: Find ancestors
  ancestors = [("mammal", ["bat", "mammal"])]

Step 3: Check for negations
  belief("mammal", "has_gills") → FOUND (polarity=NEGATIVE)
  frame.negation_blocks = True
  → RETURN "NO"

Answer: NO (bats don't have gills, inherited from mammals)
```

**Why This Matters**:

Without negation dominance:
```
"Do bats have gills?" → "I don't know"
(Unhelpful and incorrect)
```

With negation dominance:
```
"Do bats have gills?" → "No"  
(Correct and useful)
```

**Human Analogy**:

You know mammals don't have gills. When asked if a bat (mammal) has gills, you immediately say NO — you don't need to check each mammal species individually.

This is **negative inheritance** — MARC does it too.

### Step 5: Positive Inheritance

**Goal**: If no negation blocks, check if ancestor has POSITIVE belief.

**Algorithm**:
```python
def check_positive_inheritance(
    entity: str,
    predicate: str,
    ancestors: List[Tuple[str, List[str]]]
) -> Optional[Tuple[str, List[str]]]:
    """
    Check if ancestor has positive belief for predicate.
    
    Returns:
        (ancestor, path) if found
        None if not found
    """
    for ancestor, path in ancestors:
        beliefs = find_beliefs(ancestor, predicate)
        
        for belief in beliefs:
            if belief.polarity == POSITIVE:
                # FOUND POSITIVE BELIEF
                return (ancestor, path)
    
    return None  # Not found
```

**Example**:
```python
Query: "Do bats produce milk?"

Step 1: Check direct match → NOT FOUND
Step 2: Get frame → FUNCTIONAL(inherits=True)
Step 3: Find ancestors → [("mammal", ["bat", "mammal"])]
Step 4: Check negation dominance → None (no negation)
Step 5: Check positive inheritance
  belief("mammal", "produces_milk") → FOUND (polarity=POSITIVE)
  → RETURN ("mammal", ["bat", "mammal"])

Answer: YES (via inheritance from mammal)
```

**The Inference Chain**:

$$
\begin{aligned}
&\text{is\_a}(\text{bat}, \text{mammal}) \\
&\land \text{has\_property}(\text{mammal}, \text{produces\_milk}) \\
&\land \text{frame}(\text{produces\_milk}).\text{inherits} \\
&\implies \text{has\_property}(\text{bat}, \text{produces\_milk})
\end{aligned}
$$

### Step 6: Proof Construction

**Goal**: Build a formal proof trace showing HOW the answer was derived.

**Why Proofs?**

Traditional AI:
```
Query: "Do bats produce milk?"
Answer: "Yes"
Explanation: ??? (black box)
```

MARC:
```
Query: "Do bats produce milk?"
Answer: "Yes"
Proof:
  1. Direct match: NOT FOUND
  2. Frame check: produces_milk is FUNCTIONAL(inherits=True)
  3. Taxonomy: bat → mammal
  4. Negation check: No blocking negation
  5. Inheritance: mammal produces_milk (POSITIVE)
  6. Conclusion: bat inherits produces_milk
Confidence: 0.9
```

**Algorithm**:
```python
@dataclass
class ProofStep:
    """One step in the derivation"""
    rule: str                    # Rule used
    input: Any                   # Input to rule
    output: Any                  # Output from rule
    confidence: float            # Step confidence
    source: Optional[str] = None # Source belief ID

@dataclass
class AnswerProof:
    """Complete proof of answer"""
    verdict: str                 # "yes", "no", "unknown"
    steps: List[ProofStep]       # Derivation trace
    confidence: float            # Overall confidence
    conflicts: List[str]         # Conflicting beliefs (if any)
    
    def to_natural_language(self) -> str:
        """Render proof as human-readable text"""
        if self.verdict == "yes":
            return f"Yes. {self._explain_derivation()}"
        elif self.verdict == "no":
            return f"No. {self._explain_derivation()}"
        else:
            return "I do not know."
    
    def _explain_derivation(self) -> str:
        """Explain how answer was derived"""
        for step in self.steps:
            if step.rule == "direct_match":
                return "Directly stored."
            elif step.rule == "taxonomic_inheritance":
                return f"Inherited from {step.source}."
            elif step.rule == "negation_dominance":
                return f"Negation inherited from {step.source}."
        
        return "Derived through reasoning."
```

**Example Proof**:
```python
AnswerProof(
    verdict="yes",
    steps=[
        ProofStep(
            rule="direct_match",
            input=("bat", "produces_milk"),
            output=None,
            confidence=0.0
        ),
        ProofStep(
            rule="frame_check",
            input="produces_milk",
            output=RelationFrame(FUNCTIONAL, inherits=True),
            confidence=1.0
        ),
        ProofStep(
            rule="taxonomy_traversal",
            input="bat",
            output=[("mammal", ["bat", "mammal"])],
            confidence=0.95
        ),
        ProofStep(
            rule="taxonomic_inheritance",
            input=("mammal", "produces_milk"),
            output=Belief(entities=["mammal"], predicates=["produces_milk"], polarity=POSITIVE),
            confidence=0.9,
            source="belief_123"
        )
    ],
    confidence=0.9,
    conflicts=[]
)
```

**Why Formal Proofs?**

1. **Auditability**: Can verify reasoning is correct
2. **Transparency**: No black boxes
3. **Debuggability**: Can trace where reasoning went wrong
4. **Trust**: Understand WHY the system answered as it did

**Human Analogy**:

When a mathematician proves a theorem, they show EVERY STEP. Not "the answer is 42" but "here's how I derived 42."

MARC does the same for reasoning.

---

## Grounding Checks: The Honesty Principle

### The Problem: Unbounded Composition

**Naive reasoning** (no grounding checks):
```
Facts:
  - Copper is conductive
  - Objects are things

Query: "Do copper objects conduct electricity?"

Naive inference:
  1. Copper is conductive ✓
  2. Object is made of copper (ASSUMED)
  3. Therefore copper objects conduct electricity
  → Answer: YES

Problem: We never taught "objects are made of their material"!
This is HALLUCINATION via unbounded composition.
```

**MARC with grounding** (honest refusal):
```
Query: "Do copper objects conduct electricity?"

Grounding check:
  1. Do we have belief("copper object", "conductive")? NO
  2. Is "copper object" in entity index? NO
  3. Is the composition rule taught? NO
  → UNGROUNDED

Answer: "I do not know."

This is HONEST REFUSAL, not a bug.
```

### The Grounding Algorithm

**Goal**: Verify predicate is GROUNDED before answering.

**Grounding Types**:

$$
\text{Grounded}(e, p) = \begin{cases}
\text{TRUE} & \text{if } \exists \text{belief}(e, p) & \text{(direct)} \\
\text{TRUE} & \text{if } \exists a \in \text{ancestors}(e): \text{belief}(a, p) & \text{(taxonomic)} \\
\text{TRUE} & \text{if } |\text{entities}| \leq 2 \land e \in \text{index} & \text{(simple)} \\
\text{FALSE} & \text{otherwise} & \text{(refuse)}
\end{cases}
$$

**Algorithm**:
```python
def check_predicate_grounding(
    entities: List[str],
    predicate: str,
    applicable_beliefs: List[Belief]
) -> bool:
    """
    Verify predicate is grounded before answering.
    
    Returns:
        True if grounded (can answer)
        False if ungrounded (refuse to answer)
    """
    # Case 1: Direct grounding (beliefs exist)
    if applicable_beliefs:
        return True  # Already have beliefs about this
    
    # Case 2: Taxonomic grounding (ancestor has predicate)
    for entity in entities:
        ancestors = find_taxonomic_ancestors(entity)
        for ancestor, _ in ancestors:
            if has_predicate(ancestor, predicate):
                return True  # Grounded via taxonomy
    
    # Case 3: Simple property check (≤2 entities, entity exists)
    if len(entities) <= 2:
        for entity in entities:
            if entity in chitta.entity_index:
                return True  # Simple entity property
    
    # Case 4: UNGROUNDED → Refuse
    return False
```

**Examples**:

```python
# GROUNDED (direct)
Query: "Do bats have wings?"
Check: belief("bat", "has_wings") exists? YES
→ GROUNDED ✓

# GROUNDED (taxonomic)
Query: "Do bats produce milk?"
Check: ancestor("bat") = "mammal", has_predicate("mammal", "produces_milk")? YES
→ GROUNDED ✓

# GROUNDED (simple entity)
Query: "Is gold shiny?"
Check: "gold" in entity_index? YES, len(entities) = 2
→ GROUNDED ✓

# UNGROUNDED (composition not taught)
Query: "Do copper objects conduct electricity?"
Check:
  - belief("copper object", "conductive")? NO
  - ancestor("copper object") has predicate? NO (entity not even in taxonomy)
  - "copper object" in entity_index? NO
→ UNGROUNDED ✗
Answer: "I do not know."
```

**Why This Matters**:

Without grounding:
```
"Do flying purple elephants sing opera?"
→ System might try to reason about this nonsense
```

With grounding:
```
"Do flying purple elephants sing opera?"
→ "flying purple elephant" not in entity index
→ UNGROUNDED
→ "I do not know."
```

**Philosophy**: **Better to refuse honestly than hallucinate confidently.**

---

## Paraconsistency: Tolerating Contradiction

### The Problem: Real World is Inconsistent

**Classical logic** (explodes under contradiction):
```
Facts:
  1. Tweety is a bird
  2. Birds fly
  3. Tweety does not fly (penguin exception)

Classical logic:
  From (1) and (2): Tweety flies
  From (3): Tweety does not fly
  CONTRADICTION → Everything is true (explosion)
```

**Paraconsistent logic** (tolerates contradiction):
```
Facts:
  1. Tweety is a bird (confidence: 0.9)
  2. Birds fly (confidence: 0.8)
  3. Tweety does not fly (confidence: 0.95)

Paraconsistent reasoning:
  General rule: Birds fly (0.8)
  Exception: Tweety doesn't fly (0.95)
  Resolution: Trust higher confidence (exception wins)
  Answer: "Tweety does not fly" (but note the conflict)
```

### The Algorithm

**Goal**: Handle contradictions without explosion.

**Approach**: Confidence-based resolution + conflict tracking

**Algorithm**:
```python
def resolve_conflicts(beliefs: List[Belief]) -> Tuple[Belief, List[Belief]]:
    """
    Resolve contradictory beliefs.
    
    Returns:
        (winning_belief, conflicting_beliefs)
    """
    if len(beliefs) <= 1:
        return (beliefs[0] if beliefs else None, [])
    
    # Separate by polarity
    positive = [b for b in beliefs if b.polarity == POSITIVE]
    negative = [b for b in beliefs if b.polarity == NEGATIVE]
    
    if not positive or not negative:
        # No contradiction
        return (max(beliefs, key=lambda b: b.confidence), [])
    
    # CONTRADICTION DETECTED
    # Choose highest confidence
    all_sorted = sorted(beliefs, key=lambda b: b.confidence, reverse=True)
    winner = all_sorted[0]
    losers = all_sorted[1:]
    
    return (winner, losers)
```

**Example**:
```python
Beliefs:
  1. bird(tweety) → confidence=0.9
  2. fly(bird) → confidence=0.8 (general rule)
  3. NOT fly(tweety) → confidence=0.95 (specific exception)

Query: "Does Tweety fly?"

Resolution:
  Applicable beliefs: [2, 3]
  Conflict: 2 (POSITIVE) vs 3 (NEGATIVE)
  Confidence: 3 (0.95) > 2 (0.8)
  Winner: 3 (Tweety doesn't fly)
  
Answer: "No (but note: general rule says birds fly)"
```

**The Math**:

$$
\text{Resolve}(B) = \begin{cases}
\arg\max_{b \in B} \text{confidence}(b) & \text{if contradiction} \\
\max_{b \in B} \text{confidence}(b) & \text{otherwise}
\end{cases}
$$

**Why This Matters**:

Real-world knowledge contains exceptions:
- "Birds fly" (general)
- "Penguins don't fly" (exception)
- "Tweety (penguin) doesn't fly" (specific)

Classical logic can't handle this. **MARC can.**

---

## Learning Mode vs Reasoning Mode

### Two Operational Modes

**LEARNING MODE** (monotonic admission):
```python
learning_mode = True

# Accept ALL input without decay
process("Bats are mammals")
→ Add belief (no questions asked)

process("Mammals produce milk")
→ Add belief (no skepticism)

# NO lifecycle management:
  - NO decay
  - NO demotion
  - NO rejection
```

**REASONING MODE** (epistemic discipline):
```python
learning_mode = False

# Query with full reasoning
ask("Do bats produce milk?")
→ Apply all inference rules
→ Check grounding
→ Build proof
→ Return answer

# Lifecycle management active:
  - Decay unused beliefs
  - Promote frequently used
  - Demote contradicted
```

**Why Two Modes?**

**Human Analogy**:

When a teacher is TEACHING you (learning mode):
- You accept facts provisionally
- You don't question every statement
- You build a knowledge base

When you're THINKING/REASONING (reasoning mode):
- You critically evaluate
- You check for contradictions
- You apply inference rules

**Algorithm**:
```python
class Buddhi:
    def __init__(self, chitta: ChittaGraph):
        self.chitta = chitta
        self.learning_mode = True  # Default: learning
    
    def think(self, proposal: dict) -> dict:
        """Process assertion"""
        if self.learning_mode:
            # LEARNING MODE: Accept monotonically
            return self._add_belief(proposal)
        else:
            # REASONING MODE: Critically evaluate
            return self._evaluate_belief(proposal)
    
    def answer(self, query: dict) -> AnswerProof:
        """Answer query (always uses reasoning logic)"""
        # Queries ALWAYS use full reasoning
        # (even in learning mode)
        return self._reason_about_query(query)
```

**Key Insight**: Teaching and reasoning are DIFFERENT cognitive modes.

---

## The Buddhi Contract

### What Buddhi Guarantees

1. **Proofs**: Every answer includes derivation trace
2. **Grounding**: Refuses ungrounded inferences
3. **Negation Dominance**: Inherited negations block positives
4. **Frame Discipline**: Respects structural semantics of relations
5. **Paraconsistency**: Tolerates contradictions without explosion
6. **Transparency**: All reasoning steps are explicit

### What Buddhi Does NOT Do

1. **NO PERCEPTION**: Doesn't parse natural language (Manas does that)
2. **NO STORAGE**: Doesn't manage memory (Chitta does that)
3. **NO ORCHESTRATION**: Doesn't control execution (Ahankara does that)
4. **NO LEARNING**: Doesn't decide what to store (just judges truth)

**Buddhi is PURE REASONING.**

---

## Why This Design? (Feynman's Question)

### "Why not just use logic programming (Prolog)?"

**Prolog strengths**:
- ✓ Formal inference
- ✓ Backtracking search
- ✓ Unification

**Prolog weaknesses for MARC**:
- ✗ No paraconsistency (explodes on contradiction)
- ✗ No confidence tracking
- ✗ No relation frames (all predicates equal)
- ✗ No grounding checks (unbounded inference)

**MARC adds**:
- ✓ Paraconsistent reasoning
- ✓ Confidence-weighted beliefs
- ✓ Structural semantics (frames)
- ✓ Epistemic discipline (grounding)

### "Why relation frames instead of flat predicates?"

**Without frames** (flat predicates):
```
"Bat is mammal" → is_a(bat, mammal)
"London in UK" → located_in(london, uk)

Query: "Is London a mammal?" (via location inheritance)
→ london in uk, uk in europe, ...
→ Nonsense propagation
```

**With frames** (structural semantics):
```
is_a → TAXONOMIC(inherits=True)
located_in → SPATIAL(inherits=False)

Query: "Is London a mammal?"
→ Frame check: located_in doesn't inherit
→ NO propagation
→ Correct refusal
```

**Key Insight**: Humans know that location ≠ taxonomy. MARC does too (via frames).

### "Why negation dominance?"

**Without negation dominance**:
```
"Mammals don't have gills"
"Do bats (mammals) have gills?"
→ "I don't know" (useless)
```

**With negation dominance**:
```
"Mammals don't have gills"
"Do bats (mammals) have gills?"
→ "No" (inherited negation)
```

**Humans do this instinctively.** MARC does it explicitly.

---

## Summary: Buddhi in One Diagram

```
╔══════════════════════════════════════════════════════════════╗
║                   BUDDHI (Reasoning Engine)                  ║
║                                                              ║
║  INPUT: Structured Belief Proposal                           ║
║    ↓                                                         ║
║  PHASE 1: Direct Match (O(1) lookup)                         ║
║    ↓                                                         ║
║  PHASE 2: Relation Frame Check (structural semantics)        ║
║    ↓                                                         ║
║  PHASE 3: Taxonomic Traversal (find ancestors)               ║
║    ↓                                                         ║
║  PHASE 4: Negation Dominance (inherited negations block)     ║
║    ↓                                                         ║
║  PHASE 5: Positive Inheritance (ancestor properties)         ║
║    ↓                                                         ║
║  PHASE 6: Proof Construction (formal derivation)             ║
║    ↓                                                         ║
║  OUTPUT: Answer Proof (verdict + steps + confidence)         ║
║                                                              ║
║  GUARANTEES:                                                 ║
║   • Grounding discipline (refuses ungrounded inferences)     ║
║   • Negation dominance (negatives block positives)           ║
║   • Frame discipline (structural semantics)                  ║
║   • Paraconsistency (tolerates contradictions)               ║
║   • Formal proofs (every answer justified)                   ║
║                                                              ║
║  DOES NOT:                                                   ║
║   • Parse language (Manas does that)                         ║
║   • Store beliefs (Chitta does that)                         ║
║   • Orchestrate (Ahankara does that)                         ║
║                                                              ║
║  PHILOSOPHY:                                                 ║
║   "Reasoning is NOT perception"                              ║
║   "Inference is NOT retrieval"                               ║
║   "Better to refuse honestly than hallucinate confidently"   ║
╚══════════════════════════════════════════════════════════════╝
```

---

**Key Insight**: Buddhi is the intellectual core of MARC. It doesn't perceive, remember, or act — it just THINKS. And it thinks with formal rigor, structural semantics, and epistemic honesty.

**BUDDHI IS FROZEN**: No more changes. This is the production reasoning engine.
