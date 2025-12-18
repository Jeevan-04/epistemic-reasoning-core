# HRE — Hypothetical Reasoning Engine

## What is HRE?

**HRE** (Hypothetical Reasoning Engine) is MARC's **epistemically sterile sandbox** for thought experiments. It allows the system to reason about "what if" scenarios WITHOUT contaminating real beliefs.

Think of HRE like your **imagination** — you can think "What if I were a bird?" without actually believing you are a bird.

---

## The Core Problem: Why Do We Need HRE?

### The Human Analogy

When you reason about hypotheticals, your brain does something remarkable:

```
Question: "If I were a bird, could I fly?"

Your Reasoning:
1. I am NOT a bird (real belief)
2. But suppose I WERE a bird (hypothetical)
3. Birds can fly (real belief)
4. Therefore, IF I were a bird, I could fly (hypothetical conclusion)

After thinking:
- Real belief unchanged: I am still not a bird
- Hypothetical conclusion: IF-THEN relationship established
- No contamination
```

**Critical Insight**: Hypothetical reasoning happens in a SEPARATE SPACE from real beliefs.

### The Contamination Problem (Without HRE)

**Naive approach** (no sandbox):
```python
# Chitta (real beliefs):
beliefs = [
    "I am human",
    "Birds fly"
]

# Hypothetical query: "If I were a bird, could I fly?"
# Naive approach: Add hypothetical as real belief
beliefs.append("I am a bird")  # ← CONTAMINATION!

# Now reason:
buddhi.answer("Can I fly?")
→ "Yes, because you are a bird"  # ← WRONG! Hypothetical leaked into reality

# PROBLEM: Can't distinguish real from hypothetical
```

**With HRE** (epistemically sterile sandbox):
```python
# Chitta (real beliefs):
beliefs = [
    "I am human",
    "Birds fly"
]

# Hypothetical query: "If I were a bird, could I fly?"
# HRE approach: Create isolated sandbox
sandbox = HRE()
sandbox.assume("I am a bird")  # ← Hypothetical only in sandbox
sandbox.reason("Can I fly?")
→ "Yes, IF you are a bird"  # ← Correct conditional answer

# After reasoning:
# Chitta beliefs UNCHANGED
beliefs = [
    "I am human",  # ← Still true
    "Birds fly"    # ← Still true
]
# Sandbox discarded
# NO contamination ✓
```

---

## Architecture: How HRE Works

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    HYPOTHETICAL QUERY                        │
│          "If bats were insects, would they have 6 legs?"     │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                     HRE (Sandbox)                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  STEP 1: EXTRACT HYPOTHETICAL ASSUMPTION              │  │
│  │  "If bats were insects"                               │  │
│  │  → Parse: Belief(bat, insect, is_a, HYPOTHETICAL)     │  │
│  └────────────────┬───────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐  │
│  │  STEP 2: CREATE TEMPORARY BELIEF SPACE                │  │
│  │  sandbox_chitta = copy(real_chitta)                   │  │
│  │  sandbox_chitta.add("bat is insect")  # ISOLATED      │  │
│  └────────────────┬───────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐  │
│  │  STEP 3: REASON IN SANDBOX                            │  │
│  │  sandbox_buddhi.answer("Do bats have 6 legs?")        │  │
│  │  → Taxonomic traversal: bat → insect                  │  │
│  │  → Inheritance: insects have 6 legs                   │  │
│  │  → Conclusion: Yes (in sandbox)                       │  │
│  └────────────────┬───────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐  │
│  │  STEP 4: FORMAT CONDITIONAL ANSWER                    │  │
│  │  "Yes, IF bats were insects, they would have 6 legs"  │  │
│  └────────────────┬───────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐  │
│  │  STEP 5: DISCARD SANDBOX                              │  │
│  │  del sandbox_chitta                                   │  │
│  │  del sandbox_buddhi                                   │  │
│  │  → Real beliefs UNCHANGED ✓                           │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                   CONDITIONAL ANSWER                         │
│  "Yes, IF bats were insects, they would have 6 legs."       │
│                                                              │
│  Real Chitta: UNCHANGED                                      │
│  No contamination ✓                                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Deep Dive: The Sandbox Algorithm

### Step 1: Detect Hypothetical

**Goal**: Recognize when query contains a hypothetical assumption.

**Hypothetical Markers**:
- "If..."
- "Suppose..."
- "What if..."
- "Assume..."

**Algorithm**:
```python
def is_hypothetical(query: str) -> bool:
    """
    Check if query contains hypothetical marker.
    """
    hypothetical_markers = [
        'if ', 'suppose ', 'what if ', 'assume ',
        'imagine ', 'pretend ', 'let\'s say '
    ]
    
    query_lower = query.lower().strip()
    
    for marker in hypothetical_markers:
        if marker in query_lower:
            return True
    
    return False
```

**Examples**:
```python
is_hypothetical("If bats were insects, would they have 6 legs?")
→ True  (marker: "if")

is_hypothetical("Do bats have 6 legs?")
→ False  (no marker)

is_hypothetical("Suppose gold were liquid, would it flow?")
→ True  (marker: "suppose")
```

### Step 2: Extract Assumption

**Goal**: Parse hypothetical assumption from query.

**Algorithm**:
```python
def extract_assumption(query: str) -> Optional[str]:
    """
    Extract hypothetical assumption from query.
    
    Examples:
        "If bats were insects, would they fly?"
        → "bats are insects"
        
        "Suppose water were solid, would it be ice?"
        → "water is solid"
    """
    query_lower = query.lower()
    
    # Pattern: "if X were Y"
    if 'if ' in query_lower and ' were ' in query_lower:
        # Extract between "if" and comma/question
        if_start = query_lower.index('if ') + 3
        
        # Find end of assumption (comma or question mark)
        comma_idx = query_lower.find(',', if_start)
        question_idx = query_lower.find('?', if_start)
        
        if comma_idx != -1:
            assumption = query[if_start:comma_idx].strip()
        elif question_idx != -1:
            assumption = query[if_start:question_idx].strip()
        else:
            assumption = query[if_start:].strip()
        
        # Normalize "were" → "are"
        assumption = assumption.replace(' were ', ' are ')
        
        return assumption
    
    return None
```

**Examples**:
```python
extract_assumption("If bats were insects, would they have 6 legs?")
→ "bats are insects"

extract_assumption("Suppose water were solid, would it freeze?")
→ "water is solid"

extract_assumption("Do bats fly?")
→ None  (not hypothetical)
```

### Step 3: Create Sandbox

**Goal**: Create isolated belief space for hypothetical reasoning.

**The Critical Principle**: **NO SHARED STATE**

```python
class HypotheticalReasoner:
    def __init__(self, real_chitta: ChittaGraph, real_buddhi: Buddhi):
        """
        HRE: Hypothetical Reasoning Engine.
        
        Creates epistemically sterile sandboxes for thought experiments.
        """
        self.real_chitta = real_chitta
        self.real_buddhi = real_buddhi
    
    def reason_hypothetically(self, assumption: str, query: str) -> str:
        """
        Reason about hypothetical scenario.
        
        Steps:
          1. Create temporary Chitta (copy of real beliefs)
          2. Add hypothetical assumption to sandbox
          3. Create temporary Buddhi (uses sandbox Chitta)
          4. Answer query in sandbox
          5. Discard sandbox
          6. Return conditional answer
        """
        # STEP 1: Create sandbox Chitta (deep copy)
        sandbox_chitta = self._create_sandbox_chitta()
        
        # STEP 2: Add hypothetical assumption
        assumption_belief = self._parse_assumption(assumption)
        sandbox_chitta.add_belief(assumption_belief)
        
        # STEP 3: Create sandbox Buddhi
        sandbox_buddhi = Buddhi(sandbox_chitta)
        
        # STEP 4: Answer query in sandbox
        answer = sandbox_buddhi.answer(query)
        
        # STEP 5: Format conditional answer
        conditional_answer = f"IF {assumption}, then {answer}"
        
        # STEP 6: Discard sandbox (no explicit cleanup needed - Python GC)
        # sandbox_chitta and sandbox_buddhi go out of scope
        
        return conditional_answer
    
    def _create_sandbox_chitta(self) -> ChittaGraph:
        """
        Create deep copy of real Chitta for sandbox.
        
        CRITICAL: Must be DEEP copy (no shared references).
        """
        sandbox = ChittaGraph()
        
        # Copy all beliefs
        for belief_id, belief in self.real_chitta.beliefs.items():
            # Create new belief object (no shared references)
            sandbox_belief = Belief(
                entities=belief.entities.copy(),
                predicates=belief.predicates.copy(),
                polarity=belief.polarity,
                confidence=belief.confidence,
                timestamp=belief.timestamp,
                source=belief.source,
                active=belief.active,
                access_count=belief.access_count,
                last_accessed=belief.last_accessed,
                belief_id=belief_id,
                supersedes=belief.supersedes
            )
            sandbox.beliefs[belief_id] = sandbox_belief
        
        # Rebuild indices
        sandbox._rebuild_indices()
        
        return sandbox
```

**Why Deep Copy?**

```python
# WRONG (shallow copy):
sandbox_chitta = real_chitta  # ← Same object!
sandbox_chitta.add_belief(...)  # ← Contaminates real_chitta!

# CORRECT (deep copy):
sandbox_chitta = real_chitta.deep_copy()  # ← Separate object
sandbox_chitta.add_belief(...)  # ← Only affects sandbox ✓
```

### Step 4: Reason in Sandbox

**Goal**: Apply full Buddhi reasoning in isolated environment.

**Algorithm**:
```python
# Create sandbox Buddhi
sandbox_buddhi = Buddhi(sandbox_chitta)

# Answer query using sandbox
answer_proof = sandbox_buddhi.answer(query_proposal)

# Extract verdict
verdict = answer_proof.verdict  # "yes", "no", or "unknown"
```

**Example**:
```python
# Real beliefs:
real_chitta.beliefs = [
    "Bats are mammals",
    "Mammals produce milk",
    "Insects have 6 legs"
]

# Hypothetical assumption:
assumption = "Bats are insects"

# Create sandbox:
sandbox_chitta = real_chitta.deep_copy()
sandbox_chitta.add_belief(Belief(['bat', 'insect'], ['is_a'], POSITIVE))

# Sandbox now has:
sandbox_chitta.beliefs = [
    "Bats are mammals",        # From real
    "Mammals produce milk",     # From real
    "Insects have 6 legs",      # From real
    "Bats are insects"          # HYPOTHETICAL (only in sandbox)
]

# Reason in sandbox:
sandbox_buddhi = Buddhi(sandbox_chitta)
answer = sandbox_buddhi.answer("Do bats have 6 legs?")

# Reasoning:
# 1. Taxonomic traversal: bat → insect (from hypothetical)
# 2. Inheritance: insect has 6 legs
# 3. Conclusion: Yes (in sandbox)

# Answer: "Yes"

# Real beliefs UNCHANGED:
real_chitta.beliefs = [
    "Bats are mammals",        # ← Still true
    "Mammals produce milk",     # ← Still true
    "Insects have 6 legs"       # ← Still true
    # "Bats are insects" NOT added to real beliefs ✓
]
```

### Step 5: Format Conditional Answer

**Goal**: Make clear that answer is CONDITIONAL on hypothetical.

**Algorithm**:
```python
def format_conditional_answer(assumption: str, verdict: str) -> str:
    """
    Format answer to emphasize conditionality.
    
    Examples:
        ("bats are insects", "yes")
        → "Yes, IF bats were insects."
        
        ("water is solid", "no")
        → "No, even IF water were solid."
    """
    if verdict == "yes":
        return f"Yes, IF {assumption}."
    elif verdict == "no":
        return f"No, even IF {assumption}."
    else:
        return f"I do not know, even IF {assumption}."
```

**Why Emphasize Conditionality?**

Without emphasis:
```
Query: "If bats were insects, would they have 6 legs?"
Answer: "Yes"
→ Ambiguous! Does system believe bats ARE insects?
```

With emphasis:
```
Query: "If bats were insects, would they have 6 legs?"
Answer: "Yes, IF bats were insects."
→ Clear! This is a conditional answer, not a real belief.
```

---

## The Three Principles of HRE

### Principle 1: Epistemic Sterility

**Definition**: Hypothetical reasoning NEVER contaminates real beliefs.

**Mathematical Formulation**:

$$
\text{Beliefs}_{real}^{after} = \text{Beliefs}_{real}^{before}
$$

Regardless of hypothetical reasoning.

**Implementation**:
```python
# Before hypothetical reasoning
real_beliefs_before = set(real_chitta.beliefs.keys())

# Hypothetical reasoning
hre.reason_hypothetically("bats are insects", "Do bats have 6 legs?")

# After hypothetical reasoning
real_beliefs_after = set(real_chitta.beliefs.keys())

# GUARANTEE
assert real_beliefs_before == real_beliefs_after  # ✓ NO CONTAMINATION
```

### Principle 2: No Evidence Accumulation

**Definition**: Hypothetical assumptions have NO confidence, NO evidence, NO persistence.

**Why?**

```python
# WRONG: Treat hypothetical as evidence
sandbox_chitta.add_belief(Belief(
    entities=['bat', 'insect'],
    predicates=['is_a'],
    confidence=0.8,  # ← NO! Hypotheticals have no confidence
    source="hypothetical"
))

# CORRECT: Hypotheticals are confidence-free assumptions
sandbox_chitta.add_belief(Belief(
    entities=['bat', 'insect'],
    predicates=['is_a'],
    confidence=1.0,  # ← Assumed true for reasoning purposes
    source="hypothetical"
))
# But sandbox is DISCARDED, so this never affects real beliefs
```

**Key Insight**: Hypotheticals are LOGICAL ASSUMPTIONS, not EVIDENTIAL BELIEFS.

### Principle 3: No Memory

**Definition**: HRE has NO persistent state. Every hypothetical is isolated.

**Why?**

```python
# WRONG: Persistent sandbox
class HRE:
    def __init__(self):
        self.sandbox = ChittaGraph()  # ← Persistent state!
    
    def reason(self, assumption, query):
        self.sandbox.add_belief(assumption)  # ← Accumulates over time!
        return self.sandbox.answer(query)

# Hypothetical 1: "If bats were insects..."
hre.reason("bats are insects", "...")
# → sandbox.beliefs = ["bats are insects"]

# Hypothetical 2: "If water were solid..."
hre.reason("water is solid", "...")
# → sandbox.beliefs = ["bats are insects", "water is solid"]  # ← CONTAMINATION!

# CORRECT: Ephemeral sandbox
class HRE:
    def reason(self, assumption, query):
        sandbox = self._create_fresh_sandbox()  # ← New sandbox each time
        sandbox.add_belief(assumption)
        answer = sandbox.answer(query)
        del sandbox  # ← Discard after use
        return answer
```

**Key Insight**: Each hypothetical is a FRESH SLATE.

---

## Use Cases: When to Use HRE

### Use Case 1: Counterfactual Reasoning

**Question**: "If penguins could fly, would they migrate?"

**Without HRE** (contamination risk):
```python
# Add "penguins can fly" to real beliefs
chitta.add("penguins fly")  # ← Now system believes this!

# Answer
buddhi.answer("Do penguins migrate?")
→ Based on flawed assumption that penguins actually fly
```

**With HRE** (sterile sandbox):
```python
# Hypothetical reasoning
hre.reason("penguins fly", "Do penguins migrate?")
→ "IF penguins could fly, then [reasoning based on migratory birds]"

# Real beliefs UNCHANGED
chitta.has_belief("penguins fly") → False ✓
```

### Use Case 2: Scenario Exploration

**Question**: "If mammals laid eggs, would bats lay eggs?"

**HRE Approach**:
```python
hre.reason("mammals lay eggs", "Do bats lay eggs?")

# Reasoning in sandbox:
# 1. Assume: mammals lay eggs (hypothetical)
# 2. Fact: bats are mammals (real belief)
# 3. Inherit: bats lay eggs (via taxonomic inheritance)
# Answer: "Yes, IF mammals laid eggs"

# Real beliefs: Mammals do NOT lay eggs ✓
```

### Use Case 3: Detecting Inconsistencies

**Question**: "If gold were not metallic, would it conduct electricity?"

**HRE Approach**:
```python
hre.reason("gold is not metallic", "Does gold conduct electricity?")

# Reasoning in sandbox:
# 1. Assume: gold is not metallic (hypothetical)
# 2. Fact: metals conduct electricity (real belief)
# 3. Conflict: gold not metallic, but conductive property tied to metallicity
# Answer: "No, IF gold were not metallic, it would not conduct electricity"

# Reveals: Metallicity → conductivity dependency
```

---

## What HRE Does NOT Do

1. **NO LEARNING**: Hypotheticals don't update real beliefs
2. **NO EVIDENCE**: Hypotheticals have no confidence/source tracking
3. **NO PERSISTENCE**: Each hypothetical is isolated (no cumulative state)
4. **NO PROBABILISTIC REASONING**: Binary assumption (true in sandbox)

**HRE is PURE IMAGINATION** — ephemeral, sterile, isolated.

---

## Why This Design? (Feynman's Question)

### "Why not just mark hypotheticals with a flag?"

**Flagging approach** (shared belief space):
```python
# Add hypothetical to real beliefs with flag
chitta.add_belief(Belief(
    entities=['bat', 'insect'],
    predicates=['is_a'],
    is_hypothetical=True  # ← Flag
))

# Problem 1: Contamination risk
# If flag is ignored or lost, hypothetical becomes real belief

# Problem 2: Complex filtering
# Every query must filter out hypotheticals
# Easy to forget, leading to contamination

# Problem 3: No clean separation
# Hypotheticals and real beliefs share same data structures
# Hard to reason about epistemic boundaries
```

**Sandbox approach** (isolated space):
```python
# Create separate Chitta for hypothetical
sandbox_chitta = real_chitta.deep_copy()
sandbox_chitta.add_belief(...)  # ← No flag needed

# Benefit 1: Impossible to contaminate
# Sandbox is discarded after use
# Real beliefs physically separated

# Benefit 2: No filtering needed
# Real Chitta never sees hypotheticals

# Benefit 3: Clear epistemic boundary
# Real vs hypothetical is structural, not flagged
```

**Key Insight**: Physical separation > logical separation.

### "Why discard sandbox instead of caching?"

**Caching approach**:
```python
# Cache sandbox for reuse
class HRE:
    def __init__(self):
        self.sandbox_cache = {}
    
    def reason(self, assumption, query):
        key = hash(assumption)
        if key in self.sandbox_cache:
            sandbox = self.sandbox_cache[key]  # ← Reuse
        else:
            sandbox = create_sandbox()
            self.sandbox_cache[key] = sandbox
        ...
```

**Problems**:
- Memory bloat (sandboxes accumulate)
- Staleness (cached sandbox might be outdated)
- Complexity (cache invalidation, eviction policies)

**Ephemeral approach**:
```python
class HRE:
    def reason(self, assumption, query):
        sandbox = create_fresh_sandbox()  # ← Always fresh
        # ... use sandbox ...
        del sandbox  # ← Discard immediately
```

**Benefits**:
- No memory bloat (sandbox discarded)
- Always fresh (reflects current real beliefs)
- Simple (no caching logic)

**Trade-off**: Slight performance cost (copy beliefs each time). **Worth it** for epistemic cleanliness.

---

## Summary: HRE in One Diagram

```
╔═══════════════════════════════════════════════════════════════╗
║                HRE (Hypothetical Reasoning Engine)            ║
║                                                               ║
║  INPUT: Hypothetical query                                    ║
║    "If X were Y, would Z?"                                    ║
║                     ↓                                         ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │  STEP 1: Extract assumption ("X is Y")                   │ ║
║  └──────────────┬───────────────────────────────────────────┘ ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────────────────────────┐ ║
║  │  STEP 2: Create sandbox (deep copy of real beliefs)      │ ║
║  │  sandbox_chitta = real_chitta.deep_copy()                │ ║
║  └──────────────┬───────────────────────────────────────────┘ ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────────────────────────┐ ║
║  │  STEP 3: Add assumption to sandbox                       │ ║
║  │  sandbox_chitta.add("X is Y")  ← ISOLATED                │ ║
║  └──────────────┬───────────────────────────────────────────┘ ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────────────────────────┐ ║
║  │  STEP 4: Reason in sandbox                               │ ║
║  │  sandbox_buddhi = Buddhi(sandbox_chitta)                 │ ║
║  │  answer = sandbox_buddhi.answer("Z?")                    │ ║
║  └──────────────┬───────────────────────────────────────────┘ ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────────────────────────┐ ║
║  │  STEP 5: Format conditional answer                       │ ║
║  │  "Yes/No, IF X were Y"                                   │ ║
║  └──────────────┬───────────────────────────────────────────┘ ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────────────────────────┐ ║
║  │  STEP 6: Discard sandbox                                 │ ║
║  │  del sandbox_chitta, sandbox_buddhi                      │ ║
║  │  → Real beliefs UNCHANGED ✓                              │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                               ║
║  OUTPUT: Conditional answer                                   ║
║    "Answer, IF assumption"                                    ║
║                                                               ║
║  THREE PRINCIPLES:                                            ║
║   1. Epistemic Sterility (no contamination)                  ║
║   2. No Evidence (hypotheticals have no confidence)          ║
║   3. No Memory (each hypothetical is fresh slate)            ║
║                                                               ║
║  GUARANTEES:                                                  ║
║   • Real beliefs NEVER contaminated                          ║
║   • Hypotheticals physically isolated (sandbox)              ║
║   • No persistent state (ephemeral)                          ║
║                                                               ║
║  PHILOSOPHY:                                                  ║
║   "Imagination ≠ Belief"                                     ║
║   "Counterfactuals ≠ Facts"                                  ║
║   "Thinking 'what if' ≠ Believing 'it is'"                   ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**Key Insight**: HRE is MARC's imagination — the ability to think "what if" without believing it. Physical isolation (sandboxing) ensures hypotheticals NEVER contaminate reality.

**HRE is FROZEN**: This is the production hypothetical reasoning architecture.
