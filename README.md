# Epistemically Disciplined Reasoning Architecture (EDRA)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

This repository presents **EDRA** (Epistemically Disciplined Reasoning Architecture), a normative epistemic control system that enforces intellectual honesty through explicit refusal semantics.

EDRA is developed as a foundational reasoning component of a broader long-term research program called **MARC** (Mind Architecture for Reasoning and Cognition). However, **this work focuses exclusively on epistemic reasoning and belief management** — not learning, not perception, not goal-directed behavior.

**What EDRA contributes**: A refusal-first epistemic contract that prevents error classes other systems cannot:
- LLMs minimize loss → hallucinate
- Knowledge graphs lack refusal semantics → overcommit  
- Classical logic explodes under contradiction
- Probabilistic systems blur grounding

EDRA enforces grounding checks, negation dominance, and sandboxed hypotheticals with formal proof obligations.

**What EDRA is NOT**: A theory of cognition, a learning system, a competitive AI system, or a human cognitive model.

---

## Table of Contents

- [Abstract](#abstract)
- [Motivation](#motivation)
- [Core Contribution](#core-contribution)
- [Architecture Overview](#architecture-overview)
- [System Components](#system-components)
- [Design Principles](#design-principles)
- [Epistemic Guarantees](#epistemic-guarantees)
- [Known Limitations](#known-limitations)
- [Performance Characteristics](#performance-characteristics)
- [Installation & Usage](#installation--usage)
- [Benchmark Methodology](#benchmark-methodology)
- [Comparison to Other Systems](#comparison-to-other-systems)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Why build another symbolic reasoning system? Because I observed that existing systems fail to enforce epistemic discipline:

- **LLMs**: Optimize for loss minimization → produce confident hallucinations
- **Knowledge Graphs**: Lack refusal semantics → overcommit to unjustified inferences
- **Classical Logic Systems**: Explode under contradiction (principle of explosion)
- **Probabilistic Reasoners**: Blur the line between grounded and speculative reasoning

I designed EDRA to explore a different path: **What happens when we enforce refusal-first epistemic contracts?**

The result is a system that refuses to answer when evidence is insufficient, provides formal proofs for every answer, and tolerates contradictions without logical explosion.

---

## Core Contribution

### What EDRA Actually Contributes

EDRA is a **normative epistemic control architecture** with:

1. **Explicit refusal semantics** — System refuses ungrounded inferences
2. **Explicit grounding checks** — Verifies predicate support before reasoning
3. **Explicit negation dominance** — Inherited negations block positive inheritance
4. **Explicit sandboxed hypotheticals** — Counterfactuals cannot contaminate beliefs
5. **Explicit proof traces** — Every answer includes derivation path
6. **Explicit separation of observation vs inference** — Perceptual facts ≠ logical derivations

This is NOT:
- ❌ ACT-R (no psychological fidelity, no latency modeling, no error simulation)
- ❌ SOAR (no goal stacks, no task execution, no chunking)
- ❌ Expert systems (no production rules, explicit epistemic guarantees)
- ❌ Knowledge graphs (refusal semantics, proof obligations)

**EDRA is a correctness-first epistemic execution model.**

---

## Architecture Overview

EDRA uses a modular architecture. **The module naming draws from Vedantic philosophy as design metaphors, not scientific claims.**

```
╔═══════════════════════════════════════════════════════════════╗
║                         MARC SYSTEM                           ║
║                                                               ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │              CORE MODULES (1-6)                          │ ║
║  │         System CANNOT function without these             │ ║
║  │                                                          │ ║
║  │  1. MANAS         → Perception (parse language)         │ ║
║  │  2. BUDDHI        → Reasoning (logical inference)       │ ║
║  │  3. CHITTA        → Memory (belief graph)               │ ║
║  │  4. HRE           → Hypothetical reasoning (sandbox)    │ ║
║  │  5. AHANKARA      → Orchestration (coordination)        │ ║
║  │  6. SAKSHIN       → Meta-observation (audit trail)      │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                           │                                   ║
║                           │ Core cognitive loop               ║
║                           ▼                                   ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │        AUXILIARY KNOWLEDGE SOURCES (7-8)                 │ ║
║  │       Optional modules that ENRICH reasoning             │ ║
║  │                                                          │ ║
║  │  7. PERCEPTUAL PRIORS    → Observational knowledge      │ ║
║  │  8. GEOGRAPHIC MEMORY    → Spatial knowledge            │ ║
║  └──────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════╝
```

**EDRA uses a modular architecture. The module naming draws from Vedantic philosophy as design metaphors, not scientific claims.**

I use Sanskrit terms as **naming conventions** to organize components conceptually. These are NOT claims about neuroscience or cognitive modeling.

```

---

## System Components

**Naming Convention**: I use Sanskrit terms from Vedantic philosophy as module names. These are **design metaphors for conceptual organization**, not claims about cognitive modeling or neuroscience.

### 1. Manas — Input Parsing Layer

**Function**: Converts natural language into structured belief representations.

**Why It Exists**: All downstream components require structured input (entities, predicates, polarity). Without this layer, the system cannot process text.

**Operations**:
- Entity extraction (`"Bats are mammals"` → entities: `['bat', 'mammal']`)
- Relation detection (detect IS-A, HAS, PRODUCES, etc.)
- Polarity detection (positive, negative, interrogative)
- Template classification (cognitive schemas)
- Confidence estimation (parsing certainty)

**Dependencies**: None (first-layer component, accepts raw text)

**Contract**:
```python
Input:  "Do bats produce milk?"
Output: Belief(
    entities=['bat'],
    predicates=['produces_milk'],
    polarity=INTERROGATIVE,
    confidence=0.95
)
```

---

### 2. Buddhi — Inference Engine

**Function**: Determines truth through formal logical inference with explicit grounding checks.

**Core Operations**:
- Direct match (O(1) belief lookup)
- Relation frames (structural semantics: TAXONOMIC, SPATIAL, FUNCTIONAL, STATE)
- Taxonomic inheritance (bat → mammal → properties)
- Negation dominance (inherited negations block positives)
- Paraconsistent reasoning (tolerate contradictions)
- Grounding checks (refuse ungrounded inferences)
- Proof construction (formal derivation traces)

**Why This Component Matters**:

This is where epistemic guarantees are enforced:
- Grounding checks prevent ungrounded inferences
- Negation dominance handles inherited negations
- Refusal semantics enforce "I don't know" when appropriate

**Dependencies**: Chitta (requires belief storage for reasoning)

**Contract**:
```python
Input:  Belief(entities=['bat'], predicates=['produces_milk'], polarity=INTERROGATIVE)
Output: AnswerProof(
    verdict="yes",
    steps=[
        ProofStep(rule="taxonomic_inheritance", source="mammal")
    ],
    confidence=0.9
)
```

---

### 3. Chitta — Belief Storage Layer

**Function**: Stores and indexes all beliefs in a hypergraph structure.

**Operations**:
- Belief storage (canonical keys, versioning)
- Entity indexing (O(1) lookup: "what do we know about bats?")
- Predicate indexing (O(1) lookup: "all IS-A relations")
- Taxonomy graph (fast ancestor/descendant traversal)
- Lifecycle management (decay unused, promote used, demote contradicted)
- Conflict tolerance (multiple beliefs can coexist)

**Why This Component Matters**:

Provides persistent storage and fast retrieval:
- Entity indexing enables O(1) lookup
- Taxonomy graph enables efficient traversal
- Supports paraconsistent storage (contradictions coexist)

**Dependencies**: None (data layer)

**Operations**:
```python
# Store
chitta.add_belief(Belief(['bat', 'mammal'], ['is_a'], POSITIVE))

# Retrieve
chitta.get_beliefs_for_entity('bat')
→ [Belief(['bat', 'mammal'], ['is_a'], ...),
   Belief(['bat'], ['has_wings'], ...)]

# Traverse
chitta.get_ancestors('bat')
→ [('mammal', 1), ('animal', 2)]
```

---

### 4. HRE — Hypothetical Reasoning Engine

**Function**: Handles counterfactual queries in isolated sandboxes to prevent belief contamination.

**Operations**:
- Hypothetical detection (`"If bats were fish..."`)
- Assumption extraction (parse hypothetical premise)
- Sandbox creation (isolated belief space)
- Conditional reasoning (reason in sandbox)
- Sterile disposal (discard sandbox, preserve real beliefs)

**Why This Component Matters**:

Enforces epistemic sterility:
- Hypotheticals never contaminate real beliefs
- Enables "what if" queries without pollution
- Maintains separation between actual and counterfactual

**Dependencies**: Chitta (sandbox creation), Buddhi (sandbox reasoning)

**Contract**:
```python
Input:  "If bats were insects, would they have 6 legs?"
Output: "Yes, IF bats were insects, they would have 6 legs."

# Real beliefs UNCHANGED:
chitta.has_belief("bats are insects") → False ✓
```

---

### 5. Ahankara — Execution Orchestrator

**Function**: Coordinates component execution and enforces priority ordering (observation > retrieval > inference).

**Responsibilities**:
- Perception routing (always call Manas first)
- Priority enforcement (Observation > Retrieval > Inference)
- Engine selection (HRE vs Buddhi for hypotheticals)
- Answer rendering (proof → natural language)
- Meta-observation (log decisions via Sakshin)

**Why This Component Matters**:

Provides execution control:
- Enforces observation-first priority
- Routes queries to appropriate reasoning engine (Buddhi vs HRE)
- Prevents circular dependencies

**Dependencies**: All components (orchestration layer)

**Execution Flow**:
```python
Input:  "Do bats produce milk?"

Execution Flow:
1. Manas.parse() → structured proposal
2. Check Perceptual Priors → None
3. Check Geographic Memory → None
4. Buddhi.answer() → YES (taxonomic inheritance)
5. Render → "Yes. Inherited from mammal."
6. Sakshin.log() → audit trail

Output: "Yes. Inherited from mammal."
```

---

### 6. Sakshin — Audit Logger

**Function**: Logs all reasoning decisions without interference. Pure observation, zero modification.

**Operations**:
- Event logging (query, assertion, error, lifecycle)
- Passive observation (NEVER modifies reasoning)
- Structured audit trail (timestamped, queryable)
- Pattern analysis (which reasoning paths are common?)
- Comprehensive logging (successes AND failures)

**Why This Component Matters**:

Enables verification and reproducibility:
- Complete audit trail of all decisions
- Debugging path for failures
- Research data for reasoning pattern analysis

**Dependencies**: None (observation only, no interference)

**Log Entry Format**:
```python
# Ahankara calls Sakshin
ahankara.answer("Do bats fly?")
→ Sakshin logs:
{
  "timestamp": "2024-12-16T08:15:30Z",
  "query": "Do bats fly?",
  "modules_used": ["manas", "buddhi"],
  "reasoning_path": ["direct_match → found"],
  "answer": "Yes.",
  "confidence": 0.95
}

# Sakshin NEVER changes answer, just records ✓
```

---

## Auxiliary Components (Optional)

**Note**: These are external knowledge sources that enhance coverage but are NOT required for core epistemic guarantees.

### 7. Perceptual Priors — Observational Knowledge Store

**Function**: Stores non-inferable observational facts (e.g., "gold is shiny").

**Why I Made It Auxiliary**:

MARC can function perfectly without Perceptual Priors:
- Buddhi would handle all queries (via reasoning or honest refusal)
- We'd lose observational knowledge, but the core reasoning loop stays intact

**What It Adds**:
- Fast path for perceptual queries (`"Is gold shiny?"` → direct lookup, no reasoning)
- Epistemic honesty (observational ≠ inferable)
- Prevents false inheritance (`"shiny"` doesn't propagate via taxonomy)

**Integration Point**: Ahankara checks Perceptual Priors BEFORE Buddhi (Priority 1)

**Example Without Perceptual Priors**:
```python
Query: "Is gold shiny?"

# Without auxiliary module:
buddhi.answer("Is gold shiny?")
→ Grounding check: "gold" + "shiny" not stored
→ "I do not know" (correct refusal, but inefficient)

# With auxiliary module:
perceptual_priors.has_property('gold', 'shiny')
→ 0.85 (stored observation)
→ "Yes (Perceptual observation, confidence: 0.85)"
→ EARLY RETURN (no reasoning needed) ✓
```

---

### 8. Geographic Memory — Spatial Knowledge

**Role**: I implemented this as a retrieval-only module for spatial containment facts ("London is in Europe").

**Why I Made It Auxiliary**:

MARC can function without Geographic Memory:
- Buddhi would handle geographic queries using the SPATIAL frame
- We'd lose external geographic knowledge, but core loop remains functional

**What It Adds**:
- Fast path for geographic queries (`"Is London in Europe?"` → graph traversal, no reasoning)
- Authoritative external source (more trustworthy than inference)
- Negative answers (`"Tokyo is in Asia, not Europe"`)

**Integration Point**: Ahankara checks Geographic Memory BEFORE Buddhi (Priority 2)

**Example Without Geographic Memory**:
```python
Query: "Is London in Europe?"

# Without auxiliary module:
buddhi.answer("Is London in Europe?")
→ SPATIAL frame (transitive, non-inheritable)
→ Might work if beliefs stored, but expensive reasoning

# With auxiliary module:
geo_memory.is_located_in('london', 'europe')
→ Traverse: london → uk → europe ✓
→ "Yes (London → UK → Europe)"
→ EARLY RETURN (fast graph traversal) ✓
```

---

## The Complete Query Flow

### How I Designed the Execution Pipeline

When a user asks a question, I route it through this pipeline:

```
┌───────────────────────────────────────────────────────────────┐
│                    USER QUERY                                 │
│              "Do bats produce milk?"                          │
└─────────────────────┬─────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────────┐
│                 AHANKARA (Orchestrator)                       │
│                                                               │
│  PHASE 1: PERCEPTION (Required)                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  MANAS: Parse natural language                          │  │
│  │  "Do bats produce milk?"                                │  │
│  │  → Belief(entities=['bat'],                             │  │
│  │           predicates=['produces_milk'],                 │  │
│  │           polarity=INTERROGATIVE)                       │  │
│  └─────────────────┬───────────────────────────────────────┘  │
│                    │                                          │
│  PHASE 2: EXTERNAL KNOWLEDGE (Auxiliary, Optional)            │
│  ┌─────────────────▼───────────────────────────────────────┐  │
│  │  PERCEPTUAL PRIORS: Observational lookup                │  │
│  │  has_property('bat', 'produces_milk')?                  │  │
│  │  → None (not perceptual)                                │  │
│  └─────────────────┬───────────────────────────────────────┘  │
│  ┌─────────────────▼───────────────────────────────────────┐  │
│  │  GEOGRAPHIC MEMORY: Spatial lookup                      │  │
│  │  is_located_in('bat', ...)?                             │  │
│  │  → None (not geographic)                                │  │
│  └─────────────────┬───────────────────────────────────────┘  │
│                    │                                          │
│  PHASE 3: REASONING ENGINE (Required)                         │
│  ┌─────────────────▼───────────────────────────────────────┐  │
│  │  Engine Selection:                                      │  │
│  │  Hypothetical? → NO                                     │  │
│  │  → Use BUDDHI (logical reasoning)                       │  │
│  └─────────────────┬───────────────────────────────────────┘  │
│  ┌─────────────────▼───────────────────────────────────────┐  │
│  │  BUDDHI: Logical inference                              │  │
│  │  1. Direct match in CHITTA → Not found                  │  │
│  │  2. Frame check → produces_milk is FUNCTIONAL           │  │
│  │     (inherits=True, negation_blocks=True)               │  │
│  │  3. Taxonomy traversal in CHITTA → bat → mammal         │  │
│  │  4. Negation dominance → No blocking negation           │  │
│  │  5. Positive inheritance → mammal produces_milk ✓       │  │
│  │  6. Proof construction → YES (confidence: 0.9)          │  │
│  └─────────────────┬───────────────────────────────────────┘  │
│                    │                                          │
│  PHASE 4: RENDERING (Required)                                │
│  ┌─────────────────▼───────────────────────────────────────┐  │
│  │  AHANKARA: Format proof as natural language             │  │
│  │  AnswerProof(verdict="yes", steps=[...])                │  │
│  │  → "Yes. Inherited from mammal."                        │  │
│  └─────────────────┬───────────────────────────────────────┘  │
│                    │                                          │
│  PHASE 5: META-OBSERVATION (Required)                         │
│  ┌─────────────────▼───────────────────────────────────────┐  │
│  │  SAKSHIN: Log decision trace                            │  │
│  │  {query, modules_used, reasoning_path, answer, ...}     │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────┬─────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────────┐
│                    FINAL ANSWER                               │
│          "Yes. Inherited from mammal."                        │
└───────────────────────────────────────────────────────────────┘
```

---

## Inter-Module Dependencies

### Dependency Graph I Designed

I structured the modules in tiers based on their dependencies:

```
TIER 1 (No dependencies):
  • MANAS     → Takes raw text (no dependencies)
  • CHITTA    → Data layer (no dependencies)
  • SAKSHIN   → Pure observer (no dependencies)

TIER 2 (Depends on Tier 1):
  • BUDDHI    → Requires CHITTA (belief memory to reason over)
  • HRE       → Requires CHITTA (deep copy for sandbox)

TIER 3 (Depends on Tier 2):
  • HRE       → Also requires BUDDHI (reasoning in sandbox)

TIER 4 (Orchestrator):
  • AHANKARA  → Coordinates ALL modules

AUXILIARY (Independent):
  • PERCEPTUAL PRIORS  → Standalone (plugs into Ahankara)
  • GEOGRAPHIC MEMORY  → Standalone (plugs into Ahankara)
```

**Dependency Flow**:
```
MANAS ─┐
       ├──→ AHANKARA ──→ (User answer)
CHITTA ┼──→ BUDDHI ─┘         ↑
       └──→ HRE ──────────────┤
                              │
PERCEPTUAL PRIORS ────────────┤
GEOGRAPHIC MEMORY ────────────┤
                              │
SAKSHIN ←─────────────────────┘ (logs everything)
```

---

## Design Principles

### 1. Modular Separation

**Principle**: Each component has a single responsibility.

```
PERCEPTION ≠ REASONING
  Manas parses, Buddhi reasons
  
REASONING ≠ MEMORY
  Buddhi infers, Chitta stores
  
MEMORY ≠ ORCHESTRATION
  Chitta stores, Ahankara coordinates
  
OBSERVATION ≠ INTERVENTION
  Sakshin watches, never changes
```

**Why I Insisted on This**:

- **Modularity**: I can replace or upgrade modules independently
- **Testability**: I can test each module in isolation
- **Debuggability**: Clear boundaries help me diagnose failures
- **Cognitive Realism**: This mirrors how the human brain actually works

**Human Brain Analogy**:

I modeled MARC after neuroscience findings:

Your brain has:
- Sensory cortex (perception) ≠ Prefrontal cortex (reasoning)
- Hippocampus (memory formation) ≠ Neocortex (reasoning)
- Executive function (coordination) ≠ Working memory (storage)

MARC models this separation explicitly.

---

### 2. Refusal-First Epistemic Contract

**Principle**: Correctness > Coverage. Honest refusal > Confident hallucination.

**Implementation**:

1. **Grounding Checks** (Buddhi):
   ```python
   Query: "Do copper objects conduct electricity?"
   
   # Could hallucinate: "Copper is conductive → Yes"
   # But instead:
   → Grounding check: "copper object" not in entity index
   → REFUSE: "I do not know"
   ```

2. **Negation Dominance** (Buddhi):
   ```python
   Query: "Do bats have gills?"
   
   # Without negation dominance: "I don't know"
   # With negation dominance: "No" (inherited from mammals)
   ```

3. **Epistemic Sterility** (HRE):
   ```python
   Query: "If bats were fish, would they have gills?"
   
   # Sandbox reasoning, then discard
   # Real beliefs UNCHANGED ✓
   ```

4. **Confidence Ceilings** (Perceptual Priors):
   ```python
   # Perceptual observations: max 85% confidence
   # Acknowledges perceptual uncertainty
   ```

**Design Philosophy**: Better to refuse honestly than hallucinate confidently.

---

## Epistemic Guarantees

### What EDRA Enforces

1. **Formal Proofs**: Every answer includes a derivation trace
2. **Grounding Discipline**: Refuses ungrounded inferences
3. **Negation Dominance**: Inherited negations block positive inheritance
4. **Frame Discipline**: Respects structural semantics (axiomatic commitments)
5. **Paraconsistency**: Tolerates contradictions without logical explosion
6. **Epistemic Sterility**: Hypotheticals never contaminate real beliefs
7. **Auditability**: All decisions logged
8. **Modular Separation**: Clean, testable component boundaries

### What EDRA Does NOT Provide

1. ❌ **High Coverage**: Will refuse many queries (intentional restraint)
2. ❌ **Fast Inference**: Prioritizes correctness over speed
3. ❌ **Learning**: No inductive generalization (this is a reasoning core only)
4. ❌ **Compositional Reasoning**: Refuses unbounded compositions
5. ❌ **Probabilistic Inference**: Uses confidence scores, not Bayesian updating

**These are intentional design boundaries, not defects.**

---

## Known Limitations

### What I Acknowledge Explicitly

1. **Language Understanding**:
   - EDRA does NOT model natural language semantics
   - Language is treated as a controlled input encoding layer
   - This is an engineering limitation, not a cognitive claim

2. **Relation Frames**:
   - Frames (TAXONOMIC, FUNCTIONAL, SPATIAL, STATE) are **hand-designed axiomatic commitments**
   - They did NOT emerge from data
   - EDRA explores the consequences of those commitments

3. **The Benchmark**:
   - The benchmark is **architecture-aligned by design**
   - Its purpose is NOT comparative performance but **falsification of epistemic guarantees**
   - "Success" is measured against stated epistemic contracts, not external leaderboards
   - The 84% precision is meaningful only within this epistemic framework

4. **No Task Execution**:
   - EDRA has no goals, no planning, no action execution
   - It is a reasoning core, not a complete cognitive system

5. **No Learning**:
   - No inductive generalization
   - No weight updates
   - Beliefs are manually taught or retrieved from external sources

---

## Comparison to Other Systems

### EDRA vs ACT-R

| Dimension | ACT-R | EDRA |
|-----------|-------|------|
| Goal | Model human cognition | Enforce epistemic discipline |
| Focus | Psychological fidelity | Correctness guarantees |
| Error handling | Models how humans err | Prevents specific error classes |
| Learning | Activation decay, chunking | None (reasoning core only) |
| Timing | Latency modeling | Not modeled |
| Epistemic guarantees | Weak | Strong (refusal, grounding, proofs) |

**ACT-R cares about simulating human cognition. EDRA cares about enforcing epistemic contracts.**

### EDRA vs SOAR

| Dimension | SOAR | EDRA |
|-----------|------|------|
| Goal | Problem solving | Epistemic reasoning |
| Focus | Task execution | Belief management |
| Architecture | Goal stacks, operators | Refusal-first inference |
| Learning | Chunking | None |
| Planning | Yes | No |
| Epistemic guarantees | Weak | Strong |

**SOAR cares about doing things. EDRA cares about knowing things honestly.**

### EDRA vs Knowledge Graphs

| Dimension | KGs | EDRA |
|-----------|-----|------|
| Refusal semantics | No | Yes (explicit) |
| Proof obligations | No | Yes (always) |
| Contradiction handling | Varies | Paraconsistent by design |
| Grounding checks | No | Yes (always) |

### EDRA vs LLMs

| Dimension | LLMs | EDRA |
|-----------|------|------|
| Optimization | Loss minimization | Epistemic correctness |
| Hallucination | Confident hallucinations | Refuses when unsure |
| Proofs | None | Always provided |
| Grounding | Blurred | Explicit checks |

---

## Use Cases

### 1. Research Artifact

EDRA demonstrates that refusal-first epistemic contracts are implementable and testable.

### 2. Epistemic Benchmark Baseline

Compare other systems against EDRA's explicit guarantees (refusal rates, proof obligations, grounding discipline).

### 3. Educational Tool

Teach formal reasoning, epistemic humility, and audit trail construction.

### 4. Reasoning Verification

Verify logical consistency and grounding of inference chains in knowledge bases.

---

## Performance Characteristics

### Measured Query Latencies

```
FAST PATH (Direct match or external knowledge):
  • Perceptual Prior lookup: ~1ms
  • Geographic Memory traversal: ~5ms
  • Direct belief match: ~10ms
  Total: ~15ms

REASONING PATH (Buddhi inference):
  • Perception (Manas): ~10ms
  • Taxonomic traversal: ~20ms
  • Inference + proof construction: ~30ms
  • Rendering: ~5ms
  • Logging: ~2ms
  Total: ~70ms

HYPOTHETICAL PATH (HRE sandbox):
  • Perception: ~10ms
  • Sandbox creation: ~50ms
  • Reasoning in sandbox: ~50ms
  • Rendering: ~5ms
  Total: ~120ms
```

**Known Bottlenecks** (intentionally not optimized for speed):
- Manas parsing (regex-based, not optimized)
- Sandbox deep copy (HRE overhead)
- Taxonomy traversal (graph depth)

**Not Optimized For**:
- Real-time queries (70-120ms acceptable for research)
- Large knowledge bases (in-memory graph, not production-scale)
- Concurrent queries (single-threaded execution)

**Optimized For**:
- Correctness (formal proofs)
- Auditability (complete logs)
- Epistemic discipline (grounding checks)

---

## Benchmark Methodology

### Performance Metrics

Measured on epistemic discipline benchmark:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Precision | ~80-85% | Correct answers when not refusing |
| Restraint | ~85% | Refusal rate on uncertain queries |
| False Positive Rate | ~10% | Rare incorrect confident answers |
| Negation Accuracy | 81.8% | Inherited negation handling |
| Inheritance | 81.8% | Taxonomic property propagation |

### Benchmark Disclaimer

**The benchmark is architecture-aligned by design.**

Its purpose is NOT comparative performance against other systems.  
Its purpose IS **falsification of stated epistemic guarantees**.

When I report "84% precision is a success," I mean:
- Success according to the epistemic contract EDRA enforces
- NOT success on external leaderboards or standardized tests
- The benchmark tests whether EDRA upholds its refusal semantics, grounding checks, and proof obligations

This is transparent by design.

### Honest Failures (Intentional Refusals)

These demonstrate restraint, not bugs:
- **Ungrounded compositions**: "Do copper objects conduct electricity?" → REFUSED (not taught)
- **Complex spatial queries**: Handled by Geographic Memory (external retrieval)
- **Perceptual properties**: Handled by Perceptual Priors (observation layer)

---

## Future Work

###Potential Extensions (Outside Current Scope)

1. **Inductive Learning**:
   - Generalization from instances
   - Abductive reasoning
   - Analogical transfer

2. **Probabilistic Reasoning**:
   - Bayesian updates
   - Uncertainty propagation

3. **Multi-Agent Systems**:
   - Belief merging protocols
   - Collaborative reasoning

4. **Goal-Directed Planning**:
   - Action execution
   - Value alignment

**Why Not Implemented**:

EDRA is a reasoning core, not a complete cognitive system. Adding these would dilute focus and complicate verification of epistemic guarantees.

---

## Summary

```
╔═══════════════════════════════════════════════════════════════╗
║           EDRA: Epistemically Disciplined                     ║
║              Reasoning Architecture                           ║
║                                                               ║
║  PART OF: MARC (Mind Architecture for Reasoning & Cognition) ║
║  SCOPE: Epistemic reasoning core only                         ║
║                                                               ║
║  CORE CONTRIBUTION:                                           ║
║    Refusal-first epistemic contract enforcing:                ║
║    • Explicit grounding checks                               ║
║    • Explicit negation dominance                             ║
║    • Explicit sandboxed hypotheticals                        ║
║    • Explicit proof traces                                   ║
║    • Explicit observation vs inference separation            ║
║                                                               ║
║  COMPONENTS:                                                  ║
║    1. MANAS     → Input parsing                              ║
║    2. BUDDHI    → Inference engine                           ║
║    3. CHITTA    → Belief storage                             ║
║    4. HRE       → Hypothetical sandbox                       ║
║    5. AHANKARA  → Orchestrator                               ║
║    6. SAKSHIN   → Audit logger                               ║
║                                                               ║
║  DESIGN PRINCIPLES:                                           ║
║    Correctness > Speed                                       ║
║    Honest refusal > Confident hallucination                  ║
║    Proof obligations > Coverage maximization                 ║
║                                                               ║
║  EDRA IS NOT:                                                 ║
║    ❌ A cognitive/neuroscience model                         ║
║    ❌ A learning system                                      ║
║    ❌ A complete AI system                                   ║
║    ❌ Benchmark-optimized                                    ║
║                                                               ║
║  EDRA IS:                                                     ║
║    ✓ Normative epistemic control architecture                ║
║    ✓ Research artifact                                       ║
║    ✓ Educational tool                                        ║
║    ✓ Baseline for epistemic comparison                       ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**Core Claim**: EDRA prevents error classes other systems cannot address:
- LLMs hallucinate (loss minimization)
- KGs overcommit (no refusal semantics)
- Classical logic explodes (contradiction intolerance)
- Probabilistic systems blur grounding

EDRA enforces explicit epistemic boundaries.


---

### Relation Frames: Axiomatic Commitments

**Important**: Relation frames are **hand-designed axiomatic commitments**, not emergent properties.

EDRA explores the consequences of these commitments:

| Relation Type | Transitive? | Inherits? | Negation Blocks? | Example |
|--------------|-------------|-----------|------------------|---------|
| TAXONOMIC | ✓ | ✓ | ✓ | is_a, instance_of |
| SPATIAL | ✓ | ✗ | ✗ | located_in, part_of |
| FUNCTIONAL | ✗ | ✓ | ✓ | produces_milk, has_gills |
| STATE | ✗ | ✗ | ✗ | is_liquid, is_solid |

**Rationale**: These distinctions capture structural semantics I committed to axiomatically.

### Negation Dominance Rule

Inherited negations BLOCK positive inheritance:

```
Mammals do NOT have gills (NEGATIVE, FUNCTIONAL)
Bats are mammals (TAXONOMIC)
Query: "Do bats have gills?"
→ NO (negation blocks, not "unknown")
```

### Grounding Discipline

EDRA verifies predicate grounding before reasoning:
1. **Direct grounding**: Belief with (entity, predicate) exists
2. **Taxonomic grounding**: Ancestor has this predicate  
3. **Simple property check**: ≤2 entities, entity exists

EDRA refuses unbounded compositions:
- "Do copper objects conduct electricity?" → UNKNOWN (composition not taught)

This demonstrates **epistemic restraint** by design.

---

## Installation & Usage

### Installation

EDRA runs on pure Python with no external dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/epistemic-reasoning-core.git
cd epistemic-reasoning-core

# No external dependencies needed
python3 --version  # Requires 3.10+
```

### Interactive Mode

Experiment with EDRA interactively:

```bash
# Run interactive terminal
python main.py

# Or directly:
cd 0.tests
python test_interactive.py
```

### Run Benchmarks

```bash
cd 0.tests

# Epistemic discipline benchmark (brutal)
python test_benchmark.py

# Stress test
python test_stress.py

# Run all tests
python run_all_tests.py
```

---

## Benchmark Results

### Current Performance Metrics

I tested MARC on a brutal epistemic discipline benchmark:

| Metric | Score | My Interpretation |
|--------|-------|-------------------|
| Precision | ~80-85% | Principled reasoning with honest failures |
| Restraint | ~85% | Says "I don't know" when uncertain |
| False Positive Rate | ~10% | Rare hallucinations |
| Negation Accuracy | 81.8% | Correctly blocks inheritance |
| Inheritance | 81.8% | Taxonomic property propagation |

**My Philosophy**: I accept ~80-90% precision if it means maintaining high restraint.

> "If you push this system to hit 95% now, you will destroy its intellectual honesty." — My design principle

### Honest Failures (Not Bugs)

These are INTENTIONAL refusals that demonstrate cognitive realism:

- **Spatial containment**: "Is London in Europe?" → Now handled by geographic memory
- **State predicates**: "Is water liquid?" → Handled by perceptual priors
- **Ungrounded compositions**: "Do copper objects conduct electricity?" → Correctly refuses

These demonstrate **cognitive realism**: systems don't need to reason about everything.

---

> "This system models epistemic discipline: the ability to reason without overclaiming knowledge, even under contradiction."

### Key Claims

1. **Grounding discipline**: System refuses ungrounded inferences (unlike LLMs)
2. **Restraint**: Says "I don't know" when uncertain (unlike confidence-maximizing systems)
3. **Negation tolerance**: Tolerates contradictions without explosion (paraconsistent)
4. **Frame discipline**: Distinguishes logical inference from external memory
5. **Structural semantics**: Relation frames model HOW relations behave, not just what they're called

---

## Usage Examples

### Teaching EDRA

```python
from ahankara import Ahankara

edra = Ahankara()
edra.set_learning_mode()  # Disable decay during teaching

# Teach taxonomic hierarchy
edra.process("Bats are mammals.")
edra.process("Mammals produce milk.")
edra.process("Mammals do not have gills.")

# Switch to reasoning mode
edra.set_reasoning_mode()
```

### Querying EDRA

```python
# Taxonomic inheritance (FUNCTIONAL inherits=True)
answer = edra.ask("Do bats produce milk?")
# → "Yes." (inherited from mammals)

# Negation dominance (FUNCTIONAL negation_blocks=True)
answer = edra.ask("Do bats have gills?")
# → "No." (mammals don't have gills, negation blocks)

# Grounding refusal (composition not taught)
answer = edra.ask("Do copper objects conduct electricity?")
# → "I do not know." (epistemic restraint)

# Perceptual priors (observational knowledge)
answer = edra.ask("Is gold shiny?")
# → "Yes (perceptual: 85% confidence) - gold appears shiny"

# Geographic memory (external retrieval)
answer = edra.ask("Is London in Europe?")
# → "Yes (geographic memory) - london → uk → europe"
```

---

## 🔍 Testing

### Run Interactive Mode

```bash
python main.py
```

Commands:
- Type statements to teach EDRA
- Ask questions to query knowledge
- `beliefs` - show all stored beliefs
- `stats` - show system statistics
- `quit` - exit

### Run Benchmark

```bash
cd 0.tests
python test_benchmark.py
```

Tests epistemic guarantees:
- Taxonomic inheritance
- Negation dominance
- Grounding discipline
- Relation frame enforcement
- Perceptual priors
- Geographic memory
- Paraconsistency

### Run All Tests

```bash
cd 0.tests
python run_all_tests.py
```

---

## Citation

If you use EDRA in your research, please cite:

```bibtex
@software{edra2024,
  title={EDRA: Epistemically Disciplined Reasoning Architecture},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/epistemic-reasoning-core},
  note={A refusal-first epistemic control architecture}
}
```

---

## License

MIT License — See LICENSE file for details.

---

## Acknowledgments

Theoretical foundations inspired by:
- **Justification logic** (Artemov) — Formal proofs for beliefs
- **Proof assistants** (Coq, Lean) — Verified reasoning systems
- **Formal epistemology** — Knowledge representation theory
- **Paraconsistent logic** — Contradiction tolerance
- **Symbolic AI** (ACT-R, SOAR) — Modular cognitive architectures

**Naming convention**: Sanskrit terms from Vedantic philosophy used as **design metaphors**, not cognitive claims:
- **Manas** (मनस्) — Input layer
- **Buddhi** (बुद्धि) — Reasoning
- **Chitta** (चित्त) — Storage
- **Ahankara** (अहंकार) — Orchestration
- **Sakshin** (साक्षिन्) — Observation

---

## Contact & Contributing

Issues, questions, and pull requests welcome. When contributing, please maintain:
- Epistemic guarantees (refusal semantics, grounding checks, proof obligations)
- Component separation (modular boundaries)
- Test coverage (verify epistemic contracts)

**I will reject changes that sacrifice epistemic discipline for benchmark scores.**

---

**Design philosophy**: Better to refuse honestly than hallucinate confidently.
