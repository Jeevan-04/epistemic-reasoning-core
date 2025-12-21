# EDRA: Epistemically Disciplined Reasoning Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a href="https://github.com/Jeevan-04/epistemic-reasoning-core/blob/main/Details.pdf">
  <img src="https://github.com/user-attachments/assets/b1359060-3d2c-4a78-a4e5-8cc1e0b3a326" width="400">
</a>

**Click here for detailed paper 📄 ↑**

> A reasoning system that knows when to say "I don't know"

---

## About This Project

Here's the thing I wanted to build: a reasoning system that doesn't bullshit.

You know how most AI systems will give you an answer to basically anything? Ask them about copper objects and conductivity when they've only learned "copper conducts electricity," and they'll confidently tell you something. Maybe it's right, maybe it's not. But they'll answer.

I think that's backwards. 

**The intention:** Build a system that refuses to answer when it doesn't have enough evidence. Make refusal an explicit, formal part of how it works—not a bug, not a fallback, but a feature. A system that tracks what it knows, what it can infer from what it knows, and what's beyond its boundaries.

**Why this matters:** Because in the real world, "I don't know" is often the most honest answer. Medical diagnosis, legal reasoning, safety-critical systems—these need systems that know their limits. Better to refuse than to hallucinate.

**What makes it different:** Three things. First, every predicate gets grounded before inference (if you've never learned about "telepathy," you refuse questions about telepathic abilities). Second, negations are stronger than positives (if mammals don't have gills, that blocks any future claim that bats have gills). Third, every answer comes with a proof trace showing exactly which beliefs were used and how.

This is EDRA. It's a symbolic reasoning core that enforces epistemic discipline.

**A note on design philosophy:** EDRA is not a model of human cognition or moral reasoning. Its design is inspired by epistemic aspects of human reasoning—specifically the ability to withhold judgment, track justification, and respect counterevidence—but these inspirations are conceptual rather than psychological or neuroscientific. What EDRA demonstrates is not mind simulation, but **epistemic governance**: the separation of epistemic responsibility from task performance. Most systems ask "Can I produce an answer?" EDRA asks "Am I epistemically allowed to answer?"

---

## How I'm Approaching the Problem

Most reasoning systems face a trade-off: coverage vs correctness. Answer more questions → higher chance of being wrong somewhere. Answer fewer questions → more likely to be right on what you do answer.

I decided to optimize for correctness. Here's the approach:

### 1. Grounding First

Before making any inference, check: "Have I actually been taught about this predicate in this context?"

If you ask "Do bats time-travel?" and I've never seen the predicate "time-travel" in my knowledge base, I refuse. Doesn't matter if I know everything else about bats. Ungrounded predicate = automatic refusal.

This prevents the system from making up answers about concepts it's never encountered.

### 2. Negations Dominate Positives

When you teach me "Mammals don't have gills" and then "Bats are mammals," that negative fact propagates down. It blocks any future attempt to say "Bats have gills."

Even if someone later tries to teach "Flying animals have gills," the system checks: "Wait, is there a negation anywhere in the taxonomy?" Finds "mammals → NOT(gills)" → blocks the positive claim.

This implements what philosophers call defeasible reasoning—specific evidence overrides general patterns.

### 3. Explicit Refusal Semantics

Refusal isn't treated like an error. It has formal conditions:
- Grounding failure (predicate never encountered)
- Insufficient evidence (no inference path exists)
- Compositional gap (pattern not taught)

Each refusal comes with a justification explaining why. This makes the boundary between known and unknown explicit and auditable.

### 4. Proof Obligations

If the system gives an answer (YES or NO), it must provide a proof. A trace showing:
- Which stored beliefs were consulted
- Which inference rules were applied
- What the derivation chain looks like

No proof = no answer. This forces transparency.

### 5. Modular Architecture

I split the system into six independent components:
- **Manas**: Parses input into structured beliefs
- **Chitta**: Stores beliefs in a hypergraph
- **Buddhi**: Does the actual reasoning
- **HRE**: Handles hypothetical "what if" queries in isolation
- **Ahankara**: Orchestrates everything
- **Sakshin**: Logs all decisions passively

Each component has a narrow interface. They don't call each other directly—all coordination goes through the orchestrator. This makes it possible to test and verify each piece independently.

---

## What I Built & How It Works

### The System Breakdown

Think of EDRA like a pipeline with checkpoints:

**Step 1: Parse the input**  
You type: "Do bats produce milk?"  
Manas extracts: entities = [bat], predicates = [produces_milk], polarity = QUESTION  

**Step 2: Check grounding**  
Buddhi asks: "Have I seen 'produces_milk' in my knowledge base?"  
Searches beliefs: finds "mammal + produces_milk" → grounding satisfied ✓

**Step 3: Look for direct match**  
Chitta searches: "Do I have a belief about bat + produces_milk specifically?"  
Result: No direct match

**Step 4: Try taxonomic inheritance**  
Buddhi checks taxonomy: bat → mammal (distance: 1 hop)  
Chitta retrieves: "mammal + produces_milk = TRUE"  
Frame check: produces_milk is FUNCTIONAL → inheritance allowed ✓

**Step 5: Check for blocking negations**  
Buddhi searches: "Are there any negations in the path?"  
Check ancestors: bat → mammal → animal  
Search negations: None found ✓

**Step 6: Construct proof**  
Proof = [  
  (bat IS-A mammal) — taxonomic link  
  (mammal produces_milk) — stored belief  
  (FUNCTIONAL frame allows inheritance) — inference rule  
]  
Conclusion: bat produces_milk, confidence 0.9

**Step 7: Return answer with proof**  
Output: "Yes. Inherited from mammal."  
Proof available in logs  
Sakshin records everything

That's it. Every query goes through these checkpoints. If any checkpoint fails, the system refuses.

### Example: Negation Dominance

Let me walk you through how negation blocking actually works.

**Teaching phase:**
```
You: "Mammals do not have gills."
System stores: (mammal, has_gills, NEGATIVE)

You: "Bats are mammals."
System stores: (bat IS-A mammal) — taxonomic link
```

**Query phase:**
```
You: "Do bats have gills?"

Step 1: Parse → entities=[bat], predicates=[has_gills]
Step 2: Grounding check → has_gills is grounded (found in mammal belief) ✓
Step 3: Direct match? → No belief about bat+has_gills
Step 4: Taxonomy check → bat → mammal
Step 5: Negation check → FOUND: mammal has NOT(has_gills)
Step 6: Blocking rule triggers → Negative inheritance blocks any positive claim
Step 7: Answer: "No. Mammals do not have gills, inherited to bat."
```

The key: Once a negation enters the taxonomy, it propagates down and blocks contradictory positives. This prevents the system from asserting "Bats have gills" even if someone later tries to teach "Flying animals have gills."

### Example: Hypothetical Reasoning

Here's what happens when you ask a "what if" question:

**Query:** "If bats were fish, would they have gills?"

**Processing:**

```
Step 1: HRE detects hypothetical pattern → "If X were Y"
Step 2: Create sandbox:
  - Deep copy entire belief store (Chitta snapshot)
  - This takes ~50ms for 1000 beliefs
  
Step 3: Modify sandbox:
  - Original: bat IS-A mammal
  - Sandbox: bat IS-A fish (assumption added)
  
Step 4: Reason in sandbox:
  - Check: fish have gills? → Yes (if taught)
  - Inherit: bat → fish → has_gills
  - Conclusion: bat has_gills (IN SANDBOX ONLY)
  
Step 5: Return answer:
  "If bats were fish, then yes, they would have gills."
  
Step 6: Destroy sandbox:
  - Discard all sandbox beliefs
  - Original beliefs unchanged
  - Verify: bat IS-A mammal (still true in main store)
```

The critical guarantee: Hypotheticals never contaminate real beliefs. The sandbox is epistemically sterile.

### The Component Teardown

Let me break down what each piece actually does:

**MANAS (Input Parser):**
- Takes messy text → produces clean structures
- Uses regex patterns (not trying to solve NLP)
- Extracts entities, predicates, polarity
- Normalizes variations ("bat" and "bats" → "bat")
- ~150 lines of code

**CHITTA (Belief Storage):**
- Hypergraph with three indexes:
  - Entity index: bat → all beliefs mentioning bat
  - Predicate index: has_gills → all gills-related beliefs
  - Taxonomy graph: bat → mammal → animal → thing
- Supports O(1) entity lookups, O(k) taxonomy traversal
- Allows contradictions (paraconsistent design)
- ~200 lines of code

**BUDDHI (Reasoning Engine):**
- Grounding verification (refuse if predicate unseen)
- Taxonomy traversal (climb IS-A hierarchy)
- Negation blocking (search for inherited negations)
- Proof construction (build derivation trace)
- Relation frame semantics (different rules for different predicate types)
- ~400 lines of code (the biggest component)

**HRE (Hypothetical Reasoning Engine):**
- Creates isolated belief sandboxes
- Deep-copies entire knowledge base
- Makes counterfactual assumptions
- Reasons in isolation
- Guarantees epistemic sterility (no contamination)
- ~100 lines of code

**AHANKARA (Orchestrator):**
- Routes queries to appropriate components
- Enforces execution order
- Manages state transitions
- Formats responses
- Components never call each other—only Ahankara does
- ~150 lines of code

**SAKSHIN (Meta-Observer):**
- Logs everything passively
- Records: queries, beliefs consulted, inference steps, refusals
- Read-only (never modifies beliefs or influences reasoning)
- Makes system auditable and debuggable
- ~100 lines of code

Total: ~1200 lines of Python. No external dependencies.

---

## Overall System Flow

Here's the complete picture of how everything fits together:

```
USER INPUT
    ↓
[MANAS: Parse]
    → Structured belief/query
    ↓
[AHANKARA: Route]
    → Determine query type
    ↓
    ├─→ Hypothetical? → [HRE: Sandbox reasoning]
    │                      ↓
    │                   [Isolated CHITTA copy]
    │                      ↓
    │                   [BUDDHI in sandbox]
    │                      ↓
    │                   Answer + destroy sandbox
    │
    └─→ Factual? → [BUDDHI: Reason]
                      ↓
                   [Check grounding]
                      ↓
                   [Query CHITTA]
                      ↓
                   [Search direct match] ━→ Found? → Answer
                      ↓ Not found
                   [Traverse taxonomy]
                      ↓
                   [Check negations] ━→ Found? → Block & refuse
                      ↓ None
                   [Apply inference]
                      ↓
                   [Build proof]
                      ↓
                   Answer with trace
    ↓
[SAKSHIN: Log]
    → Record decision trace
    ↓
[AHANKARA: Format]
    → Human-readable response
    ↓
USER OUTPUT
```

Every path goes through grounding checks. Every answer has a proof or a refusal justification. Every decision gets logged.

---

## What This Actually Achieves

I tested EDRA on 44 queries designed to break epistemic guarantees. Here's what happened:

**Grounding discipline:** Perfect. All 8 ungrounded queries were correctly refused. Zero false refusals (queries that should've been answered but weren't).

**Negation handling:** 82% accurate. The system correctly blocked inherited positives in most cases. Failures happened when multiple inheritance paths conflicted (bat → mammal AND bat → flying_animal, both trying to assert opposite things).

**Taxonomic inference:** 82% accurate. Successfully propagated properties down IS-A hierarchies. Failures mostly in deep chains (>3 hops) where confidence decay triggers refusal.

**Hypothetical isolation:** 100%. Not a single hypothetical assumption leaked into the main belief store. Epistemic sterility guaranteed.

**Overall precision:** 84% on answerable queries, with 85% restraint rate.

What this means: The system maintains high precision by refusing aggressively. It answers 15% of what it's asked, but gets those answers right 84% of the time.

Is 84% good enough? Depends. For safety-critical systems where a wrong answer is worse than no answer, yes. For general question-answering where coverage matters, no.

The point isn't to maximize accuracy. The point is to demonstrate that epistemic discipline—knowing when to shut up—is compatible with high precision on what you do answer.

---

## The Finish Line (For Now)

This is where the project stands:

**What works:** Grounding checks, negation dominance, proof traces, hypothetical sandboxing, modular architecture. The core mechanisms for epistemic discipline are functional and verified.

**What doesn't:** Natural language robustness (brittle parsing), learning (no inductive generalization), composition (refuses "copper objects conduct electricity" even though it knows "copper conducts electricity"), scaling (everything's in memory).

**What it's good for:** Research into epistemic reasoning, verification substrate for hybrid neural-symbolic systems, demonstrating that refusal can be formalized as a first-class outcome.

**What it's not good for:** General-purpose question answering, real-world deployment without additional infrastructure, anything requiring common sense or inductive reasoning.

**The core insight:** You can build a reasoning system that refuses confidently, provides formal justifications, and maintains epistemic integrity—by restricting expressiveness and accepting lower coverage.

**What I learned:** Epistemic discipline is possible. It requires formal semantics for refusal, strict grounding checks, and proof obligations. The trade-off is expressiveness for honesty. Whether that trade-off is worth it depends on your application.

**Next steps:** Add learning mechanisms (inductive pattern recognition), improve composition (teach the system about "X objects" patterns), implement probabilistic confidence (Bayesian belief updating), test on real-world domains (medical diagnosis, legal reasoning).

But for now, this is EDRA. A reasoning system that knows when it doesn't know. Built to refuse with pride.

---

## Installation & Usage

```bash
# Clone and run
git clone https://github.com/Jeevan-04/epistemic-reasoning-core.git
cd epistemic-reasoning-core
python main.py  # Requires Python 3.10+, no dependencies
```

### Quick Start

```python
from ahankara import Ahankara

edra = Ahankara()
edra.set_learning_mode()

# Teach facts
edra.process("Bats are mammals.")
edra.process("Mammals produce milk.")
edra.process("Mammals do not have gills.")

edra.set_reasoning_mode()

# Query
edra.ask("Do bats produce milk?")  # → "Yes. Inherited from mammal."
edra.ask("Do bats have gills?")     # → "No. Mammals do not have gills."
edra.ask("Do bats time-travel?")    # → "I do not know." (ungrounded)
```

### Run Tests

```bash
cd 0.tests
python test_benchmark.py  # Full benchmark (44 queries)
```

---

## System Architecture

EDRA consists of six core components working together:

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
│                 "Do bats produce milk?"                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MANAS: Input Parser                                        │
│  Converts natural language → structured belief proposals    │
│  Output: Belief(entities=['bat'],                           │
│                 predicates=['produces_milk'],               │
│                 polarity=INTERROGATIVE)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  AHANKARA: Query Orchestrator                               │
│  Routes query through priority levels:                      │
│    1. Check Perceptual Priors (observed facts)             │
│    2. Check Geographic Memory (spatial lookups)             │
│    3. Route to inference engine                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  BUDDHI: Inference Engine                                   │
│  1. Check if predicate is grounded                          │
│  2. Look for direct matches in CHITTA                       │
│  3. Try taxonomic inheritance (bat → mammal)                │
│  4. Check for blocking negations                            │
│  5. Construct formal proof or refuse                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  CHITTA: Belief Storage                                     │
│  Hypergraph storing all beliefs with:                       │
│    • Entity index (bat → all beliefs about bats)           │
│    • Predicate index (produces_milk → all beliefs)          │
│    • Taxonomy graph (bat → mammal → animal)                │
│  Returns: "mammals produce milk" (stored belief)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  SAKSHIN: Audit Logger                                      │
│  Records decision trace: query → beliefs consulted →        │
│  inference path → final answer                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Final Answer                            │
│  "Yes. Inherited from mammal."                              │
│  Proof: [bat IS-A mammal, mammal produces_milk]            │
└─────────────────────────────────────────────────────────────┘
```

---

## Relation Frames

### The Four Frame Types

| Frame | Transitive? | Inherits? | Negation Blocks? | Examples |
|-------|-------------|-----------|------------------|----------|
| TAXONOMIC | Yes | Yes | Yes | is_a, instance_of |
| FUNCTIONAL | No | Yes | Yes | produces_milk, has_wings |
| SPATIAL | Yes | No | No | located_in, part_of |
| STATE | No | No | No | is_liquid, is_shiny |

### Frame Inference Rules

**TAXONOMIC frame** (is_a):
```
IF:   X is_a Y
AND:  Y is_a Z
THEN: X is_a Z  (transitivity)
```

**FUNCTIONAL frame** (produces_milk):
```
IF:   bat is_a mammal
AND:  mammal produces_milk
THEN: bat produces_milk  (positive inheritance)

IF:   mammal NOT(has_gills)
AND:  bat is_a mammal
THEN: bat NOT(has_gills)  (negative inheritance + blocking)
```

**SPATIAL frame** (located_in):
```
IF:   Paris located_in France
AND:  France located_in Europe
THEN: Paris located_in Europe  (transitivity, no inheritance)
```

**STATE frame** (is_shiny):
```
No inference rules. States don't propagate.
Must be directly observed or asserted.
```

---

## Comparison to Related Work

### EDRA vs Knowledge Graphs

Traditional KGs (DBpedia, Wikidata, Freebase):
- Store facts as triples
- Support SPARQL queries
- No refusal mechanism (return empty set or nothing)
- No formal proof traces
- Contradictions cause query failures

EDRA:
- Stores beliefs with provenance
- Supports natural language queries
- Refuses when ungrounded
- Always provides proof traces
- Contradictions coexist (paraconsistent)

### EDRA vs LLMs

Large Language Models (GPT, Claude, Llama):
- Optimize for next-token prediction
- High coverage, low restraint
- Hallucinate on edge cases
- No formal proofs
- Probabilistic confidence

EDRA:
- Optimizes for correctness
- Lower coverage, high restraint
- Refuses on edge cases
- Formal proof traces
- Explicit grounding checks

### EDRA vs ACT-R / SOAR

ACT-R:
- Simulates human cognition
- Models reaction time, errors
- Goal-directed behavior
- Learning through experience

SOAR:
- Problem-solving architecture
- Goal stacks, operators
- Chunking for learning
- Task execution focus

EDRA:
- Enforces epistemic discipline
- No timing models
- Refusal-first design
- Reasoning focus, no task execution

The key difference: ACT-R and SOAR model *how humans think*. EDRA models *how to think carefully*.

---

## Limitations

### What EDRA Doesn't Do

**Natural Language Understanding**: EDRA treats language as a controlled encoding layer. It uses regex patterns and simple parsing. It doesn't understand context, pragmatics, or ambiguity the way humans do.

**Learning**: There's no inductive generalization, no learning from examples, no weight updates. You teach facts explicitly. The system doesn't discover patterns on its own.

**Compositional Reasoning**: EDRA refuses queries involving compositional patterns it hasn't been taught. "Do copper objects conduct electricity?" gets refused even though "copper conducts electricity" is known.

**Scaling**: The system stores everything in memory. Works fine for thousands of beliefs. Doesn't scale to millions. This is a research prototype, not a production system.

**Common Sense**: EDRA doesn't have common sense. It knows what you teach it. Nothing more.

---

## Future Directions

### Potential Extensions

**Inductive Learning**: Currently, EDRA only stores what you teach it. We could add pattern recognition—observe many instances of "X is_a Y" and "Y has_property P" implies "X has P", then generalize the rule.

**Probabilistic Reasoning**: Right now, confidence scores are simple floats. We could implement Bayesian belief propagation for uncertainty handling.

**Multi-Agent Systems**: What happens when two EDRAs with different belief stores communicate? How do they merge knowledge? How do they handle disagreements?

**Compositional Semantics**: Teaching EDRA about compositional patterns ("X objects have properties of X") would expand coverage without sacrificing grounding.

**Natural Language Improvement**: Replace regex parsing with proper NLP. This would improve robustness on varied inputs.

**Normative Reasoning Integration**: Future extensions could integrate normative or value-based layers atop EDRA's epistemic core, allowing moral or policy reasoning systems to operate over beliefs that are explicitly grounded and justified.

### Research Questions

- Can we prove formal bounds on EDRA's refusal rate given knowledge coverage?
- What's the relationship between grounding strictness and reasoning power?
- How do paraconsistent belief stores affect downstream reasoning quality?
- Can we automatically learn relation frame assignments from data?

---

## Citation

If you use EDRA in your research:

```bibtex
@software{edra2024,
  title={EDRA: Epistemically Disciplined Reasoning Architecture},
  author={Jeevan Naidu},
  year={2024},
  url={https://github.com/Jeevan-04/epistemic-reasoning-core},
  note={A refusal-first epistemic reasoning system}
}
```

---

## Acknowledgments

Theoretical foundations:
- Justification logic (Artemov)
- Paraconsistent logic (da Costa, Priest)
- Formal epistemology (Hintikka, van Fraassen)
- Proof assistants (Coq, Lean)

Naming inspired by Vedantic philosophy:
- Manas (मनस्) - Input processing
- Buddhi (बुद्धि) - Reasoning
- Chitta (चित्त) - Storage
- Ahankara (अहंकार) - Orchestration
- Sakshin (साक्षिन्) - Observation

---

## License

MIT License - See LICENSE file

---

*Built with intellectual honesty. Refuses with pride.*
