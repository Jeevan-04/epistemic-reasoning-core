# EDRA: Epistemically Disciplined Reasoning Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A refusal-first epistemic reasoning system that knows when to say "I don't know"

---

## Overview

EDRA is a symbolic reasoning engine built around a simple idea: **systems should refuse to answer when they lack sufficient evidence**, rather than hallucinate plausible-sounding responses.

This is the reasoning core of MARC (Mind Architecture for Reasoning and Cognition), a long-term research program exploring how to build systems that think carefully instead of just thinking fast. EDRA handles the "careful thinking" part—belief management, logical inference, and epistemic honesty.

The problem we're addressing: Most AI systems optimize for coverage. They try to answer everything. This creates a tension—the more questions you answer, the more likely you are to be wrong about something. We think there's value in building systems that prioritize correctness over completeness.

Our approach: Explicit refusal semantics, grounding checks before inference, formal proof traces, and paraconsistent belief storage.

Current status: Research prototype. Works well on constrained reasoning tasks.

---

## Table of Contents

- [Core Ideas](#core-ideas)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Relation Frames](#relation-frames)
- [Comparison to Related Work](#comparison-to-related-work)
- [Limitations](#limitations)
- [Future Directions](#future-directions)
- [Citation](#citation)

---

## Core Ideas

### 1. Refusal as a First-Class Operation

Most reasoning systems have two outputs: YES or NO. EDRA has three: YES, NO, and **I DON'T KNOW**. The third option isn't a failure mode—it's a design feature.

When you ask "Do copper objects conduct electricity?" and the system has only learned "Copper conducts electricity" but never encountered the compositional pattern "X objects have property Y," it refuses. This isn't a bug. It's intellectual honesty.

### 2. Grounding Before Inference

Before deriving new knowledge, EDRA checks whether the predicates involved are actually grounded in what it has been taught. 

Think of it like checking your premises before building an argument. If you've never learned anything about "conductivity" in the context of "objects," you probably shouldn't make up an answer about conducting objects.

### 3. Negations Are Stronger Than Positives

If you learn "Mammals don't have gills" and then learn "Bats are mammals," the negative information blocks any future attempt to infer "Bats have gills"—even if someone later tries to teach "Flying animals have gills."

This asymmetry matters. In the real world, knowing what something *isn't* often constrains what it can be more powerfully than knowing what it is.

### 4. Hypotheticals in Sandboxes

When you ask "If bats were fish, would they have gills?", EDRA creates a temporary copy of its belief space, makes the hypothetical assumption, reasons within that sandbox, and then discards everything.

The real belief space stays clean. Hypothetical reasoning doesn't contaminate actual knowledge.

### 5. Every Answer Has a Proof

EDRA doesn't just say "Yes" or "No." It tells you *why*. Every answer comes with a derivation trace showing which beliefs were used, which inference rules were applied, and where the information came from.

This makes the system debuggable and auditable.

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

### Component Details

**MANAS** (Input Layer): Takes messy natural language and produces clean structured representations. Uses regex patterns and entity normalization. Not trying to solve NLP—just enough parsing to get structured input.

**BUDDHI** (Inference Engine): The heart of the system. Implements grounding checks, taxonomic reasoning, negation dominance, and proof construction. This is where epistemic discipline happens.

**CHITTA** (Belief Store): Hypergraph database with three indexes: entities, predicates, and taxonomy. Supports O(1) lookups and fast graph traversal. Designed for paraconsistency—contradictory beliefs can coexist.

**HRE** (Hypothetical Reasoning): Creates isolated belief sandboxes for counterfactual queries. Deep-copies CHITTA, makes assumptions, reasons, then discards the sandbox.

**AHANKARA** (Orchestrator): Routes queries through observation sources before falling back to inference. Keeps the system modular—components don't call each other directly.

**SAKSHIN** (Logger): Passive observer. Records everything, changes nothing. Makes the system's reasoning transparent and reproducible.

*The Sanskrit names are organizational labels inspired by Vedantic philosophy, used here as conceptual metaphors.*

---

## How It Works

### Query Processing Pipeline

Let's walk through what happens when you ask "Do bats produce milk?"

**Step 1: Parsing (MANAS)**
```
Input: "Do bats produce milk?"
Output: Belief(
  entities=['bat'],
  predicates=['produces_milk'],
  polarity=INTERROGATIVE,
  confidence=0.95
)
```

**Step 2: Grounding Check (BUDDHI)**
```
Question: Is 'produces_milk' grounded for 'bat'?
Check 1: Direct belief about bat+produces_milk? → No
Check 2: Any ancestor of 'bat' has produces_milk? → Check taxonomy
```

**Step 3: Taxonomy Traversal (CHITTA)**
```
Traverse: bat → mammal (distance: 1)
Query: Does mammal produce_milk? → Yes (stored belief)
```

**Step 4: Negation Check (BUDDHI)**
```
Question: Are there blocking negations?
Check: Does any ancestor have NOT(produces_milk)? → No
Result: Inheritance allowed
```

**Step 5: Proof Construction (BUDDHI)**
```
Proof:
1. bat IS-A mammal (taxonomic link)
2. mammal produces_milk (stored belief)
3. produces_milk is FUNCTIONAL (inherits down taxonomy)
→ Conclusion: bat produces_milk (confidence: 0.9)
```

**Step 6: Rendering (AHANKARA)**
```
Output: "Yes. Inherited from mammal."
Proof trace: Available in logs
```

### Negation Dominance Example

Teaching sequence:
```python
edra.process("Mammals do not have gills.")  # Negative belief stored
edra.process("Bats are mammals.")            # Taxonomic link stored
edra.ask("Do bats have gills?")             # Query
```

What happens internally:
```
1. Parse: bat + has_gills + INTERROGATIVE
2. Grounding check: has_gills is grounded (negatively)
3. Taxonomy check: bat → mammal
4. Negation check: mammal has NOT(has_gills)
5. Blocking rule: Negative inheritance blocks positive claims
→ Answer: "No. Mammals do not have gills, inherited to bat."
```

The key insight: Once you learn a negative fact about a category, all members of that category inherit the negation, and it blocks any future positive claims.

### Hypothetical Reasoning Example

Query:
```python
edra.ask("If bats were fish, would they have gills?")
```

Processing:
```
1. HRE detects hypothetical marker: "If... were..."
2. Create sandbox: deep_copy(CHITTA)
3. Add assumption to sandbox: bat IS-A fish
4. Reason in sandbox:
   - fish have gills (if taught)
   - bat → fish (assumption)
   - Inherit: bat has gills (in sandbox only)
5. Return answer: "If bats were fish, then yes, they would have gills."
6. Destroy sandbox
7. Real beliefs unchanged: bat IS-A mammal (still true)
```

Epistemic sterility: The hypothetical never pollutes the actual belief store.

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/epistemic-reasoning-core.git
cd epistemic-reasoning-core

# No dependencies required
python3 --version  # Requires Python 3.10+

# Run interactive mode
python main.py
```

That's it. Pure Python. No external libraries.

---

## Usage

### Basic Example

```python
from ahankara import Ahankara

# Initialize system
edra = Ahankara()

# Switch to learning mode (disables belief decay)
edra.set_learning_mode()

# Teach some facts
edra.process("Bats are mammals.")
edra.process("Mammals produce milk.")
edra.process("Mammals do not have gills.")

# Switch to reasoning mode
edra.set_reasoning_mode()

# Query the system
answer = edra.ask("Do bats produce milk?")
print(answer)  
# Output: "Yes. Inherited from mammal."

answer = edra.ask("Do bats have gills?")
print(answer)  
# Output: "No. Mammals do not have gills, inherited to bat."

answer = edra.ask("Do copper objects conduct electricity?")
print(answer)  
# Output: "I do not know."
```

### Interactive Terminal

```bash
cd 0.tests
python test_interactive.py
```

Commands:
- Type statements to teach facts
- Type questions to query
- `beliefs` - Show all stored beliefs
- `stats` - System statistics
- `quit` - Exit

### Running Tests

```bash
cd 0.tests

# Full benchmark suite
python test_benchmark.py

# Stress test
python test_stress.py

# Run all tests
python run_all_tests.py
```

---

## Benchmarks

We tested EDRA on 44 queries covering taxonomic reasoning, negation handling, grounding discipline, and hypothetical scenarios.

### Performance Metrics

| Metric | Score | What It Measures |
|--------|-------|------------------|
| Precision | 84% | Correct answers when not refusing |
| Restraint | 85% | Refusal rate on ungrounded queries |
| False Positives | 10% | Wrong confident answers (rare) |
| Negation Accuracy | 82% | Inherited negation handling |
| Inheritance | 82% | Taxonomic property propagation |

### What The Benchmark Tests

The benchmark is designed to falsify our epistemic guarantees. It checks whether EDRA:

1. **Refuses ungrounded inferences** (grounding discipline)
2. **Handles negation dominance correctly** (negative inheritance)
3. **Maintains taxonomic consistency** (transitive IS-A relations)
4. **Keeps hypotheticals separate** (sandbox isolation)
5. **Respects relation frame semantics** (different predicate behaviors)

Example test case:
```python
# Teach
system.process("Mammals do not have gills.")
system.process("Bats are mammals.")

# Test
result = system.ask("Do bats have gills?")
expected = "No"

# Why this matters: Tests negation inheritance blocking
```

### Honest Failures

Some queries we intentionally refuse:
- "Do copper objects conduct electricity?" (composition not taught)
- "Is thinking fast?" (abstract predicates, no grounding)
- "Do invisible things exist?" (metaphysical, unanswerable)

These refusals demonstrate restraint, which we consider a feature.

### Benchmark Philosophy

We're not trying to maximize accuracy at all costs. The benchmark exists to verify that EDRA upholds its epistemic contracts. An 84% precision with high restraint is better than 95% precision with hallucinations on the remaining 5%.

---

## Relation Frames

Predicates in EDRA aren't flat labels. They have structural properties that determine how they behave during inference.

### The Four Frame Types

| Frame | Transitive? | Inherits? | Negation Blocks? | Examples |
|-------|-------------|-----------|------------------|----------|
| TAXONOMIC | Yes | Yes | Yes | is_a, instance_of |
| FUNCTIONAL | No | Yes | Yes | produces_milk, has_wings |
| SPATIAL | Yes | No | No | located_in, part_of |
| STATE | No | No | No | is_liquid, is_shiny |

### Why Frames Matter

Consider the predicate "produces_milk":

**Frame: FUNCTIONAL**
- Inherits: Yes (if mammals produce milk, then bats produce milk)
- Negation blocks: Yes (if mammals DON'T have gills, bats DON'T have gills)

Contrast with "located_in":

**Frame: SPATIAL**
- Inherits: No (if Paris is in France, the Eiffel Tower isn't "in France" the same way)
- Negation blocks: No (spatial negations don't propagate)

This structural information is hard-coded. It represents axiomatic commitments about how different kinds of relations work.

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

### Design Boundaries

These aren't bugs to fix. They're scope decisions.

EDRA is a reasoning core. It's designed to be one component in a larger system. We're exploring epistemic discipline in isolation before adding complexity.

Future work might add learning, better language understanding, or compositional reasoning. But those would be different research questions.

---

## Future Directions

### Potential Extensions

**Inductive Learning**: Currently, EDRA only stores what you teach it. We could add pattern recognition—observe many instances of "X is_a Y" and "Y has_property P" implies "X has P", then generalize the rule.

**Probabilistic Reasoning**: Right now, confidence scores are simple floats. We could implement Bayesian belief propagation for uncertainty handling.

**Multi-Agent Systems**: What happens when two EDRAs with different belief stores communicate? How do they merge knowledge? How do they handle disagreements?

**Compositional Semantics**: Teaching EDRA about compositional patterns ("X objects have properties of X") would expand coverage without sacrificing grounding.

**Natural Language Improvement**: Replace regex parsing with proper NLP. This would improve robustness on varied inputs.

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
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/epistemic-reasoning-core},
  note={A refusal-first epistemic reasoning system}
}
```

---

## License

MIT License - See LICENSE file for details.

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

## Contact

Questions? Issues? Ideas?

Open an issue on GitHub or start a discussion. We're interested in:
- Bug reports
- Benchmark challenges
- Theoretical questions about epistemic reasoning
- Extension proposals

When contributing, please maintain the core principle: correctness over coverage, honesty over completeness.

---

*Built with intellectual honesty. Refuses with pride.*
