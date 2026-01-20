# Episteme: Persistant Epistemic Reasoning Core

> **"Logic First, Numbers Second."**

Episteme is a persistent **epistemic reasoning system** that layers quantitative belief lifecycle management (confidence, decay) on top of a rigorous non-monotonic logic engine. Unlike neuro-symbolic hybrids that blend logic and probability into a "soup", Episteme maintains a strict separation: **Logic determines validity; Numbers determine availability.**

## 1. Technical Architecture (MARC)

The system follows the **MARC** (Modular Architecture for Reasoning and Cognition) design, composed of four distinct layers:

### 🧠 Manas (Acquisition & Normalization)
*   **Role**: Stateless sensory processing.
*   **Function**: Parses natural language into structured `BeliefProposals`.
*   **Key Feature**: **Strict Entity Validation**. Manas rejects non-semantic tokens (e.g., numeric entities like "1") before they pollute the graph, ensuring a clean separation between symbolic logic and mathematical constants.

### 💾 Chitta (Memory & Persistence)
*   **Role**: The Persistent Knowledge Graph.
*   **Function**: Stores beliefs as `(Entity, Predicate, Object)` triples with explicit epistemic metadata.
*   **Key Feature**: **Quantitative Lifecycle**. Beliefs are not static; they:
    *   **Reinforce** with repeated evidence (Asymptotic confidence boost).
    *   **Decay** over time if acceptable evidence is absent.
    *   **Gate** logic: Beliefs below a confidence threshold become `INACTIVE` and invisible to the logic engine.

### 💡 Buddhi (Intellect & Inference)
*   **Role**: The Pure Logic Engine.
*   **Function**: Performs deductive and defeasible inference over the Chitta graph.
*   **Key Feature**: **The Lattice of Truth**. Resolves conflicts using a strict hierarchy:
    `AXIOM` > `OBSERVATION` > `EXCEPTION` > `DEFAULT` > `HYPOTHESIS`.

### 👤 Ahankara (System Controller)
*   **Role**: The "Self" or Agent Loop.
*   **Function**: Orchestrates the Perceive-Store-Reason loop. Manages the event log and persistence.

---

## 2. Core Logic Capabilities

### The Lattice of Truth
Episteme rejects flat belief spaces. Truth is determined by structural rank, not just weight.
*   **AXIOM**: Immutable truths (Rules of Logic/Nature).
*   **EXCEPTION**: Specific overrides (Penguins don't fly).
*   **DEFAULT**: General rules (Birds fly).

### Conflict Resolution
The system explicitly handles contradictory information using two mechanisms:
1.  **Vertical Conflict (Specificity)**: Specific knowledge overrides general knowledge (Subclass wins).
2.  **Horizontal Conflict (Ambiguity)**: Mutually exclusive paths of equal rank result in a `CONFLICT` verdict (The Nixon Diamond).

---

## 3. System Showcase (Screenshots)

The following outputs are actual traces from `showcase_episteme.py`.

### A. Acquisition & Internal State
Parsing natural language into structured, normalized beliefs.

```text
➤ 1. Teaching Basic Taxonomy
  Inputting natural language facts. Manas normalizes entities/predicates.
  USER: 'Socrates is a human.'
  USER: 'A human is a mammal.'
  USER: 'A mammal is an animal.'
```

**Internal Graph Visualization (Chitta):**
```text
Internal Storage: Beliefs about 'socrates'
ID              Statement                                Type            Conf   Status    
------------------------------------------------------------------------------------------
7d7c697c        Socrates is a human.                     DEFAULT         0.90   Active    
------------------------------------------------------------------------------------------
```

### B. Logical Inference (Buddhi)
Deriving new truths via transitive entailment.

```text
➤ 3. Asking a Question
  Query: 'Is Socrates a mammal?'
  Reasoning Trace:
    [focus] Found 3 relevant belief(s)
    [grounding_check] Taxonomic grounding: ancestor 'human' has predicate {'is_a'}
    [taxonomic_entailment] Entailment: socrates is a mammal (Conf: 1.0)

  VERDICT: YES
```

### C. Specificity Conflict (Penguins)
Demonstrating **Defeasible Reasoning**: A specific Exception overrides a General Default.

```text
➤ 5. Resolving Specificity
  Query: 'Does Tweety fly?' (Tweety is a Penguin, Penguins don't fly, Birds fly)
  
  VERDICT: NO
  Reason: Specificity Win: Negative penguin (Dist 1) overrides Positive bird (Dist 2)
```

### D. The Nixon Diamond (Horizontal Conflict)
Handling ambiguity where no clear logical winner exists.

```text
➤ 7. Resolving Nixon Diamond
  Query: 'Does Nixon fly?' (Quaker [Yes] vs Republican [No])
  
  VERDICT: CONFLICT
  Conflict Detected: Horizontal Conflict: quaker (Pos) vs republican (Neg) at equal distance 1.
```

### E. Quantitative Decay
Beliefs fade if not reinforced, eventually becoming inactive.

```text
➤ 9. Temporal Decay & Logic Gating
  Time passes...
  [Chitta] 📉 Deactivating 'Market will crash' (Conf 0.050 < 0.1)
```

---

## 4. Technical Philosophy & Claims

**What Episteme IS:**
*   **Post-Logical**: Logic constraints are primary; probability is secondary.
*   **Non-Monotonic**: New knowledge (Exceptions) can invalidate old inferences (Defaults).
*   **Epistemic**: It explicitly models *why* it believes something (Rank, Path, Confidence).

**What Episteme is NOT:**
*   **NOT a Probabilistic Reasoner**: It does not average contradictions into a "0.5 truth".
*   **NOT "Commonsense" AI**: It relies only on what it is taught or can strictly derive.
*   **NOT a vector DB**: It uses structured graph representations, not semantic similarity embeddings.

## License
MIT
