# Episteme: Layered Epistemic Reasoning Core

> **"Logic First, Numbers Second."**

Episteme is a persistent **epistemic reasoning system** that layers quantitative belief lifecycle management (confidence, decay) on top of a rigorous non-monotonic logic engine. Unlike neuro-symbolic hybrids that blend logic and probability into a "soup", Episteme maintains a strict separation: **Logic determines validity; Numbers determine availability.**

---

## 🔬 Deep Dive: How Episteme Understands
Episteme treats understanding ("Manas") and reasoning ("Buddhi") as distinct processes. Here is the exact lifecycle of a belief from raw text to logical node.

### 1. The Manas Pipeline (Acquisition)
Input: *"Socrates is a human."*

**Step A: Stateless Parsing (`BeliefProposal`)**
Manas first converts the text into a raw, untrusted proposal. It detects the *intent* (Assertion) and the *template* (Is-A Relation).
```json
{
  "template": "is_a",
  "raw_text": "Socrates is a human.",
  "entities": ["Socrates", "human"],
  "confidence": 0.9,
  "polarity": 1
}
```

**Step B: Entity Normalization & Sanitation**
Before entering the graph, entities are rigorously scrubbed:
*   **Normalization**: "Humans" → "human", " The Socrates " → "socrates".
*   **Sanitation**: Strict rejection of numeric entities ("1") or leaked verbs ("is").

**Step C: Epistemic Classification**
Manas infers the *Epistemic Type* based on the structure:
*   `Is-A` → **DEFAULT** (Class membership is generally true).
*   `Values` → **OBSERVATION** (Specific property).
*   `Rules` → **AXIOM** (If explicitly marked).

### 2. Graph Storage (Chitta)
The cleaned belief is stored in the persistent **Chitta Graph**.
```python
Belief(
    id="7d7c697c",
    subject="socrates",
    predicate="is_a",
    object="human",
    epistemic_state=EpistemicType.DEFAULT,
    confidence=0.9,
    active=True
)
```

---

## � Benchmark Performance
Episteme is rigorously tested against a **Brutal Benchmark** suite of 1,050 test cases designed to break fragile logic systems.

### Overall Accuracy: **84.7%**
*(Metric: Epistemic Logic Correctness)*

### Category Breakdown
| Category | Cases | Accuracy | Status |
| :--- | :---: | :---: | :--- |
| **Compositional Logic** | 70 | **100.0%** | ✅ Perfect Chain Inference |
| **Ungrounded Queries** | 150 | **100.0%** | ✅ Perfect Refusal Discipline |
| **Entity Ambiguity** | 50 | **100.0%** | ✅ Perfect Resolution |
| **Cross-Frame Isolation** | 150 | **98.0%** | ✅ Robust Context Handling |
| **Explicit Contradiction** | 350 | **74.6%** | ⚠️ Polarity Conflicts Detected |
| **Inheritance Exception** | 150 | **60.7%** | ⚠️ Specificity Logic (Improved in V1.0) |

> **Note on "Failures":** Many "failures" in earlier versions were actually *Epistemic Refusals* (The system refusing to guess). In V1.0, we distinguish `REFUSED` (Correct Humble) from `UNKNOWN` (Failure).

---

## 🛠 Technical Architecture (MARC)

### 🧠 Manas (Acquisition)
*   **Stateless**: No memory access. Pure text processing.
*   **Strict Validators**: Rejects "1" as an entity.

### 💾 Chitta (Memory)
*   **Lifecycle**: Evidence (+Boost) / Time (-Decay).
*   **Logic Gating**: Low confidence = Invisible to Logic.

### 💡 Buddhi (Intellect)
*   **The Lattice of Truth**: `AXIOM` > `EXCEPTION` > `DEFAULT`.
*   **Verdict Engine**:
    *   `YES`: Entailed.
    *   `NO`: Explicit negation or Specificity Override.
    *   `CONFLICT`: Nixon Diamond (Ambiguous).
    *   `UNKNOWN`: Insufficient Grounding.

### 👤 Ahankara (Controller)
*   **The Self**: Manages the loop. Persists state to `showcase_db`.

---

## 🚀 Key Output Examples
*Actual outputs from `showcase_episteme.py`*

### A. Specificity Override (Penguins)
*Context: Birds fly (Default). Penguins are birds. Penguins don't fly (Exception).*

```text
➤ Query: 'Does Tweety fly?'
  VERDICT: NO
  Reason: Specificity Win: Negative penguin (Dist 1) overrides Positive bird (Dist 2)
```

### B. The Nixon Diamond
*Context: Nixon is Quaker (Pacifist) & Republican (Warhawk).*

```text
➤ Query: 'Does Nixon fly?' (Metaphor for War Support)
  VERDICT: CONFLICT
  Conflict Detected: Horizontal Conflict: quaker (Pos) vs republican (Neg) at equal distance 1.
```

---

## Technical Philosophy
1.  **Contradiction Blocks Inference, Not Retrieval**: If you taught it "Sky is Green", it will repeat it. But it won't use it to prove "Grass is Blue".
2.  **Logic is Structural**: Semantics (`is_a`) matter more than vector similarity.
3.  **Humble AI**: It is better to say `UNKNOWN` than to hallucinate.

## License
MIT
