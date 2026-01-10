# Episteme: Layered Epistemic Reasoning Core

## Overview
Episteme is a persistent **epistemic reasoning system** with non-monotonic logic, explicit conflict detection, belief revision, and quantitative belief lifecycle management layered on top of symbolic inference. 

It is designed as a **post-logical** system where:
1.  **Logic comes first**: Truth is determined by structural entailment and defeasible rules.
2.  **Numbers come second**: Confidence, decay, and evidence aggregation support logical decisions but **never override** logical truth.

## Core Capabilities
- **Non-Monotonic Logic**: Supports Defeasible Reasoning (Exceptions override Defaults).
- **Explicit Conflict Detection**: Recognizes and reports contradictions (e.g., Nixon Diamond) as `CONFLICT`, not minimal-energy states.
- **Belief Revision**: New higher-rank information (AXIOMS) overrides lower-rank beliefs (DEFAULTS).
- **Quantitative Lifecycle**: Beliefs fade over time (Temporal Decay) and reinforce with evidence, but strictly within logical boundaries.
- **Persistent Knowledge Graph**: All beliefs are stored in a semantic graph with full serialization support.

## Architecture (MARC)
- **Manas (Acquisition)**: Stateless parsing of natural language into structured belief proposals. Hardened inputs.
- **Chitta (Memory)**: The Persistent Knowledge Graph. Handles storage, indexing, and quantitative state (decay/reinforcement).
- **Buddhi (Intellect)**: The Logic Engine. Performs deductive inference, checks entailment, and resolves conflicts using the Lattice of Truth.
- **Ahankara (Self)**: The System Controller. Manages the loop between perception, memory, and logic.

## Usage
### Quick Start
```bash
# Run the Grand Showcase to see all layers in action
python3 showcase_episteme.py
```

### Example Output
```text
➤ Query: 'Does Tweety fly?'
  VERDICT: NO
  Reason: Specificity Win: Negative penguin (Dist 1) overrides Positive bird (Dist 2)
```

## Philosophy
Episteme rejects the "probabilistic soup" approach. It asserts that:
*   $A \to B$ is a structural claim, not a statistical correlation.
*   Conflicts are features of a complex world, not errors to be averaged out.
*   System hygiene (validating inputs vs logic) is paramount.
