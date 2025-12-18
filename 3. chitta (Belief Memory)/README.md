"""
CHITTA — BELIEF MEMORY SYSTEM
=================================

Production-grade hypergraph memory store for MARC cognitive architecture.

## Philosophy

Chitta is Sanskrit for "mind-stuff" or "memory substrate." In MARC, Chitta is:
- The ONLY storage for beliefs (single node type)
- Pure data structure (no reasoning logic)
- Hypergraph with explicit relations
- Fully traceable and inspectable
- Serializable and recoverable

## Architecture

```
Belief (atomic unit)
  ↓
ChittaGraph (hypergraph storage)
  ↓
Indexes (optimized queries)
  ↓
Persistence (checkpoints, export)
```

## Core Components

### 1. Belief (`belief.py`)
- Single node type for all knowledge
- Immutable semantic content (canonical)
- Mutable epistemic state (confidence, activation)
- Full provenance tracking
- Validation and serialization

### 2. ChittaGraph (`graph.py`)
- Hypergraph storage engine
- Edge management (relations between beliefs)
- Multi-index support (template, entity, predicate)
- Confidence updates (log-odds math)
- Activation tracking and decay
- Statistics and debugging

### 3. Indexes (`indexes.py`)
- Composite indexes (multi-field queries)
- Range queries (confidence, time)
- Full-text search
- Semantic similarity (embedding hooks)

### 4. Persistence (`persistence.py`)
- JSON serialization
- Incremental checkpointing
- Export formats (CSV, GraphML, DOT)
- State recovery

### 5. Utils (`utils.py`)
- ID generation
- Timestamp handling
- Log-odds mathematics
- Validation helpers
- Entity/predicate extraction

## Usage

### Basic Usage

```python
from chitta import Belief, ChittaGraph

# Create graph
graph = ChittaGraph()

# Create belief
belief = Belief(
    template="relation",
    canonical={
        "relation_type": "can_fly",
        "entities": ["bird"]
    },
    confidence=0.85,
    statement_text="Birds can fly"
)

# Add to graph
belief_id = graph.add_belief(belief)

# Query
birds = graph.find_by_entity("bird")
high_conf = graph.query(min_confidence=0.8)

# Add relations
graph.add_edge(id1, "supports", id2, weight=0.9)

# Update confidence (Bayesian log-odds)
graph.update_confidence_evidence(belief_id, evidence_score=1.5)

# Save/load
graph.save("chitta_state.json")
loaded = ChittaGraph.load("chitta_state.json")
```

### Advanced Usage

```python
# Complex queries
result = graph.query(
    template="relation",
    entity="penguin",
    min_confidence=0.6,
    epistemic_state="asserted"
)

# Neighbor traversal
neighbors = graph.neighbors(belief_id, relation="supports", direction="both")

# Export
from chitta.persistence import ChittaExporter
ChittaExporter.to_dot(graph, "graph.dot")
ChittaExporter.to_csv(graph, "beliefs.csv")

# Checkpointing
from chitta.persistence import ChittaCheckpoint
checkpoint = ChittaCheckpoint("./checkpoints")
checkpoint.save(graph, name="state_v1")
restored = checkpoint.load("state_v1")
```

## Belief Schema

```json
{
  "id": "b_a1b2c3d4",
  "epistemic_state": "asserted | unknown | hypothetical",
  "template": "relation | event | is_a | has_attr | ...",
  "canonical": {
    "relation_type": "can_fly",
    "entities": ["bird"]
  },
  "original_text": "Birds can fly",
  "statement_text": "Birds can fly",
  "confidence": 0.85,
  "moral_value": null,
  "source": {"input": "user", "parser": "manas_v0.1"},
  "provenance": [
    {"op": "parsed", "from": "manas", "score": 0.9}
  ],
  "edges_out": {"supports": ["b_xyz123"]},
  "edges_in": {"contradicts": ["b_abc456"]},
  "activation": 5.2,
  "created_at": "2025-12-13T...",
  "updated_at": "2025-12-13T...",
  "active": true,
  "metadata": {}
}
```

## Edge Types

- `supports`: increases confidence
- `contradicts`: triggers revision
- `derived_from`: provenance link
- `refines`: specialization/exception
- `causes`: causal relation
- `answers`: answers unknown belief
- `related`: weak semantic link
- `is_a`: taxonomic relation
- `part_of`: mereological relation
- `temporal_before/after`: temporal ordering

## Key Principles

1. **Single Node Type**: Only beliefs, no rules/questions/hypotheses as separate types
2. **No Logic in Storage**: Graph stores structure, Buddhi does reasoning
3. **Immutable Content**: Canonical structure never changes after creation
4. **Mutable Epistemic State**: Confidence and activation updated by Buddhi
5. **Full Provenance**: Every belief tracks its origins and transformations
6. **Inspectable**: All state visible and debuggable
7. **Recoverable**: Save/load preserves complete state

## Integration with MARC

```
Input → Manas → propositions
                    ↓
                 Chitta.add_belief()
                    ↓
                 Buddhi.reason()
                    ↓
                 Chitta.update_confidence()
                    ↓
                 HRE.hypothesize()
                    ↓
                 Chitta.add_belief(epistemic_state="hypothetical")
                    ↓
                 Sakshin.log()
```

## Testing

Run comprehensive test suite:

```bash
cd "3. chitta (Belief Memory)"
python test_chitta.py
```

Tests cover:
- Belief creation and validation
- Graph operations
- Edge management
- Confidence updates (log-odds)
- Activation tracking
- Complex queries
- Serialization
- Export formats
- Stress tests (1000+ beliefs)
- Penguin scenario (contradiction handling)

## Performance

- **Belief storage**: O(1) by ID
- **Entity lookup**: O(1) via index
- **Template lookup**: O(1) via index
- **Complex queries**: O(n) with index intersection
- **Neighbor traversal**: O(k) where k = degree
- **Serialization**: O(n) beliefs + O(m) edges

## Files

```
chitta/
├── __init__.py          # Public API
├── belief.py            # Belief class (450 lines)
├── graph.py             # ChittaGraph storage (600 lines)
├── indexes.py           # Advanced indexing (350 lines)
├── persistence.py       # Serialization & export (350 lines)
├── utils.py             # Utilities & math (250 lines)
├── test_chitta.py       # Test suite (550 lines)
└── README.md            # This file
```

Total: ~2550 lines of production code

## Design Decisions

### Why Single Node Type?

Clean epistemology. A "question" is just a belief with `epistemic_state="unknown"`. 
A "hypothesis" is `epistemic_state="hypothetical"`. No need for separate node types.

### Why Immutable Canonical?

Semantic content should never change. If you need to modify meaning, create a new 
belief with `refines` edge. This preserves history and prevents silent corruption.

### Why Log-Odds Confidence?

Bayesian updates require additive evidence combination. Log-odds space enables:
- `logit(p_new) = logit(p_old) + evidence`
- Numerically stable
- Principled probability updates
- No ad-hoc heuristics

### Why Hypergraph?

Relations can connect multiple beliefs (N-ary relations). Hyperedges via relation 
nodes provide flexibility while maintaining explainability.

### Why No Reasoning in Chitta?

Separation of concerns. Chitta stores, Buddhi reasons. This enables:
- Swappable reasoning engines
- Multiple reasoning strategies
- Clear debugging (storage vs logic)
- Testable components

## Future Extensions

- [ ] Embedding-based semantic search (FAISS integration)
- [ ] Temporal reasoning (Allen's interval algebra)
- [ ] Probabilistic inference (belief propagation)
- [ ] Graph neural networks on belief graph
- [ ] Distributed storage (sharding for large graphs)
- [ ] Real-time sync (multi-agent belief sharing)
- [ ] Conflict-free replicated data types (CRDTs)

## License

Part of MARC (Mind Architecture for Reasoning & Cognition)

---

**Status**: ✅ Production-ready for MARC v0.1

**Next Steps**: Integrate with Buddhi (reasoning engine)
