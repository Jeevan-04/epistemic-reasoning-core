# Chitta — The Belief Memory Graph (चित्त)

## What is Chitta?

**Chitta** (Sanskrit: चित्त, "memory" or "storehouse consciousness") is MARC's **long-term belief memory**. It stores everything the system "knows" — not as flat rows in a database, but as a **living graph** of interconnected beliefs.

Think of Chitta like your **hippocampus + semantic memory** — not just storing facts, but organizing them into meaningful structures that enable reasoning.

---

## The Core Problem: Why Do We Need Chitta?

### The Human Analogy

When you learn "Bats are mammals," your brain doesn't just store that sentence. It:

1. **Links** bat → mammal in your conceptual taxonomy
2. **Indexes** both "bat" and "mammal" for fast retrieval
3. **Connects** to related knowledge (mammals produce milk, have fur, etc.)
4. **Remembers** when you learned it (episodic metadata)
5. **Tracks** how confident you are (uncertainty)

This isn't a DATABASE. It's a **knowledge graph** with semantic structure.

### Why Not Just Use a Database?

**❌ Flat database approach**:
```sql
CREATE TABLE facts (
    id INT,
    subject VARCHAR,
    predicate VARCHAR,
    object VARCHAR,
    confidence FLOAT
);

INSERT INTO facts VALUES (1, 'bat', 'is_a', 'mammal', 0.9);
```

**Problems**:
- No structure (bat and mammal are just strings)
- No taxonomy (can't traverse bat → mammal → animal)
- No fast lookup (need full table scan for "what do we know about bats?")
- No version control (can't track belief evolution)
- No conflict resolution (two facts about same thing → database error)

**✅ Graph approach (Chitta)**:
```python
# Belief node
Belief(
    entities=['bat', 'mammal'],
    predicates=['is_a'],
    polarity=POSITIVE,
    confidence=0.9,
    timestamp=datetime.now(),
    source="user_input"
)

# Entity index
entity_index = {
    'bat': [belief_123, belief_456],     # Fast lookup: "what about bats?"
    'mammal': [belief_123, belief_789]   # Fast lookup: "what about mammals?"
}

# Predicate index
predicate_index = {
    'is_a': [belief_123, belief_999],    # Fast lookup: "all taxonomic facts"
    'produces_milk': [belief_789]
}

# Taxonomy graph
taxonomy = {
    'bat': {'parents': ['mammal'], 'children': []},
    'mammal': {'parents': ['animal'], 'children': ['bat', 'dog', 'whale']}
}
```

**Benefits**:
- ✓ Fast lookups (O(1) entity/predicate indexing)
- ✓ Structured traversal (bat → mammal → animal)
- ✓ Versioning (beliefs have timestamps, can track evolution)
- ✓ Conflict tolerance (multiple beliefs can coexist with different confidences)

---

## Architecture: How Chitta Works

### High-Level Structure

```
┌───────────────────────────────────────────────────────────────┐
│                    CHITTA (Belief Memory)                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            BELIEF STORE (Core Memory)                   │ │
│  │                                                         │ │
│  │  {                                                      │ │
│  │    "belief_001": Belief(                               │ │
│  │      entities=['bat', 'mammal'],                       │ │
│  │      predicates=['is_a'],                              │ │
│  │      polarity=POSITIVE,                                │ │
│  │      confidence=0.9,                                   │ │
│  │      timestamp=2024-12-16 07:39:21,                    │ │
│  │      source="user_input"                               │ │
│  │    ),                                                  │ │
│  │    "belief_002": Belief(...),                          │ │
│  │    ...                                                 │ │
│  │  }                                                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         ENTITY INDEX (Fast Entity Lookup)               │ │
│  │                                                         │ │
│  │  {                                                      │ │
│  │    'bat': [belief_001, belief_005, belief_023],        │ │
│  │    'mammal': [belief_001, belief_007, belief_012],     │ │
│  │    'gold': [belief_041],                               │ │
│  │    ...                                                 │ │
│  │  }                                                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │       PREDICATE INDEX (Fast Predicate Lookup)           │ │
│  │                                                         │ │
│  │  {                                                      │ │
│  │    'is_a': [belief_001, belief_002, belief_003],       │ │
│  │    'produces_milk': [belief_007],                      │ │
│  │    'has_wings': [belief_005],                          │ │
│  │    ...                                                 │ │
│  │  }                                                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         TAXONOMY GRAPH (IS-A Hierarchy)                 │ │
│  │                                                         │ │
│  │  bat ──────is_a────→ mammal ──────is_a────→ animal     │ │
│  │   ↑                    ↑                      ↑         │ │
│  │   └─children─────────┘ └─children──────────┘           │ │
│  │                                                         │ │
│  │  Structure:                                             │ │
│  │  {                                                      │ │
│  │    'bat': {                                             │ │
│  │      'parents': ['mammal'],                             │ │
│  │      'children': []                                     │ │
│  │    },                                                   │ │
│  │    'mammal': {                                          │ │
│  │      'parents': ['animal'],                             │ │
│  │      'children': ['bat', 'dog', 'whale']                │ │
│  │    }                                                    │ │
│  │  }                                                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │       LIFECYCLE MANAGER (Belief Evolution)              │ │
│  │                                                         │ │
│  │  • Decay unused beliefs (time-based)                    │ │
│  │  • Promote frequently accessed (usage tracking)         │ │
│  │  • Demote contradicted (conflict detection)             │ │
│  │  • Archive old beliefs (soft deletion)                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

---

## Deep Dive: The Belief Object

### Anatomy of a Belief

```python
@dataclass
class Belief:
    """
    A single belief in memory.
    
    Represents one fact or relation in the knowledge graph.
    """
    # CONTENT
    entities: List[str]          # Normalized entity names
    predicates: List[str]        # Normalized predicate names
    polarity: Polarity           # POSITIVE, NEGATIVE, or INTERROGATIVE
    
    # METADATA
    confidence: float            # [0.0, 1.0]
    timestamp: datetime          # When belief was created
    source: str                  # Where it came from
    
    # LIFECYCLE
    active: bool = True          # Is belief currently valid?
    access_count: int = 0        # How many times accessed
    last_accessed: datetime = None  # When last used
    
    # VERSIONING
    belief_id: str               # Unique identifier
    supersedes: Optional[str] = None  # Previous version (if updated)
    
    def key(self) -> str:
        """
        Canonical key for belief indexing.
        
        Format: "entity1|entity2|...|predicate1|predicate2|...|polarity"
        """
        entity_str = '|'.join(sorted(self.entities))
        pred_str = '|'.join(sorted(self.predicates))
        return f"{entity_str}||{pred_str}||{self.polarity.value}"
```

**Example**:
```python
Belief(
    entities=['bat', 'mammal'],
    predicates=['is_a'],
    polarity=POSITIVE,
    confidence=0.9,
    timestamp=datetime(2024, 12, 16, 7, 39, 21),
    source="user_input",
    active=True,
    access_count=5,
    last_accessed=datetime(2024, 12, 16, 8, 15, 30),
    belief_id="belief_001",
    supersedes=None
)
```

### The Canonical Key

**Problem**: How do you check if two beliefs are the same?

```python
belief1 = Belief(entities=['bat', 'mammal'], predicates=['is_a'], ...)
belief2 = Belief(entities=['mammal', 'bat'], predicates=['is_a'], ...)
# Are these the same? YES (order doesn't matter)

belief3 = Belief(entities=['bat'], predicates=['has_wings'], ...)
# Same as belief1? NO
```

**Solution**: Canonical key — order-independent hash

```python
def key(self) -> str:
    """
    Generate canonical key for belief.
    
    Ensures:
      - Order independence: [bat, mammal] == [mammal, bat]
      - Polarity sensitivity: POSITIVE ≠ NEGATIVE
      - Predicate sensitivity: is_a ≠ has_wings
    """
    # Sort entities (order-independent)
    entity_str = '|'.join(sorted(self.entities))
    
    # Sort predicates (order-independent)
    pred_str = '|'.join(sorted(self.predicates))
    
    # Include polarity (polarity-sensitive)
    return f"{entity_str}||{pred_str}||{self.polarity.value}"
```

**Examples**:
```python
Belief(['bat', 'mammal'], ['is_a'], POSITIVE).key()
→ "bat|mammal||is_a||positive"

Belief(['mammal', 'bat'], ['is_a'], POSITIVE).key()
→ "bat|mammal||is_a||positive"  # Same key (order-independent)

Belief(['bat', 'mammal'], ['is_a'], NEGATIVE).key()
→ "bat|mammal||is_a||negative"  # Different key (polarity matters)
```

---

## Deep Dive: Entity Indexing

### The Problem: Fast Entity Lookup

**Without indexing**:
```python
# Find all beliefs about "bat"
beliefs_about_bat = []
for belief in all_beliefs:
    if 'bat' in belief.entities:
        beliefs_about_bat.append(belief)

# Complexity: O(n) where n = total beliefs
# For 10,000 beliefs, 10,000 comparisons
```

**With indexing**:
```python
# Entity index: entity → [belief_ids]
entity_index = {
    'bat': [belief_001, belief_005, belief_023],
    'mammal': [belief_001, belief_007, belief_012],
    ...
}

# Find all beliefs about "bat"
beliefs_about_bat = entity_index.get('bat', [])

# Complexity: O(1) for lookup + O(k) where k = beliefs about entity
# For 10,000 beliefs but only 3 about "bat", only 3 comparisons
```

**Speedup**: 10,000x faster for typical queries!

### The Algorithm

```python
class ChittaGraph:
    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_belief(self, belief: Belief) -> str:
        """
        Add belief to memory and update indices.
        """
        # Generate unique ID
        belief_id = f"belief_{len(self.beliefs):06d}"
        belief.belief_id = belief_id
        
        # Store in main belief store
        self.beliefs[belief_id] = belief
        
        # Update entity index
        for entity in belief.entities:
            self.entity_index[entity].add(belief_id)
        
        # Update predicate index
        for predicate in belief.predicates:
            self.predicate_index[predicate].add(belief_id)
        
        return belief_id
    
    def get_beliefs_for_entity(self, entity: str) -> List[Belief]:
        """
        Fast lookup: all beliefs mentioning entity.
        
        Complexity: O(1) for index lookup + O(k) for k beliefs
        """
        belief_ids = self.entity_index.get(entity, set())
        return [self.beliefs[bid] for bid in belief_ids if self.beliefs[bid].active]
    
    def get_beliefs_for_predicate(self, predicate: str) -> List[Belief]:
        """
        Fast lookup: all beliefs with predicate.
        
        Complexity: O(1) for index lookup + O(k) for k beliefs
        """
        belief_ids = self.predicate_index.get(predicate, set())
        return [self.beliefs[bid] for bid in belief_ids if self.beliefs[bid].active]
```

**Example**:
```python
# Initial state
chitta = ChittaGraph()

# Add beliefs
chitta.add_belief(Belief(['bat', 'mammal'], ['is_a'], POSITIVE))
# → Updates:
#   entity_index['bat'] = {belief_000001}
#   entity_index['mammal'] = {belief_000001}
#   predicate_index['is_a'] = {belief_000001}

chitta.add_belief(Belief(['bat'], ['has_wings'], POSITIVE))
# → Updates:
#   entity_index['bat'] = {belief_000001, belief_000002}
#   predicate_index['has_wings'] = {belief_000002}

# Fast lookup
chitta.get_beliefs_for_entity('bat')
→ [Belief(['bat', 'mammal'], ['is_a'], ...),
   Belief(['bat'], ['has_wings'], ...)]
# Complexity: O(1) + O(2) = O(1)
```

**Human Analogy**:

Your brain doesn't scan ALL memories when you think "bat". It directly accesses the "bat" concept node and retrieves connected knowledge.

Chitta does the same with **entity indexing**.

---

## Deep Dive: Taxonomy Graph

### The Problem: Efficient Ancestor Lookup

**Without taxonomy graph**:
```python
# Find ancestors of "bat"
# Must scan all beliefs for IS-A relations
ancestors = []
for belief in all_beliefs:
    if 'is_a' in belief.predicates and 'bat' in belief.entities:
        # Extract parent from belief
        parent = extract_parent(belief, 'bat')
        ancestors.append(parent)
        # Now recursively find parents of parent... O(n²)
```

**With taxonomy graph**:
```python
# Taxonomy graph: entity → {parents, children}
taxonomy = {
    'bat': {
        'parents': ['mammal'],
        'children': []
    },
    'mammal': {
        'parents': ['animal'],
        'children': ['bat', 'dog', 'whale']
    },
    'animal': {
        'parents': [],
        'children': ['mammal', 'bird', 'fish']
    }
}

# Find ancestors of "bat"
def get_ancestors(entity):
    ancestors = []
    current = [entity]
    while current:
        node = current.pop(0)
        parents = taxonomy[node]['parents']
        ancestors.extend(parents)
        current.extend(parents)
    return ancestors

get_ancestors('bat')
→ ['mammal', 'animal']  # Direct traversal, no belief scanning
```

### The Algorithm

```python
class ChittaGraph:
    def __init__(self):
        self.taxonomy: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {'parents': [], 'children': []}
        )
    
    def add_taxonomic_relation(self, child: str, parent: str):
        """
        Add IS-A relation to taxonomy graph.
        
        Updates:
          - child's parents
          - parent's children
        """
        # Add to graph
        if parent not in self.taxonomy[child]['parents']:
            self.taxonomy[child]['parents'].append(parent)
        
        if child not in self.taxonomy[parent]['children']:
            self.taxonomy[parent]['children'].append(child)
    
    def get_ancestors(self, entity: str, max_depth: int = 10) -> List[Tuple[str, int]]:
        """
        Get all taxonomic ancestors with depth.
        
        Returns:
            List of (ancestor, depth) tuples
            
        Example:
            get_ancestors('bat')
            → [('mammal', 1), ('animal', 2)]
        """
        ancestors = []
        visited = set()
        queue = [(entity, 0)]
        
        while queue and len(ancestors) < max_depth:
            current, depth = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Get parents
            parents = self.taxonomy[current]['parents']
            for parent in parents:
                if parent not in visited:
                    ancestors.append((parent, depth + 1))
                    queue.append((parent, depth + 1))
        
        return ancestors
    
    def get_descendants(self, entity: str, max_depth: int = 10) -> List[Tuple[str, int]]:
        """
        Get all taxonomic descendants with depth.
        
        Returns:
            List of (descendant, depth) tuples
            
        Example:
            get_descendants('mammal')
            → [('bat', 1), ('dog', 1), ('whale', 1)]
        """
        descendants = []
        visited = set()
        queue = [(entity, 0)]
        
        while queue and len(descendants) < max_depth:
            current, depth = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Get children
            children = self.taxonomy[current]['children']
            for child in children:
                if child not in visited:
                    descendants.append((child, depth + 1))
                    queue.append((child, depth + 1))
        
        return descendants
```

**Example Usage**:
```python
# Build taxonomy
chitta.add_taxonomic_relation('bat', 'mammal')
chitta.add_taxonomic_relation('mammal', 'animal')
chitta.add_taxonomic_relation('dog', 'mammal')

# Query ancestors
chitta.get_ancestors('bat')
→ [('mammal', 1), ('animal', 2)]

# Query descendants
chitta.get_descendants('mammal')
→ [('bat', 1), ('dog', 1)]

# Fast traversal (no belief scanning)
```

**Why This Matters**:

Taxonomic reasoning is CENTRAL to Buddhi's inference. Without a fast taxonomy graph, every query would require expensive belief scanning.

**Speedup**: O(n) → O(k) where k = depth of hierarchy (typically k << n)

---

## Deep Dive: Lifecycle Management

### The Problem: Memory Isn't Static

**Human Memory Evolution**:
- Facts you use frequently → become stronger
- Facts you never recall → fade away
- Contradicted facts → lose confidence
- Old facts → might become outdated

**MARC Memory Evolution** (same principles):
- Frequently accessed beliefs → higher confidence
- Unused beliefs → decay over time
- Contradicted beliefs → demoted
- Superseded beliefs → archived

### Decay: Forgetting Unused Beliefs

**The Math**:

$$
\text{confidence}_{new} = \text{confidence}_{old} \times e^{-\lambda \cdot \Delta t}
$$

Where:
- $\lambda$ = decay rate (default: 0.1)
- $\Delta t$ = time since last access (in days)

**Algorithm**:
```python
def apply_decay(self, decay_rate: float = 0.1):
    """
    Apply time-based decay to unused beliefs.
    
    Only applies in REASONING MODE (not learning mode).
    """
    if self.learning_mode:
        return  # No decay during learning
    
    now = datetime.now()
    
    for belief in self.beliefs.values():
        if not belief.active:
            continue
        
        # Calculate time since last access
        last_access = belief.last_accessed or belief.timestamp
        delta_days = (now - last_access).days
        
        if delta_days > 0:
            # Apply exponential decay
            decay_factor = math.exp(-decay_rate * delta_days)
            belief.confidence *= decay_factor
            
            # Deactivate if confidence too low
            if belief.confidence < 0.1:
                belief.active = False
```

**Example**:
```python
Belief(confidence=0.9, last_accessed=30 days ago)
→ confidence_new = 0.9 × e^(-0.1 × 30)
                = 0.9 × e^(-3)
                = 0.9 × 0.0498
                = 0.045
→ confidence < 0.1 → DEACTIVATE
```

**Why Decay?**

- Prevents memory bloat (unused beliefs fade)
- Models human forgetting
- Prioritizes frequently used knowledge

**When NOT to Decay**:

```python
# LEARNING MODE: No decay
learning_mode = True
→ Beliefs accumulate without decay

# REASONING MODE: Decay active
learning_mode = False
→ Unused beliefs fade over time
```

### Promotion: Strengthening Used Beliefs

**The Math**:

$$
\text{confidence}_{new} = \min(1.0, \text{confidence}_{old} + \alpha)
$$

Where:
- $\alpha$ = promotion boost (default: 0.01)

**Algorithm**:
```python
def access_belief(self, belief_id: str, promote: bool = True):
    """
    Mark belief as accessed and optionally promote.
    
    Simulates strengthening through retrieval (like human memory).
    """
    belief = self.beliefs[belief_id]
    
    # Update access metadata
    belief.access_count += 1
    belief.last_accessed = datetime.now()
    
    # Promote confidence (if in reasoning mode)
    if promote and not self.learning_mode:
        belief.confidence = min(1.0, belief.confidence + 0.01)
```

**Example**:
```python
# Belief initially at 0.8 confidence
belief = Belief(confidence=0.8)

# Access 5 times
for i in range(5):
    chitta.access_belief(belief.belief_id)

# confidence = 0.8 + (5 × 0.01) = 0.85
```

**Why Promotion?**

**Human Memory Strengthening**: Facts you recall frequently become more confident.

MARC does the same: frequently queried beliefs gain confidence.

### Demotion: Weakening Contradicted Beliefs

**The Math**:

$$
\text{confidence}_{new} = \text{confidence}_{old} \times (1 - \beta)
$$

Where:
- $\beta$ = demotion factor (default: 0.2)

**Algorithm**:
```python
def handle_conflict(self, conflicting_beliefs: List[str]):
    """
    Demote conflicting beliefs.
    
    When multiple beliefs contradict, lower-confidence ones are demoted.
    """
    # Sort by confidence
    sorted_beliefs = sorted(
        [self.beliefs[bid] for bid in conflicting_beliefs],
        key=lambda b: b.confidence,
        reverse=True
    )
    
    # Winner: highest confidence (no demotion)
    winner = sorted_beliefs[0]
    
    # Losers: demote
    for belief in sorted_beliefs[1:]:
        belief.confidence *= 0.8  # 20% demotion
        
        # Deactivate if confidence too low
        if belief.confidence < 0.3:
            belief.active = False
```

**Example**:
```python
Conflicting beliefs:
  - belief_A: "Birds fly" (confidence=0.8)
  - belief_B: "Penguins don't fly" (confidence=0.95)

Resolution:
  - Winner: belief_B (higher confidence)
  - Loser: belief_A demoted to 0.8 × 0.8 = 0.64
```

**Why Demotion?**

- Resolves contradictions without deleting beliefs
- Models human doubt (when contradicted, you become less certain)
- Allows paraconsistent reasoning (both beliefs coexist, but winner is trusted)

---

## The Chitta Contract

### What Chitta Guarantees

1. **Fast Lookup**: O(1) entity/predicate indexing
2. **Structured Storage**: Beliefs organized in semantic graph
3. **Versioning**: Can track belief evolution over time
4. **Lifecycle Management**: Decay, promotion, demotion
5. **Taxonomy Graph**: Efficient ancestor/descendant traversal
6. **Conflict Tolerance**: Multiple beliefs can coexist

### What Chitta Does NOT Do

1. **NO REASONING**: Doesn't infer (Buddhi does that)
2. **NO PERCEPTION**: Doesn't parse language (Manas does that)
3. **NO ORCHESTRATION**: Doesn't control execution (Ahankara does that)
4. **NO JUDGMENT**: Doesn't decide what to believe (just stores)

**Chitta is PURE MEMORY.**

---

## Why This Design? (Feynman's Question)

### "Why not just use a SQL database?"

**SQL strengths**:
- ✓ ACID guarantees
- ✓ Mature ecosystem
- ✓ Query language (SQL)

**SQL weaknesses for MARC**:
- ✗ No graph structure (need joins for traversal)
- ✗ No native versioning (need manual schema)
- ✗ No lifecycle management (no decay/promotion)
- ✗ No semantic indexing (just B-trees)

**MARC graph approach**:
- ✓ Native graph traversal (taxonomy)
- ✓ Built-in versioning (supersedes field)
- ✓ Lifecycle management (decay/promotion)
- ✓ Semantic indexing (entity/predicate)

### "Why not use a graph database (Neo4j)?"

**Neo4j strengths**:
- ✓ Native graph storage
- ✓ Cypher query language
- ✓ Production-ready

**Why MARC rolls its own**:
- **Control**: Custom lifecycle logic (decay/promotion)
- **Simplicity**: No external dependencies
- **Performance**: In-memory graph (no disk I/O)
- **Transparency**: Full visibility into internals

**Trade-off**: Less mature, but perfectly suited to MARC's needs.

### "Why learning mode vs reasoning mode?"

**Without mode separation**:
```python
# Add belief: "Bats are mammals"
chitta.add(Belief(['bat', 'mammal'], ['is_a'], ...))
→ Should this decay over time? Should it be questioned?

# Confusion: Is this teaching or reasoning?
```

**With mode separation**:
```python
# LEARNING MODE: Teaching phase
chitta.learning_mode = True
chitta.add(Belief(['bat', 'mammal'], ['is_a'], ...))
→ Accept without question, no decay

# REASONING MODE: Thinking phase
chitta.learning_mode = False
chitta.query("Do bats fly?")
→ Apply decay, promote used beliefs, critically evaluate
```

**Human Analogy**:

When you're in class (learning mode):
- You accept what the teacher says
- You don't question every fact
- You build knowledge provisionally

When you're studying for exam (reasoning mode):
- You critically evaluate what you "know"
- You notice contradictions
- You strengthen facts you use frequently

**MARC separates these modes explicitly.**

---

## Summary: Chitta in One Diagram

```
╔═══════════════════════════════════════════════════════════════╗
║                  CHITTA (Belief Memory Graph)                 ║
║                                                               ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  BELIEF STORE                                           │  ║
║  │  • Canonical keys for deduplication                     │  ║
║  │  • Metadata (timestamp, source, confidence)             │  ║
║  │  • Lifecycle tracking (access count, last accessed)     │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                            ↓                                  ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  INDEXING LAYER                                         │  ║
║  │  • Entity index: entity → [beliefs]  (O(1) lookup)      │  ║
║  │  • Predicate index: predicate → [beliefs]  (O(1) lookup)│  ║
║  │  • Taxonomy graph: entity → {parents, children}         │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                            ↓                                  ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  LIFECYCLE MANAGEMENT (Reasoning Mode Only)             │  ║
║  │  • Decay: confidence × e^(-λΔt)  (forget unused)        │  ║
║  │  • Promote: confidence + α  (strengthen used)            │  ║
║  │  • Demote: confidence × (1-β)  (weaken contradicted)    │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                               ║
║  MODES:                                                       ║
║  • LEARNING MODE: Monotonic addition (no decay)              ║
║  • REASONING MODE: Lifecycle management (decay/promote)      ║
║                                                               ║
║  GUARANTEES:                                                  ║
║   • Fast lookup (O(1) entity/predicate indexing)             ║
║   • Structured traversal (taxonomy graph)                    ║
║   • Versioning (belief evolution tracking)                   ║
║   • Conflict tolerance (multiple beliefs coexist)            ║
║                                                               ║
║  DOES NOT:                                                    ║
║   • Reason (Buddhi does that)                                ║
║   • Parse language (Manas does that)                         ║
║   • Orchestrate (Ahankara does that)                         ║
║                                                               ║
║  PHILOSOPHY:                                                  ║
║   "Memory is NOT reasoning"                                  ║
║   "Storage is NOT inference"                                 ║
║   "Knowledge graphs > flat databases"                        ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**Key Insight**: Chitta is the semantic memory of MARC. It doesn't reason or perceive — it just REMEMBERS. And it remembers with structure, indexing, and lifecycle management that mirrors human semantic memory.

**Chitta is FROZEN**: This is the production memory architecture.
