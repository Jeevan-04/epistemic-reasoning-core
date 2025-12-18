# Geographic Memory — External Retrieval Knowledge

**STATUS**: AUXILIARY KNOWLEDGE SOURCE (not core module)

**MARC Architecture**:
- **CORE MODULES** (1-6): Manas, Buddhi, Chitta, HRE, Ahankara, Sakshin — system cannot function without these
- **AUXILIARY KNOWLEDGE SOURCES** (7-8): Perceptual Priors, Geographic Memory — optional external knowledge that enriches reasoning

---

## What is Geographic Memory?

**Geographic Memory** is MARC's **retrieval-only external knowledge base** for geographic facts. It stores locations, containment relations, and spatial hierarchies WITHOUT inference.

Think of Geographic Memory like a **world atlas** — you look up "Where is London?" and get "London → UK → Europe". You don't DERIVE this through reasoning.

**Critical Distinction**: Geographic Memory is NOT a core reasoning module. It is an **auxiliary knowledge source** that plugs into the core architecture. MARC can function without Geographic Memory (it would just refuse geographic queries and fall back to Buddhi).

---

## The Core Problem: Why Do We Need Geographic Memory?

### The Human Analogy

When asked "Is London in Europe?", your brain doesn't reason:

```
❌ INFERENCE APPROACH (Wrong):
1. London is a city
2. Cities are... in countries?
3. Countries are... in continents?
4. Therefore... ??? (no logical path)

Problem: Geographic containment is NOT inferrable from taxonomy.
```

Instead, your brain RETRIEVES:

```
✓ RETRIEVAL APPROACH (Correct):
1. RECALL: London → UK (stored geographic fact)
2. RECALL: UK → Europe (stored geographic fact)
3. TRAVERSE: London → UK → Europe
4. CONCLUDE: Yes, London is in Europe

This is SPATIAL MEMORY, not logical inference.
```

**Key Insight**: Geographic knowledge is **retrieval-based** (stored external facts), not **inference-based** (derived through reasoning).

### The Composition Problem (Without Geographic Memory)

**Scenario**: System taught "London is in UK" and "UK is in Europe".

**Without Geographic Memory**:
```python
# Store as normal beliefs in Chitta
chitta.add(Belief(['london', 'uk'], ['located_in'], POSITIVE))
chitta.add(Belief(['uk', 'europe'], ['located_in'], POSITIVE))

# Query: "Is London in Europe?"
buddhi.answer("Is London in Europe?")
→ Might try to use SPATIAL frame (transitive, but non-inheritable)
→ Transitivity: london → uk → europe
→ Might work, BUT:

# Query: "Is Tokyo in Europe?"
buddhi.answer("Is Tokyo in Europe?")
→ Grounding check: "tokyo" + "europe" not stored, no path found
→ Refuses with "I do not know" (correct, but inefficient)

# Problem: Every geographic query requires Buddhi's reasoning
# → Expensive when we have external authoritative source
```

**With Geographic Memory**:
```python
# Store in geographic memory (separate from beliefs)
geo_memory.add_location('london', 'uk')
geo_memory.add_location('uk', 'europe')
geo_memory.add_location('tokyo', 'japan')
geo_memory.add_location('japan', 'asia')

# Query: "Is London in Europe?"
geo_memory.is_located_in('london', 'europe')
→ Traverse: london → uk → europe ✓
→ "Yes (London → UK → Europe)"

# Query: "Is Tokyo in Europe?"
geo_memory.is_located_in('tokyo', 'europe')
→ Traverse: tokyo → japan → asia (no path to europe) ✗
→ "No (Tokyo is in Asia, not Europe)"

# Benefit: Fast retrieval, no reasoning overhead ✓
```

---

## Architecture: How Geographic Memory Works

### High-Level Structure

```
┌───────────────────────────────────────────────────────────────┐
│           GEOGRAPHIC MEMORY (Spatial Knowledge Graph)         │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         CONTAINMENT GRAPH (Location Hierarchy)          │  │
│  │                                                         │  │
│  │  london ──contains→ uk ──contains→ europe               │  │
│  │    ↑                  ↑                ↑                │  │
│  │    └─contained_in────┘                │                │  │
│  │                                        │                │  │
│  │  paris ──contains→ france ────contains→┘                │  │
│  │                                                         │  │
│  │  tokyo ──contains→ japan ──contains→ asia               │  │
│  │                                                         │  │
│  │  Structure:                                             │  │
│  │  {                                                      │  │
│  │    'london': {'container': 'uk', 'type': 'city'},      │  │
│  │    'uk': {'container': 'europe', 'type': 'country'},   │  │
│  │    'europe': {'container': None, 'type': 'continent'}, │  │
│  │    'tokyo': {'container': 'japan', 'type': 'city'},    │  │
│  │    'japan': {'container': 'asia', 'type': 'country'},  │  │
│  │    'asia': {'container': None, 'type': 'continent'}    │  │
│  │  }                                                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  PROPERTIES:                                                  │
│  • Retrieval-Only: No inference, just lookup + traversal     │
│  • Transitive: London → UK → Europe (path following)         │
│  • Non-Inheritable: Location doesn't inherit properties      │
│  • External Source: Authoritative geographic data            │
│                                                               │
│  QUERY INTERFACE:                                             │
│  • is_located_in(place, container) → bool                    │
│  • get_location_path(place) → [place, country, continent]   │
│  • format_answer(place, container, path) → string            │
└───────────────────────────────────────────────────────────────┘
```

---

## Deep Dive: The Core Principles

### Principle 1: Retrieval-Only (No Inference)

**Definition**: Geographic Memory does NOT reason. It only RETRIEVES stored facts and TRAVERSES containment paths.

**Examples**:

```python
# STORED (retrieval works):
geo_memory.add_location('london', 'uk')
geo_memory.is_located_in('london', 'uk')
→ True (direct lookup)

# TRANSITIVE (traversal works):
geo_memory.add_location('uk', 'europe')
geo_memory.is_located_in('london', 'europe')
→ True (traverse: london → uk → europe)

# NOT STORED (no inference):
geo_memory.is_located_in('new_york', 'usa')
→ False (not stored, no attempt to infer)

# MARC refuses to hallucinate geographic facts
```

**Mathematical Property**:

$$
\text{Geographic}(q) \iff \exists \text{path in containment graph}
$$

(Query succeeds IFF path exists in graph, no inference)

**Implementation**:
```python
class GeographicMemory:
    def __init__(self):
        self.locations: Dict[str, Dict] = {}
    
    def is_located_in(self, place: str, container: str) -> bool:
        """
        Check if place is in container (via transitive containment).
        
        NO INFERENCE - only path traversal.
        """
        place_lower = place.lower()
        container_lower = container.lower()
        
        # Traverse containment path
        current = place_lower
        visited = set()
        
        while current:
            if current == container_lower:
                return True  # Found path
            
            if current in visited:
                break  # Cycle detection
            visited.add(current)
            
            # Get parent container
            if current in self.locations:
                current = self.locations[current].get('container')
            else:
                break  # Not in graph
        
        return False  # No path found (honest refusal)
```

### Principle 2: Transitive Containment

**Definition**: If A is in B, and B is in C, then A is in C (spatial transitivity).

**Example**:

```python
# Stored facts:
geo_memory.add_location('london', 'uk')
geo_memory.add_location('uk', 'europe')

# Query with transitivity:
geo_memory.is_located_in('london', 'europe')
→ Traverse: london → uk → europe ✓
→ True

# Multi-hop transitivity:
geo_memory.add_location('westminster', 'london')
geo_memory.is_located_in('westminster', 'europe')
→ Traverse: westminster → london → uk → europe ✓
→ True
```

**Why Transitivity?**

Humans understand spatial nesting:
- "Westminster is in London" (direct)
- "London is in UK" (direct)
- "UK is in Europe" (direct)
- **Therefore**: "Westminster is in Europe" (transitive)

Geographic Memory models this with graph traversal.

**Algorithm**:
```python
def get_location_path(self, place: str) -> List[str]:
    """
    Get full containment path from place to root.
    
    Example:
      get_location_path('london')
      → ['london', 'uk', 'europe']
    """
    place_lower = place.lower()
    path = [place_lower]
    visited = set([place_lower])
    
    current = place_lower
    while current in self.locations:
        container = self.locations[current].get('container')
        if not container or container in visited:
            break
        
        path.append(container)
        visited.add(container)
        current = container
    
    return path
```

### Principle 3: Non-Inheritable

**Definition**: Geographic location does NOT inherit properties from container.

**Example**:

```python
# Suppose we had: "Europe is cold" (hypothetical)
# And: "London is in Europe"

# WRONG (inheritance):
"Is London cold?"
→ london → europe → cold (inherited)
→ "Yes"
# Problem: Not all places in Europe are equally cold!

# CORRECT (no inheritance):
geo_memory.is_located_in('london', 'europe')
→ True (containment confirmed)

# But property query:
"Is London cold?"
→ Geographic Memory: Not applicable (only spatial queries)
→ Falls back to Perceptual Priors or Buddhi
→ Honest refusal if not stored
```

**Why Non-Inheritable?**

**Location ≠ Property Transfer**

- London is in Europe (spatial fact)
- Europe has property X (hypothetical)
- London does NOT inherit X (no property flow)

Geographic Memory ONLY handles spatial containment, not property inheritance.

### Principle 4: Negative Queries (Absence Detection)

**Definition**: Geographic Memory can answer NEGATIVE queries when place is in a DIFFERENT container.

**Algorithm**:
```python
def format_answer(self, place: str, container: str) -> str:
    """
    Format geographic answer.
    
    Handles:
      - Positive: "Yes (London → UK → Europe)"
      - Negative: "No (Tokyo is in Asia, not Europe)"
      - Unknown: None (not in geographic memory)
    """
    if self.is_located_in(place, container):
        # Positive case
        path = self.get_location_path(place)
        path_str = ' → '.join(path[:path.index(container) + 1])
        return f"Yes ({path_str})"
    
    # Check if place exists but in different location
    place_lower = place.lower()
    if place_lower in self.locations:
        # Get actual location
        actual_path = self.get_location_path(place)
        if len(actual_path) >= 2:
            actual_container = actual_path[-1]  # Top-level container
            return f"No ({place} is in {actual_container}, not {container})"
    
    return None  # Not in geographic memory
```

**Example**:
```python
# Query: "Is Tokyo in Europe?"
geo_memory.is_located_in('tokyo', 'europe')
→ False

# But format_answer provides context:
geo_memory.format_answer('tokyo', 'europe')
→ "No (Tokyo is in Asia, not Europe)"
# Helpful refusal with actual location ✓
```

---

## Deep Dive: The Query Algorithm

### Spatial Containment Check

**Goal**: Determine if place is (transitively) contained in container.

**Algorithm**:
```python
class GeographicMemory:
    def add_location(self, place: str, container: str, place_type: str = 'place'):
        """
        Add geographic containment fact.
        
        Args:
          place: Location name (city, country, etc.)
          container: Containing location (country, continent, etc.)
          place_type: Type of location (city, country, continent)
        """
        place_lower = place.lower()
        container_lower = container.lower()
        
        self.locations[place_lower] = {
            'container': container_lower,
            'type': place_type
        }
    
    def is_located_in(self, place: str, container: str, max_depth: int = 10) -> bool:
        """
        Check if place is (transitively) in container.
        
        Args:
          place: Location to check
          container: Container to check against
          max_depth: Maximum traversal depth (prevent infinite loops)
        
        Returns:
          True if path exists, False otherwise
        """
        place_lower = place.lower()
        container_lower = container.lower()
        
        # Direct match
        if place_lower == container_lower:
            return True
        
        # Traverse containment path
        current = place_lower
        visited = set()
        depth = 0
        
        while current and depth < max_depth:
            if current == container_lower:
                return True  # Found path
            
            if current in visited:
                break  # Cycle detected
            visited.add(current)
            
            # Get parent container
            if current in self.locations:
                current = self.locations[current].get('container')
                depth += 1
            else:
                break  # Not in graph
        
        return False  # No path found
```

**Example Traversal**:
```python
# Build graph
geo = GeographicMemory()
geo.add_location('westminster', 'london', 'district')
geo.add_location('london', 'uk', 'city')
geo.add_location('uk', 'europe', 'country')

# Query: "Is Westminster in Europe?"
geo.is_located_in('westminster', 'europe')

# Traversal:
# Step 1: current='westminster', target='europe' → not match
# Step 2: current='london', target='europe' → not match
# Step 3: current='uk', target='europe' → not match
# Step 4: current='europe', target='europe' → MATCH ✓
# Return: True
```

### Handling Edge Cases

**Case 1: Place not in memory**:
```python
geo.is_located_in('unknown_city', 'europe')
→ False (not stored, no inference)
→ Falls back to Buddhi (which will refuse via grounding)
```

**Case 2: Cycle detection**:
```python
# Corrupt data (hypothetical):
geo.add_location('a', 'b')
geo.add_location('b', 'c')
geo.add_location('c', 'a')  # ← Cycle!

geo.is_located_in('a', 'z')
→ Detects cycle via visited set
→ Returns False (safe termination)
```

**Case 3: Reverse containment**:
```python
# Query: "Is Europe in London?"
geo.is_located_in('europe', 'london')
→ False (no path from europe to london)
# Correct: Containment is directional
```

---

## Integration with MARC

### Priority in Query Resolution

**Ahankara's Query Pipeline**:

```python
def query_answer(self, query_text: str) -> str:
    # Phase 1: Perception
    proposal = self.manas.parse(query_text)
    
    # Phase 2a: Perceptual Priors (Priority 1)
    if self.perceptual_priors:
        answer = self.perceptual_priors.check_query(...)
        if answer:
            return answer
    
    # Phase 2b: GEOGRAPHIC MEMORY (Priority 2)
    if self.geographic_memory:
        answer = self._check_geographic_memory(proposal)
        if answer:
            return answer  # ← EARLY RETURN
    
    # Phase 3: Buddhi (Priority 3)
    # ...
```

**Why Priority 2?**

**Epistemology**:
$$
\text{Observation} > \text{External Retrieval} > \text{Inference}
$$

- Perceptual observation (highest priority)
- External authoritative source (geographic memory)
- Logical inference (lowest priority)

**Example**:
```python
# Query: "Is London in Europe?"

# WITHOUT geographic memory:
buddhi.answer("Is London in Europe?")
→ Tries to use SPATIAL frame (transitive)
→ Might work, but expensive reasoning

# WITH geographic memory:
geo_memory.is_located_in('london', 'europe')
→ Fast path traversal (O(depth))
→ Returns immediately ✓

# Benefit: External source > Reasoning (faster + authoritative)
```

---

## What Geographic Memory Is NOT

### NOT a Reasoning Engine

```python
# Geographic memory does NOT infer
geo_memory.is_located_in('new_city', 'country')
→ False (not stored, no inference attempted)

# Buddhi might try to reason (but would fail via grounding)
```

### NOT Property-Aware

```python
# Geographic memory ONLY handles spatial containment
geo_memory.is_located_in('london', 'europe')
→ True (spatial fact)

# But property query:
"Is London cold?"
→ Geographic Memory: Not applicable
→ Falls back to other modules
```

### NOT Compositional

```python
# Geographic memory stored:
# - 'london' → 'uk'
# - 'paris' → 'france'

# Query: "Is London-Paris in Europe?"
# Manas: entities=['london', 'paris'], predicates=['located_in', 'europe']

geo_memory.check_query(['london', 'paris'], ...)
→ None (compound entity 'london-paris' not stored)

# Benefit: No composition → No hallucination ✓
```

---

## Summary: Geographic Memory in One Diagram

```
╔═══════════════════════════════════════════════════════════════╗
║          GEOGRAPHIC MEMORY (Spatial Knowledge Graph)          ║
║                                                               ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │       CONTAINMENT GRAPH (Transitive Hierarchy)           │ ║
║  │                                                          │ ║
║  │  westminster → london → uk → europe                     │ ║
║  │  paris → france → europe                                │ ║
║  │  tokyo → japan → asia                                   │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                           │                                   ║
║                           ▼                                   ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │  QUERY: is_located_in(place, container)                 │ ║
║  │  • Graph traversal (BFS/path following)                 │ ║
║  │  • NO inference                                         │ ║
║  │  • Transitive (multi-hop paths)                         │ ║
║  │  • Cycle detection (safe termination)                   │ ║
║  └──────────────┬───────────────────────────────────────────┘ ║
║                 │                                             ║
║                 ▼                                             ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │  ANSWER (if path exists)                                │ ║
║  │  "Yes (London → UK → Europe)"                           │ ║
║  │                                                          │ ║
║  │  NEGATIVE (if in different location)                    │ ║
║  │  "No (Tokyo is in Asia, not Europe)"                    │ ║
║  │                                                          │ ║
║  │  REFUSAL (if not in memory)                             │ ║
║  │  None → Falls back to Buddhi                            │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                               ║
║  FOUR PRINCIPLES:                                             ║
║   1. Retrieval-Only (no inference)                           ║
║   2. Transitive Containment (multi-hop paths)                ║
║   3. Non-Inheritable (location ≠ property transfer)          ║
║   4. Negative Queries (absence detection)                    ║
║                                                               ║
║  INTEGRATION:                                                 ║
║   Priority 2 in Ahankara's query pipeline                    ║
║   (External Retrieval > Inference)                           ║
║                                                               ║
║  EXAMPLES:                                                    ║
║   ✓ "Is London in Europe?" → Geographic                     ║
║   ✓ "Is Tokyo in Asia?" → Geographic                        ║
║   ✗ "Is gold shiny?" → NOT geographic (perceptual)          ║
║   ✗ "Do bats fly?" → NOT geographic (inference)             ║
║                                                               ║
║  PHILOSOPHY:                                                  ║
║   "Retrieval ≠ Inference"                                    ║
║   "Some knowledge is LOOKED UP, not DERIVED"                 ║
║   "External sources are authoritative"                       ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**Key Insight**: Geographic Memory is MARC's "world atlas" — authoritative spatial knowledge retrieved via graph traversal, not derived through reasoning. It represents the boundary between external knowledge and internal inference.

**IMPORTANT**: Geographic Memory is an **AUXILIARY** module, not CORE. The 6 core modules (Manas, Buddhi, Chitta, HRE, Ahankara, Sakshin) are sufficient for MARC to function. Geographic Memory ENRICHES the system with spatial knowledge but is NOT required for basic operation.

**Geographic Memory is FROZEN**: This is the production geographic knowledge architecture.
