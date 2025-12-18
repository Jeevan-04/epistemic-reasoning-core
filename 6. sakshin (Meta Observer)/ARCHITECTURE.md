# Sakshin — The Meta-Observer (साक्षिन्)

## What is Sakshin?

**Sakshin** (Sanskrit: साक्षिन्, "witness" or "observer") is MARC's **meta-observer**. It watches, records, and logs — but NEVER intervenes.

Think of Sakshin like your **metacognition** — the part of your mind that observes your own thinking without changing it.

---

## The Core Problem: Why Do We Need Sakshin?

### The Human Analogy

When you solve a math problem, part of your brain is DOING the problem. But another part is WATCHING:

```
Problem: "What is 17 × 23?"

Working Memory (Ahankara, Buddhi):
  17 × 23
  = 17 × 20 + 17 × 3
  = 340 + 51
  = 391

Metacognition (Sakshin):
  "I noticed I used the distributive method"
  "I was confident about 17 × 20"
  "I had to think longer about 51"
  "I double-checked the addition"
```

**The Observer** (Sakshin) doesn't CHANGE the calculation. It just WATCHES and REMEMBERS how you thought.

### The Auditability Problem (Without Sakshin)

**Without observation**:
```python
# System answers question
ahankara.query_answer("Do bats produce milk?")
→ "Yes. Inherited from mammal."

# User asks: "How did you get that answer?"
# System: ??? (no trace of reasoning process)

# Developer asks: "Why did the system fail on this edge case?"
# No logs, no trace → impossible to debug
```

**With Sakshin**:
```python
# System answers question
ahankara.query_answer("Do bats produce milk?")
→ "Yes. Inherited from mammal."

# Sakshin logged:
{
  "query": "Do bats produce milk?",
  "timestamp": "2024-12-16T08:15:30Z",
  "modules_used": ["manas", "buddhi"],
  "perception": {
    "entities": ["bat"],
    "predicates": ["produces_milk"],
    "confidence": 0.95
  },
  "reasoning_path": [
    "direct_match → not found",
    "taxonomic_traversal → bat → mammal",
    "inheritance → mammal produces_milk (positive)",
    "verdict → yes"
  ],
  "answer": "Yes. Inherited from mammal.",
  "confidence": 0.9
}

# User asks: "How did you get that answer?"
→ Show reasoning_path

# Developer asks: "Why did system fail?"
→ Analyze logs, find where reasoning went wrong
```

**Key Insight**: Observation enables **auditability** and **debuggability**.

---

## Architecture: How Sakshin Works

### High-Level Flow

```
┌───────────────────────────────────────────────────────────────┐
│                  MARC COGNITIVE PROCESS                       │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Manas: Parse "Do bats produce milk?"                   │  │
│  │  → entities=['bat'], predicates=['produces_milk']       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Buddhi: Reason about query                             │  │
│  │  → Taxonomic inheritance via mammal                     │  │
│  │  → Verdict: YES                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Ahankara: Render answer                                │  │
│  │  → "Yes. Inherited from mammal."                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│                  ┌────────────────┐                           │
│                  │    SAKSHIN     │                           │
│                  │  (Observer)    │                           │
│                  └────────┬───────┘                           │
│                           │                                   │
│                           │  [Watches, Records, NEVER Changes]│
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              AUDIT LOG (Read-Only)                      │  │
│  │                                                         │  │
│  │  {                                                      │  │
│  │    "timestamp": "2024-12-16T08:15:30Z",                │  │
│  │    "query": "Do bats produce milk?",                   │  │
│  │    "modules_used": ["manas", "buddhi"],                │  │
│  │    "perception": {...},                                │  │
│  │    "reasoning_path": [...],                            │  │
│  │    "answer": "Yes. Inherited from mammal.",            │  │
│  │    "confidence": 0.9                                   │  │
│  │  }                                                      │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

**Critical Properties**:
1. **Read-Only**: Sakshin NEVER modifies beliefs, proofs, or answers
2. **Non-Intrusive**: Sakshin doesn't slow down reasoning (async logging)
3. **Comprehensive**: Logs EVERY decision, not just successes

---

## Deep Dive: The Observation Algorithm

### Step 1: Passive Listening

**Goal**: Receive events from Ahankara without interfering.

**Algorithm**:
```python
class Sakshin:
    """
    The meta-observer. Watches without intervening.
    """
    def __init__(self):
        self.logs: List[Dict] = []
    
    def observe(self, event: Dict):
        """
        Log an event.
        
        CRITICAL: This method NEVER raises exceptions.
        If logging fails, silently continue (don't break reasoning).
        """
        try:
            # Add timestamp
            event['timestamp'] = datetime.now().isoformat()
            
            # Append to log
            self.logs.append(event)
            
        except Exception as e:
            # NEVER raise (would break reasoning)
            # Just silently fail
            pass
```

**Why Never Raise?**

```python
# WRONG: Raising exception
def observe(event):
    if invalid(event):
        raise ValueError("Invalid event")  # ← BREAKS reasoning!

# Ahankara calls Sakshin
ahankara.answer(query)
sakshin.observe(event)  # ← Exception raised!
# → Answer never returned to user (BAD)

# CORRECT: Silent failure
def observe(event):
    try:
        log(event)
    except:
        pass  # ← Logging failed, but reasoning continues

# Ahankara calls Sakshin
ahankara.answer(query)
sakshin.observe(event)  # ← Even if logging fails...
# → Answer still returned to user (GOOD)
```

**Philosophy**: **Observation must NEVER interfere with cognition.**

### Step 2: Structured Logging

**Goal**: Log events in structured, queryable format.

**Log Schema**:
```python
@dataclass
class CognitiveEvent:
    """
    One decision/action in MARC's cognitive process.
    """
    timestamp: str              # ISO 8601 timestamp
    event_type: str             # "query", "assertion", "error"
    query_text: str             # Original user input
    
    # Perception
    perception: Dict            # Manas output
    
    # Reasoning
    modules_used: List[str]     # Which modules were called
    reasoning_path: List[str]   # Step-by-step reasoning trace
    
    # Answer
    answer: str                 # Final rendered answer
    confidence: float           # Overall confidence
    
    # Metadata
    execution_time_ms: float    # How long it took
    errors: List[str]           # Any errors encountered
```

**Example Event**:
```json
{
  "timestamp": "2024-12-16T08:15:30.123Z",
  "event_type": "query",
  "query_text": "Do bats produce milk?",
  
  "perception": {
    "entities": ["bat"],
    "predicates": ["produces_milk"],
    "polarity": "interrogative",
    "confidence": 0.95
  },
  
  "modules_used": ["manas", "buddhi"],
  
  "reasoning_path": [
    "phase:perception → manas.parse()",
    "phase:reasoning → buddhi.answer()",
    "step:direct_match → not found",
    "step:frame_check → produces_milk is FUNCTIONAL(inherits=True)",
    "step:taxonomy_traversal → bat → mammal",
    "step:negation_check → no blocking negation",
    "step:inheritance → mammal produces_milk (POSITIVE)",
    "step:conclusion → YES (confidence: 0.9)"
  ],
  
  "answer": "Yes. Inherited from mammal.",
  "confidence": 0.9,
  
  "execution_time_ms": 67.3,
  "errors": []
}
```

### Step 3: Query Interface (Optional)

**Goal**: Allow analysis of past observations.

**Algorithm**:
```python
class Sakshin:
    def get_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Query audit logs.
        
        Examples:
          - Get all logs from today
          - Get all errors
          - Get all queries about "bat"
        """
        filtered = self.logs
        
        if start_time:
            filtered = [
                log for log in filtered
                if datetime.fromisoformat(log['timestamp']) >= start_time
            ]
        
        if end_time:
            filtered = [
                log for log in filtered
                if datetime.fromisoformat(log['timestamp']) <= end_time
            ]
        
        if event_type:
            filtered = [
                log for log in filtered
                if log['event_type'] == event_type
            ]
        
        return filtered
    
    def get_reasoning_patterns(self) -> Dict[str, int]:
        """
        Analyze which reasoning patterns are most common.
        
        Returns:
          {
            "direct_match": 145,
            "taxonomic_inheritance": 89,
            "negation_dominance": 23,
            ...
          }
        """
        patterns = defaultdict(int)
        
        for log in self.logs:
            for step in log.get('reasoning_path', []):
                # Extract pattern name
                if 'step:' in step:
                    pattern = step.split('→')[0].strip().replace('step:', '')
                    patterns[pattern] += 1
        
        return dict(patterns)
```

**Example Queries**:
```python
# Get all logs from today
today = datetime.now().replace(hour=0, minute=0, second=0)
logs = sakshin.get_logs(start_time=today)

# Get all errors
errors = sakshin.get_logs(event_type='error')

# Analyze reasoning patterns
patterns = sakshin.get_reasoning_patterns()
→ {
    "direct_match": 145,
    "taxonomic_inheritance": 89,
    "negation_dominance": 23,
    "grounding_refusal": 12
  }
```

---

## The Three Principles of Sakshin

### Principle 1: Observe, Never Intervene

**Definition**: Sakshin WATCHES cognitive processes but NEVER changes them.

**What Sakshin Does NOT Do**:
- ❌ Modify beliefs
- ❌ Alter reasoning
- ❌ Change answers
- ❌ Reject queries
- ❌ Raise exceptions that break cognition

**What Sakshin DOES**:
- ✓ Record decisions
- ✓ Log events
- ✓ Track patterns
- ✓ Enable debugging

**Human Analogy**:

When you notice yourself making a mistake:
- ❌ **Intervention**: "Wait, I'm doing this wrong!" → Change behavior
- ✓ **Observation**: "I notice I made an error" → Log for later reflection

Sakshin is PURE observation (no intervention).

### Principle 2: Comprehensive Logging

**Definition**: Log EVERYTHING, not just successes.

**Traditional Logging** (success-only):
```python
# Only log successful answers
if answer.verdict == "yes":
    log(answer)  # ← Only logs successes

# Problems:
# - Can't debug failures
# - Can't see refusals
# - Can't analyze error patterns
```

**Sakshin Logging** (comprehensive):
```python
# Log EVERY query, regardless of outcome
sakshin.observe({
    'query': query,
    'answer': answer,  # ← Even if answer is "I don't know"
    'errors': errors   # ← Even if errors occurred
})

# Benefits:
# - See where system refuses (epistemic discipline)
# - Debug failures (trace reasoning path)
# - Analyze error patterns (improve system)
```

**Philosophy**: **Failures are data.** Log them.

### Principle 3: Read-Only Audit Trail

**Definition**: Logs are IMMUTABLE. Once written, never modified.

**Why Immutability?**

```python
# WRONG: Mutable logs
def observe(event):
    log_id = len(self.logs)
    self.logs[log_id] = event
    
    # Later: Modify log
    self.logs[log_id]['answer'] = "REDACTED"  # ← Tampering!

# Problems:
# - Logs no longer trustworthy
# - Audit trail corrupted
# - Can't reproduce past behavior

# CORRECT: Immutable logs
def observe(event):
    self.logs.append(event)  # ← Append-only

# Logs never modified after creation
# → Trustworthy audit trail
```

**Benefit**: Logs are **forensic evidence** of MARC's reasoning. Immutability ensures they're trustworthy.

---

## What Sakshin Logs

### Event Types

1. **Query Events** (user asks question):
```json
{
  "event_type": "query",
  "query_text": "Do bats fly?",
  "perception": {...},
  "reasoning_path": [...],
  "answer": "Yes.",
  "confidence": 0.95
}
```

2. **Assertion Events** (user teaches fact):
```json
{
  "event_type": "assertion",
  "assertion_text": "Bats are mammals",
  "perception": {...},
  "belief_added": {...},
  "conflicts": []
}
```

3. **Error Events** (something went wrong):
```json
{
  "event_type": "error",
  "error_type": "parsing_failure",
  "input_text": "Askdjfh asdf?",
  "error_message": "Could not parse query",
  "stack_trace": "..."
}
```

4. **Lifecycle Events** (belief decay, promotion):
```json
{
  "event_type": "lifecycle",
  "action": "decay",
  "beliefs_affected": 127,
  "average_confidence_change": -0.05
}
```

---

## Why This Design? (Feynman's Question)

### "Why not just use print() statements?"

**Print statements**:
```python
def answer(query):
    print(f"Answering: {query}")  # ← Unstructured
    result = buddhi.reason(query)
    print(f"Result: {result}")    # ← No metadata
    return result

# Problems:
# - Unstructured (hard to parse)
# - No timestamps
# - No queryability
# - Mixes with user output (cluttered)
```

**Sakshin logging**:
```python
def answer(query):
    sakshin.observe({
        'event_type': 'query',
        'query': query,
        'timestamp': datetime.now(),
        'result': result,
        'confidence': result.confidence
    })
    return result

# Benefits:
# - Structured (JSON/dict)
# - Queryable (filter by time, type, etc.)
# - Separate from user output
# - Machine-readable (can analyze programmatically)
```

### "Why not modify logs for privacy/redaction?"

**Mutable logs** (privacy via redaction):
```python
# Log event
sakshin.observe(event)

# Later: Redact sensitive info
for log in sakshin.logs:
    if 'sensitive' in log['query']:
        log['query'] = "REDACTED"  # ← Modifying history!

# Problem: Audit trail corrupted
# Can't trust logs (might be tampered)
```

**Immutable logs** (privacy via filtering):
```python
# Log event (immutable)
sakshin.observe(event)

# Query with filter (doesn't modify logs)
public_logs = sakshin.get_logs(exclude_sensitive=True)
# → Returns filtered view, original logs unchanged

# Benefit: Original audit trail preserved
# Public view respects privacy
# Forensic trail still available for debugging
```

### "Why async logging (if implemented)?"

**Synchronous logging** (blocks reasoning):
```python
def answer(query):
    result = buddhi.reason(query)  # Fast: 50ms
    sakshin.observe(event)          # Slow: 20ms (disk I/O)
    return result
# Total: 70ms (logging adds overhead)
```

**Asynchronous logging** (non-blocking):
```python
def answer(query):
    result = buddhi.reason(query)  # Fast: 50ms
    sakshin.observe_async(event)    # Returns immediately
    return result
# Total: 50ms (logging doesn't block)
# Logging happens in background
```

**Trade-off**: Slight delay in log availability, but reasoning stays fast.

**Note**: Current MARC implementation is synchronous (simple). Async can be added later if needed.

---

## Summary: Sakshin in One Diagram

```
╔═══════════════════════════════════════════════════════════════╗
║                   SAKSHIN (Meta-Observer)                     ║
║                                                               ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │                  MARC COGNITIVE PROCESS                  │ ║
║  │  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐     │ ║
║  │  │ Manas  │ → │ Buddhi │ → │Ahankara│ → │ Answer │     │ ║
║  │  └────────┘   └────────┘   └────────┘   └────────┘     │ ║
║  │       ↓            ↓            ↓            ↓          │ ║
║  │       └────────────┴────────────┴────────────┘          │ ║
║  │                        │                                │ ║
║  └────────────────────────┼────────────────────────────────┘ ║
║                           │                                  ║
║                           ▼                                  ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │              SAKSHIN (Passive Observer)                  │ ║
║  │                                                          │ ║
║  │  observe(event):                                         │ ║
║  │    - NEVER modifies event                               │ ║
║  │    - NEVER raises exception                             │ ║
║  │    - NEVER changes reasoning                            │ ║
║  │    - ONLY logs to audit trail                           │ ║
║  └──────────────────┬───────────────────────────────────────┘ ║
║                     │                                         ║
║                     ▼                                         ║
║  ┌──────────────────────────────────────────────────────────┐ ║
║  │         AUDIT TRAIL (Immutable, Append-Only)             │ ║
║  │                                                          │ ║
║  │  [                                                       │ ║
║  │    {event_1: query, answer, reasoning_path, ...},       │ ║
║  │    {event_2: assertion, belief_added, ...},             │ ║
║  │    {event_3: error, error_message, ...},                │ ║
║  │    ...                                                   │ ║
║  │  ]                                                       │ ║
║  │                                                          │ ║
║  │  Properties:                                             │ ║
║  │  • Immutable (never modified after creation)            │ ║
║  │  • Comprehensive (logs successes AND failures)          │ ║
║  │  • Structured (queryable JSON/dict format)              │ ║
║  │  • Timestamped (ISO 8601)                               │ ║
║  └──────────────────────────────────────────────────────────┘ ║
║                                                               ║
║  THREE PRINCIPLES:                                            ║
║   1. Observe, Never Intervene                                ║
║   2. Comprehensive Logging (successes + failures)            ║
║   3. Read-Only Audit Trail (immutable)                       ║
║                                                               ║
║  USE CASES:                                                   ║
║   • Debugging (why did system answer X?)                     ║
║   • Auditability (what did system decide?)                   ║
║   • Research (analyze reasoning patterns)                    ║
║   • Reproducibility (replay past queries)                    ║
║                                                               ║
║  DOES NOT:                                                    ║
║   • Modify beliefs (read-only)                               ║
║   • Change reasoning (non-intrusive)                         ║
║   • Raise exceptions (silent failure)                        ║
║   • Filter/redact (logs raw events)                          ║
║                                                               ║
║  PHILOSOPHY:                                                  ║
║   "Observation ≠ Intervention"                               ║
║   "Watching ≠ Changing"                                      ║
║   "The witness doesn't judge, just records"                  ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**Key Insight**: Sakshin is intentionally "boring" — it does ONE thing (observe) and does it without interfering. Like a flight recorder that logs everything but never touches the controls.

**Sakshin is FROZEN**: This is the production meta-observation architecture.
