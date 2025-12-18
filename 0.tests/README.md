# MARC Test Suite

Comprehensive testing for all MARC modules with **NO hardcoded responses** - everything uses real cognition.

## Test Files

### Module Tests
- **`test_manas.py`** - Understanding engine (parsing, normalization, intent detection)
- **`test_buddhi.py`** - Reasoning engine (judgment, decay, promotion, contradictions)
- **`test_chitta.py`** - Belief memory (storage, versioning, query index, edges)
- **`test_ahankara.py`** - Orchestrator (phases, pipeline, query answering)

### Integration Tests
- **`test_integration.py`** - Full system test (all refinements working together)
- **`test_penguin.py`** - Classic penguin scenario (contradiction detection)

### Interactive Test
- **`test_interactive.py`** - **Terminal input mode** - teach MARC yourself!

## Running Tests

### Run All Tests
```bash
cd 0.tests
python run_all_tests.py
```

### Run Individual Module Tests
```bash
python test_manas.py      # Test understanding
python test_buddhi.py     # Test reasoning
python test_chitta.py     # Test memory
python test_ahankara.py   # Test orchestration
```

### Run Integration Tests
```bash
python test_integration.py  # Comprehensive system test
python test_penguin.py      # Penguin contradiction scenario
```

### Run Interactive Test (Recommended!)
```bash
python test_interactive.py
```

This lets you:
- Type statements to teach MARC
- Ask questions to query MARC's knowledge
- Type `beliefs` to see what MARC knows
- Type `stats` to see system statistics
- Type `quit` to exit

**MARC starts with ZERO knowledge** - everything is learned from YOUR input!

## Test Philosophy

### NO Hardcoding
- ✅ No hardcoded responses
- ✅ No hardcoded beliefs
- ✅ No mock data
- ✅ All tests use real parsing and reasoning

### Real Cognition
- Manas actually parses text (rule-based, no LLM needed for tests)
- Buddhi actually judges beliefs using full judgment function
- Chitta actually stores and versions beliefs
- Ahankara actually enforces phase transitions

### What We Test

#### Manas
- Entity normalization (`"Birds"` → `"bird"`)
- Intent detection (query vs assertion)
- Modality detection (strong/weak/default)
- Assertion: all entities are lowercase

#### Buddhi
- Belief decay (0.995 per cycle)
- Hypothesis promotion (confidence > 0.65)
- Contradiction detection (tentative edges)
- Coherence bonus calculation
- Asymmetric conflict weighting

#### Chitta
- Belief versioning (full epistemic trace)
- Query index (open/answered queries)
- Edge management (4-tuple format with metadata)
- Soft-merge detection (NO auto-merge)

#### Ahankara
- Phase enforcement (IDLE→PERCEIVE→JUDGE→COMMIT)
- Process pipeline (full cognitive cycle)
- Query answering through Buddhi
- Deliberation budget (max depth)
- Language rendering centralization

## Example: Interactive Test

```bash
$ python test_interactive.py

======================================================================
MARC INTERACTIVE TEST - Real Cognition, NO Hardcoding
======================================================================

Commands:
  - Type any statement to teach MARC
  - Ask questions to query MARC's knowledge
  - 'beliefs' - show all beliefs
  - 'stats' - show system statistics
  - 'quit' or 'exit' - stop

MARC starts with ZERO knowledge. Everything is learned from YOU!
======================================================================

You: Birds can fly
MARC: ✓ Understood and stored.

You: Penguins are birds
MARC: ✓ Understood and stored.

You: Penguins cannot fly
MARC: ⚠ Uncertain - stored as hypothesis.

You: Can penguins fly?
MARC: I am not certain. General rule: Penguins are birds. Exception hypothesis: Penguins cannot fly.

You: beliefs

======================================================================
BELIEF GRAPH
======================================================================
Total active beliefs: 3

1. Birds can fly
   Confidence: 0.837
   State: asserted
   Template: relation
   Edges:
     contradicts: 1 connections

2. Penguins are birds
   Confidence: 0.891
   State: asserted
   Template: is_a

3. Penguins cannot fly
   Confidence: 0.298
   State: hypothetical
   Template: relation
   Edges:
     contradicts: 1 connections

======================================================================

You: quit
👋 Goodbye! MARC shutting down...
```

## Expected Results

All tests should pass with ✅:

```
======================================================================
TEST SUMMARY
======================================================================
✅ PASSED - test_manas.py
✅ PASSED - test_buddhi.py
✅ PASSED - test_chitta.py
✅ PASSED - test_ahankara.py
✅ PASSED - test_integration.py

======================================================================
Results: 5 passed, 0 failed, 0 skipped
======================================================================

🎉 All tests passed!
```

## Debugging Failed Tests

If a test fails:
1. Check the error message for which assertion failed
2. Run that specific test file individually for detailed output
3. Use `test_interactive.py` to manually verify behavior
4. Check `STATUS.md` in root for module specifications

## Notes

- Tests use rule-based parsing (no LLM required)
- All parsing logic is in `manas_utils.py`
- Judgment function is fully implemented in `buddhi.py`
- Phase enforcement uses assertions (will raise if violated)
- Interactive test is the best way to understand MARC's cognition!
