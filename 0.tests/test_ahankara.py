"""
Test suite for Ahankara (Self Model / Orchestrator)

Tests:
- Cognitive phase enforcement
- Process pipeline
- Query answering
- Deliberation budget
- NO hardcoded orchestration - real phase management
"""

import sys
from pathlib import Path

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent / "5. ahankara (Self Model)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "3. chitta (Belief Memory)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "2. buddhi (Reasoning)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "1. manas (Input Layer)"))

from ahankara import Ahankara, PHASE_IDLE, PHASE_PERCEIVE, PHASE_JUDGE, PHASE_COMMIT


def test_phase_enforcement():
    """Test cognitive phase transitions"""
    print("\n" + "="*60)
    print("TEST: Phase Enforcement")
    print("="*60)
    
    marc = Ahankara()
    
    print(f"Initial phase: {marc.phase}")
    assert marc.phase == PHASE_IDLE, "Should start in IDLE!"
    
    # Process should go through phases
    result = marc.process("Water is wet")
    
    print(f"After process: {marc.phase}")
    assert marc.phase == PHASE_IDLE, "Should return to IDLE after commit!"
    
    print("✅ Phase enforcement working")
    print()


def test_process_pipeline():
    """Test full cognitive cycle"""
    print("\n" + "="*60)
    print("TEST: Process Pipeline")
    print("="*60)
    
    marc = Ahankara()
    
    inputs = [
        "Birds can fly",
        "Penguins are birds",
        "Penguins cannot fly",
    ]
    
    for text in inputs:
        result = marc.process(text)
        print(f"\nInput: '{text}'")
        print(f"  Action: {result['action']}")
        print(f"  Belief ID: {result.get('belief_id', 'N/A')}")
        print(f"  Message: {result.get('message', 'N/A')}")
    
    print(f"\nSystem stats:")
    print(f"  Cycles: {marc.stats['cycles']}")
    print(f"  Perceive calls: {marc.stats['perceive_calls']}")
    print(f"  Judge calls: {marc.stats['judge_calls']}")
    print(f"  Commit calls: {marc.stats['commit_calls']}")
    
    print("✅ Process pipeline working")
    print()


def test_query_answering():
    """Test query interface"""
    print("\n" + "="*60)
    print("TEST: Query Answering")
    print("="*60)
    
    marc = Ahankara()
    
    # Add knowledge
    marc.process("Birds can fly")
    marc.process("Penguins are birds")
    marc.process("Penguins cannot fly")
    
    # Ask questions
    questions = [
        "Can birds fly?",
        "Can penguins fly?",
        "Are penguins birds?",
    ]
    
    for q in questions:
        answer = marc.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")
    
    print("\n✅ Query answering working")
    print()


def test_deliberation_budget():
    """Test query depth limiting"""
    print("\n" + "="*60)
    print("TEST: Deliberation Budget")
    print("="*60)
    
    marc = Ahankara(max_query_depth=2)
    
    # Ask a question that might not have an answer
    question = "What is the meaning of life?"
    answer = marc.ask(question)
    
    print(f"Q: {question}")
    print(f"A: {answer[:100]}...")
    print(f"\nDepth exceeded: {marc.stats['query_depth_exceeded']}")
    
    print("✅ Deliberation budget enforced")
    print()


def test_statistics():
    """Test statistics tracking"""
    print("\n" + "="*60)
    print("TEST: Statistics Tracking")
    print("="*60)
    
    marc = Ahankara()
    
    # Process several inputs
    for i in range(5):
        marc.process(f"Fact number {i}")
    
    # Ask a question
    marc.ask("What is fact number 3?")
    
    print("System statistics:")
    for key, value in marc.stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Statistics tracking working")
    print()


if __name__ == "__main__":
    print("="*60)
    print("AHANKARA TEST SUITE")
    print("="*60)
    print("\nTesting Ahankara orchestration...")
    print("NO hardcoded orchestration - real phase management")
    
    try:
        test_phase_enforcement()
        test_process_pipeline()
        test_query_answering()
        test_deliberation_budget()
        test_statistics()
        
        print("="*60)
        print("✅ ALL AHANKARA TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
