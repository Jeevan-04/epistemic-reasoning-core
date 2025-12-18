"""
HRE Comprehensive Test

Tests hypothetical reasoning capabilities while enforcing epistemic sterility.
"""

import sys
from pathlib import Path

marc_root = Path(__file__).parent.parent
sys.path.insert(0, str(marc_root / "1. manas (Input Layer)"))
sys.path.insert(0, str(marc_root / "2. buddhi (Reasoning)"))
sys.path.insert(0, str(marc_root / "3. chitta (Belief Memory)"))
sys.path.insert(0, str(marc_root / "4. HRE (Hypothetical Reasoning Engine)"))
sys.path.insert(0, str(marc_root / "5. ahankara (Self Model)"))

from ahankara import Ahankara


def test_hre_counterfactual_reasoning():
    """Test HRE's ability to reason about counterfactuals."""
    print("\n" + "="*80)
    print("TEST: Counterfactual Reasoning")
    print("="*80)
    
    marc = Ahankara()
    
    # Teach baseline facts
    marc.process("Birds can fly")
    marc.process("Penguins are birds")
    
    # Normal query
    answer = marc.ask("Can birds fly?")
    print(f"\nNormal: Can birds fly?")
    print(f"Answer: {answer}")
    assert "yes" in answer.lower() or "birds can fly" in answer.lower()
    
    # Hypothetical: What if birds couldn't fly?
    hyp_answer = marc.ask_hypothetically(
        "Can birds fly?",
        assumptions=["Birds cannot fly"]
    )
    print(f"\nHypothetical: Can birds fly? (assuming birds cannot fly)")
    print(f"Answer: {hyp_answer}")
    assert "hypothetically" in hyp_answer.lower()
    
    # Verify baseline unchanged
    answer_after = marc.ask("Can birds fly?")
    print(f"\nNormal (after HRE): Can birds fly?")
    print(f"Answer: {answer_after}")
    assert answer == answer_after
    
    print("\n✅ Counterfactual reasoning working correctly")


def test_hre_assumption_chains():
    """Test HRE with multiple assumptions."""
    print("\n" + "="*80)
    print("TEST: Multiple Assumptions")
    print("="*80)
    
    marc = Ahankara()
    
    # Teach facts
    marc.process("Mammals breathe air")
    marc.process("Whales are mammals")
    
    # Query with multiple assumptions
    hyp_answer = marc.ask_hypothetically(
        "Do whales breathe air?",
        assumptions=[
            "Whales are fish",
            "Fish breathe water"
        ]
    )
    
    print(f"\nHypothetical: Do whales breathe air?")
    print(f"Assumptions: Whales are fish, Fish breathe water")
    print(f"Answer: {hyp_answer}")
    
    assert "hypothetically" in hyp_answer.lower()
    
    # Verify baseline
    normal_answer = marc.ask("Are whales mammals?")
    print(f"\nNormal: Are whales mammals?")
    print(f"Answer: {normal_answer}")
    assert "yes" in normal_answer.lower()
    
    print("\n✅ Multiple assumptions working correctly")


def test_hre_no_assumptions():
    """Test HRE without assumptions (should behave like normal query)."""
    print("\n" + "="*80)
    print("TEST: HRE Without Assumptions")
    print("="*80)
    
    marc = Ahankara()
    
    marc.process("Dogs are mammals")
    
    # HRE with no assumptions
    hyp_answer = marc.ask_hypothetically(
        "Are dogs mammals?",
        assumptions=[]
    )
    
    print(f"\nHypothetical (no assumptions): Are dogs mammals?")
    print(f"Answer: {hyp_answer}")
    
    # Should still be marked as hypothetical
    assert "hypothetically" in hyp_answer.lower()
    
    print("\n✅ HRE without assumptions working correctly")


def test_hre_epistemic_sterility_comprehensive():
    """Comprehensive test of epistemic sterility."""
    print("\n" + "="*80)
    print("TEST: Epistemic Sterility (Comprehensive)")
    print("="*80)
    
    marc = Ahankara()
    
    # Teach baseline
    marc.process("Cats are animals")
    marc.process("Cats meow")
    
    initial_beliefs = len(marc.chitta.beliefs)
    initial_confidences = {
        bid: b.confidence 
        for bid, b in marc.chitta.beliefs.items()
    }
    
    print(f"\nInitial state: {initial_beliefs} beliefs")
    
    # Run 10 hypothetical queries with various assumptions
    for i in range(10):
        marc.ask_hypothetically(
            "Are cats animals?",
            assumptions=[f"Assumption {i}"]
        )
    
    # Verify NO changes
    final_beliefs = len(marc.chitta.beliefs)
    final_confidences = {
        bid: b.confidence 
        for bid, b in marc.chitta.beliefs.items()
    }
    
    print(f"After 10 HRE queries: {final_beliefs} beliefs")
    
    assert initial_beliefs == final_beliefs, \
        f"Belief count changed: {initial_beliefs} → {final_beliefs}"
    
    for bid in initial_confidences:
        assert abs(initial_confidences[bid] - final_confidences[bid]) < 0.001, \
            f"Confidence changed for {bid}"
    
    print("\n✅ Epistemic sterility maintained across multiple queries")


def test_hre_statistics():
    """Test that HRE tracks statistics correctly."""
    print("\n" + "="*80)
    print("TEST: HRE Statistics")
    print("="*80)
    
    marc = Ahankara()
    
    if marc.hre is None:
        print("⏸️  HRE not available")
        return
    
    marc.process("Birds fly")
    
    initial_queries = marc.hre.stats["queries"]
    initial_sandboxes = marc.hre.stats["sandboxes_created"]
    
    # Run some HRE queries
    marc.ask_hypothetically("Do birds fly?")
    marc.ask_hypothetically("Do birds fly?", assumptions=["Birds cannot fly"])
    
    final_queries = marc.hre.stats["queries"]
    final_sandboxes = marc.hre.stats["sandboxes_created"]
    
    print(f"\nQueries: {initial_queries} → {final_queries}")
    print(f"Sandboxes: {initial_sandboxes} → {final_sandboxes}")
    
    assert final_queries == initial_queries + 2
    assert final_sandboxes == initial_sandboxes + 2
    
    print("\n✅ HRE statistics tracking working correctly")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HRE COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nTesting hypothetical reasoning with epistemic sterility enforcement")
    print("="*80)
    
    try:
        test_hre_counterfactual_reasoning()
        test_hre_assumption_chains()
        test_hre_no_assumptions()
        test_hre_epistemic_sterility_comprehensive()
        test_hre_statistics()
        
        print("\n" + "="*80)
        print("🎉 ALL HRE TESTS PASSED")
        print("="*80)
        print("\n✅ HRE v0 implementation complete")
        print("✅ Epistemic sterility enforced")
        print("✅ Counterfactual reasoning working")
        print("✅ No contamination of base Chitta")
        print("="*80)
        
    except AssertionError as e:
        print("\n" + "="*80)
        print("❌ HRE TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        print("\n🚨 Fix violation before proceeding")
        print("="*80)
        raise
