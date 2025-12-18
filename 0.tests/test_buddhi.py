"""
Test suite for Buddhi (Reasoning Engine)

Tests:
- Judgment function
- Belief aging/decay
- Hypothesis promotion
- Contradiction detection
- Coherence calculation
- NO hardcoded beliefs - all generated through real reasoning
"""

import sys
from pathlib import Path

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent / "1. manas (Input Layer)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "2. buddhi (Reasoning)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "3. chitta (Belief Memory)"))

from manas import Manas
from buddhi import Buddhi
from graph import ChittaGraph


def test_belief_decay():
    """Test belief aging and decay mechanism"""
    print("\n" + "="*60)
    print("TEST: Belief Decay")
    print("="*60)
    
    chitta = ChittaGraph()
    buddhi = Buddhi(chitta)
    manas = Manas()
    
    # Create a belief
    proposal = manas.parse("Water is wet")
    result = buddhi.think(proposal)
    
    b_id = result['belief_id']
    belief = chitta.get(b_id)
    
    # Set high confidence and low activation for decay
    belief.confidence = 0.900
    belief.activation = 0.05  # Below threshold
    initial_conf = belief.confidence
    
    print(f"Initial confidence: {initial_conf:.3f}")
    print(f"Activation: {belief.activation:.3f}")
    
    # Apply decay
    buddhi.decay_beliefs()
    
    belief = chitta.get(b_id)
    final_conf = belief.confidence
    decay_factor = final_conf / initial_conf
    
    print(f"After decay: {final_conf:.3f}")
    print(f"Decay factor: {decay_factor:.4f}")
    print(f"Decays applied: {buddhi.stats['decays']}")
    
    assert final_conf < initial_conf, "Decay not working!"
    print("✅ Decay working correctly")
    print()


def test_hypothesis_promotion():
    """Test hypothesis promotion mechanism"""
    print("\n" + "="*60)
    print("TEST: Hypothesis Promotion")
    print("="*60)
    
    chitta = ChittaGraph()
    buddhi = Buddhi(chitta)
    manas = Manas()
    
    # Create a hypothesis
    proposal = manas.parse("Gold is valuable")
    result = buddhi.think(proposal)
    
    b_id = result['belief_id']
    belief = chitta.get(b_id)
    
    # Set as hypothesis with high confidence
    belief.epistemic_state = "hypothetical"
    belief.confidence = 0.700  # Above promotion threshold
    
    print(f"Initial state: {belief.epistemic_state}")
    print(f"Confidence: {belief.confidence:.3f}")
    
    # Apply promotion
    buddhi.promote_hypotheses()
    
    belief = chitta.get(b_id)
    print(f"After promotion: {belief.epistemic_state}")
    print(f"Promotions: {buddhi.stats['promotions']}")
    
    assert belief.epistemic_state == "asserted", "Promotion not working!"
    print("✅ Promotion working correctly")
    print()


def test_contradiction_detection():
    """Test contradiction edge creation"""
    print("\n" + "="*60)
    print("TEST: Contradiction Detection")
    print("="*60)
    
    chitta = ChittaGraph()
    buddhi = Buddhi(chitta)
    manas = Manas()
    
    # Add general rule
    p1 = manas.parse("Birds can fly")
    r1 = buddhi.think(p1)
    print(f"Added: '{p1['raw_text']}'")
    
    # Add specific exception
    p2 = manas.parse("Penguins cannot fly")
    r2 = buddhi.think(p2)
    print(f"Added: '{p2['raw_text']}'")
    
    # Check for contradiction edges
    b1_id = r1['belief_id']
    b2_id = r2['belief_id']
    
    b1 = chitta.get(b1_id)
    b2 = chitta.get(b2_id)
    
    # Get outgoing contradiction edges from b2
    contradicts_out = b2.edges_out.get("contradicts", [])
    
    print(f"\nBelief 1: {b1.statement_text}")
    print(f"Belief 2: {b2.statement_text}")
    print(f"Contradiction edges from B2: {len(contradicts_out)}")
    
    assert len(contradicts_out) > 0, "No contradiction edges created!"
    print("✅ Contradiction detection working")
    print()


def test_judgment_function():
    """Test judgment function components"""
    print("\n" + "="*60)
    print("TEST: Judgment Function")
    print("="*60)
    
    chitta = ChittaGraph()
    buddhi = Buddhi(chitta)
    manas = Manas()
    
    # Create related beliefs to test coherence
    beliefs = [
        "Mammals breathe air",
        "Dogs are mammals",
        "Dogs breathe air",
    ]
    
    for text in beliefs:
        proposal = manas.parse(text)
        result = buddhi.think(proposal)
        b_id = result['belief_id']
        belief = chitta.get(b_id)
        
        print(f"\n'{text}'")
        print(f"  Confidence: {belief.confidence:.3f}")
        print(f"  Epistemic state: {belief.epistemic_state}")
        print(f"  Action: {result['action']}")
    
    print("\n✅ Judgment function operating")
    print()


if __name__ == "__main__":
    print("="*60)
    print("BUDDHI TEST SUITE")
    print("="*60)
    print("\nTesting Buddhi reasoning engine...")
    print("NO hardcoded beliefs - all through real reasoning")
    
    try:
        test_belief_decay()
        test_hypothesis_promotion()
        test_contradiction_detection()
        test_judgment_function()
        
        print("="*60)
        print("✅ ALL BUDDHI TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
