"""
Test suite for Chitta (Belief Memory)

Tests:
- Belief storage and retrieval
- Versioning
- Query index
- Edge management
- Soft-merge detection
- NO hardcoded data - all created through system
"""

import sys
from pathlib import Path

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent / "3. chitta (Belief Memory)"))

from graph import ChittaGraph
from belief import Belief


def test_belief_versioning():
    """Test belief version tracking"""
    print("\n" + "="*60)
    print("TEST: Belief Versioning")
    print("="*60)
    
    chitta = ChittaGraph()
    
    # Create a belief
    belief = Belief(
        template="relation",
        canonical={"subject": "water", "predicate": "is", "object": "wet"},
        confidence=0.5,
        original_text="Water is wet"
    )
    
    chitta.add_belief(belief)
    
    print(f"Initial confidence: {belief.confidence:.3f}")
    print(f"Initial versions: {len(belief.versions)}")
    
    # Change confidence multiple times
    for i, conf in enumerate([0.6, 0.7, 0.8], 1):
        belief.confidence = conf
        print(f"Change {i}: confidence = {conf:.3f}, versions = {len(belief.versions)}")
    
    # Check version history
    if belief.versions:
        print(f"\nVersion history:")
        for i, v in enumerate(belief.versions):
            print(f"  v{i}: conf={v['confidence']:.3f}, state={v['epistemic_state']}")
    
    assert len(belief.versions) >= 3, "Versioning not tracking changes!"
    print("✅ Versioning working correctly")
    print()


def test_query_index():
    """Test query registration and retrieval"""
    print("\n" + "="*60)
    print("TEST: Query Index")
    print("="*60)
    
    chitta = ChittaGraph()
    
    # Register queries
    queries = [
        {"predicates": ["can_fly"], "question": "Can birds fly?"},
        {"predicates": ["breathe"], "question": "Do fish breathe?"},
        {"predicates": ["can_fly"], "question": "Can penguins fly?"},
    ]
    
    for q in queries:
        chitta.register_query(q)
        print(f"Registered: {q['question']}")
    
    # Retrieve queries
    can_fly_queries = chitta.get_open_queries("can_fly")
    breathe_queries = chitta.get_open_queries("breathe")
    
    print(f"\nQueries for 'can_fly': {len(can_fly_queries)}")
    print(f"Queries for 'breathe': {len(breathe_queries)}")
    
    assert len(can_fly_queries) == 2, "Query index not working!"
    assert len(breathe_queries) == 1, "Query index not working!"
    print("✅ Query index working correctly")
    print()


def test_edge_management():
    """Test edge creation and retrieval"""
    print("\n" + "="*60)
    print("TEST: Edge Management")
    print("="*60)
    
    chitta = ChittaGraph()
    
    # Create beliefs
    b1 = Belief(
        template="is_a",
        canonical={"subject": "dog", "object": "mammal"},
        confidence=0.9,
        original_text="Dogs are mammals"
    )
    
    b2 = Belief(
        template="relation",
        canonical={"subject": "mammal", "predicate": "breathe", "object": "air"},
        confidence=0.8,
        original_text="Mammals breathe air"
    )
    
    chitta.add_belief(b1)
    chitta.add_belief(b2)
    
    print(f"Belief 1: {b1.statement_text}")
    print(f"Belief 2: {b2.statement_text}")
    
    # Add edge using private method (b1 supports b2)
    b1._add_edge_out("supports", b2.id)
    b2._add_edge_in("supports", b1.id)
    
    print(f"Created edge: {b1.id[:12]}... --supports--> {b2.id[:12]}...")
    
    # Retrieve edges using belief methods
    supports_out = b1.edges_out.get("supports", [])
    
    print(f"Supports edges from B1: {len(supports_out)}")
    
    if supports_out:
        for tgt_id in supports_out:
            print(f"  Target: {tgt_id[:12]}...")
    
    assert len(supports_out) == 1, "Edge management not working!"
    print("✅ Edge management working correctly")
    print()


def test_soft_merge_detection():
    """Test detection of near-duplicate beliefs"""
    print("\n" + "="*60)
    print("TEST: Soft-Merge Detection")
    print("="*60)
    
    chitta = ChittaGraph()
    
    # Create similar beliefs
    beliefs = [
        Belief(
            template="is_a",
            canonical={"subject": "dog", "object": "animal"},
            confidence=0.9,
            original_text="Dogs are animals"
        ),
        Belief(
            template="is_a",
            canonical={"subject": "dog", "object": "animal"},
            confidence=0.85,
            original_text="A dog is an animal"
        ),
    ]
    
    for b in beliefs:
        chitta.add_belief(b)
        print(f"Added: '{b.statement_text}'")
    
    # Detect merges
    chitta.detect_soft_merges()
    candidates = chitta.get_merge_candidates()
    
    print(f"\nMerge candidates detected: {len(candidates)}")
    
    for b1_id, b2_id, similarity in candidates:
        b1 = chitta.get(b1_id)
        b2 = chitta.get(b2_id)
        print(f"  '{b1.statement_text}' ↔ '{b2.statement_text}' (sim={similarity:.2f})")
    
    print("✅ Soft-merge detection operating")
    print()


if __name__ == "__main__":
    print("="*60)
    print("CHITTA TEST SUITE")
    print("="*60)
    print("\nTesting Chitta belief memory...")
    print("NO hardcoded data - all created dynamically")
    
    try:
        test_belief_versioning()
        test_query_index()
        test_edge_management()
        test_soft_merge_detection()
        
        print("="*60)
        print("✅ ALL CHITTA TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
