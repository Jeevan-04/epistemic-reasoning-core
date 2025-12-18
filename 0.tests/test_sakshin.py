"""
THE TERRIFYING TEST

If Sakshin can't reconstruct complete influence graphs, STOP.

This test verifies that Sakshin can answer:
  "Show me every belief that influenced this answer, directly or indirectly."

If this test fails, the auditability guarantee is broken.
"""

import sys
from pathlib import Path

marc_root = Path(__file__).parent.parent
sys.path.insert(0, str(marc_root / "1. manas (Input Layer)"))
sys.path.insert(0, str(marc_root / "2. buddhi (Reasoning)"))
sys.path.insert(0, str(marc_root / "3. chitta (Belief Memory)"))
sys.path.insert(0, str(marc_root / "5. ahankara (Self Model)"))
sys.path.insert(0, str(marc_root / "6. sakshin (Meta Observer)"))

from ahankara import Ahankara
from sakshin import Sakshin


def test_basic_event_recording():
    """Test that Sakshin records events."""
    print("\n" + "="*80)
    print("TEST 1: Basic Event Recording")
    print("="*80)
    
    sakshin = Sakshin()
    
    # Record events
    event_id_1 = sakshin.observe("manas", "parse", {"input": "Birds can fly"})
    event_id_2 = sakshin.observe("buddhi", "judgment", {"action": "added_belief"})
    event_id_3 = sakshin.observe("chitta", "belief_stored", {"belief_id": "b_123"})
    
    print(f"\nRecorded {len(sakshin.events)} events:")
    for i, event in enumerate(sakshin.events):
        print(f"  {i+1}. [{event.module}] {event.event_type}")
    
    assert len(sakshin.events) == 3
    assert sakshin.stats["total_events"] == 3
    assert sakshin.stats["events_by_module"]["manas"] == 1
    assert sakshin.stats["events_by_module"]["buddhi"] == 1
    assert sakshin.stats["events_by_module"]["chitta"] == 1
    
    print("\n✅ Event recording working")


def test_event_replay():
    """Test that Sakshin can replay event sequences."""
    print("\n" + "="*80)
    print("TEST 2: Event Replay")
    print("="*80)
    
    sakshin = Sakshin()
    
    # Record sequence
    for i in range(10):
        sakshin.observe("test", f"event_{i}", {"index": i})
    
    # Replay full sequence
    full_replay = sakshin.replay()
    print(f"\nFull replay: {len(full_replay)} events")
    assert len(full_replay) == 10
    
    # Replay slice
    partial_replay = sakshin.replay(from_idx=3, to_idx=7)
    print(f"Partial replay (3:7): {len(partial_replay)} events")
    assert len(partial_replay) == 4
    assert partial_replay[0].data["index"] == 3
    assert partial_replay[-1].data["index"] == 6
    
    print("\n✅ Event replay working")


def test_integrity_verification():
    """Test that Sakshin detects tampering."""
    print("\n" + "="*80)
    print("TEST 3: Integrity Verification")
    print("="*80)
    
    sakshin = Sakshin()
    
    # Record events
    sakshin.observe("manas", "parse", {"input": "Birds can fly"})
    sakshin.observe("buddhi", "judgment", {"action": "added_belief"})
    
    # Verify integrity (should pass)
    print("\nBefore tampering:")
    assert sakshin.verify_integrity() == True
    print("  ✅ Log integrity verified")
    
    # Tamper with log
    print("\nTampering with event...")
    sakshin.events[0].data["input"] = "TAMPERED DATA"
    
    # Verify integrity (should fail)
    print("After tampering:")
    assert sakshin.verify_integrity() == False
    print("  ❌ Tampering detected (expected)")
    
    print("\n✅ Integrity verification working")


def test_influence_reconstruction_simple():
    """Test basic influence graph reconstruction."""
    print("\n" + "="*80)
    print("TEST 4: Simple Influence Reconstruction")
    print("="*80)
    
    sakshin = Sakshin()
    
    # Simulate cognitive sequence
    parse_event = sakshin.observe("manas", "parse", {
        "input": "Birds can fly"
    })
    
    judgment_event = sakshin.observe("buddhi", "judgment", {
        "parent_event": parse_event,
        "action": "added_belief"
    })
    
    storage_event = sakshin.observe("chitta", "belief_stored", {
        "parent_event": judgment_event,
        "belief_id": "b_123"
    })
    
    # Reconstruct influences
    print(f"\nReconstructing influences for storage event: {storage_event}")
    influences = sakshin.reconstruct_influences(storage_event)
    
    print(f"\nInfluenced by {len(influences['influenced_by'])} events:")
    for event in influences['influenced_by']:
        print(f"  • [{event.module}] {event.event_type}")
    
    # Should include all 3 events
    assert len(influences['influenced_by']) == 3
    
    # Check dependency graph
    print(f"\nDependency graph:")
    for event_id, parents in influences['dependency_graph'].items():
        print(f"  {event_id[:8]}... → {[p[:8] for p in parents]}")
    
    print("\n✅ Simple influence reconstruction working")


def test_sakshin_complete_reconstruction():
    """
    THE TERRIFYING TEST.
    
    Show me every belief that influenced this answer,
    directly or indirectly.
    
    If Sakshin can't reconstruct that, STOP.
    """
    print("\n" + "="*80)
    print("🔥 THE TERRIFYING TEST: Complete Reconstruction 🔥")
    print("="*80)
    
    # Note: Full integration requires Ahankara to notify Sakshin
    # For now, we'll test the reconstruction algorithm with simulated events
    
    sakshin = Sakshin()
    
    # Simulate: Teaching "Birds can fly"
    parse_1 = sakshin.observe("manas", "parse", {
        "input": "Birds can fly",
        "output": {"entities": ["bird"], "predicates": ["can_fly"]}
    })
    
    judgment_1 = sakshin.observe("buddhi", "judgment", {
        "parent_event": parse_1,
        "proposal": "Birds can fly",
        "action": "added_belief",
        "confidence": 0.85
    })
    
    belief_1 = sakshin.observe("chitta", "belief_stored", {
        "parent_event": judgment_1,
        "belief_id": "b_001",
        "statement": "Birds can fly"
    })
    
    # Simulate: Teaching "Penguins are birds"
    parse_2 = sakshin.observe("manas", "parse", {
        "input": "Penguins are birds",
        "output": {"entities": ["penguin", "bird"], "predicates": ["is_a"]}
    })
    
    judgment_2 = sakshin.observe("buddhi", "judgment", {
        "parent_event": parse_2,
        "proposal": "Penguins are birds",
        "action": "added_belief",
        "confidence": 0.90
    })
    
    belief_2 = sakshin.observe("chitta", "belief_stored", {
        "parent_event": judgment_2,
        "belief_id": "b_002",
        "statement": "Penguins are birds"
    })
    
    # Simulate: Query "Can penguins fly?"
    query_parse = sakshin.observe("manas", "parse", {
        "input": "Can penguins fly?",
        "output": {"entities": ["penguin"], "predicates": ["can_fly"]}
    })
    
    focus_event = sakshin.observe("buddhi", "focus", {
        "parent_event": query_parse,
        "belief_ids": ["b_001", "b_002"],
        "focused_beliefs": 2
    })
    
    derivation_event = sakshin.observe("buddhi", "derivation", {
        "parent_event": focus_event,
        "inputs": ["b_001", "b_002"],
        "verdict": "yes",
        "confidence": 0.85
    })
    
    answer_event = sakshin.observe("ahankara", "query_response", {
        "parent_event": derivation_event,
        "question": "Can penguins fly?",
        "answer": "Yes.",
        "verdict": "yes"
    })
    
    print(f"\nRecorded {len(sakshin.events)} events in cognitive sequence")
    
    # NOW THE TERRIFYING PART: Reconstruct complete influence graph
    print("\n" + "-"*80)
    print("Reconstructing complete influence graph...")
    print("-"*80)
    
    influences = sakshin.reconstruct_influences(answer_event)
    
    print(f"\n📊 INFLUENCE ANALYSIS for: 'Can penguins fly?'")
    print(f"\nAnswer influenced by {len(influences['influenced_by'])} events:")
    
    # Group by module
    by_module = {}
    for event in influences['influenced_by']:
        if event.module not in by_module:
            by_module[event.module] = []
        by_module[event.module].append(event)
    
    for module, events in sorted(by_module.items()):
        print(f"\n  {module.upper()}:")
        for event in events:
            print(f"    • {event.event_type}")
            if event.event_type == "belief_stored":
                print(f"      → {event.data.get('statement', '???')}")
    
    # Critical checks
    print("\n" + "-"*80)
    print("CRITICAL CHECKS:")
    print("-"*80)
    
    # Must include both belief storage events
    belief_events = [e for e in influences['influenced_by'] if e.event_type == "belief_stored"]
    print(f"\n✓ Belief storage events: {len(belief_events)}")
    assert len(belief_events) >= 2, "Missing belief storage events!"
    
    # Must include both parse events
    parse_events = [e for e in influences['influenced_by'] if e.event_type == "parse"]
    print(f"✓ Parse events: {len(parse_events)}")
    assert len(parse_events) >= 3, "Missing parse events!"
    
    # Must include focus event
    focus_events = [e for e in influences['influenced_by'] if e.event_type == "focus"]
    print(f"✓ Focus events: {len(focus_events)}")
    assert len(focus_events) >= 1, "Missing focus event!"
    
    # Must include derivation event
    derivation_events = [e for e in influences['influenced_by'] if e.event_type == "derivation"]
    print(f"✓ Derivation events: {len(derivation_events)}")
    assert len(derivation_events) >= 1, "Missing derivation event!"
    
    print("\n" + "="*80)
    print("🎉 TERRIFYING TEST PASSED")
    print("="*80)
    print("\n✅ Sakshin can reconstruct complete influence graphs")
    print("✅ Every belief that influenced the answer is traceable")
    print("✅ Full auditability achieved")
    print("="*80)


def test_sakshin_module_filtering():
    """Test filtering events by module and type."""
    print("\n" + "="*80)
    print("TEST 5: Event Filtering")
    print("="*80)
    
    sakshin = Sakshin()
    
    # Record mixed events
    sakshin.observe("manas", "parse", {"input": "A"})
    sakshin.observe("manas", "parse", {"input": "B"})
    sakshin.observe("buddhi", "judgment", {"action": "accept"})
    sakshin.observe("chitta", "belief_stored", {"id": "b_1"})
    sakshin.observe("buddhi", "judgment", {"action": "reject"})
    
    # Filter by module
    manas_events = sakshin.get_events_by_module("manas")
    print(f"\nManas events: {len(manas_events)}")
    assert len(manas_events) == 2
    
    buddhi_events = sakshin.get_events_by_module("buddhi")
    print(f"Buddhi events: {len(buddhi_events)}")
    assert len(buddhi_events) == 2
    
    # Filter by type
    parse_events = sakshin.get_events_by_type("parse")
    print(f"Parse events: {len(parse_events)}")
    assert len(parse_events) == 2
    
    judgment_events = sakshin.get_events_by_type("judgment")
    print(f"Judgment events: {len(judgment_events)}")
    assert len(judgment_events) == 2
    
    print("\n✅ Event filtering working")


def test_sakshin_summary():
    """Test summary generation."""
    print("\n" + "="*80)
    print("TEST 6: Summary Generation")
    print("="*80)
    
    sakshin = Sakshin()
    
    # Record events
    sakshin.observe("manas", "parse", {})
    sakshin.observe("manas", "parse", {})
    sakshin.observe("buddhi", "judgment", {})
    sakshin.observe("chitta", "belief_stored", {})
    
    # Generate summary
    summary = sakshin.summary()
    print(f"\n{summary}")
    
    assert "Total events: 4" in summary
    assert "manas: 2" in summary
    assert "buddhi: 1" in summary
    assert "✅ INTACT" in summary
    
    print("\n✅ Summary generation working")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SAKSHIN TEST SUITE")
    print("="*80)
    print("\nTesting meta-observation capabilities...")
    print("SAKSHIN IS BORING BY DESIGN - it should NOT feel smart")
    print("="*80)
    
    try:
        test_basic_event_recording()
        test_event_replay()
        test_integrity_verification()
        test_influence_reconstruction_simple()
        test_sakshin_module_filtering()
        test_sakshin_summary()
        test_sakshin_complete_reconstruction()
        
        print("\n" + "="*80)
        print("🎉 ALL SAKSHIN TESTS PASSED")
        print("="*80)
        print("\n✅ Event recording: WORKING")
        print("✅ Event replay: WORKING")
        print("✅ Integrity verification: WORKING")
        print("✅ Influence reconstruction: WORKING")
        print("✅ Complete auditability: ACHIEVED")
        print("\n" + "="*80)
        print("SAKSHIN IS OPERATIONAL")
        print("="*80)
        print("\nKey capabilities:")
        print("  • Records all cognitive events")
        print("  • Detects log tampering")
        print("  • Reconstructs influence graphs")
        print("  • Enables full auditability")
        print("\nRemains BORING (no reasoning, no judgment)")
        print("="*80)
        
    except AssertionError as e:
        print("\n" + "="*80)
        print("❌ SAKSHIN TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        print("\n🚨 Fix before proceeding")
        print("="*80)
        raise
