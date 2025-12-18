#!/usr/bin/env python3
"""
CHITTA QUICK DEMO
===================

Fast demonstration of Chitta capabilities for MARC.
"""

from belief import Belief
from graph import ChittaGraph


def main():
    print("\n" + "=" * 70)
    print("CHITTA — BELIEF MEMORY SYSTEM")
    print("Production-ready hypergraph storage for MARC")
    print("=" * 70 + "\n")
    
    # Create graph
    chitta = ChittaGraph()
    print("✓ ChittaGraph initialized\n")
    
    # Create beliefs
    print("📝 Creating beliefs...\n")
    
    b1 = Belief(
        template="relation",
        canonical={"relation_type": "can_fly", "entities": ["bird"]},
        confidence=0.85,
        statement_text="Birds can fly",
    )
    
    b2 = Belief(
        template="is_a",
        canonical={"relation_type": "is_a", "entities": ["penguin", "bird"]},
        confidence=0.98,
        statement_text="Penguins are birds",
    )
    
    b3 = Belief(
        template="relation",
        canonical={"relation_type": "can_fly", "entities": ["penguin"]},
        confidence=0.95,
        statement_text="Penguins cannot fly",
    )
    
    # Add to graph
    id1 = chitta.add_belief(b1)
    id2 = chitta.add_belief(b2)
    id3 = chitta.add_belief(b3)
    
    print(f"  1. {b1.statement_text} (conf={b1.confidence:.2f}) [{id1}]")
    print(f"  2. {b2.statement_text} (conf={b2.confidence:.2f}) [{id2}]")
    print(f"  3. {b3.statement_text} (conf={b3.confidence:.2f}) [{id3}]")
    
    # Add relations
    print("\n🔗 Adding relations...\n")
    chitta.add_edge(id2, "supports", id1, weight=0.7)
    chitta.add_edge(id3, "contradicts", id1, weight=0.9)
    chitta.add_edge(id3, "refines", id1, weight=0.8)
    
    print("  - 'Penguins are birds' SUPPORTS 'Birds can fly'")
    print("  - 'Penguins cannot fly' CONTRADICTS 'Birds can fly'")
    print("  - 'Penguins cannot fly' REFINES 'Birds can fly' (exception)")
    
    # Query
    print("\n🔍 Querying beliefs...\n")
    
    bird_beliefs = chitta.find_by_entity("bird")
    print(f"  Beliefs about 'bird': {len(bird_beliefs)}")
    for b in bird_beliefs:
        print(f"    - {b.statement_text} (conf={b.confidence:.2f})")
    
    high_conf = chitta.query(min_confidence=0.9)
    print(f"\n  High confidence beliefs (≥0.9): {len(high_conf)}")
    for b in high_conf:
        print(f"    - {b.statement_text} (conf={b.confidence:.2f})")
    
    # Update confidence
    print("\n📊 Updating confidence (log-odds)...\n")
    print(f"  Before: {b1.confidence:.4f}")
    chitta.update_confidence_evidence(id1, evidence_score=1.0)
    print(f"  After +1.0 evidence: {b1.confidence:.4f}")
    
    # Statistics
    print("\n📈 Graph statistics:\n")
    stats = chitta.get_stats()
    print(f"  Total beliefs: {stats['total_beliefs']}")
    print(f"  Active beliefs: {stats['active_beliefs']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Templates: {stats['templates']}")
    print(f"  Entities: {stats['entities']}")
    print(f"  Predicates: {stats['predicates']}")
    
    # Save
    print("\n💾 Saving to JSON...\n")
    chitta.save("demo_chitta.json")
    print("  ✓ Saved to demo_chitta.json")
    
    # Load
    loaded = ChittaGraph.load("demo_chitta.json")
    print(f"  ✓ Loaded: {loaded}")
    
    # Export
    print("\n📤 Exporting to formats...\n")
    from persistence import ChittaExporter
    ChittaExporter.to_dot(chitta, "demo_chitta.dot")
    print("  ✓ Exported to demo_chitta.dot (Graphviz)")
    print("    Run: dot -Tpng demo_chitta.dot -o demo_chitta.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("CHITTA DEMO COMPLETE ✓")
    print("=" * 70)
    print("\nCapabilities demonstrated:")
    print("  ✅ Belief creation with validation")
    print("  ✅ Hypergraph storage")
    print("  ✅ Multi-index queries")
    print("  ✅ Edge relations (supports, contradicts, refines)")
    print("  ✅ Log-odds confidence updates")
    print("  ✅ JSON serialization")
    print("  ✅ Export to Graphviz")
    print("\nChitta is ready for MARC integration!")
    print("Next: Build Buddhi (reasoning engine)\n")
    
    # Cleanup
    import os
    os.remove("demo_chitta.json")
    os.remove("demo_chitta.dot")


if __name__ == "__main__":
    main()
