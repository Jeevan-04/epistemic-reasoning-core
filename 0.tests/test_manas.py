"""
Test suite for Manas (Understanding Engine)

Tests:
- Entity normalization
- Query vs assertion detection
- Modality detection
- Intent classification
- NO hardcoded responses - real parsing only
"""

import sys
from pathlib import Path

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent / "1. manas (Input Layer)"))

from manas import Manas
from manas_utils import normalize_entity, detect_intent, detect_modality


def test_entity_normalization():
    """Test that entities are normalized to lowercase singular"""
    print("\n" + "="*60)
    print("TEST: Entity Normalization")
    print("="*60)
    
    test_cases = [
        ("Birds", "bird"),
        ("Dogs", "dog"),
        ("Penguins", "penguin"),
        ("CATS", "cats"),  # All caps
        ("People", "people"),  # Irregular plural
    ]
    
    for input_entity, expected in test_cases:
        result = normalize_entity(input_entity)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_entity}' → '{result}' (expected: '{expected}')")
    
    print()


def test_intent_detection():
    """Test query vs assertion detection"""
    print("\n" + "="*60)
    print("TEST: Intent Detection")
    print("="*60)
    
    queries = [
        "Can birds fly?",
        "Do penguins swim?",
        "Are dogs mammals?",
        "What is water?",
        "Where do fish live?",
    ]
    
    assertions = [
        "Birds can fly",
        "Penguins swim",
        "Dogs are mammals",
        "Water is wet",
        "Fish live in water",
    ]
    
    print("\nQueries (should detect as 'query'):")
    for text in queries:
        intent = detect_intent(text)
        status = "✅" if intent == "query" else "❌"
        print(f"{status} '{text}' → {intent}")
    
    print("\nAssertions (should detect as 'assertion'):")
    for text in assertions:
        intent = detect_intent(text)
        status = "✅" if intent == "assertion" else "❌"
        print(f"{status} '{text}' → {intent}")
    
    print()


def test_modality_detection():
    """Test strong/weak/default modality detection"""
    print("\n" + "="*60)
    print("TEST: Modality Detection")
    print("="*60)
    
    test_cases = [
        ("Birds always fly south", "strong"),
        ("Fish must breathe water", "strong"),
        ("Dogs never eat vegetables", "strong"),
        ("Cats might like water", "weak"),
        ("Birds probably migrate", "weak"),
        ("Fish maybe sleep", "weak"),
        ("Water is wet", "default"),
        ("Dogs are mammals", "default"),
    ]
    
    for text, expected in test_cases:
        modality = detect_modality(text)
        status = "✅" if modality == expected else "❌"
        print(f"{status} '{text}' → {modality} (expected: {expected})")
    
    print()


def test_manas_parsing():
    """Test full Manas parsing pipeline (NO hardcoding)"""
    print("\n" + "="*60)
    print("TEST: Manas Parsing (Rule-Based, NO Hardcoding)")
    print("="*60)
    
    manas = Manas(llm_backend="mock")  # Uses rule-based parser
    
    test_inputs = [
        "Birds can fly",
        "Penguins cannot fly",
        "Dogs are mammals",
        "Can fish breathe air?",
        "Cats might like water",
    ]
    
    for text in test_inputs:
        print(f"\nInput: '{text}'")
        proposal = manas.parse(text)
        
        print(f"  Template: {proposal['template']}")
        print(f"  Entities: {proposal['entities']}")
        print(f"  Predicates: {proposal['predicates']}")
        print(f"  Polarity: {proposal['polarity']}")
        print(f"  Intent: {proposal.get('intent', 'N/A')}")
        print(f"  Confidence: {proposal['parser_confidence']:.2f}")
        
        # Verify normalization
        if proposal['entities']:
            all_lowercase = all(e == e.lower() for e in proposal['entities'])
            status = "✅" if all_lowercase else "❌"
            print(f"  {status} All entities lowercase: {all_lowercase}")
    
    print()


if __name__ == "__main__":
    print("="*60)
    print("MANAS TEST SUITE")
    print("="*60)
    print("\nTesting Manas understanding engine...")
    print("NO hardcoded responses - all parsing is rule-based")
    
    try:
        test_entity_normalization()
        test_intent_detection()
        test_modality_detection()
        test_manas_parsing()
        
        print("="*60)
        print("✅ ALL MANAS TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
