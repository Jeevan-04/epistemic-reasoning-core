"""
Interactive MARC Test - User Input from Terminal

This test allows you to interact with MARC directly:
- Type assertions to teach MARC
- Ask questions to query MARC's knowledge
- Type 'quit' or 'exit' to stop
- Type 'beliefs' to see all beliefs
- Type 'stats' to see system statistics

NO hardcoded knowledge - everything learned from YOUR input!
"""

import sys
from pathlib import Path

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent / "5. ahankara (Self Model)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "3. chitta (Belief Memory)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "2. buddhi (Reasoning)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "1. manas (Input Layer)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "7. perceptual_priors"))
sys.path.insert(0, str(Path(__file__).parent.parent / "8. geographic_memory"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ahankara import Ahankara


def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print("MARC INTERACTIVE TEST - Real Cognition, NO Hardcoding")
    print("="*70)
    print("\nCommands:")
    print("  - Type any statement to teach MARC")
    print("  - Ask questions to query MARC's knowledge")
    print("  - 'beliefs' - show all beliefs")
    print("  - 'stats' - show system statistics")
    print("  - 'quit' or 'exit' - stop")
    print("\nMARC starts with ZERO knowledge. Everything is learned from YOU!")
    print("="*70 + "\n")


def show_beliefs(marc: Ahankara):
    """Display all active beliefs"""
    print("\n" + "="*70)
    print("BELIEF GRAPH")
    print("="*70)
    
    active_beliefs = [b for b in marc.chitta.beliefs.values() if b.active]
    
    if not active_beliefs:
        print("No beliefs yet. MARC has learned nothing.")
    else:
        print(f"Total active beliefs: {len(active_beliefs)}\n")
        
        for i, belief in enumerate(active_beliefs, 1):
            print(f"{i}. {belief.statement_text}")
            print(f"   Confidence: {belief.confidence:.3f}")
            print(f"   State: {belief.epistemic_state}")
            print(f"   Template: {belief.template}")
            
            # Show edges
            if belief.edges_out:
                print(f"   Edges:")
                for rel_type, targets in belief.edges_out.items():
                    print(f"     {rel_type}: {len(targets)} connections")
            print()
    
    print("="*70 + "\n")


def show_stats(marc: Ahankara):
    """Display system statistics"""
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)
    
    print("\nAhankara (Orchestrator):")
    for key, value in marc.stats.items():
        print(f"  {key}: {value}")
    
    print("\nBuddhi (Reasoning):")
    for key, value in marc.buddhi.stats.items():
        print(f"  {key}: {value}")
    
    print("\nManas (Understanding):")
    print(f"  parses: {marc.manas.stats['parses']}")
    if 'success_rate' in marc.manas.stats:
        print(f"  success_rate: {marc.manas.stats['success_rate']*100:.1f}%")
    
    print("\nChitta (Memory):")
    print(f"  total_beliefs: {len(marc.chitta.beliefs)}")
    active = len([b for b in marc.chitta.beliefs.values() if b.active])
    print(f"  active_beliefs: {active}")
    print(f"  total_edges: {sum(len(v) for edges in marc.chitta.edges.values() for v in edges.values())}")
    
    print("\n" + "="*70 + "\n")


def interactive_loop():
    """Main interactive loop"""
    marc = Ahankara()
    
    print_header()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! MARC shutting down...")
                break
            
            elif user_input.lower() == 'beliefs':
                show_beliefs(marc)
                continue
            
            elif user_input.lower() == 'stats':
                show_stats(marc)
                continue
            
            # Check if it's a question
            is_question = '?' in user_input or any(
                user_input.lower().startswith(q) 
                for q in ['what', 'who', 'where', 'when', 'why', 'how', 'can', 'do', 'is', 'are']
            )
            
            if is_question:
                # Query mode
                answer = marc.ask(user_input)
                print(f"MARC: {answer}\n")
            else:
                # Assertion mode
                result = marc.process(user_input)
                action = result.get('action', 'unknown')
                
                if action == 'added_belief':
                    print(f"MARC: ✓ Understood and stored.\n")
                elif action == 'existing_belief':
                    print(f"MARC: ✓ I already know that.\n")
                elif action == 'hypothetical':
                    print(f"MARC: ⚠ Uncertain - stored as hypothesis.\n")
                elif action == 'unknown':
                    print(f"MARC: ? I don't understand that.\n")
                else:
                    print(f"MARC: Processed ({action})\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. MARC shutting down...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    interactive_loop()
