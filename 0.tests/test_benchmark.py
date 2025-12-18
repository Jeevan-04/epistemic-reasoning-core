"""
MARC BRUTAL BENCHMARK TEST

Comprehensive epistemic discipline benchmark covering:
- Grounding discipline (refuses ungrounded inferences)
- Negation dominance (inherited negations block positives)
- Relation frames (TAXONOMIC vs SPATIAL vs FUNCTIONAL vs STATE)
- Paraconsistency (tolerates contradictions without explosion)
- Perceptual priors (observational knowledge)
- Geographic memory (external memory, retrieval-only)
- Inheritance correctness (taxonomic property propagation)
- Restraint (says "I do not know" when uncertain)

This is NOT a "pass 95%" test. This is an EPISTEMIC INTEGRITY test.
The goal is to demonstrate principled behavior, not maximize accuracy.

Philosophy:
"This system models epistemic discipline: the ability to reason 
without overclaiming knowledge, even under contradiction."
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent / "5. ahankara (Self Model)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "3. chitta (Belief Memory)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "2. buddhi (Reasoning)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "1. manas (Input Layer)"))
sys.path.insert(0, str(Path(__file__).parent.parent / "7. perceptual_priors"))
sys.path.insert(0, str(Path(__file__).parent.parent / "8. geographic_memory"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ahankara import Ahankara


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with expected behavior"""
    question: str
    expected: str  # "yes", "no", "unknown"
    category: str
    rationale: str  # Why this answer is expected
    
    
@dataclass
class BenchmarkResult:
    """Result of benchmark run"""
    total: int
    correct: int
    precision: float
    restraint: float  # % of unknowns correctly identified
    false_positives: int
    false_negatives: int
    by_category: Dict[str, dict]


class BrutalBenchmark:
    """
    Brutal epistemic discipline benchmark.
    
    Tests principled reasoning, not trivia recall.
    """
    
    def __init__(self):
        self.marc = None
        self.questions: List[BenchmarkQuestion] = []
        self._setup_questions()
    
    def _print_belief_trace(self, fact: str, result: dict):
        """Print detailed trace of how MARC processed a belief through each module."""
        print(f"\n  ┌─ COGNITIVE TRACE: '{fact}'")
        
        # PHASE 1: Manas (Perception)
        if 'proposal' in result:
            proposal = result['proposal']
            print(f"  │")
            print(f"  ├─ [MANAS] Perception & Parsing:")
            print(f"  │  ├─ Template: {proposal.get('template', 'N/A')}")
            print(f"  │  ├─ Entities: {proposal.get('entities', [])}")
            print(f"  │  ├─ Predicates: {proposal.get('predicates', [])}")
            print(f"  │  ├─ Polarity: {'+' if proposal.get('polarity', 1) > 0 else '-'}")
            print(f"  │  ├─ Parser Confidence: {proposal.get('parser_confidence', 0):.3f}")
            print(f"  │  ├─ Intent: {proposal.get('intent', 'assertion')}")
            print(f"  │  └─ Modality: {proposal.get('modality', 'default')}")
        
        # PHASE 2: Buddhi (Reasoning/Judgment)
        print(f"  │")
        print(f"  ├─ [BUDDHI] Reasoning & Judgment:")
        if 'decision' in result:
            decision = result['decision']
            print(f"  │  ├─ Decision: {decision.get('outcome', 'N/A')}")
            print(f"  │  └─ Action: {result.get('action', 'N/A')}")
        
        # PHASE 3: Chitta (Memory Storage)
        print(f"  │")
        print(f"  └─ [CHITTA] Belief Storage:")
        if 'belief_id' in result:
            print(f"     ├─ Stored as: {result['belief_id'][:16]}...")
            
            # Try to get the actual belief from Chitta
            if self.marc and hasattr(self.marc, 'chitta'):
                belief = self.marc.chitta.beliefs.get(result['belief_id'])
                if belief:
                    print(f"     ├─ Final Confidence: {belief.confidence:.3f}")
                    print(f"     ├─ Epistemic State: {belief.epistemic_state}")
                    print(f"     └─ Template: {belief.template}")
        else:
            print(f"     └─ (No belief stored - may have been merged or rejected)")
        
        print()
    
    def _setup_questions(self):
        """Setup comprehensive benchmark questions"""
        
        # ═══════════════════════════════════════════════════════════
        # CATEGORY 1: TAXONOMIC INHERITANCE
        # ═══════════════════════════════════════════════════════════
        self.questions.extend([
            BenchmarkQuestion(
                "Are bats mammals?",
                "yes",
                "taxonomic",
                "Direct taxonomic fact (taught)"
            ),
            BenchmarkQuestion(
                "Do bats produce milk?",
                "yes",
                "inheritance_positive",
                "Mammals produce milk → bats inherit (FUNCTIONAL inherits=True)"
            ),
            BenchmarkQuestion(
                "Do bats have backbones?",
                "yes",
                "inheritance_positive",
                "Mammals have backbones → bats inherit (FUNCTIONAL inherits=True)"
            ),
            BenchmarkQuestion(
                "Do eagles have beaks?",
                "yes",
                "inheritance_positive",
                "Birds have beaks → eagles inherit (FUNCTIONAL inherits=True)"
            ),
            BenchmarkQuestion(
                "Do dolphins breathe air?",
                "yes",
                "inheritance_positive",
                "Mammals breathe air → dolphins inherit (FUNCTIONAL inherits=True)"
            ),
        ])
        
        # ═══════════════════════════════════════════════════════════
        # CATEGORY 2: NEGATION DOMINANCE
        # ═══════════════════════════════════════════════════════════
        self.questions.extend([
            BenchmarkQuestion(
                "Do bats have gills?",
                "no",
                "negation_dominance",
                "Mammals do NOT have gills → negation blocks (frame.negation_blocks=True)"
            ),
            BenchmarkQuestion(
                "Do whales have gills?",
                "no",
                "negation_dominance",
                "Mammals do NOT have gills → negation blocks whale (FUNCTIONAL negation_blocks=True)"
            ),
            BenchmarkQuestion(
                "Do penguins have gills?",
                "no",
                "negation_dominance",
                "Birds do NOT have gills → negation blocks (if taught)"
            ),
        ])
        
        # ═══════════════════════════════════════════════════════════
        # CATEGORY 3: GROUNDING DISCIPLINE (CORRECT REFUSALS)
        # ═══════════════════════════════════════════════════════════
        self.questions.extend([
            BenchmarkQuestion(
                "Do copper objects conduct electricity?",
                "unknown",
                "grounding_refusal",
                "Ungrounded composition - 'copper objects' not taught, refuses correctly"
            ),
            BenchmarkQuestion(
                "Are platypuses mammals?",
                "unknown",
                "grounding_refusal",
                "Platypus not in entity index - refuses correctly (entity not taught)"
            ),
            BenchmarkQuestion(
                "Do unicorns have horns?",
                "unknown",
                "grounding_refusal",
                "Unicorn not taught - refuses correctly (fictional entity)"
            ),
        ])
        
        # ═══════════════════════════════════════════════════════════
        # CATEGORY 4: RELATION FRAMES (NON-INHERITABLE RELATIONS)
        # ═══════════════════════════════════════════════════════════
        self.questions.extend([
            BenchmarkQuestion(
                "Is London in Europe?",
                "yes",
                "geographic_memory",
                "Geographic memory: London → UK → Europe (external memory, not inference)"
            ),
            BenchmarkQuestion(
                "Is Tokyo in Europe?",
                "no",
                "geographic_memory",
                "Geographic memory: Tokyo → Japan → Asia (not Europe)"
            ),
            BenchmarkQuestion(
                "Is Paris in France?",
                "yes",
                "geographic_memory",
                "Geographic memory: Paris → France (direct containment)"
            ),
        ])
        
        # ═══════════════════════════════════════════════════════════
        # CATEGORY 5: PERCEPTUAL PRIORS
        # ═══════════════════════════════════════════════════════════
        self.questions.extend([
            BenchmarkQuestion(
                "Is gold shiny?",
                "yes",
                "perceptual_prior",
                "Perceptual prior: gold appears shiny (observational, confidence < 1.0)"
            ),
            BenchmarkQuestion(
                "Is water liquid?",
                "yes",
                "perceptual_prior",
                "Perceptual prior: water appears liquid at room temp (observational)"
            ),
            BenchmarkQuestion(
                "Is copper conductive?",
                "yes",
                "perceptual_prior",
                "Perceptual prior: copper appears conductive (empirical property)"
            ),
        ])
        
        # ═══════════════════════════════════════════════════════════
        # CATEGORY 6: PARACONSISTENCY (CONTRADICTION TOLERANCE)
        # ═══════════════════════════════════════════════════════════
        self.questions.extend([
            BenchmarkQuestion(
                "Is Tweety a bird?",
                "yes",
                "paraconsistency",
                "Direct belief wins over exception (if Tweety taught as bird)"
            ),
            # Note: Add more paraconsistency tests if contradictions taught
        ])
        
        # ═══════════════════════════════════════════════════════════
        # CATEGORY 7: STATE PREDICATES (NON-INHERITABLE)
        # ═══════════════════════════════════════════════════════════
        self.questions.extend([
            BenchmarkQuestion(
                "Is ice solid?",
                "yes",
                "perceptual_prior",
                "Perceptual prior: ice appears solid (STATE, context-dependent)"
            ),
        ])
    
    def _print_belief_trace(self, fact: str, result: dict):
        """Print detailed trace of how MARC processed a belief through each module."""
        print(f"\n  ┌─ COGNITIVE TRACE: '{fact}'")
        
        # PHASE 1: Manas (Perception)
        if 'proposal' in result:
            proposal = result['proposal']
            print(f"  │")
            print(f"  ├─ [MANAS] Perception & Parsing:")
            print(f"  │  ├─ Template: {proposal.get('template', 'N/A')}")
            print(f"  │  ├─ Entities: {proposal.get('entities', [])}")
            print(f"  │  ├─ Predicates: {proposal.get('predicates', [])}")
            print(f"  │  ├─ Polarity: {'+' if proposal.get('polarity', 1) > 0 else '-'}")
            print(f"  │  ├─ Parser Confidence: {proposal.get('parser_confidence', 0):.3f}")
            print(f"  │  ├─ Intent: {proposal.get('intent', 'assertion')}")
            print(f"  │  ├─ Modality: {proposal.get('modality', 'default')}")
            
            if 'canonical' in proposal:
                canonical = proposal['canonical']
                print(f"  │  └─ Canonical:")
                for key, val in canonical.items():
                    if val and key not in ['entities']:  # entities already shown
                        print(f"  │     └─ {key}: {val}")
        
        # PHASE 2: Buddhi (Reasoning/Judgment)
        print(f"  │")
        print(f"  ├─ [BUDDHI] Reasoning & Judgment:")
        if 'decision' in result:
            decision = result['decision']
            print(f"  │  ├─ Decision: {decision.get('outcome', 'N/A')}")
            print(f"  │  ├─ Action: {result.get('action', 'N/A')}")
            
            if 'winner' in decision:
                winner = decision['winner']
                print(f"  │  ├─ Winner Belief:")
                print(f"  │  │  ├─ ID: {winner.get('id', 'N/A')[:12]}...")
                print(f"  │  │  ├─ Confidence: {winner.get('confidence', 0):.3f}")
                print(f"  │  │  └─ Epistemic State: {winner.get('epistemic_state', 'N/A')}")
            
            if 'scores' in decision:
                print(f"  │  └─ Judgment Scores:")
                for candidate, score in list(decision['scores'].items())[:3]:  # top 3
                    print(f"  │     └─ {candidate[:20]}...: {score:.3f}")
        
        # PHASE 3: Chitta (Memory Storage)
        print(f"  │")
        print(f"  └─ [CHITTA] Belief Storage:")
        if 'belief_id' in result:
            print(f"     ├─ Stored as: {result['belief_id'][:16]}...")
            
            # Try to get the actual belief from Chitta
            if self.marc and hasattr(self.marc, 'chitta'):
                belief = self.marc.chitta.beliefs.get(result['belief_id'])
                if belief:
                    print(f"     ├─ Final Confidence: {belief.confidence:.3f}")
                    print(f"     ├─ Epistemic State: {belief.epistemic_state}")
                    print(f"     ├─ Template: {belief.template}")
                    
                    if hasattr(belief, 'provenance') and belief.provenance:
                        print(f"     └─ Provenance:")
                        for i, prov in enumerate(belief.provenance[:2]):  # first 2
                            print(f"        └─ {prov.op} (score: {prov.score:.3f})")
        else:
            print(f"     └─ (No belief stored - may have been merged or rejected)")
        
        print()
    
    def teach_knowledge_base(self):
        """Teach MARC the base knowledge required for benchmark"""
        print("\n" + "="*70)
        print("TEACHING KNOWLEDGE BASE")
        print("="*70 + "\n")
        
        # Initialize MARC
        self.marc = Ahankara()
        self.marc.set_learning_mode()  # NO decay, NO demotion during teaching
        
        # Taxonomic hierarchy
        facts = [
            # Mammal taxonomy
            "Bats are mammals.",
            "Whales are mammals.",
            "Dolphins are mammals.",
            "Cats are mammals.",
            
            # Bird taxonomy
            "Eagles are birds.",
            "Penguins are birds.",
            "Sparrows are birds.",
            
            # Fish taxonomy
            "Sharks are fish.",
            "Salmon are fish.",
            
            # Mammal properties
            "Mammals produce milk.",
            "Mammals have backbones.",
            "Mammals breathe air.",
            "Mammals do not have gills.",
            
            # Bird properties
            "Birds have beaks.",
            "Birds have feathers.",
            "Birds lay eggs.",
            
            # Fish properties
            "Fish have gills.",
            "Fish live in water.",
        ]
        
        for fact in facts:
            result = self.marc.process(fact)
            status = "✓" if result.get('action') == 'added_belief' else "✗"
            print(f"{status} {fact}")
            
            # Print detailed cognitive trace for each fact
            self._print_belief_trace(fact, result)
        
        print(f"\n✓ Taught {len(facts)} facts")
        print("="*70 + "\n")
        
        # Switch to reasoning mode for queries
        self.marc.set_reasoning_mode()
    
    def _print_answer_trace(self, question: str, answer: str, proof_data: dict = None):
        """Print detailed trace of how MARC derived an answer."""
        print(f"\n  ┌─ ANSWER DERIVATION: '{question}'")
        print(f"  │")
        
        # Get the full answer result if available
        if self.marc and hasattr(self.marc, 'last_answer_result'):
            result = self.marc.last_answer_result
            
            # PHASE 1: Query Parsing (Manas)
            if 'proposal' in result:
                proposal = result['proposal']
                print(f"  ├─ [MANAS] Query Understanding:")
                print(f"  │  ├─ Parsed Entities: {proposal.get('entities', [])}")
                print(f"  │  ├─ Parsed Predicates: {proposal.get('predicates', [])}")
                print(f"  │  ├─ Query Intent: {proposal.get('intent', 'query')}")
                print(f"  │  └─ Template: {proposal.get('template', 'N/A')}")
            
            # PHASE 2: Reasoning Trace (Buddhi)
            print(f"  │")
            print(f"  ├─ [BUDDHI] Reasoning Process:")
            
            # Check for proof/derivation steps
            if 'proof' in result:
                proof = result['proof']
                
                # Show derivation steps
                if hasattr(proof, 'steps') and proof.steps:
                    print(f"  │  ├─ Derivation Steps ({len(proof.steps)} total):")
                    for i, step in enumerate(proof.steps[:5], 1):  # show first 5
                        print(f"  │  │  └─ Step {i}: {step.rule}")
                        print(f"  │  │     ├─ Output: {step.output[:60]}..." if len(step.output) > 60 else f"  │  │     ├─ Output: {step.output}")
                        if step.confidence:
                            print(f"  │  │     └─ Confidence: {step.confidence:.3f}")
                
                # Show conflicts if any
                if hasattr(proof, 'conflicts') and proof.conflicts:
                    print(f"  │  ├─ Conflicts Detected ({len(proof.conflicts)} total):")
                    for conflict in proof.conflicts[:2]:  # show first 2
                        print(f"  │  │  └─ Predicate: {conflict.predicate}")
                        print(f"  │  │     ├─ Positive belief: {conflict.positive[:12]}...")
                        print(f"  │  │     ├─ Negative belief: {conflict.negative[:12]}...")
                        print(f"  │  │     └─ Resolution: {conflict.resolution}")
                
                # Show final verdict
                if hasattr(proof, 'verdict'):
                    print(f"  │  └─ Verdict: {proof.verdict.upper()}")
            
            # PHASE 3: Memory Access (Chitta)
            print(f"  │")
            print(f"  └─ [CHITTA] Beliefs Accessed:")
            
            if 'beliefs_consulted' in result:
                beliefs = result['beliefs_consulted']
                if beliefs:
                    print(f"     ├─ Total beliefs examined: {len(beliefs)}")
                    for i, belief_id in enumerate(beliefs[:3], 1):  # show first 3
                        belief = self.marc.chitta.beliefs.get(belief_id)
                        if belief:
                            print(f"     ├─ Belief {i}:")
                            print(f"     │  ├─ Text: {belief.statement_text[:50]}..." if len(belief.statement_text) > 50 else f"     │  ├─ Text: {belief.statement_text}")
                            print(f"     │  ├─ Confidence: {belief.confidence:.3f}")
                            print(f"     │  └─ Entities: {belief.entities}")
                    if len(beliefs) > 3:
                        print(f"     └─ ... and {len(beliefs) - 3} more beliefs")
                else:
                    print(f"     └─ No existing beliefs matched the query")
            else:
                print(f"     └─ (Belief access info not available)")
        
        # Final answer
        print(f"\n  ➜ FINAL ANSWER: {answer}")
        print()
    
    def run(self) -> BenchmarkResult:
        """Run full benchmark and return results"""
        print("\n" + "="*70)
        print("MARC BRUTAL BENCHMARK")
        print("Epistemic Discipline Test")
        print("="*70 + "\n")
        
        # Teach knowledge base
        self.teach_knowledge_base()
        
        # Run queries
        print("\n" + "="*70)
        print("RUNNING BENCHMARK QUERIES")
        print("="*70 + "\n")
        
        results = []
        by_category: Dict[str, dict] = {}
        
        for i, q in enumerate(self.questions, 1):
            print(f"\n{'='*70}")
            print(f"QUERY {i}/{len(self.questions)}: {q.question}")
            print(f"{'='*70}")
            
            # Store last answer result for tracing
            if hasattr(self.marc, '_last_result'):
                delattr(self.marc, '_last_result')
            
            answer = self.marc.ask(q.question)
            
            # Print detailed answer derivation trace
            self._print_answer_trace(q.question, answer)
            
            # Parse answer
            actual = self._parse_answer(answer)
            correct = (actual == q.expected)
            
            # Record result
            results.append({
                'question': q.question,
                'expected': q.expected,
                'actual': actual,
                'correct': correct,
                'category': q.category,
                'rationale': q.rationale,
                'raw_answer': answer
            })
            
            # Update category stats
            if q.category not in by_category:
                by_category[q.category] = {
                    'total': 0,
                    'correct': 0,
                    'questions': []
                }
            
            by_category[q.category]['total'] += 1
            if correct:
                by_category[q.category]['correct'] += 1
            by_category[q.category]['questions'].append(results[-1])
            
            # Print verdict summary
            status = "✓" if correct else "✗"
            print(f"\n{'─'*70}")
            print(f"{status} VERDICT: Expected [{q.expected}] | Got [{actual}] | {'CORRECT' if correct else 'INCORRECT'}")
            print(f"  Category: {q.category}")
            print(f"  Rationale: {q.rationale}")
            if not correct:
                print(f"  ⚠️  Mismatch - reviewing answer logic needed")
            print(f"{'─'*70}\n")
            
            # Print verdict summary
            status = "✓" if correct else "✗"
            print(f"\n{'─'*70}")
            print(f"{status} VERDICT: Expected [{q.expected}] | Got [{actual}] | {'CORRECT' if correct else 'INCORRECT'}")
            print(f"  Category: {q.category}")
            print(f"  Rationale: {q.rationale}")
            if not correct:
                print(f"  ⚠️  Mismatch Details:")
                print(f"     Expected: {q.expected}")
                print(f"     Actual: {actual}")
                print(f"     Raw Answer: {answer[:100]}..." if len(answer) > 100 else f"     Raw Answer: {answer}")
            print(f"{'─'*70}")
            print()
        
        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        precision = correct / total if total > 0 else 0.0
        
        # Calculate restraint (% of "unknown" expected that were answered "unknown")
        unknown_expected = [r for r in results if r['expected'] == 'unknown']
        restraint = 0.0
        if unknown_expected:
            unknown_correct = sum(1 for r in unknown_expected if r['actual'] == 'unknown')
            restraint = unknown_correct / len(unknown_expected)
        
        # Calculate false positives/negatives
        false_positives = sum(1 for r in results 
                            if r['expected'] == 'unknown' and r['actual'] != 'unknown')
        false_negatives = sum(1 for r in results 
                            if r['expected'] != 'unknown' and r['actual'] == 'unknown')
        
        # Calculate category precision
        for cat, data in by_category.items():
            data['precision'] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
        
        result = BenchmarkResult(
            total=total,
            correct=correct,
            precision=precision,
            restraint=restraint,
            false_positives=false_positives,
            false_negatives=false_negatives,
            by_category=by_category
        )
        
        # Print summary
        self._print_summary(result)
        
        # Save detailed results
        self._save_results(results, result)
        
        return result
    
    def _parse_answer(self, answer: str) -> str:
        """Parse MARC answer to yes/no/unknown"""
        answer_lower = answer.lower()
        
        # Check for unknown/uncertain (BEFORE checking yes/no to avoid substring matches)
        unknown_phrases = [
            'i do not know',
            'i don\'t know',
            'not certain',
            'uncertain',
            'unknown',
            'insufficient',
            'need more'
        ]
        
        for phrase in unknown_phrases:
            if phrase in answer_lower:
                return 'unknown'
        
        # Check for yes
        if 'yes' in answer_lower or answer_lower.startswith('✓'):
            return 'yes'
        
        # Check for no
        if 'no' in answer_lower or answer_lower.startswith('✗'):
            return 'no'
        
        # Default: unknown
        return 'unknown'
    
    def _print_summary(self, result: BenchmarkResult):
        """Print benchmark summary"""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70 + "\n")
        
        print(f"Overall Precision: {result.precision*100:.1f}% ({result.correct}/{result.total})")
        print(f"Restraint (unknown correctly identified): {result.restraint*100:.1f}%")
        print(f"False Positives (hallucinations): {result.false_positives}")
        print(f"False Negatives (excessive restraint): {result.false_negatives}")
        
        print("\n" + "-"*70)
        print("BY CATEGORY")
        print("-"*70 + "\n")
        
        for cat, data in sorted(result.by_category.items(), key=lambda x: -x[1]['precision']):
            precision = data['precision'] * 100
            print(f"{cat:30s} {precision:5.1f}% ({data['correct']}/{data['total']})")
        
        print("\n" + "="*70)
        
        # Epistemic philosophy statement
        print("\nEPISTEMIC PHILOSOPHY:")
        print("This system models epistemic discipline:")
        print("- Grounding: refuses ungrounded inferences")
        print("- Restraint: says 'I do not know' when uncertain")
        print("- Negation tolerance: blocks conclusions under contradiction")
        print("- Frame discipline: distinguishes inference from memory")
        print("\nGoal: principled reasoning, not benchmark chasing.")
        print("="*70 + "\n")
    
    def _save_results(self, results: List[dict], summary: BenchmarkResult):
        """Save detailed results to JSON"""
        output = {
            'summary': {
                'total': summary.total,
                'correct': summary.correct,
                'precision': summary.precision,
                'restraint': summary.restraint,
                'false_positives': summary.false_positives,
                'false_negatives': summary.false_negatives,
            },
            'by_category': {
                cat: {
                    'total': data['total'],
                    'correct': data['correct'],
                    'precision': data['precision']
                }
                for cat, data in summary.by_category.items()
            },
            'detailed_results': results
        }
        
        output_path = Path(__file__).parent / "benchmark_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Detailed results saved to: {output_path}")


def main():
    """Run brutal benchmark"""
    benchmark = BrutalBenchmark()
    result = benchmark.run()
    
    # Exit code based on epistemic integrity (not precision)
    # We accept ~80-90% precision with high restraint
    # We reject >90% precision with low restraint (likely hallucinating)
    
    if result.restraint < 0.5:
        print("\n❌ EPISTEMIC FAILURE: Low restraint - system is hallucinating!")
        return 1
    
    if result.precision < 0.6:
        print("\n⚠️  Low precision, but maintaining epistemic discipline")
        return 0
    
    print("\n✓ Epistemic discipline maintained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
