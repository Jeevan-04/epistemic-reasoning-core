"""
Run All MARC Tests

Runs all essential tests in sequence:
1. Module tests (Manas, Buddhi, Chitta, HRE, Ahankara, Sakshin)
2. Integration test
3. Stress test
4. Benchmark test (epistemic discipline)
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_file: str) -> bool:
    """Run a single test file"""
    print(f"\n{'='*70}")
    print(f"Running {test_file}...")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, test_file],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Run all tests"""
    print("="*70)
    print("MARC TEST SUITE")
    print("="*70)
    print("\nRunning all module tests...")
    print("NO hardcoded data - all tests use real cognition")
    
    tests = [
        "test_manas.py",          # Input layer
        "test_chitta.py",         # Memory
        "test_buddhi.py",         # Reasoning core (FROZEN)
        "test_hre.py",            # Hypothetical reasoning
        "test_ahankara.py",       # Orchestrator
        "test_sakshin.py",        # Meta-observer
        "test_integration.py",    # Full integration
        "test_stress.py",         # Stress testing
        "test_benchmark.py",      # Epistemic discipline benchmark
    ]
    
    results = {}
    
    for test in tests:
        test_path = Path(__file__).parent / test
        if test_path.exists():
            success = run_test(str(test_path))
            results[test] = success
        else:
            print(f"⚠️  {test} not found, skipping...")
            results[test] = None
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test, success in results.items():
        if success is None:
            status = "⚠️  SKIPPED"
        elif success:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        print(f"{status} - {test}")
    
    # Overall result
    passed = sum(1 for s in results.values() if s is True)
    failed = sum(1 for s in results.values() if s is False)
    skipped = sum(1 for s in results.values() if s is None)
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")


if __name__ == "__main__":
    main()
