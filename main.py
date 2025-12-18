"""
MARC — Memory, Attention, Reasoning, and Cognition

Main entry point for MARC cognitive architecture.

Usage:
  python main.py                 # Start interactive mode
  python main.py --test          # Run test suite
  python main.py --help          # Show help

For testing:
  cd 0.tests
  python test_interactive.py     # Interactive terminal mode
  python run_all_tests.py        # Run all tests
"""

import sys
import subprocess
from pathlib import Path


def show_help():
    """Display usage information and available commands."""
    print(__doc__)
    print("\nAvailable commands:")
    print("  python main.py              # Start interactive question-answering mode")
    print("  python main.py --test       # Run the full test suite")
    print("  python main.py --interactive # Start interactive mode (same as default)")
    print("  python main.py --help       # Show this help message")
    print("\nFor more testing options, see:")
    print("  cd 0.tests && python test_interactive.py")
    print("  cd 0.tests && python run_all_tests.py")


def run_interactive():
    """Launch the interactive question-answering mode."""
    tests_dir = Path(__file__).parent / "0.tests"
    interactive_test = tests_dir / "test_interactive.py"
    
    if not interactive_test.exists():
        print(f"❌ Error: {interactive_test} not found!")
        return 1
    
    # Run the interactive test (which provides a REPL)
    result = subprocess.run(
        [sys.executable, str(interactive_test)],
        cwd=str(tests_dir)
    )
    
    return result.returncode


def run_tests():
    """Run the complete test suite."""
    tests_dir = Path(__file__).parent / "0.tests"
    run_all = tests_dir / "run_all_tests.py"
    
    if not run_all.exists():
        print(f"❌ Error: {run_all} not found!")
        return 1
    
    # Execute all tests
    result = subprocess.run(
        [sys.executable, str(run_all)],
        cwd=str(tests_dir)
    )
    
    return result.returncode


def main():
    """Main entry point - parses arguments and dispatches to appropriate mode."""
    args = sys.argv[1:]
    
    if not args or "--interactive" in args:
        # Default behavior: start interactive mode
        return run_interactive()
    
    elif "--test" in args:
        # Run the test suite
        return run_tests()
    
    elif "--help" in args or "-h" in args:
        # Show help and exit
        show_help()
        return 0
    
    else:
        # Unknown argument
        print(f"❌ Unknown argument: {args[0]}")
        print("Use --help for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
