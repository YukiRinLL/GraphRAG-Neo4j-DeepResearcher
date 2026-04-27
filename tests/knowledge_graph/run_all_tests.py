"""
Knowledge Graph Test Runner

This script runs all knowledge graph tests in sequence.
"""
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

def run_test(test_file: str) -> bool:
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run {test_file}: {e}")
        return False

def main():
    """Run all knowledge graph tests."""
    test_dir = Path(__file__).parent
    
    # Define test files in order of complexity
    test_files = [
        "test_neo4j_connection.py",      # Basic connection test
        "test_neo4j_variants.py",        # Different URI formats
        "test_sync_neo4j.py",            # Synchronous connection
        "test_builder.py",                 # Builder component
        "test_kg_simple.py",              # Simple integration
        "test_kg_final.py",               # Complete integration
    ]
    
    results = {}
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"Warning: {test_file} not found, skipping...")
            continue
        
        success = run_test(str(test_path))
        results[test_file] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_file, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_file:30} {status}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)