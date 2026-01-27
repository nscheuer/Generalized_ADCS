#!/usr/bin/env python3
"""
Run all 6 LP (non-planner) Monte Carlo tests for the Planner paper.

Tests:
1. 3MTQ+0RW LP Reduced Attitude
2. 3MTQ+0RW LP Full 180° Slew
3. 3MTQ+0RW LP Multi-Goal
4. 3MTQ+1RW LP Reduced Attitude
5. 3MTQ+1RW LP Full 180° Slew
6. 3MTQ+1RW LP Multi-Goal

Parameters match dissertation Chapter 7:
- 500 second duration
- Satellite params from Table mc_sat_params
- ISS orbit (429 km, 51.5° inclination)
"""
import subprocess
import sys
import os
from pathlib import Path

# Get script directory
SCRIPT_DIR = Path(__file__).parent

# List of test scripts to run
LP_TESTS = [
    "generate_mc_3mtq+0rw_lp_reduced.py",
    "generate_mc_3mtq+0rw_lp_full.py",
    "generate_mc_3mtq+0rw_lp_multi.py",
    "generate_mc_3mtq+1rw_lp_reduced.py",
    "generate_mc_3mtq+1rw_lp_full.py",
    "generate_mc_3mtq+1rw_lp_multi.py",
]


def run_test(script_name: str, python_path: str = None) -> bool:
    """Run a single test script."""
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    if python_path is None:
        python_path = sys.executable
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [python_path, str(script_path)],
            cwd=str(SCRIPT_DIR.parent.parent),  # Run from repo root
            timeout=1800,  # 30 minute timeout per test
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ {script_name} failed with error: {e}")
        return False


def main():
    print("=" * 60)
    print("Running all LP (non-planner) Monte Carlo tests")
    print("=" * 60)
    
    # Check for venv python
    venv_python = SCRIPT_DIR.parent.parent / "venv" / "bin" / "python"
    if venv_python.exists():
        python_path = str(venv_python)
        print(f"Using venv Python: {python_path}")
    else:
        python_path = sys.executable
        print(f"Using system Python: {python_path}")
    
    results = {}
    for test in LP_TESTS:
        results[test] = run_test(test, python_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for test, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {test}")
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
