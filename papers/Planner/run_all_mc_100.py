#!/usr/bin/env python3
"""
Run all 6 ALTRO planner MC scripts with 100 seeds each.
Headless — saves data but doesn't show plots.

Usage:
    python papers/Planner/run_all_mc_100.py          # All 6 configs, 100 seeds
    python papers/Planner/run_all_mc_100.py -n 10    # Quick test with 10 seeds
    python papers/Planner/run_all_mc_100.py --only 1rw_full 0rw_reduced  # Subset
"""
import os
os.environ['MPLBACKEND'] = 'Agg'  # Headless — must be before any matplotlib import
import sys
import time
import argparse
import subprocess

SCRIPTS = {
    "1rw_full":    "papers/Planner/generate_mc_3mtq+1rw_plan_full.py",
    "1rw_reduced": "papers/Planner/generate_mc_3mtq+1rw_plan_reduced.py",
    "1rw_multi":   "papers/Planner/generate_mc_3mtq+1rw_plan_multi.py",
    "0rw_full":    "papers/Planner/generate_mc_3mtq+0rw_plan_full.py",
    "0rw_reduced": "papers/Planner/generate_mc_3mtq+0rw_plan_reduced.py",
    "0rw_multi":   "papers/Planner/generate_mc_3mtq+0rw_plan_multi.py",
}

def main():
    parser = argparse.ArgumentParser(description="Run all 6 ALTRO planner MC configs")
    parser.add_argument("-n", "--num-runs", type=int, default=100, help="Number of MC seeds (default: 100)")
    parser.add_argument("--only", nargs="+", choices=list(SCRIPTS.keys()),
                        help="Run only these configs")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers (default: 4)")
    args = parser.parse_args()

    configs = args.only if args.only else list(SCRIPTS.keys())
    
    python = sys.executable
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print(f"=" * 70)
    print(f"ALTRO Planner Monte Carlo — {args.num_runs} seeds × {len(configs)} configs")
    print(f"Python: {python}")
    print(f"Root: {root}")
    print(f"=" * 70)
    
    results = {}
    total_start = time.time()
    
    for i, name in enumerate(configs):
        script = SCRIPTS[name]
        script_path = os.path.join(root, script)
        
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(configs)}] {name} — {script}")
        print(f"{'='*70}")
        
        t0 = time.time()
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        
        cmd = [python, script_path, "-n", str(args.num_runs)]
        
        try:
            proc = subprocess.run(
                cmd, cwd=root, env=env,
                timeout=7200,  # 2 hour timeout per config
                capture_output=False,  # Stream output live
            )
            elapsed = time.time() - t0
            success = (proc.returncode == 0)
            results[name] = {"elapsed": elapsed, "success": success, "returncode": proc.returncode}
            
            status = "✅" if success else f"❌ (rc={proc.returncode})"
            print(f"\n→ {name}: {status} in {elapsed:.0f}s ({elapsed/60:.1f}min)")
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            results[name] = {"elapsed": elapsed, "success": False, "returncode": -1}
            print(f"\n→ {name}: ⏰ TIMEOUT after {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = {"elapsed": elapsed, "success": False, "returncode": -1}
            print(f"\n→ {name}: 💥 ERROR: {e}")
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"SUMMARY — {args.num_runs} seeds each")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Status':<8} {'Time':>10}")
    print(f"{'-'*15} {'-'*8} {'-'*10}")
    for name in configs:
        r = results[name]
        status = "✅" if r["success"] else "❌"
        t = f"{r['elapsed']:.0f}s"
        print(f"{name:<15} {status:<8} {t:>10}")
    
    n_ok = sum(1 for r in results.values() if r["success"])
    print(f"\n{n_ok}/{len(configs)} succeeded in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min total)")
    print(f"\nData saved to: papers/Planner/output_data/")

if __name__ == "__main__":
    main()
